# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanFLF2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        memory_profiler=None,
        quantization=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.memory_profiler = memory_profiler

        base_memory = torch.cuda.memory_allocated()
        shard_fn = partial(shard_model, device_id=device_id)
        
        if self.memory_profiler:
            # Record T5 model loading start time
            t5_load_start_time = time.time()
            self.memory_profiler.log_event('t5_load_start', {'timestamp': t5_load_start_time})
        
        # Choose device for T5 model based on t5_cpu parameter
        t5_device = torch.device('cpu') if t5_cpu else self.device
        
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=t5_device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )
        if self.memory_profiler:
            # Record T5 model loading end time and calculate duration
            t5_load_end_time = time.time()
            t5_load_duration = t5_load_end_time - t5_load_start_time
            gpu_memory_after_t5 = torch.cuda.memory_allocated()
            
            if t5_cpu:
                # T5 on CPU: no GPU memory used, set base_memory to current memory
                self.memory_profiler.log_event('t5_load_end', {
                    'timestamp': t5_load_end_time,
                    'duration': t5_load_duration,
                    'base_memory': base_memory
                })
                self.memory_profiler.log_event('t5_loaded', {'base_memory': 0, 'incremental_memory': 0})
            else:
                # T5 on GPU: use base_memory from before T5 loading for proper incremental calculation
                self.memory_profiler.log_event('t5_load_end', {
                    'timestamp': t5_load_end_time,
                    'duration': t5_load_duration,
                    'base_memory': base_memory
                })
                self.memory_profiler.log_event('t5_loaded', {'base_memory': base_memory})
            base_memory = gpu_memory_after_t5

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)
        if self.memory_profiler:
            self.memory_profiler.log_event('vae_loaded', {'base_memory': base_memory})
            base_memory = torch.cuda.memory_allocated()

        self.clip = CLIPModel(
            dtype=config.clip_dtype,
            device=self.device,
            checkpoint_path=os.path.join(checkpoint_dir,
                                         config.clip_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer))
        if self.memory_profiler:
            self.memory_profiler.log_event('clip_loaded', {'base_memory': base_memory})
            base_memory = torch.cuda.memory_allocated()

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        model_kwargs = {}
        if quantization:
            model_kwargs['load_in_4bit'] = True
        self.model = WanModel.from_pretrained(checkpoint_dir, **model_kwargs)
        self.model.eval().requires_grad_(False)

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)
        if self.memory_profiler:
            self.memory_profiler.log_event('dit_loaded', {'base_memory': base_memory})
            base_memory = torch.cuda.memory_allocated()

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 first_frame,
                 last_frame,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=16,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.5,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from input first-last frame and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            first_frame (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            last_frame (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
                [NOTE] If the sizes of first_frame and last_frame are mismatched, last_frame will be cropped & resized
                to match first_frame.
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """
        first_frame_size = first_frame.size
        last_frame_size = last_frame.size
        first_frame = TF.to_tensor(first_frame).sub_(0.5).div_(0.5).to(
            self.device)
        last_frame = TF.to_tensor(last_frame).sub_(0.5).div_(0.5).to(
            self.device)

        F = frame_num
        first_frame_h, first_frame_w = first_frame.shape[1:]
        aspect_ratio = first_frame_h / first_frame_w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        first_frame_h = lat_h * self.vae_stride[1]
        first_frame_w = lat_w * self.vae_stride[2]
        if first_frame_size != last_frame_size:
            # 1. resize
            last_frame_resize_ratio = max(
                first_frame_size[0] / last_frame_size[0],
                first_frame_size[1] / last_frame_size[1])
            last_frame_size = [
                round(last_frame_size[0] * last_frame_resize_ratio),
                round(last_frame_size[1] * last_frame_resize_ratio),
            ]
            # 2. center crop
            last_frame = TF.center_crop(last_frame, last_frame_size)

        max_seq_len = ((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, (F - 1) // 4 + 1,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, 81, lat_h, lat_w, device=self.device)
        msk[:, 1:-1] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        self.clip.model.to(self.device)
        clip_context = self.clip.visual(
            [first_frame[:, None, :, :], last_frame[:, None, :, :]])
        if offload_model:
            self.clip.model.cpu()

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    first_frame[None].cpu(),
                    size=(first_frame_h, first_frame_w),
                    mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 2, first_frame_h, first_frame_w),
                torch.nn.functional.interpolate(
                    last_frame[None].cpu(),
                    size=(first_frame_h, first_frame_w),
                    mode='bicubic').transpose(0, 1),
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        if self.memory_profiler:
            base_memory = torch.cuda.memory_allocated()
            self.memory_profiler.log_event('before_generate', {'base_memory': base_memory})

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise

            arg_c = {
                'context': [context[0]],
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            arg_null = {
                'context': context_null,
                'clip_fea': clip_context,
                'seq_len': max_seq_len,
                'y': [y],
            }

            if offload_model:
                torch.cuda.empty_cache()

            self.model.to(self.device)
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0].to(
                        torch.device('cpu') if offload_model else self.device)
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                if self.memory_profiler:
                    self.memory_profiler.log_event(f'step_{_}', {'base_memory': base_memory})

                latent = latent.to(
                    torch.device('cpu') if offload_model else self.device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = [latent.to(self.device)]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)
                if self.memory_profiler:
                    self.memory_profiler.log_event('after_decode', {'base_memory': base_memory})

        del noise, latent
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
