# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys

import time
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

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
        quantization=False,
        memory_profiler=None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

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
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.quantization = quantization

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.memory_profiler = memory_profiler

        if self.memory_profiler:
            base_memory = torch.cuda.memory_allocated()
            # Record T5 model loading start time
            t5_load_start_time = time.time()
            self.memory_profiler.log_event('t5_load_start', {'timestamp': t5_load_start_time})

        # Choose device for T5 model based on t5_cpu parameter
        t5_device = 'cpu' if t5_cpu else self.device
        
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=t5_device,
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)
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

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        if self.quantization:
            self.model = WanModel.from_pretrained(
                checkpoint_dir, 
                load_in_4bit=True
            )
        else:
            self.model = WanModel.from_pretrained(checkpoint_dir)
            self.model.to(self.device)
        self.model.eval().requires_grad_(False)

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
            self.model.to(self.device)

        if self.memory_profiler:
            self.memory_profiler.log_event('dit_loaded', {'base_memory': base_memory})
            base_memory = torch.cuda.memory_allocated()

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1):
        if self.memory_profiler:
            self.memory_profiler.log_event('generate_start', {'timestamp': time.time()})
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """


        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # preprocess
        base_memory_before_encode = torch.cuda.memory_allocated()
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            if self.memory_profiler:
                self.memory_profiler.log_event('t5_encode_start', {'timestamp': time.time()})
            context = self.text_encoder([input_prompt])
            context_null = self.text_encoder([n_prompt])
            if self.memory_profiler:
                self.memory_profiler.log_event('t5_encode_end', {'timestamp': time.time(), 'base_memory': base_memory_before_encode})
        else:
            # T5 CPU mode: model runs on CPU, no GPU memory usage for T5
            if self.memory_profiler:
                self.memory_profiler.log_event('t5_encode_start', {'timestamp': time.time()})
            # Move T5 model to CPU for encoding
            self.text_encoder.model.cpu()
            context = self.text_encoder([input_prompt])
            context_null = self.text_encoder([n_prompt])
            # Move results back to GPU
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
            if self.memory_profiler:
                # In CPU mode, T5 uses 0 GPU memory
                self.memory_profiler.log_event('t5_encode_end', {'timestamp': time.time(), 'base_memory': base_memory_before_encode})

        if self.memory_profiler:
            if self.t5_cpu:
                self.memory_profiler.log_event('kv_cache', metadata={'model_name': 'T5', 'base_memory': 0})
            else:
                self.memory_profiler.log_event('kv_cache', metadata={'model_name': 'T5', 'base_memory': base_memory_before_encode})



        # forward pass
        if self.memory_profiler:
            base_memory = torch.cuda.memory_allocated()
            self.memory_profiler.log_event('before_generate', {'base_memory': base_memory})

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]
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
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            if self.memory_profiler:
                self.memory_profiler.log_event('dit_forward_start', {'timestamp': time.time()})
            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                self.model.to(self.device)
                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                if self.memory_profiler:
                    self.memory_profiler.log_event(f'step_{_}', {'base_memory': base_memory})

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
            if self.memory_profiler:
                self.memory_profiler.log_event('dit_forward_end', {'timestamp': time.time()})

            z = latents

            if self.memory_profiler:
                base_memory_after_forward = torch.cuda.memory_allocated()
                self.memory_profiler.log_event('kv_cache', metadata={'model_name': 'DiT', 'base_memory': base_memory_after_forward})

        if self.t5_cpu:
            self.model.cpu()
            torch.cuda.empty_cache()

        videos = None
        if self.rank == 0:
            base_memory_before_decode = torch.cuda.memory_allocated()
            if self.memory_profiler:
                self.memory_profiler.log_event('vae_decode_start', {'timestamp': time.time()})
            videos = self.vae.decode(z)
            if self.memory_profiler:
                self.memory_profiler.log_event('vae_decode_end', {'timestamp': time.time(), 'base_memory': base_memory_before_decode})
                self.memory_profiler.log_event('generate_end', {'timestamp': time.time()})
            if self.memory_profiler:
                self.memory_profiler.log_event('after_decode', {'base_memory': base_memory})

        # release memory
        del context
        del context_null
        del noise
        del latents
        del z
        gc.collect()
        torch.cuda.empty_cache()

        return videos
