# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
import yaml
import time
import json
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda import memory_stats, memory_allocated, memory_reserved

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt":
            "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
    "flf2v-14B": {
        "prompt":
            "CG动画风格，一只蓝色的小鸟从地面起飞，煽动翅膀。小鸟羽毛细腻，胸前有独特的花纹，背景是蓝天白云，阳光明媚。镜跟随小鸟向上移动，展现出小鸟飞翔的姿态和天空的广阔。近景，仰视视角。",
        "first_frame":
            "examples/flf2v_input_first_frame.png",
        "last_frame":
            "examples/flf2v_input_last_frame.png",
    },
    "vace-1.3B": {
        "src_ref_images":
            'examples/girl.png,examples/snake.png',
        "prompt":
            "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    },
    "vace-14B": {
        "src_ref_images":
            'examples/girl.png,examples/snake.png',
        "prompt":
            "在一个欢乐而充满节日气氛的场景中，穿着鲜艳红色春服的小女孩正与她的可爱卡通蛇嬉戏。她的春服上绣着金色吉祥图案，散发着喜庆的气息，脸上洋溢着灿烂的笑容。蛇身呈现出亮眼的绿色，形状圆润，宽大的眼睛让它显得既友善又幽默。小女孩欢快地用手轻轻抚摸着蛇的头部，共同享受着这温馨的时刻。周围五彩斑斓的灯笼和彩带装饰着环境，阳光透过洒在她们身上，营造出一个充满友爱与幸福的新年氛围。"
    }
}


class MemoryTracker:
    def __init__(self, logger):
        self.logger = logger
        self.baseline_stats = {}

    def set_baseline(self):
        torch.cuda.synchronize()
        self.baseline_stats = memory_stats()
        self.log_memory("Baseline")

    def log_memory(self, tag):
        torch.cuda.synchronize()
        current_stats = memory_stats()
        
        allocated = current_stats['allocated_bytes.all.current'] / (1024**2)
        reserved = current_stats['reserved_bytes.all.current'] / (1024**2)
        
        runtime_peak = (current_stats['allocated_bytes.all.peak'] - self.baseline_stats.get('allocated_bytes.all.peak', 0)) / (1024**2)
        
        self.logger.info(f"[Memory] {tag}: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB, Runtime Peak={runtime_peak:.2f}MB")
        return {'allocated': allocated, 'reserved': reserved, 'runtime_peak': runtime_peak}

    def get_memory_stats(self):
        return torch.cuda.memory_stats()

class MemoryProfiler:
    def __init__(self, name, logger, trace_path):
        self.name = name
        self.logger = logger
        self.trace_path = trace_path
        self.events = []
        self.memory_tracker = MemoryTracker(logger)
        self.profiler = None
        self.last_peak_stats = {}

    def start_profiling(self):
        self.logger.info(f"Starting memory profiling for '{self.name}'")
        self.memory_tracker.set_baseline()
        self.last_peak_stats = torch.cuda.memory_stats()
        if self.trace_path:
            self.profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.trace_path),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            self.profiler.start()

    def stop_profiling(self):
        if self.profiler:
            self.profiler.stop()
        self.logger.info(f"Finished memory profiling for '{self.name}'")

    def log_event(self, event_name, metadata=None):
        torch.cuda.synchronize()
        current_stats = memory_stats()
        event_data = {"event": event_name}
        log_message = f"[Event] {event_name}"

        if metadata and 'model_name' in metadata:
            event_data['model_name'] = metadata['model_name']
            log_message += f" ({metadata['model_name']})"

        # Calculate incremental memory (tensor memory)
        incremental_memory_bytes = 0
        if metadata and 'base_memory' in metadata:
            base_memory = metadata['base_memory']
            current_memory = torch.cuda.memory_allocated()
            incremental_memory_bytes = current_memory - base_memory
            event_data["incremental_memory"] = incremental_memory_bytes

        # Calculate peak increase since last event
        peak_allocated_since_last = current_stats['allocated_bytes.all.peak'] - self.last_peak_stats.get('allocated_bytes.all.peak', 0)
        peak_reserved_since_last = current_stats['reserved_bytes.all.peak'] - self.last_peak_stats.get('reserved_bytes.all.peak', 0)

        # Runtime overhead is the part of the peak that is not tensor memory
        cuda_runtime_peak_bytes = peak_allocated_since_last - incremental_memory_bytes
        pytorch_runtime_peak_bytes = peak_reserved_since_last - incremental_memory_bytes

        event_data['cuda_runtime_peak'] = cuda_runtime_peak_bytes / (1024**2)
        event_data['pytorch_runtime_peak'] = pytorch_runtime_peak_bytes / (1024**2)
        
        log_message += f": CUDA Runtime Peak={event_data['cuda_runtime_peak']:.2f}MB, PyTorch Runtime Peak={event_data['pytorch_runtime_peak']:.2f}MB"

        if metadata and 'base_memory' in metadata:
            log_message += f", Incremental Memory = {incremental_memory_bytes / (1024*1024):.2f} MB"
        else:
            peak_memory = torch.cuda.max_memory_allocated()
            event_data["peak_memory"] = peak_memory
            log_message += f", Peak Memory = {peak_memory / (1024*1024):.2f} MB"

        self.events.append(event_data)
        self.logger.info(log_message)
        
        # Update last stats for next event
        self.last_peak_stats = current_stats



def _validate_args(args):
    # Basic check
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.task in WAN_CONFIGS, f"Unsupport task: {args.task}"
    assert args.task in EXAMPLE_PROMPT, f"Unsupport task: {args.task}"

    # The default sampling steps are 40 for image-to-video tasks and 50 for text-to-video tasks.
    if args.sample_steps is None:
        args.sample_steps = 50
        if "i2v" in args.task:
            args.sample_steps = 40

    if args.sample_shift is None:
        args.sample_shift = 5.0
        if "i2v" in args.task and args.size in ["832*480", "480*832"]:
            args.sample_shift = 3.0
        elif "flf2v" in args.task or "vace" in args.task:
            args.sample_shift = 16

    # The default number of frames are 1 for text-to-image tasks and 81 for other tasks.
    if args.frame_num is None:
        args.frame_num = 1 if "t2i" in args.task else 81

    # T2I frame_num check
    if "t2i" in args.task:
        assert args.frame_num == 1, f"Unsupport frame_num {args.frame_num} for task {args.task}"

    args.base_seed = args.base_seed if args.base_seed >= 0 else random.randint(
        0, sys.maxsize)
    # Size check
    assert args.size in SUPPORTED_SIZES[
        args.
        task], f"Unsupport size {args.size} for task {args.task}, supported sizes are: {', '.join(SUPPORTED_SIZES[args.task])}"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="The path to the configuration file."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--size",
        type=str,
        default="1280*720",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video. For the I2V task, the aspect ratio of the output video will follow that of the input image."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=None,
        help="How many frames to sample from a image or video. The number should be 4n+1"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument(
        "--offload_model",
        type=str2bool,
        default=None,
        help="Whether to offload the model to CPU after each model forward, reducing GPU memory usage."
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--save_file",
        type=str,
        default=None,
        help="The file to save the generated image or video to.")
    parser.add_argument(
        "--src_video",
        type=str,
        default=None,
        help="The file of the source video. Default None.")
    parser.add_argument(
        "--src_mask",
        type=str,
        default=None,
        help="The file of the source mask. Default None.")
    parser.add_argument(
        "--src_ref_images",
        type=str,
        default=None,
        help="The file list of the source reference images. Separated by ','. Default None."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to generate the image or video from.")
    parser.add_argument(
        "--use_prompt_extend",
        action="store_true",
        default=False,
        help="Whether to use prompt extend.")
    parser.add_argument(
        "--prompt_extend_method",
        type=str,
        default="local_qwen",
        choices=["dashscope", "local_qwen"],
        help="The prompt extend method to use.")
    parser.add_argument(
        "--prompt_extend_model",
        type=str,
        default=None,
        help="The prompt extend model to use.")
    parser.add_argument(
        "--prompt_extend_target_lang",
        type=str,
        default="zh",
        choices=["zh", "en"],
        help="The target language of prompt extend.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="The seed to use for generating the image or video.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="[image to video] The image to generate the video from.")
    parser.add_argument(
        "--first_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (first frame) to generate the video from."
    )
    parser.add_argument(
        "--last_frame",
        type=str,
        default=None,
        help="[first-last frame to video] The image (last frame) to generate the video from."
    )
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--sample_steps", type=int, default=None, help="The sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=None,
        help="Sampling shift factor for flow matching schedulers.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")

    args = parser.parse_args()

    _validate_args(args)

    return args


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate(args, memory_profiler=None):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    _init_logging(rank)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True
        logging.info(
            f"offload_model is not specified, set to {args.offload_model}.")
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert args.ulysses_size * args.ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size())

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )

    if args.use_prompt_extend:
        if args.prompt_extend_method == "dashscope":
            prompt_expander = DashScopePromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task or "flf2v" in args.task)
        elif args.prompt_extend_method == "local_qwen":
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl="i2v" in args.task,
                device=rank)
        else:
            raise NotImplementedError(
                f"Unsupport prompt_extend_method: {args.prompt_extend_method}")

    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert cfg.num_heads % args.ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{args.ulysses_size=}`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    if "t2v" in args.task or "t2i" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanT2V pipeline.")
        logging.info(f"WanT2V config: {cfg}")
        wan_t2v = wan.WanT2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            memory_profiler=memory_profiler,
        )
        if memory_profiler:
            memory_profiler.log_event('model_loaded', {'timestamp': time.time()})

        logging.info(
            f"Generating {'image' if 't2i' in args.task else 'video'} ...")
        if memory_profiler:
            base_memory = torch.cuda.memory_allocated()
            memory_profiler.log_event('before_forward_pass', {'base_memory': base_memory, 'model_name': 't2v', 'timestamp': time.time()})
        video = wan_t2v.generate(
            args.prompt,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        if memory_profiler:
            memory_profiler.log_event('after_forward_pass', {'base_memory': base_memory, 'model_name': 't2v', 'timestamp': time.time()})

    elif "i2v" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.image is None:
            args.image = EXAMPLE_PROMPT[args.task]["image"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input image: {args.image}")

        img = Image.open(args.image).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=img,
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanI2V pipeline.")
        wan_i2v = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            memory_profiler=memory_profiler,
        )
        if memory_profiler:
            memory_profiler.log_event('model_loaded')

        logging.info("Generating video ...")
        if memory_profiler:
            base_memory = torch.cuda.memory_allocated()
            memory_profiler.log_event('before_forward_pass', {'base_memory': base_memory, 'model_name': 'i2v'})
        video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        if memory_profiler:
            memory_profiler.log_event('after_forward_pass', {'base_memory': base_memory, 'model_name': 'i2v'})
    elif "flf2v" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
        if args.first_frame is None or args.last_frame is None:
            args.first_frame = EXAMPLE_PROMPT[args.task]["first_frame"]
            args.last_frame = EXAMPLE_PROMPT[args.task]["last_frame"]
        logging.info(f"Input prompt: {args.prompt}")
        logging.info(f"Input first frame: {args.first_frame}")
        logging.info(f"Input last frame: {args.last_frame}")
        first_frame = Image.open(args.first_frame).convert("RGB")
        last_frame = Image.open(args.last_frame).convert("RGB")
        if args.use_prompt_extend:
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt_output = prompt_expander(
                    args.prompt,
                    tar_lang=args.prompt_extend_target_lang,
                    image=[first_frame, last_frame],
                    seed=args.base_seed)
                if prompt_output.status == False:
                    logging.info(
                        f"Extending prompt failed: {prompt_output.message}")
                    logging.info("Falling back to original prompt.")
                    input_prompt = args.prompt
                else:
                    input_prompt = prompt_output.prompt
                input_prompt = [input_prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating WanFLF2V pipeline.")
        wan_flf2v = wan.WanFLF2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            memory_profiler=memory_profiler,
        )

        logging.info("Generating video ...")
        if memory_profiler:
            base_memory = torch.cuda.memory_allocated()
            memory_profiler.log_event('before_forward_pass', {'base_memory': base_memory, 'model_name': 'flf2v'})
        video = wan_flf2v.generate(
            args.prompt,
            first_frame,
            last_frame,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        if memory_profiler:
            memory_profiler.log_event('after_forward_pass', {'base_memory': base_memory, 'model_name': 'flf2v'})
    elif "vace" in args.task:
        if args.prompt is None:
            args.prompt = EXAMPLE_PROMPT[args.task]["prompt"]
            args.src_video = EXAMPLE_PROMPT[args.task].get("src_video", None)
            args.src_mask = EXAMPLE_PROMPT[args.task].get("src_mask", None)
            args.src_ref_images = EXAMPLE_PROMPT[args.task].get(
                "src_ref_images", None)

        logging.info(f"Input prompt: {args.prompt}")
        if args.use_prompt_extend and args.use_prompt_extend != 'plain':
            logging.info("Extending prompt ...")
            if rank == 0:
                prompt = prompt_expander.forward(args.prompt)
                logging.info(
                    f"Prompt extended from '{args.prompt}' to '{prompt}'")
                input_prompt = [prompt]
            else:
                input_prompt = [None]
            if dist.is_initialized():
                dist.broadcast_object_list(input_prompt, src=0)
            args.prompt = input_prompt[0]
            logging.info(f"Extended prompt: {args.prompt}")

        logging.info("Creating VACE pipeline.")
        wan_vace = wan.WanVace(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            memory_profiler=memory_profiler,
        )

        src_video, src_mask, src_ref_images = wan_vace.prepare_source(
            [args.src_video], [args.src_mask], [
                None if args.src_ref_images is None else
                args.src_ref_images.split(',')
            ], args.frame_num, SIZE_CONFIGS[args.size], device)

        logging.info(f"Generating video...")
        if memory_profiler:
            base_memory = torch.cuda.memory_allocated()
            memory_profiler.log_event('before_forward_pass', {'base_memory': base_memory, 'model_name': 'vace'})
        video = wan_vace.generate(
            args.prompt,
            src_video,
            src_mask,
            src_ref_images,
            size=SIZE_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
        if memory_profiler:
            memory_profiler.log_event('after_forward_pass', {'base_memory': base_memory, 'model_name': 'vace'})
    else:
        raise ValueError(f"Unkown task type: {args.task}")

    video_tensor = video
    if isinstance(video, list):
        video_tensor = video[0]

    if rank == 0:
        if args.save_file is None:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            formatted_prompt = args.prompt.replace(" ", "_").replace("/",
                                                                     "_")[:50]
            suffix = '.png' if "t2i" in args.task else '.mp4'
            task_name = getattr(args, 'task', 't2v-1.3B')
            args.save_file = f"{task_name}_{args.size.replace('*','x') if sys.platform=='win32' else args.size}_{args.ulysses_size}_{args.ring_size}_{formatted_prompt}_{formatted_time}" + suffix

        if "t2i" in args.task:
            logging.info(f"Saving generated image to {args.save_file}")
            cache_image(
                tensor=video_tensor.squeeze(1)[None],
                save_file=args.save_file,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
        else:
            logging.info(f"Saving generated video to {args.save_file}")
            cache_video(
                tensor=video_tensor[None],
                save_file=args.save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
    logging.info("Finished.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared PyTorch CUDA cache.")
    return video_tensor


if __name__ == '__main__':
    args = _parse_args()
    memory_profiler = None
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update args with config values
        if 'model_config' in config:
            for key, value in config['model_config'].items():
                setattr(args, key, value)
        if 'optimization' in config:
            for key, value in config['optimization'].items():
                setattr(args, key, value)
        if 'logging' in config:
            for key, value in config['logging'].items():
                setattr(args, key, value)

        # Setup logger
        log_path = os.path.join(config['logging']['trace_path'], 'memory_profile.log')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = logging.getLogger(config['name'])
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        memory_profiler = MemoryProfiler(config['name'], logger, config['logging']['trace_path'])
        memory_profiler.start_profiling()
        memory_profiler.log_event('init')

    video = generate(args, memory_profiler=memory_profiler)

    if memory_profiler:
        memory_profiler.log_event('inference_end')
        memory_profiler.stop_profiling()
