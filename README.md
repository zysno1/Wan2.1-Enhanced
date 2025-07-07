# Wan2.1-Enhanced 显存优化与性能分析

本项目基于 Wan2.1 文本生成视频模型，专注于显存优化技术的研究与实现。通过集成多种显存优化策略，我们旨在降低模型的显存需求，使其能够在资源受限的环境中高效运行。

## 目录

1. [项目背景](#1-项目背景)
2. [测试目标](#2-测试目标)
3. [使用方法](#3-使用方法)
4. [测试场景和报告](#4-测试场景和报告)
5. [技术方案](#5-技术方案)
6. [工具设计](#6-工具设计)
7. [参数说明](#7-参数说明)
8. [模型架构与默认配置](#8-模型架构与默认配置)

## 1. 项目背景

Wan2.1 是一个视频生成项目，在生成过程中涉及多个大型模型的加载和推理，显存使用是一个关键的性能指标。为了优化显存使用，需要对不同推理配置下的显存消耗进行系统的测试和分析。并沉淀视频生成模型的显存估算公式

本项目的工作目录为 `/workspace/Wan2.1-Enhanced`，所有的代码、工具、文档和生成的日志都应存放在此目录下。

步骤 1: 文本编码 (Text Encoding)

主要工作： 将输入的文字（Prompt）翻译成可供机器理解的数学指令。

所用模型： 文本编码器 (Text Encoder)，如 CLIP 或 T5。

步骤 2: 潜在视频生成 (Latent Video Generation)

主要工作： 在一个压缩的潜在空间中，根据文本指令，从随机噪声出发，生成视频的动态结构。

所用模型： 扩散Transformer (DiT)。

步骤 3: 视频解码 (Video Decoding)

主要工作： 将潜在空间中的视频结构，解码还原成最终用户可见的像素视频文件。

所用模型： VAE 解码器 (VAE Decoder)，可选配视频超分模型。

## 2. 测试目标

- 精确测量不同推理配置下的显存使用情况
- 识别单次运行中显存使用的瓶颈
- 为后续优化提供数据支持和方向指导


## 3. 使用方法

### 3.1 测试执行

1. **执行测试**

   所有测试都应在指定的虚拟环境下执行，以确保依赖隔离和环境一致性。

   ```bash
   # 激活虚拟环境
   source /workspace/venv/bin/activate

   # 执行测试脚本
   # （请注意：run_memory_tests.sh 脚本需要您根据要运行的配置进行相应修改）
   bash tests/scripts/run_memory_tests.sh
   ```

2. **数据分析**

   测试脚本执行完成后，会自动调用 `tests/scripts/analyze.py` 来分析日志并生成增强的内存分析报告。

   报告将以 `enhanced_memory_analysis_report_YYYYMMDD_HHMMSS.md` 的格式保存在项目根目录下。


## 4. 测试场景和报告

### 4.1 场景一：1.3B模型生成5秒480P视频，L40S单卡

#### 4.1.1 第一部分：测试基线

**配置 (`baseline.yaml`)**

```yaml
name: "baseline"
description: "Baseline configuration without optimizations"

model_config:
  task: "t2v-1.3B"
  size: "832*480"
  load_strategy: "block"     # 模型加载策略：full/block
  offload_model: "false"    # 卸载到CPU 
  precision: "fp16"         # 计算精度：fp32/fp16/bf16
  device: "cuda"           # 运行设备
  offload: false           # CPU 卸载开关
  ckpt_dir: /workspace/Wan2.1-Enhanced/Wan2.1-T2V-1.3B

optimization:
  attention_slicing: false   # 注意力切片
  gradient_checkpointing: false
  batch_size: 1
  micro_batch_size: 1
  parallel_degree: 1       # 模型并行度

logging:
  profile_memory: true
  log_interval: 10         # 记录间隔（步数）
  trace_path: "profiler_logs/baseline"
```

**测试提示词**
```
Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.
```

**执行命令**
```bash
bash tests/scripts/run_memory_tests.sh tests/configs/baseline.yaml
```

**测试结果**
 
 - **显存峰值**：23.12 GB（总峰值显存）
 - **模型组件显存**：19.89 GB (基础显存: 16.76 GB + 激活显存: 3.61 GB + KV缓存: 0.03 GB)
 - **纯运行时开销**：3.28 GB
 - **生成时间**：242.15 秒
   - 模型加载：89.23 秒
   - T5编码：0.88 秒  
   - DiT前向推理：223.94 秒
   - VAE解码：9.10 秒
 - **详细显存分布**：
   - DiT模型：5439.81 MB (基础) + 3226.79 MB (激活)
   - T5模型：10836.48 MB (基础) + 17.00 MB (激活) + 17.00 MB (KV缓存)
   - VAE模型：485.95 MB (基础) + 370.20 MB (激活)
   - PyTorch/CUDA运行时开销：3281.73 MB
 
 **生成的视频文件**
 
 - 输出视频：`t2v-1.3B_832*480_1_1_Two_anthropomorphic_cats_in_comfy_boxing_gear_and__20250706_174858.mp4`
 - 视频规格：832x480, 5秒, 16fps
 - 实际生成视频预览：

**Video Demos**

<div align="center">
  <video src="https://github.com/user-attachments/assets/d801973a-d037-4917-a79e-430a55f60b36" width="70%" poster=""> </video>
</div>

*展示了两只拟人化猫咪穿着舒适的拳击装备在聚光灯舞台上激烈对战的场景，动作流畅自然，细节丰富*

#### 4.1.2 第二部分：CPU模型卸载

**配置 (`cpu_offload.yaml`)**

```yaml
name: "CPU-offload"
description: "T5 Model offload to CPU"

model_config:
  task: "t2v-1.3B"
  size: "832*480"
  load_strategy: "block"     # 模型加载策略：full/block
  offload_model: true       # 卸载到CPU
  precision: "fp16"         # 计算精度：fp32/fp16/bf16
  device: "cuda"           # 运行设备
  ckpt_dir: /workspace/Wan2.1-Enhanced/Wan2.1-T2V-1.3B

optimization:
  attention_slicing: false   # 注意力切片
  gradient_checkpointing: false
  batch_size: 1
  micro_batch_size: 1
  parallel_degree: 1       # 模型并行度

logging:
  profile_memory: true
  log_interval: 10         # 记录间隔（步数）
  trace_path: "profiler_logs"
```

**测试提示词**
```
Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.
```

**执行命令**
```bash
bash tests/scripts/run_memory_tests.sh tests/configs/cpu_offload.yaml
```

**测试结果**
 
 - **显存峰值**：8.20 GB（总峰值显存）
 - **模型组件显存**：5.92 GB (基础显存) + 9.90 GB (激活显存) + 5.92 GB (KV缓存)
 - **纯运行时开销**：2.47 GB
 - **生成时间**：633.89 秒
   - 模型加载：142.58 秒
   - T5编码：397.61 秒
   - DiT前向推理：224.19 秒
   - VAE解码：9.08 秒
 - **详细显存分布**：
   - DiT模型：5413.04 MB (基础) + 3608.07 MB (激活)
   - T5模型：0 MB (基础) + 5920.00 MB (激活) + 5920.00 MB (KV缓存)
   - VAE模型：505.75 MB (基础) + 370.20 MB (激活)
   - PyTorch/CUDA运行时开销：2472.56 MB
 
 #### 4.1.3 第三部分：注意力切片

**配置 (`attention_slicing.yaml`)**

```yaml
name: "attention_slicing"
description: "Attention slicing is an optimization technique that reduces the memory footprint of AI inference by breaking down the computationally intensive attention mechanism into sequential, smaller chunks, enabling models to run on hardware with limited VRAM at the cost of slightly slower performance."

model_config:
  task: "t2v-1.3B"
  size: "832*480"
  load_strategy: "block"     # 模型加载策略：full/block
  offload_model: true       # 卸载到CPU
  precision: "fp16"         # 计算精度：fp32/fp16/bf16
  device: "cuda"           # 运行设备
  ckpt_dir: /workspace/Wan2.1-Enhanced/Wan2.1-T2V-1.3B

optimization:
  attention_slicing: true   # 注意力切片
  gradient_checkpointing: false
  batch_size: 1
  micro_batch_size: 1
  parallel_degree: 1       # 模型并行度

logging:
  profile_memory: true
  log_interval: 10         # 记录间隔（步数）
  trace_path: "profiler_logs"
  ```

**测试提示词**
```
Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.
```

**执行命令**
```bash
bash tests/scripts/run_memory_tests.sh tests/configs/attention_slicing.yaml
```

**测试结果**
 
 - 显存峰值：待测试（CPU卸载+注意力切片组合优化）
 - 生成时间：待测试（预计比单纯CPU卸载略长）
 - 显存分析报告：待生成
 
 **生成的视频文件**
 
 - 输出视频：`outputs/attention_slicing_video.mp4`
 - 视频规格：832x480, 5秒, 16fps
 - 视频预览：
   
   ![注意力切片优化视频](outputs/attention_slicing_video.gif)
   
   *展示了都市夜景的繁华与动感，注意力切片技术在保持视频质量的同时有效降低了显存峰值*
 
 
 
 ### 4.2 场景二：多卡并行测试，2x A100
 
#### 4.2.1 FSDP并行策略


## 5. 技术方案

### 5.1 显存消耗采集与性能分析技术方案

#### 5.1.1 显存监控架构设计

为了精确测量和分析显存使用情况，我们设计了一套完整的显存监控系统，将显存消耗拆分为 **基础显存** 和 **激活显存** 两部分：

- **基础显存**：模型加载后，在推理前占用的显存，主要包括模型权重、优化器状态等
- **激活显存**：在模型前向传播过程中，因计算产生的中间变量（激活值）所占用的显存

#### 5.1.2 技术实现方案

**1. 核心监控组件**

我们实现了两个核心类来处理显存监控：

```python
class MemoryTracker:
    """基础显存跟踪器，负责记录和格式化显存信息"""
    def __init__(self, logger):
        self.logger = logger

    def log_memory(self, tag):
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        self.logger.info(f"[Memory] {tag}: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")
        return {'allocated': allocated, 'reserved': reserved}

class MemoryProfiler:
    """高级显存分析器，集成PyTorch Profiler和事件记录"""
    def __init__(self, config_name, logger, output_dir):
        self.config_name = config_name
        self.logger = logger
        self.output_dir = output_dir
        self.memory_tracker = MemoryTracker(logger)
        self.events = []
        self.profiler = None
```

**2. 增量显存测量**

通过在关键节点记录基准显存，计算每个组件的增量显存消耗：

```python
def log_event(self, event_name, metadata=None):
    torch.cuda.synchronize()  # 确保CUDA操作完成
    if metadata and 'base_memory' in metadata:
        base_memory = metadata['base_memory']
        current_memory = torch.cuda.memory_allocated()
        incremental_memory = current_memory - base_memory
        self.events.append({"event": event_name, "incremental_memory": incremental_memory})
    else:
        peak_memory = torch.cuda.max_memory_allocated()
        self.events.append({"event": event_name, "peak_memory": peak_memory})
```

**3. 模型加载监控实现**

在模型加载的各个阶段插入监控点：

```python
# T5文本编码器加载
base_memory = torch.cuda.memory_allocated()
self.text_encoder = T5EncoderModel(...)
self.memory_profiler.log_event('t5_loaded', {'base_memory': base_memory})

# VAE模型加载
base_memory = torch.cuda.memory_allocated()
self.vae = WanVAE(...)
self.memory_profiler.log_event('vae_loaded', {'base_memory': base_memory})

# DiT模型加载
base_memory = torch.cuda.memory_allocated()
self.model = WanModel.from_pretrained(checkpoint_dir)
self.memory_profiler.log_event('dit_loaded', {'base_memory': base_memory})
```

**4. PyTorch Profiler集成**

集成PyTorch原生性能分析器，提供详细的计算图和显存分析：

```python
def start_profiling(self):
    self.profiler = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=3
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
    self.profiler.start()
```

**5. 数据持久化**

所有显存事件自动保存为JSON格式，便于后续分析：

```python
def stop_profiling(self):
    if self.profiler:
        self.profiler.stop()
    # 显存事件数据会自动保存到指定目录
```

### 5.2 推理配置方案

#### 5.2.1 配置文件结构

推理配置文件采用YAML格式，包含以下主要部分：

```yaml
name: "配置名称"  # 配置的唯一标识符
description: "配置描述"  # 对该配置的简要说明

model_config:  # 模型相关配置
  task: "t2v-1.3B"  # 任务类型
  size: "832*480"  # 输出分辨率
  load_strategy: "block"  # 模型加载策略
  offload_model: true/false  # 是否启用CPU卸载
  precision: "fp16"  # 计算精度
  device: "cuda"  # 运行设备
  ckpt_dir: "/path/to/model"  # 模型权重路径

optimization:  # 优化策略配置
  attention_slicing: true/false  # 是否启用注意力切片
  gradient_checkpointing: true/false  # 是否启用梯度检查点
  batch_size: 1  # 批处理大小
  micro_batch_size: 1  # 微批处理大小
  parallel_degree: 1  # 模型并行度

logging:  # 日志和分析配置
  profile_memory: true  # 是否记录显存使用
  log_interval: 10  # 日志记录间隔
  trace_path: "profiler_logs"  # 分析日志保存路径
```

#### 5.2.2 测试配置矩阵

| 配置名称 | 加载策略 | 精度 | 优化特性 | 并行设置 |
|---|---|---|---|---|
| baseline | block | fp16 | 无 | 单卡 |
| cpu_offload | block | fp16 | CPU卸载 | 单卡 |
| attention_slicing | block | fp16 | 注意力切片 | 单卡 |
| combined_opt | block | fp16 | CPU卸载+注意力切片 | 单卡 |
| fsdp | block | fp16 | FSDP并行 | 多卡 |

## 6. 工具设计

### 6.1 数据收集器

本项目使用5.1.2节中描述的`MemoryTracker`和`MemoryProfiler`类来收集显存使用数据。这些工具在模型加载和推理的关键节点插入监控点，记录显存使用情况，并将数据保存为JSON格式以便后续分析。

主要功能包括：

- 记录模型加载过程中各组件的显存增量
- 监控推理过程中的显存峰值
- 区分基础显存和激活显存
- 集成PyTorch Profiler提供更详细的性能分析
- 自动保存显存事件数据
```

#### 5.1.3 显存分析工具

我们提供了专门的分析工具 `analyze_memory.py` 来处理收集到的显存数据，该工具可以解析JSON格式的显存事件记录，提取关键指标，并生成可视化报告：

```python
def analyze_log_file(file_path: str) -> Dict[str, float]:
    """解析memory_events.json文件，提取关键事件的峰值显存"""
    with open(file_path, 'r') as f:
        events = json.load(f)
    
    memory_data = {}
    for event in events:
        event_name = event.get("event")
        peak_memory = event.get("peak_memory", 0)
        incremental_memory = event.get("incremental_memory", 0)
        if event_name:
            memory_data[event_name] = format_b_to_mb(peak_memory or incremental_memory)
    
    return memory_data
```

分析工具的主要功能包括：

- 解析显存事件日志文件
- 计算各组件的显存占用
- 区分基础显存和激活显存
- 生成可视化图表和Markdown格式报告
- 提供优化建议

### 6.2 日志格式和目录管理

#### 6.2.1 目录结构规划

为了系统地管理测试过程中产生的各种文件，我们定义以下目录结构：

```
Wan2.1-Enhanced/
├── ... (项目原有文件)
├── tests/                          # 测试脚本和相关资源
│   ├── configs/                    # 推理配置文件目录
│   │   ├── baseline.yaml
│   │   └── memory_opt.yaml
│   ├── scripts/                    # 测试执行脚本
│   │   └── run_memory_tests.sh
│   └── README.md
├── profiler_logs/                  # 性能分析日志
│   ├── t2v-1.3B-fp32/
│   │   ├── memory_events.json
```

## 7. 参数说明

以下是 `generate.py` 脚本中使用的主要超参数的详细说明，这些参数控制着视频生成的各个方面。

- `--task` (str): 指定要运行的任务类型。不同的任务会加载不同的模型配置。
  - 可选项: `t2v-1.3B`, `t2v-14B`, `t2i-14B`, `i2v-14B`, `flf2v-14B`, `vace-1.3B`, `vace-14B`。
  - 默认值: `t2v-14B`。

- `--size` (str): 生成视频的分辨率，格式为 `宽度*高度`。
  - 示例: `1280*720`, `720*1280`。
  - 默认值: `1280*720`。

- `--frame_num` (int): 生成视频的总帧数。对于文生视频和图生视频，推荐使用 `4n+1` 格式的帧数。
  - 默认值: 对于 `t2i` 任务为 `1`，其他任务为 `81`。

- `--ckpt_dir` (str): 存放模型权重文件的目录路径。这是加载预训练模型的必需参数。

- `--offload_model` (bool): 是否在每次模型前向传播后将模型权重卸载到 CPU。这可以显著减少 GPU 显存使用，但会增加推理时间。
  - 默认值: 在单卡环境下为 `True`，多卡环境下为 `False`。

- `--ulysses_size` (int): Ulysses 张量并行的大小。用于在多个 GPU 上对序列维度进行切分，以支持更长序列的推理。
  - 默认值: `1`。

- `--ring_size` (int): Ring Attention 并行的大小。与 Ulysses 结合使用，进一步优化长序列的注意力计算。
  - 默认值: `1`。

- `--t5_fsdp` (bool): 是否对 T5 文本编码器使用 FSDP (Fully Sharded Data Parallel)。这会将模型参数分片到多个 GPU 上，以降低单卡显存峰值。
  - 默认值: `False`。

- `--dit_fsdp` (bool): 是否对 DiT (Diffusion Transformer) 模型使用 FSDP。
  - 默认值: `False`。

- `--t5_cpu` (bool): 是否将 T5 模型放置在 CPU 上运行。这可以节省大量 GPU 显存，但会因为数据传输而降低速度。
  - 默认值: `False`。

- `--prompt` (str): 用于生成视频或图像的文本描述。

- `--use_prompt_extend` (bool): 是否启用提示词扩展功能。该功能会使用大语言模型（如 Qwen）来丰富和优化原始提示词。
  - 默认值: `False`。

- `--base_seed` (int): 随机种子，用于确保生成结果的可复现性。如果设置为 `-1`，则会随机选择一个种子。
  - 默认值: `-1`。

- `--image` (str): [图生视频] 输入图像的路径。

- `--first_frame` (str): [首末帧生视频] 输入的第一帧图像的路径。

- `--last_frame` (str): [首末帧生视频] 输入的最后一帧图像的路径。

- `--sample_solver` (str): 扩散过程的采样器类型。
  - 可选项: `unipc`, `dpm++`。
  - 默认值: `unipc`。

- `--sample_steps` (int): 采样步数。步数越多，生成质量可能越高，但耗时也越长。
  - 默认值: `t2v` 任务为 `50`，`i2v` 任务为 `40`。

- `--sample_shift` (float): Flow Matching 调度器的移位因子。这是一个高级参数，用于调整采样过程的动态。

- `--sample_guide_scale` (float): 无分类器指导（Classifier-Free Guidance）的缩放因子。该值越高，生成结果与提示词的关联性越强，但可能会牺牲多样性。
  - 默认值: `5.0`。

- `--precision` (str): 推理时使用的计算精度。
  - 可选项: `fp16`, `bf16`, `fp32`。
  - 默认值: `fp16`。
  - 说明: 使用较低的精度（如 `fp16`）可以显著减少显存占用并加速推理，但可能会有微小的精度损失。

- `--attention_slicing` (bool): 是否启用注意力切片（Attention Slicing）。这是一种显存优化技术，通过将注意力计算分解为多个小块来降低峰值显存，但可能会略微增加计算时间。
  - 默认值: `False`。

- `--gradient_checkpointing` (bool): 是否启用梯度检查点。这是一种在训练中用于节省显存的技术，它通过重新计算而不是存储中间激活值来减少显存。在推理中，它的应用场景较少，但对于超长序列可能有用。
  - 默认值: `False`。

- `--load_strategy` (str): 模型权重的加载策略。
  - 默认值: `block`。
  - 说明: `block` 策略可能指的是一次性将整个模型加载到设备中。

## 8. 模型架构与默认配置

本节提供 `WanT2V-1.3B` 模型的核心架构参数和默认配置，这些信息派生自模型的配置文件，为高级用户和开发者提供参考。

### 8.1 核心架构参数

- **文本编码器 (T5)**:
  - **模型类型**: `umt5_xxl` - 指定了使用的 T5 模型的具体变体，`xxl` 表示这是一个超大规模版本，具有更多的参数和更强的语言理解能力。
  - **Tokenizer**: `google/umt5-xxl` - 用于将输入文本转换为模型可处理的 token ID 的分词器。此分词器与 `umt5_xxl` 模型配套使用。
  - **权重精度**: `bfloat16` - T5 模型权重加载时使用的精度。`bfloat16` 是一种节省显存的浮点数格式，同时能保持较好的模型性能。
  - **最大文本长度**: `512` tokens - 模型能够处理的输入文本的最大长度（以 token 计）。超过此长度的文本将被截断。
  - **权重文件**: `models_t5_umt5-xxl-enc-bf16.pth` - 存储 T5 编码器预训练权重的文件名。

- **视频 VAE (Variational Autoencoder)**:
  - **权重文件**: `Wan2.1_VAE.pth` - 存储视频 VAE 预训练权重的文件名。VAE 负责将视频帧编码到潜在空间，以及从潜在空间解码生成视频帧。
  - **降采样步长**: `(4, 8, 8)` - VAE 在三个维度（时间、高度、宽度）上的降采样因子。这意味着视频在潜在空间中的尺寸会显著小于原始尺寸，从而降低计算复杂度。

- **扩散模型 (DiT - Diffusion Transformer)**:
  - **模型维度 (dim)**: `1536` - Transformer 模型内部的主要工作维度。这是模型中嵌入向量和隐藏状态的大小。
  - **FFN 维度 (ffn_dim)**: `8960` - Transformer 块中前馈神经网络（FFN）的隐藏层维度。通常远大于模型维度，以增强模型的表达能力。
  - **频率编码维度 (freq_dim)**: `256` - 用于时间步（timestep）编码的维度。这使得模型能够感知在扩散过程中的当前位置。
  - **注意力头数 (num_heads)**: `12` - 多头注意力机制中的头的数量。多头允许模型在不同子空间中并行关注不同的信息。
  - **层数 (num_layers)**: `30` - DiT 模型中堆叠的 Transformer 块的数量。更深的模型通常具有更强的学习能力。
  - **Patch Size**: `(1, 2, 2)` - 将输入视频（或潜在表示）切分为小块（patch）的大小。`1` 表示在时间维度上不切分，`2x2` 表示在空间维度上切分。
  - **Window Size**: `(-1, -1)` - 注意力计算的窗口大小。`-1` 表示使用全局注意力，即每个 patch 都可以关注所有的其他 patch。
  - **QK Normalization**: `True` - 是否对查询（Query）和键（Key）进行归一化。这有助于稳定注意力的计算过程。
  - **Cross Attention Normalization**: `True` - 是否对交叉注意力模块的输出进行归一化。
  - **Epsilon (eps)**: `1e-06` - 在归一化操作（如 LayerNorm）中为了防止除以零而添加的一个小常数。

### 8.2 默认配置

- **模型参数精度**: `bfloat16` - 整个推理流程中默认使用的浮点数精度，以平衡性能和显存消耗。
- **训练总步数**: `1000` - 模型预训练时所经历的总步数。这是一个参考信息，表明了模型的训练程度。
- **采样帧率**: `16` fps - 生成视频的默认帧率（每秒帧数）。
- **默认负向提示词 (Negative Prompt)**: 
  ```
  色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走
  ```
  - 这是一组用于指导模型避免生成特定不希望出现的特征的文本。它有助于提高生成视频的整体质量和美感。
