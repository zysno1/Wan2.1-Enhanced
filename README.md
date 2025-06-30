# Wan2.1 视频生成显存优化测试方案设计文档

## 目录
- [1. 项目背景](#1-项目背景)
- [2. 测试目标](#2-测试目标)
- [3. 测试方案](#3-测试方案)
- [4. 工具设计](#4-工具设计)
- [5. 实施计划](#5-实施计划)
- [6. 日志格式和目录管理](#6-日志格式和目录管理)
- [7. 预期成果](#7-预期成果)

## 1. 项目背景

Wan2.1 是一个视频生成项目，在生成过程中涉及多个大型模型的加载和推理，显存使用是一个关键的性能指标。为了优化显存使用，需要对不同推理配置下的显存消耗进行系统的测试和分析。

本项目的工作目录为 `/workspace/Wan2.1-Enhanced`，所有的代码、工具、文档和生成的日志都应存放在此目录下。

## 2. 测试目标

- 精确测量不同推理配置下的显存使用情况
- 识别单次运行中显存使用的瓶颈
- 为后续优化提供数据支持和方向指导

## 3. 测试方案

### 3.1 PyTorch 性能分析实现

#### 3.1.1 显存监控

为了更精确地分析显存使用情况，我们将显存消耗拆分为 **基础显存** 和 **激活显存** 两部分：

- **基础显存**：模型加载后，在推理前占用的显存，主要包括模型权重、优化器状态等。
- **激活显存**：在模型前向传播过程中，因计算产生的中间变量（激活值）所占用的显存。

我们将通过在模型加载和推理的关键节点记录显存快照来分别统计这两部分。具体来说，我们会记录每个模型加载后的**增量显存**，以及在推理过程中每个步骤的**激活显存**。所有的显存事件都将保存到 JSON 文件中，以便后续分析。




            warmup=1,
            active=3
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )
```

### 3.2 推理配置方案

#### 3.2.1 配置文件结构

```yaml
name: "optimization_test_1"
description: "使用模型分块加载和注意力切片的优化配置"

model_config:
  load_strategy: "block"     # 模型加载策略：full/block
  precision: "fp16"         # 计算精度：fp32/fp16/bf16
  device: "cuda"           # 运行设备
  offload: false           # CPU 卸载开关

optimization:
  attention_slicing: true   # 注意力切片
  gradient_checkpointing: false
  batch_size: 1
  micro_batch_size: 1
  parallel_degree: 1       # 模型并行度

logging:
  profile_memory: true
  log_interval: 10         # 记录间隔（步数）
  trace_path: "traces/opt_test_1"
```

#### 3.2.2 测试配置矩阵

| 配置名称 | 加载策略 | 精度 | 优化特性 | 并行设置 |
|---------|---------|------|---------|----------|
| 配置名称 | 加载策略 | 精度 | 优化特性 | 并行设置 |
|---|---|---|---|---|
| t2v-1.3B-fp32 | full | fp32 | 无 | 单卡 |

## 4. 工具设计

### 4.1 数据收集器

```python
class MemoryProfiler:
    def __init__(self, config_name, output_dir):
        self.config_name = config_name
        self.output_dir = output_dir
        self.memory_tracker = MemoryTracker(output_dir)
        self.events = []

    def start_profiling(self):
        self.profiler = setup_profiler(self.output_dir)
        self.profiler.start()

    def log_event(self, event_name, metadata=None):
        # 如果提供了 base_memory，则计算增量显存
        # 否则，记录当前的峰值显存
        if metadata and 'base_memory' in metadata:
            incremental_memory = torch.cuda.memory_allocated() - metadata['base_memory']
            memory_log = f"{event_name}: incremental_memory={incremental_memory / (1024**2):.2f} MB"
        else:
            peak_memory = torch.cuda.max_memory_allocated()
            memory_log = f"{event_name}: peak_memory={peak_memory / (1024**2):.2f} MB"

        event = {
            "name": event_name,
            "timestamp": time.time(),
            "memory_log": memory_log,
            "metadata": metadata
        }
        self.events.append(event)

    def stop_profiling(self):
        self.profiler.stop()
        self.memory_tracker.save_logs()
        self.save_events()
```

### 4.2 分析工具功能

#### 4.2.1 基础分析

- 显存使用统计
  - **基础显存**：分析每个核心组件（如 T5, VAE, DiT）加载时所引起的**增量显存**消耗。
  - **激活显存**：分析在视频生成（前向传播）的每个阶段（例如，`forward pass` 之前、`transformer` 块内部、`vae.decode` 之后）的显存使用情况。
  - **总峰值显存**：整个生命周期内的最大显存占用。
  - 显存碎片率和波动情况。

- 阶段分析
  - 模型加载显存
  - 推理过程显存
  - 模型切换显存



### 4.3 可视化组件

- 显存使用时序图

- 关键事件标注
- 异常点检测

## 5. 实施计划

### 5.1 环境准备

1. 确保测试环境
   - 清理 GPU 进程
   - 预热 GPU
   - 检查 CUDA 版本

2. 工具准备
   - 安装依赖
   - 配置日志目录
   - 准备测试数据

### 5.2 测试执行

1. **执行测试**

   所有测试都应在指定的虚拟环境下执行，以确保依赖隔离和环境一致性。

   ```bash
   # 激活虚拟环境
   source /workspace/env/bin/activate

   # 执行测试脚本
   bash tests/scripts/run_memory_tests.sh
   ```

3. 数据分析
   ```bash
   python Wan-Test/analyze_trace_file.py profiler_logs/t2v_1.3B_480p_5s_baseline/trace.json
   ```

## 6. 日志格式和目录管理

### 6.1 目录结构规划

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
