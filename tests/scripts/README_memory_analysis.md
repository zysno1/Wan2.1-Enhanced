# 显存分析工具使用说明

## 概述

本工具用于分析 Wan2.1-Enhanced 模型的显存使用情况，生成详细的 Markdown 格式分析报告。

## 功能特性

### 🔍 深度分析功能
- **模型组件分析**: 详细分析 T5编码器、VAE模型、DiT模型的显存占用
- **阶段性监控**: 跟踪初始化、模型加载、推理等关键阶段的显存变化
- **生成步骤分析**: 监控每个生成步骤的增量显存消耗
- **性能指标计算**: 自动计算显存利用率、峰值稳定性等关键指标

### 📊 报告生成功能
- **完整的Markdown报告**: 包含图表、数据表格和详细分析
- **优化建议**: 基于分析结果提供具体的性能优化建议
- **部署指导**: 根据显存需求推荐合适的GPU配置
- **基准对比**: 提供性能评级和效率指标

## 使用方法

### 基本用法

```bash
# 分析显存日志并生成报告
python tests/scripts/analyze_memory.py --log <日志文件路径> --output <输出报告路径>
```

### 参数说明

- `--log`: 必需参数，指定 `memory_events.json` 文件路径
- `--output`: 可选参数，指定输出的 Markdown 报告文件名（默认: `memory_analysis_report.md`）

### 使用示例

```bash
# 示例1: 分析基线测试数据
python tests/scripts/analyze_memory.py \
    --log profiler_logs/baseline/memory_events.json \
    --output baseline_analysis.md

# 示例2: 分析优化后的测试数据
python tests/scripts/analyze_memory.py \
    --log profiler_logs/optimized/memory_events.json \
    --output optimized_analysis.md

# 示例3: 使用默认输出文件名
python tests/scripts/analyze_memory.py \
    --log profiler_logs/baseline/memory_events.json
```

## 输入数据格式

工具期望的输入文件为 JSON 格式，包含以下事件类型：

```json
[
    {
        "event": "init",
        "peak_memory": 0
    },
    {
        "event": "t5_loaded",
        "incremental_memory": 11362869248
    },
    {
        "event": "model_loaded",
        "peak_memory": 24824987648
    },
    {
        "event": "step_0",
        "incremental_memory": 35122176
    }
]
```

### 支持的事件类型

| 事件名称 | 数据字段 | 说明 |
|:---|:---|:---|
| `init` | `peak_memory` | 系统初始化时的显存状态 |
| `t5_loaded` | `incremental_memory` | T5编码器加载的增量显存 |
| `vae_loaded` | `incremental_memory` | VAE模型加载的增量显存 |
| `dit_loaded` | `incremental_memory` | DiT模型加载的增量显存 |
| `model_loaded` | `peak_memory` | 所有模型加载完成后的峰值显存 |
| `forward_pass` | `peak_memory` | 前向传播阶段的峰值显存 |
| `step_N` | `incremental_memory` | 第N步生成的增量显存 |
| `inference_end` | `peak_memory` | 推理结束时的峰值显存 |

## 输出报告结构

生成的报告包含以下主要章节：

### 1. 概述和测试环境
- 测试配置信息
- 日志文件路径
- 模型配置说明

### 2. 显存使用分析
- **模型加载阶段显存消耗**: 各组件的显存占用和比例
- **关键阶段显存峰值**: 不同阶段的峰值显存统计
- **生成步骤显存变化**: 生成过程中的显存变化模式
- **显存使用模式分析**: 深入分析各组件的使用特征

### 3. 性能优化建议
- **短期优化策略**: 立即可实施的优化方案
- **中期优化策略**: 需要一定开发工作的优化方案
- **长期优化策略**: 架构级别的优化建议

### 4. 基准对比
- **显存效率指标**: 量化的性能指标
- **性能评级**: 基于指标的评级结果

### 5. 结论与建议
- **主要发现**: 关键的分析结果
- **优先级建议**: 按优先级排序的优化建议
- **部署建议**: GPU选择和配置建议

## 测试工具

### 运行功能测试

```bash
# 运行完整的功能测试套件
python tests/scripts/test_memory_analysis.py
```

测试套件包含：
- 字节转换功能测试
- 日志分析功能测试
- 报告生成功能测试
- 错误处理测试

### 测试输出示例

```
=== 显存分析脚本功能测试 ===

测试字节转换功能...
  ✓ 1048576 bytes -> 1.00 MB
  ✓ 1073741824 bytes -> 1024.00 MB
字节转换测试通过!

测试日志分析功能...
  ✓ 解析了 11 个事件
  ✓ 峰值显存事件: 3 个
  ✓ 增量显存事件: 6 个
日志分析测试通过!

🎉 所有测试通过!
```

## 故障排除

### 常见问题

**Q: 提示"无法读取或解析文件"**
A: 检查文件路径是否正确，确保文件存在且为有效的JSON格式

**Q: 报告中某些数据显示为0**
A: 检查输入数据是否包含对应的事件类型和数据字段

**Q: 生成的报告内容不完整**
A: 确保输入数据包含足够的事件信息，特别是模型加载相关的事件

### 调试模式

如需调试，可以在脚本中添加详细的日志输出：

```python
# 在 analyze_memory.py 中添加调试信息
print(f"解析到的事件数量: {len(events)}")
print(f"峰值显存事件: {list(memory_data.keys())}")
print(f"增量显存事件: {list(incremental_data.keys())}")
```

## 扩展功能

### 自定义分析

可以通过修改 `generate_report` 函数来添加自定义的分析内容：

```python
# 添加自定义分析章节
def add_custom_analysis(report, analysis_data):
    # 自定义分析逻辑
    custom_section = "### 自定义分析\n\n"
    # ... 添加分析内容
    return report + custom_section
```

### 多格式输出

可以扩展工具支持其他输出格式：

```python
# 添加HTML输出支持
def generate_html_report(analysis_data, log_path):
    # HTML报告生成逻辑
    pass

# 添加PDF输出支持
def generate_pdf_report(analysis_data, log_path):
    # PDF报告生成逻辑
    pass
```

## 版本信息

- **当前版本**: 2.0
- **兼容性**: Python 3.6+
- **依赖**: 标准库（无外部依赖）

## 更新日志

### v2.0 (当前版本)
- ✨ 重构分析引擎，支持更详细的显存分析
- ✨ 新增完整的Markdown报告生成
- ✨ 添加性能优化建议和部署指导
- ✨ 增加基准对比和性能评级
- ✨ 完善的测试套件和错误处理
- 🐛 修复字节转换精度问题
- 🐛 修复报告格式问题

### v1.0
- 基础的显存数据解析
- 简单的表格式报告输出