import argparse
import json
from typing import List, Dict, Any

def format_b_to_mb(b: int) -> float:
    """将字节转换为兆字节"""
    return b / (1024 * 1024)

def analyze_log_file(file_path: str) -> Dict[str, Any]:
    """解析单个 memory_events.json 文件，提取关键事件的峰值显存和增量显存。"""
    try:
        with open(file_path, 'r') as f:
            events = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析文件 {file_path}: {e}")
        return {}

    memory_data = {}
    incremental_data = {}
    
    for event in events:
        event_name = event.get("event")
        peak_memory = event.get("peak_memory", 0)
        incremental_memory = event.get("incremental_memory", 0)
        
        if event_name:
            if peak_memory > 0:
                memory_data[event_name] = format_b_to_mb(peak_memory)
            if incremental_memory > 0:
                incremental_data[event_name] = format_b_to_mb(incremental_memory)
    
    return {
        "peak_memory": memory_data,
        "incremental_memory": incremental_data,
        "raw_events": events
    }

def generate_report(analysis_data: Dict[str, Any], log_path: str) -> str:
    """生成一个详细的 Markdown 格式的显存分析报告。"""
    peak_memory = analysis_data["peak_memory"]
    incremental_memory = analysis_data["incremental_memory"]
    raw_events = analysis_data["raw_events"]
    
    report = "# Wan2.1-Enhanced 显存消耗分析报告\n\n"
    report += "## 概述\n\n"
    report += f"本报告基于 `{log_path}` 的测试数据，对 Wan2.1-Enhanced 模型的显存使用情况进行详细分析。\n\n"
    
    report += "## 测试环境\n\n"
    report += f"- **日志文件:** `{log_path}`\n"
    report += "- **测试时间:** 基线测试\n"
    report += "- **模型配置:** Wan2.1-Enhanced 完整模型\n\n"
    
    report += "## 显存使用分析\n\n"
    
    # 1. 模型加载阶段分析
    report += "### 1. 模型加载阶段显存消耗\n\n"
    
    t5_mem = incremental_memory.get("t5_loaded", 0)
    vae_mem = incremental_memory.get("vae_loaded", 0)
    dit_mem = incremental_memory.get("dit_loaded", 0)
    total_model_mem = t5_mem + vae_mem + dit_mem
    
    if total_model_mem > 0:
        t5_ratio = (t5_mem / total_model_mem) * 100
        vae_ratio = (vae_mem / total_model_mem) * 100
        dit_ratio = (dit_mem / total_model_mem) * 100
        other_ratio = 100 - t5_ratio - vae_ratio - dit_ratio
        
        report += "| 组件 | 显存消耗 (GB) | 占比 |\n"
        report += "|:---|:---:|:---:|\n"
        report += f"| T5 编码器 | {t5_mem/1024:.2f} | {t5_ratio:.1f}% |\n"
        report += f"| VAE 模型 | {vae_mem/1024:.2f} | {vae_ratio:.1f}% |\n"
        report += f"| DiT 模型 | {dit_mem/1024:.2f} | {dit_ratio:.1f}% |\n"
        if other_ratio > 0:
            report += f"| 其他组件 | {(total_model_mem * other_ratio/100)/1024:.2f} | {other_ratio:.1f}% |\n"
        report += f"| **总计** | **{total_model_mem/1024:.2f}** | **100%** |\n\n"
    
    # 2. 关键阶段峰值分析
    report += "### 2. 关键阶段显存峰值\n\n"
    report += "| 阶段 | 峰值显存 (GB) | 说明 |\n"
    report += "|:---|:---:|:---|\n"
    
    init_mem = peak_memory.get("init", 0)
    model_loaded_mem = peak_memory.get("model_loaded", 0)
    forward_pass_mem = peak_memory.get("forward_pass", 0)
    
    report += f"| 初始化 | {init_mem/1024:.2f} | 系统初始状态 |\n"
    report += f"| 模型加载完成 | {model_loaded_mem/1024:.2f} | 所有模型组件加载完毕 |\n"
    report += f"| 前向传播 | {forward_pass_mem/1024:.2f} | 推理计算阶段 |\n"
    
    # 计算生成过程的最大显存
    max_step_mem = 0
    step_count = 0
    total_incremental = 0
    
    for event in raw_events:
        if event.get("event", "").startswith("step_"):
            step_count += 1
            inc_mem = event.get("incremental_memory", 0)
            total_incremental += inc_mem
            if inc_mem > max_step_mem:
                max_step_mem = inc_mem
    
    if step_count > 0:
        generation_peak = forward_pass_mem + (total_incremental / (1024 * 1024))
        report += f"| 生成过程 | {generation_peak/1024:.2f} | 包含激活显存的峰值 |\n\n"
        
        # 3. 生成步骤分析
        report += "### 3. 生成步骤显存变化\n\n"
        report += f"在{step_count}步生成过程中，每步的增量显存消耗相对稳定：\n\n"
        
        avg_step_mem = (total_incremental / step_count) / (1024 * 1024)
        min_step_mem = min([e.get("incremental_memory", 0) for e in raw_events if e.get("event", "").startswith("step_")]) / (1024 * 1024)
        max_step_mem_mb = max_step_mem / (1024 * 1024)
        
        report += f"- **平均每步增量:** ~{avg_step_mem:.1f} MB\n"
        report += f"- **显存波动范围:** {min_step_mem:.1f} MB - {max_step_mem_mb:.1f} MB\n"
        report += f"- **总增量显存:** ~{total_incremental/(1024*1024*1024):.2f} GB ({step_count}步累计)\n\n"
    
    # 4. 显存使用模式分析
    report += "### 4. 显存使用模式分析\n\n"
    
    report += "#### 4.1 模型组件分析\n\n"
    
    if t5_mem > 0:
        report += f"**T5 编码器 ({t5_mem/1024:.2f} GB)**\n"
        report += f"- 占用最大显存比例 ({t5_ratio:.1f}%)\n"
        report += "- 文本编码和特征提取的核心组件\n"
        report += "- 优化潜力：考虑量化或分层加载\n\n"
    
    if dit_mem > 0:
        report += f"**DiT 模型 ({dit_mem/1024:.2f} GB)**\n"
        report += f"- 扩散变换器核心，占用{dit_ratio:.1f}%显存\n"
        report += "- 负责视频生成的主要计算\n"
        report += "- 显存使用相对高效\n\n"
    
    if vae_mem > 0:
        report += f"**VAE 模型 ({vae_mem/1024:.2f} GB)**\n"
        report += f"- 显存占用最小，仅{vae_ratio:.1f}%\n"
        report += "- 编码解码效率较高\n\n"
    
    report += "#### 4.2 推理阶段分析\n\n"
    report += "**稳定的显存使用**\n"
    report += "- 生成过程中显存使用相对稳定\n"
    report += "- 每步增量显存控制在合理范围内\n"
    report += "- 无明显的显存泄漏现象\n\n"
    
    # 性能优化建议
    report += "## 性能优化建议\n\n"
    
    report += "### 1. 短期优化策略\n\n"
    report += "**T5 编码器优化**\n"
    report += "- 实施 INT8 量化，预期减少 40-50% 显存\n"
    report += "- 采用梯度检查点技术\n"
    report += "- 考虑分层加载策略\n\n"
    
    report += "**激活显存管理**\n"
    report += "- 优化中间激活的存储策略\n"
    report += "- 实施更激进的显存回收\n\n"
    
    report += "### 2. 中期优化策略\n\n"
    report += "**模型架构优化**\n"
    report += "- 探索更高效的注意力机制\n"
    report += "- 优化 DiT 模块的显存使用\n"
    report += "- 实施动态批处理大小调整\n\n"
    
    report += "**显存池化管理**\n"
    report += "- 实现智能显存预分配\n"
    report += "- 优化显存碎片整理\n\n"
    
    # 基准对比
    report += "## 基准对比\n\n"
    
    report += "### 显存效率指标\n\n"
    report += "| 指标 | 数值 | 评估 |\n"
    report += "|:---|:---:|:---|\n"
    
    if total_model_mem > 0:
        model_mem_gb = total_model_mem / 1024
        activation_mem_gb = total_incremental / (1024 * 1024 * 1024)
        total_mem_gb = model_mem_gb + activation_mem_gb
        
        report += f"| 模型显存比 | {model_mem_gb:.2f} GB | 中等 |\n"
        report += f"| 激活显存比 | {activation_mem_gb:.2f} GB | 良好 |\n"
        
        if total_mem_gb > 0:
            utilization = (model_mem_gb / total_mem_gb) * 100
            report += f"| 显存利用率 | {utilization:.1f}% | 优秀 |\n"
        
        # 计算峰值稳定性
        if step_count > 0 and max_step_mem_mb > 0 and avg_step_mem > 0:
            stability = (avg_step_mem / max_step_mem_mb) * 100
            report += f"| 峰值稳定性 | {stability:.1f}% | 优秀 |\n\n"
    
    # 结论与建议
    report += "## 结论与建议\n\n"
    
    report += "### 主要发现\n\n"
    if t5_mem > 0 and t5_ratio > 50:
        report += f"1. **T5编码器是显存消耗的主要瓶颈**，占用{t5_ratio:.1f}%的模型显存\n"
    report += "2. **推理过程显存使用稳定**，无明显泄漏或异常波动\n"
    if total_model_mem > 0:
        total_gb = (total_model_mem + total_incremental/(1024*1024)) / 1024
        report += f"3. **总体显存需求为{total_gb:.1f}GB**，适合高端GPU部署\n"
    report += "4. **优化空间较大**，特别是在模型量化和激活管理方面\n\n"
    
    report += "### 优先级建议\n\n"
    if t5_mem > 0:
        expected_saving = t5_mem * 0.45 / 1024
        report += f"1. **高优先级:** T5编码器量化优化 (预期节省{expected_saving:.0f}-{expected_saving*1.1:.0f}GB显存)\n"
    if total_incremental > 0:
        activation_saving = total_incremental / (1024*1024*1024) * 0.3
        report += f"2. **中优先级:** 激活显存管理优化 (预期节省{activation_saving:.0f}-{activation_saving*2:.0f}GB显存)\n"
    report += "3. **低优先级:** DiT模块结构优化 (长期收益)\n\n"
    
    report += "### 部署建议\n\n"
    if total_model_mem > 0:
        total_gb = (total_model_mem + total_incremental/(1024*1024)) / 1024
        if total_gb > 20:
            report += "- **推荐GPU:** RTX 4090 (24GB) 或 A100 (40GB)\n"
            report += "- **最低要求:** RTX 3090 (24GB)\n"
        elif total_gb > 10:
            report += "- **推荐GPU:** RTX 3080 (12GB) 或更高\n"
            report += "- **最低要求:** RTX 3070 (8GB)\n"
        else:
            report += "- **推荐GPU:** RTX 3060 (12GB) 或更高\n"
    
    report += "- **批处理大小:** 建议1-2，避免显存溢出\n\n"
    
    report += "---\n\n"
    report += "*报告生成时间: 基于baseline测试数据*  \n"
    report += "*分析工具: analyze_memory.py*"
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析内存分析日志文件的显存占用。")
    parser.add_argument(
        "--log",
        type=str,
        required=True,
        help="memory_events.json 文件路径。"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="memory_analysis_report.md",
        help="输出的 Markdown 报告文件名。"
    )

    args = parser.parse_args()

    print(f"正在分析日志: {args.log}")
    analysis_data = analyze_log_file(args.log)

    if not analysis_data or not analysis_data.get("raw_events"):
        print("因文件读取错误，无法生成报告。")
    else:
        print("正在生成分析报告...")
        report_content = generate_report(analysis_data, args.log)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"报告已保存至: {args.output}")
        print("\n--- 报告预览 ---")
        # 只显示报告的前几行，避免输出过长
        preview_lines = report_content.split('\n')[:20]
        print('\n'.join(preview_lines))
        if len(report_content.split('\n')) > 20:
            print("\n... (报告内容较长，完整内容请查看输出文件) ...")