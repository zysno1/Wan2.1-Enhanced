import json
import argparse
from typing import List, Dict, Any

def format_b_to_mb(b: int) -> float:
    """将字节转换为兆字节"""
    return b / (1024 * 1024)

def analyze_log_file(file_path: str) -> Dict[str, float]:
    """解析单个 memory_events.json 文件，提取关键事件的峰值显存。"""
    try:
        with open(file_path, 'r') as f:
            events = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法读取或解析文件 {file_path}: {e}")
        return {}

    memory_data = {}
    for event in events:
        event_name = event.get("event")
        peak_memory = event.get("peak_memory", 0)
        if event_name:
            memory_data[event_name] = format_b_to_mb(peak_memory)
    
    return memory_data

def generate_report(log_data: Dict[str, float], log_path: str) -> str:
    """生成一个 Markdown 格式的显存分析报告。"""
    report = f"# 显存分析报告\n\n"
    report += f"- **日志文件:** `{log_path}`\n\n"
    report += "| 事件 (Event) | 峰值显存 (MB) |\n"
    report += "|:---|:---:|\n"

    # 计算模型基础显存和激活显存
    init_mem = log_data.get("init", 0)
    model_load_mem = log_data.get("model_loaded", 0)
    forward_pass_mem = log_data.get("forward_pass", 0)

    base_model_mem = model_load_mem - init_mem
    activation_mem = forward_pass_mem - model_load_mem

    # 插入计算出的指标
    log_data["model_base_memory"] = base_model_mem
    log_data["activation_memory"] = activation_mem

    # 排序并生成报告
    sorted_keys = sorted(log_data.keys(), key=lambda x: str(x))

    for key in sorted_keys:
        mem_val = log_data[key]
        report += f"| `{key}` | {mem_val:.2f} |\n"

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
    memory_data = analyze_log_file(args.log)

    if not memory_data:
        print("因文件读取错误，无法生成报告。")
    else:
        print("正在生成分析报告...")
        report_content = generate_report(memory_data, args.log)
        
        with open(args.output, 'w') as f:
            f.write(report_content)
        
        print(f"报告已保存至: {args.output}")
        print("\n--- 报告预览 ---")
        print(report_content)