#!/usr/bin/env python3
"""
测试显存分析脚本的功能
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.scripts.analyze_memory import analyze_log_file, generate_report, format_b_to_mb

def create_test_data():
    """创建测试用的内存事件数据"""
    test_events = [
        {"event": "init", "peak_memory": 0},
        {"event": "t5_loaded", "incremental_memory": 11362869248},  # ~10.6GB
        {"event": "vae_loaded", "incremental_memory": 509557760},   # ~486MB
        {"event": "dit_loaded", "incremental_memory": 5704053248},  # ~5.3GB
        {"event": "model_loaded", "peak_memory": 24824987648},      # ~23.1GB
        {"event": "forward_pass", "peak_memory": 24824987648},     # ~23.1GB
        {"event": "before_generate", "incremental_memory": 0},
        {"event": "step_0", "incremental_memory": 35122176},        # ~33.5MB
        {"event": "step_1", "incremental_memory": 51903488},        # ~49.5MB
        {"event": "step_2", "incremental_memory": 68668416},        # ~65.5MB
        {"event": "inference_end", "peak_memory": 25893642240}     # ~24.1GB
    ]
    return test_events

def test_format_conversion():
    """测试字节到MB的转换"""
    print("测试字节转换功能...")
    
    # 测试用例
    test_cases = [
        (1024 * 1024, 1.0),        # 1MB
        (1024 * 1024 * 1024, 1024.0),  # 1GB
        (0, 0.0),                   # 0字节
        (512 * 1024, 0.5)          # 0.5MB
    ]
    
    for bytes_val, expected_mb in test_cases:
        result = format_b_to_mb(bytes_val)
        assert abs(result - expected_mb) < 0.01, f"转换错误: {bytes_val} bytes -> {result} MB (期望: {expected_mb} MB)"
        print(f"  ✓ {bytes_val} bytes -> {result:.2f} MB")
    
    print("字节转换测试通过!\n")

def test_log_analysis():
    """测试日志分析功能"""
    print("测试日志分析功能...")
    
    # 创建临时测试文件
    test_events = create_test_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_events, f, indent=2)
        temp_file = f.name
    
    try:
        # 分析测试数据
        analysis_result = analyze_log_file(temp_file)
        
        # 验证分析结果
        assert "peak_memory" in analysis_result, "缺少峰值显存数据"
        assert "incremental_memory" in analysis_result, "缺少增量显存数据"
        assert "raw_events" in analysis_result, "缺少原始事件数据"
        
        peak_memory = analysis_result["peak_memory"]
        incremental_memory = analysis_result["incremental_memory"]
        
        # 验证关键数据
        assert "model_loaded" in peak_memory, "缺少模型加载峰值数据"
        assert "t5_loaded" in incremental_memory, "缺少T5增量数据"
        assert "vae_loaded" in incremental_memory, "缺少VAE增量数据"
        assert "dit_loaded" in incremental_memory, "缺少DiT增量数据"
        
        print(f"  ✓ 解析了 {len(analysis_result['raw_events'])} 个事件")
        print(f"  ✓ 峰值显存事件: {len(peak_memory)} 个")
        print(f"  ✓ 增量显存事件: {len(incremental_memory)} 个")
        
        # 验证数值范围
        t5_mem = incremental_memory.get("t5_loaded", 0)
        assert 10000 < t5_mem < 12000, f"T5显存数值异常: {t5_mem} MB"
        
        model_peak = peak_memory.get("model_loaded", 0)
        assert 20000 < model_peak < 30000, f"模型峰值显存异常: {model_peak} MB"
        
        print("日志分析测试通过!\n")
        
    finally:
        # 清理临时文件
        os.unlink(temp_file)

def test_report_generation():
    """测试报告生成功能"""
    print("测试报告生成功能...")
    
    # 创建测试数据
    test_events = create_test_data()
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_events, f, indent=2)
        temp_file = f.name
    
    try:
        # 分析数据
        analysis_result = analyze_log_file(temp_file)
        
        # 生成报告
        report = generate_report(analysis_result, temp_file)
        
        # 验证报告内容
        assert "# Wan2.1-Enhanced 显存消耗分析报告" in report, "缺少报告标题"
        assert "## 概述" in report, "缺少概述部分"
        assert "## 显存使用分析" in report, "缺少分析部分"
        assert "### 1. 模型加载阶段显存消耗" in report, "缺少模型加载分析"
        assert "### 2. 关键阶段显存峰值" in report, "缺少峰值分析"
        assert "## 性能优化建议" in report, "缺少优化建议"
        assert "## 结论与建议" in report, "缺少结论部分"
        
        # 验证数据表格
        assert "T5 编码器" in report, "缺少T5编码器分析"
        assert "VAE 模型" in report, "缺少VAE模型分析"
        assert "DiT 模型" in report, "缺少DiT模型分析"
        
        # 验证优化建议
        assert "INT8 量化" in report, "缺少量化建议"
        assert "激活显存管理" in report, "缺少激活显存建议"
        
        print(f"  ✓ 报告长度: {len(report)} 字符")
        report_lines = len(report.split('\n'))
        print(f"  ✓ 报告行数: {report_lines} 行")
        print("  ✓ 包含所有必要章节")
        print("  ✓ 包含详细的分析数据")
        print("  ✓ 包含优化建议")
        
        print("报告生成测试通过!\n")
        
    finally:
        # 清理临时文件
        os.unlink(temp_file)

def test_error_handling():
    """测试错误处理"""
    print("测试错误处理功能...")
    
    # 测试不存在的文件
    result = analyze_log_file("nonexistent_file.json")
    assert not result or not result.get("raw_events"), "应该返回空结果"
    print("  ✓ 不存在文件的错误处理")
    
    # 测试无效JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("invalid json content")
        temp_file = f.name
    
    try:
        result = analyze_log_file(temp_file)
        assert not result or not result.get("raw_events"), "应该返回空结果"
        print("  ✓ 无效JSON的错误处理")
    finally:
        os.unlink(temp_file)
    
    print("错误处理测试通过!\n")

def main():
    """运行所有测试"""
    print("=== 显存分析脚本功能测试 ===\n")
    
    try:
        test_format_conversion()
        test_log_analysis()
        test_report_generation()
        test_error_handling()
        
        print("🎉 所有测试通过!")
        print("\n分析脚本功能正常，可以用于生成详细的显存分析报告。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()