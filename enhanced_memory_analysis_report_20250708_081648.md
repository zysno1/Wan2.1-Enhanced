# Memory and Performance Analysis Report

This report details the memory consumption and performance for different model components and configurations.

## Test Configuration: `memory_events_baseline_20250708_081647`

### Memory Analysis

#### Total Peak Memory

| Metric | Memory (MB) | Memory (GB) |
|--------|-------------|-------------|
| **Total Peak Memory** | **24656.08** | **24.08** |

#### Model Components

| Component | Base Memory (MB) | Activation Memory (MB) | KV Cache (MB) |
|-----------|------------------|------------------------|---------------|
| DiT | 5439.81 | 3226.38 | - |
| T5 | 10836.48 | 17.00 | 17.00 |
| VAE | 485.95 | 370.20 | - |
|-----------|------------------|------------------------|---------------|
| **Total** | **16762.24** | **3613.57** | **17.00** |

#### Runtime Memory

*Note: This represents pure PyTorch/CUDA runtime overhead, excluding model base memory, activation memory, and KV cache.*

| Component | Memory (MB) |
|-----------|-------------|
| PyTorch/CUDA Runtime Overhead | 4263.27 |

### Performance Analysis

#### Timing Information

| Stage              | Duration (s) |
|--------------------|--------------|
| All Models Loading | 80.93       |
| T5 Model Loading   | 50.82       |
| T5 Encode          | 0.90       |
| DiT Forward        | 223.40       |
| VAE Decode         | 9.13       |
| Total Generation   | 233.63       |
|--------------------|--------------|
