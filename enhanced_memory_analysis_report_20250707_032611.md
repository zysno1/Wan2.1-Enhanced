# Memory and Performance Analysis Report

This report details the memory consumption and performance for different model components and configurations.

## Test Configuration: `memory_events_baseline_20250707_032609`

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
| All Models Loading | 105.21       |
| T5 Model Loading   | 66.98       |
| T5 Encode          | 0.95       |
| DiT Forward        | 224.06       |
| VAE Decode         | 9.15       |
| Total Generation   | 234.37       |
|--------------------|--------------|
