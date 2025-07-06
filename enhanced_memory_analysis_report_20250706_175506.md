# Memory and Performance Analysis Report

This report details the memory consumption and performance for different model components and configurations.

## Test Configuration: `memory_events_baseline_20250706_175505`

### Memory Analysis

#### Model Components

| Component | Base Memory (MB) | Activation Memory (MB) | KV Cache (MB) |
|-----------|------------------|------------------------|---------------|
| DiT | 5439.81 | 3226.79 | - |
| T5 | 10836.48 | 17.00 | 17.00 |
| VAE | 485.95 | 370.20 | - |
|-----------|------------------|------------------------|---------------|
| **Total** | **16762.24** | **3613.99** | **17.00** |

#### Runtime Memory

*Note: This represents pure PyTorch/CUDA runtime overhead, excluding model base memory, activation memory, and KV cache.*

| Component | Memory (MB) |
|-----------|-------------|
| PyTorch/CUDA Runtime Overhead | 3281.73 |

### Performance Analysis

#### Timing Information

| Stage              | Duration (s) |
|--------------------|--------------|
| Model Loading      | 10.38       |
| T5 Encode          | 0.42       |
| DiT Forward        | 223.07       |
| VAE Decode         | 8.99       |
| Total Generation   | 241.18       |
|--------------------|--------------|
