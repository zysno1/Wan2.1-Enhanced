# Memory and Performance Analysis Report

This report details the memory consumption and performance for different model components and configurations.

## Test Configuration: `memory_events_baseline_20250707_015258`

### Memory Analysis

#### Total Peak Memory

| Metric | Memory (MB) | Memory (GB) |
|--------|-------------|-------------|
| **Total Peak Memory** | **8392.35** | **8.20** |

#### Model Components

| Component | Base Memory (MB) | Activation Memory (MB) | KV Cache (MB) |
|-----------|------------------|------------------------|---------------|
| DiT | 5413.04 | 3608.07 | - |
| T5 | - | 5920.00 | 5920.00 |
| VAE | 505.75 | 370.20 | - |
|-----------|------------------|------------------------|---------------|
| **Total** | **5918.79** | **9898.26** | **5920.00** |

#### Runtime Memory

*Note: This represents pure PyTorch/CUDA runtime overhead, excluding model base memory, activation memory, and KV cache.*

| Component | Memory (MB) |
|-----------|-------------|
| PyTorch/CUDA Runtime Overhead | 8392.35 |

### Performance Analysis

#### Timing Information

| Stage              | Duration (s) |
|--------------------|--------------|
| All Models Loading | 123.92       |
| T5 Model Loading   | 103.04       |
| T5 Encode          | 275.34       |
| DiT Forward        | 223.00       |
| VAE Decode         | 9.05       |
| Total Generation   | 510.55       |
|--------------------|--------------|
