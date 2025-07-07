# Memory and Performance Analysis Report

This report details the memory consumption and performance for different model components and configurations.

## Test Configuration: `memory_events_CPU-offload_20250707_034548`

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
| All Models Loading | 147.94       |
| T5 Model Loading   | 117.46       |
| T5 Encode          | 267.92       |
| DiT Forward        | 223.96       |
| VAE Decode         | 9.12       |
| Total Generation   | 504.14       |
|--------------------|--------------|
