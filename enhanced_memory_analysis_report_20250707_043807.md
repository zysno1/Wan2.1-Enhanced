# Memory and Performance Analysis Report

This report details the memory consumption and performance for different model components and configurations.

## Test Configuration: `memory_events_CPU-offload_20250707_043806`

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

### Performance Analysis

#### Timing Information

| Stage              | Duration (s) |
|--------------------|--------------|
| All Models Loading | 142.58       |
| T5 Model Loading   | 111.08       |
| T5 Encode          | 397.61       |
| DiT Forward        | 224.19       |
| VAE Decode         | 9.08       |
| Total Generation   | 633.89       |
|--------------------|--------------|
