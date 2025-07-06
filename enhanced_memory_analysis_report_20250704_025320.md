# Memory and Performance Analysis Report

This report details the memory consumption and performance for different model components and configurations.

## Test Configuration: `memory_usage`

### Memory Analysis

| Component      | Model Load (MB) | Activation (MB) | KV Cache (MB) | PyTorch/CUDA Runtime (MB) |
|----------------|-----------------|-----------------|---------------|---------------------------|
|----------------|-----------------|-----------------|---------------|---------------------------|
| **Total**      | **0.00**    | **0.00**      | **0.00**    | **0.00**           |

### Performance Analysis

| Stage              | Duration (s) |
|--------------------|--------------|
| DiT Forward        | 223.44       |
| T5 Encode          | 0.43       |
| Total Generation   | 241.22       |
| VAE Decode         | 9.03       |
|--------------------|--------------|

