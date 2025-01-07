# llm.c

# LLM Computation Acceleration with Tensor Core

## Project Overview

We are accelerating Large Language Model (LLM) computation using NVIDIA Tensor Cores, focusing on mixed precision techniques via cuBLAS. We'll use [Karpathy's llm.c](https://github.com/karpathy/llm.c) as our base implementation.

## Team Members and Responsibilities

### Member A (Tingting): Background Research and Problem Analysis

- **Tasks:**
  - Research Tensor Core technology and mixed precision computation.
  - Identify performance bottlenecks using profiling tools (e.g., NVIDIA Nsight).
  - Write the **Introduction**, **Result** and **References** sections of the report.

### Member B (Tianyi): Optimization Design and Implementation

- **Tasks:**
  - Design optimization strategies using cuBLAS.
  - Implement mixed precision computation in the code.
  - Write the **Methodology** section of the report.

### Member C (Jiacheng): Experimentation and Results Analysis

- **Tasks:**
  - Set up the GPU environment and necessary software.
  - Analyze results and compare with the original implementation.
  - Write the **Experimental Setup**, **Results**, and **Discussion and Conclusion** sections of the report.




## quick start (1 GPU, BF16 only)

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2cu PRECISION=BF16 NO_MULTI_GPU=1
./train_gpt2cu 
```

## (1 GPU, FP32 only)

```bash
make clean
make train_gpt2fp32cu PRECISION=FP32 NO_MULTI_GPU=1
./train_gpt2fp32cu
```