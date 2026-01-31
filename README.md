# SC4064 Assignment 1: CUDA Programming

This repository contains CUDA implementations for basic parallel computing operations: vector addition, matrix addition, and matrix multiplication.

## Project Structure

```
.
├── environment.yaml    # Conda environment specification
├── vec_add.cu      # Task 1 source code
├── vec_add         # Task 1 compiled binary
├── mat_add.cu      # Task 2 source code
├── mat_add         # Task 2 compiled binary
├── mat_mul.cu      # Task 3 source code
├── mat_mul         # Task 3 compiled binary
├── report.pdf      # Assignment 1 pdf report
└── README.md
```

---

## System Requirements

The code was developed and tested on the following system:

| Component        | Specification                          |
|------------------|----------------------------------------|
| OS               | Ubuntu 24.04.3 LTS                     |
| Kernel           | 5.14.0-284.25.1.el9_2.x86_64           |
| Architecture     | x86_64                                 |
| GPU              | NVIDIA H100 80GB HBM3                  |
| NVIDIA Driver    | 550.90.07                              |
| CUDA Version     | 13.1                                   |

---

## Environment Setup

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- NVIDIA GPU with appropriate drivers installed

### Setup with the Provided Environment File

Clone the repository and create the environment from the provided `environment.yaml`:

```bash
# Create the conda environment from the YAML file
conda env create -f environment.yaml

# Activate the environment
conda activate cuda_build
```

### Configure Environment Variables

After activating the environment, set the required environment variables:

```bash
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
```

### Verify Installation

Confirm that the CUDA compiler and C++ compiler are correctly installed:

```bash
nvcc --version
x86_64-conda-linux-gnu-g++ --version
```

---

## Compilation & Execution

All programs use the `-arch=sm_90` flag optimized for the H100 GPU. Adjust this flag according to your GPU's compute capability if needed (e.g., `sm_80` for A100, `sm_86` for RTX 3090).

### Task 1: Vector Addition

```bash
nvcc -O3 -arch=sm_90 -o vec_add vec_add.cu
./vec_add
```

**Results:**

| Block Size | Time (ms) | GFLOPS  |
|------------|-----------|---------|
| 32         | 21.1654   | 50.73   |
| 64         | 10.0874   | 106.44  |
| 128        | 5.0725    | 211.68  |
| 256        | 4.5934    | 233.76  |

### Task 2: Matrix Addition

```bash
nvcc -O3 -arch=sm_90 -o mat_add mat_add.cu
./mat_add
```

**Results:**

| Kernel Type | Time (ms) | FLOPS       |
|-------------|-----------|-------------|
| 1D Kernel   | 1.4634    | 9.17e+10    |
| 2D Kernel   | 0.3508    | 3.83e+11    |

### Task 3: Matrix Multiplication

```bash
nvcc -O3 -arch=sm_90 -o mat_mul mat_mul.cu
./mat_mul
```

**Results:**

| Block Size | Time (ms) | TFLOPS |
|------------|-----------|--------|
| 8×8        | 328.22    | 3.35   |
| 16×16      | 212.96    | 5.16   |
| 32×32      | 185.26    | 5.94   |

---
