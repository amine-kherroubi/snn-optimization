# Memory Management Strategies for GPU-Accelerated Shallow Neural Network Training

This project evaluates memory management optimizations for CUDA-based shallow neural network training, conducted at the École Nationale Supérieure d'Informatique (ESI), Algiers, in February 2026. Building upon the reference implementation by Brouthen and Akeb [1], we investigate three optimization strategies that address GPU memory allocation overhead and transfer efficiency.

## Overview

The reference implementation allocates and frees GPU memory for every matrix operation, leading to tens of thousands of allocation-deallocation cycles during training. We implement and evaluate three alternatives: a streams-based approach using CUDA streams for concurrent execution, a pinned memory strategy using page-locked host memory, and a combined approach integrating both optimizations with pre-allocated GPU resources. Experiments on a Tesla T4 GPU demonstrate that the combined strategy achieves approximately 1.65× to 1.73× speedup over the reference implementation for the baseline network configuration (256 neurons, 1 hidden layer). Additional scalability analysis reveals that optimization effectiveness is highly configuration-dependent, with benefits diminishing for larger networks (1024 neurons) and deeper architectures (3 hidden layers).

## Key Results

For the baseline network configuration across three dataset sizes, the combined strategy consistently outperforms the reference implementation (1.65× to 1.73× speedup), while streams alone provide minimal improvement (0.99× to 1.12×) and pinned memory yields modest gains (1.04× to 1.18×). Scalability experiments show that speedup varies from 2.14× for moderate configurations (128 neurons) to 0.93× for large configurations (1024 neurons), demonstrating that memory pooling optimizations are most effective when allocation overhead dominates relative to computation time. Network depth analysis reveals progressive degradation: two hidden layers maintain 1.52× speedup, while three hidden layers achieve only 1.01× speedup due to increased computational workload.

## Requirements

- CUDA Toolkit 11.8 or later
- GCC 9.4.0 or compatible host compiler
- NVIDIA GPU with compute capability 7.5 or higher (tested on Tesla T4)
- OpenMP for timing (`-fopenmp`)

## Building

Compile any implementation variant with:
```bash
nvcc -O3 -Xcompiler -fopenmp <filename>.cu -o <output_name>
```

For example:
```bash
nvcc -O3 -Xcompiler -fopenmp alternatives/nn_cuda_combined.cu -o nn_cuda_combined
```

## Usage

Run training with a synthetic dataset:
```bash
./<executable> <dataset.csv>
```

Example:
```bash
./nn_cuda_combined reference/data/synthetic_convex_large.csv
```

The program outputs average training time and final MSE over 10 independent runs.

## Repository Structure

- `alternatives/` — Optimized CUDA implementations
  - `nn_cuda_reference.cu` — Baseline reference implementation
  - `nn_cuda_streams.cu` — Streams-based optimization
  - `nn_cuda_pinned.cu` — Pinned memory optimization
  - `nn_cuda_combined.cu` — Combined optimization strategy
  - `nn_cuda_*_two_layers.cu` / `nn_cuda_*_three_layers.cu` — Network depth variants
- `reference/data/` — Synthetic convex datasets (small, medium, large)
- `report/` — Technical report (Typst source and figures)

## Authors

Mohamed El Amine Kherroubi, Badis Khalef, Mounir Sofiane Mostefai, Youcef Tati, Mohamed Ishak Messadia  
2CS-SIQ/SID, École Nationale Supérieure d'Informatique (ESI), Algiers

## Acknowledgments

This work builds upon the baseline CUDA implementation by Brouthen Kamel and Akeb Abdelaziz [1]. We thank Professor Dr. Amina Selma Haichour for her guidance throughout this project.

## References

[1] Brouthen, K., & Akeb, A. (2024). Exploring parallelization of shallow neural network using CUDA.