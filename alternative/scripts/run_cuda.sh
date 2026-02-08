#!/usr/bin/env bash
set -e

# Compile CUDA version
nvcc ../nn_cuda_optimized.cu -o ../nn_cuda_optimized -Xcompiler -fopenmp \
  -gencode arch=compute_75,code=sm_75 \
  -Xptxas=-v

echo "Compiled nn_cuda_optimized.cu successfully"
echo

# Run CUDA version on datasets
echo "--- Small Dataset (256 samples) ---"
../nn_cuda_optimized ../../reference/data/synthetic_convex_small.csv
echo

echo "--- Medium Dataset (2560 samples) ---"
../nn_cuda_optimized ../../reference/data/synthetic_convex_medium.csv
echo

echo "--- Large Dataset (25600 samples) ---"
../nn_cuda_optimized ../../reference/data/synthetic_convex_large.csv
