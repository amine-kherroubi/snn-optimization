#!/usr/bin/env bash
set -e

# Array of CUDA variants to test
variants=("reference" "streams" "pinned" "combined")

# Compile CUDA variants
for variant in "${variants[@]}"; do
  nvcc -o nn_cuda_${variant} nn_cuda_${variant}.cu -Xcompiler -fopenmp \
    -gencode arch=compute_75,code=sm_75 \
    -Xptxas=-v
done

# Datasets to test
datasets=("small" "medium" "large")

# Run each variant on each dataset
for variant in "${variants[@]}"; do
  echo "Testing: nn_cuda_${variant}"
  
  for dataset in "${datasets[@]}"; do
    echo "Dataset: $dataset"
    ./nn_cuda_${variant} ../reference/data/synthetic_convex_${dataset}.csv
    echo
  done
  echo
done