#!/usr/bin/env bash
set -e

# Array of CUDA variants to test
variants=("ref" "streams" "pinned" "persistent")

# Compile all variants
echo "=============================="
echo "Compiling CUDA variants..."
echo "=============================="

for variant in "${variants[@]}"; do
  echo "Compiling nn_cuda_${variant}.cu..."
  nvcc -o ../nn_cuda_${variant} ../nn_cuda_${variant}.cu -Xcompiler -fopenmp \
    -gencode arch=compute_75,code=sm_75 \
    -Xptxas=-v && echo "✓ Compiled nn_cuda_${variant}.cu successfully"
  echo
done

# Datasets to test
datasets=("small" "medium" "large")

# Run each variant on each dataset
echo "=============================="
echo "Running Performance Tests"
echo "=============================="
echo

for variant in "${variants[@]}"; do
  echo "=============================="
  echo "Testing: nn_cuda_${variant}"
  echo "=============================="
  
  for dataset in "${datasets[@]}"; do
    case $dataset in
      small)
        echo "→ Small Dataset (256 samples)"
        ;;
      medium)
        echo "→ Medium Dataset (2560 samples)"
        ;;
      large)
        echo "→ Large Dataset (25600 samples)"
        ;;
    esac
    
    ../nn_cuda_${variant} ../data/synthetic_convex_${dataset}.csv
    echo
  done
  
  echo
done