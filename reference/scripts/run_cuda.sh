#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Compile CUDA version
nvcc -o "$ROOT_DIR/nn_cuda" "$ROOT_DIR/nn_cuda.cu" -Xcompiler -fopenmp \
  -gencode arch=compute_75,code=sm_75 \
  -Xptxas=-v && echo "Compiled nn_cuda.cu successfully"

# Run CUDA version on the datasets

echo "Small Dataset (256 samples)"
"$ROOT_DIR/nn_cuda" "$ROOT_DIR/data/synthetic_convex_small.csv"
echo

echo "Medium Dataset (1024 samples)"
"$ROOT_DIR/nn_cuda" "$ROOT_DIR/data/synthetic_convex_medium.csv"
echo

echo "Large Dataset (4096 samples)"
"$ROOT_DIR/nn_cuda" "$ROOT_DIR/data/synthetic_convex_large.csv"

rm -f "$ROOT_DIR/nn_cuda"