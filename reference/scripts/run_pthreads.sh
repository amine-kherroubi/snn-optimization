#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Compile pthreads version
gcc -o "$ROOT_DIR/nn_pthreads" "$ROOT_DIR/nn_pthreads.c" -lm -pthread -pg -fopenmp

# Run pthreads version on the datasets

echo "Small Dataset (256 samples)"
"$ROOT_DIR/nn_pthreads" "$ROOT_DIR/data/synthetic_convex_small.csv"
echo 

echo "Medium Dataset (1024 samples)"
"$ROOT_DIR/nn_pthreads" "$ROOT_DIR/data/synthetic_convex_medium.csv"
echo

echo "Large Dataset (4096 samples)"
"$ROOT_DIR/nn_pthreads" "$ROOT_DIR/data/synthetic_convex_large.csv"

rm -rf "$ROOT_DIR/nn_pthreads" "$ROOT_DIR/gmon.out"