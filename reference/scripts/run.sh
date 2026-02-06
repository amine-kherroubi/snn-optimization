#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Compile sequential version
gcc -o "$ROOT_DIR/nn" "$ROOT_DIR/nn.c" -lm -pg -fopenmp

# Run sequential version on the datasets

echo "Small Dataset (256 samples)"
"$ROOT_DIR/nn" "$ROOT_DIR/data/synthetic_convex_small.csv"
echo

echo "Medium Dataset (1024 samples)"
"$ROOT_DIR/nn" "$ROOT_DIR/data/synthetic_convex_medium.csv"
echo

echo "Large Dataset (4096 samples)"
"$ROOT_DIR/nn" "$ROOT_DIR/data/synthetic_convex_large.csv"

# Run sequential version and log output

"$ROOT_DIR/nn" "$ROOT_DIR/data/synthetic_convex_large.csv" > "$ROOT_DIR/log/log.txt"

rm -rf "$ROOT_DIR/nn" "$ROOT_DIR/gmon.out"
