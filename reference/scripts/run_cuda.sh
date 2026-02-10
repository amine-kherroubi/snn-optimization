#!/usr/bin/env bash
set -e

# Compile sequential version
gcc -o ../nn ../nn.c -lm -pg -fopenmp

# Run sequential version on the datasets

echo "Small Dataset (256 samples)"
../nn ../data/synthetic_convex_small.csv
echo

# echo "Medium Dataset (1024 samples)"
# ../nn ../data/synthetic_convex_medium.csv
# echo

# echo "Large Dataset (4096 samples)"
# ../nn ../data/synthetic_convex_large.csv

# Run sequential version and log output
# ../nn ../data/synthetic_convex_large.csv > ../log/log.txt