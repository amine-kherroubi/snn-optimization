#!/usr/bin/env bash
set -e

# Compile pthreads version
gcc -o ../nn_pthreads ../nn_pthreads.c -lm -pthread -pg -fopenmp

# Run pthreads version on the datasets

echo "Small Dataset (256 samples)"
../nn_pthreads ../data/synthetic_convex_small.csv
echo 

# echo "Medium Dataset (1024 samples)"
# ../nn_pthreads ../data/synthetic_convex_medium.csv
# echo

# echo "Large Dataset (4096 samples)"
# ../nn_pthreads ../data/synthetic_convex_large.csv