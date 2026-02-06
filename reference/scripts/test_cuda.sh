#!/usr/bin/env bash
set -e

# Compile test_cuda
nvcc -o ../test_cuda ../test_cuda.cu

# Run it
../test_cuda