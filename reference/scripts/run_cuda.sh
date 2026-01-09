# Compile CUDA version
nvcc -o reference/nn_cuda reference/nn_cuda.cu -Xcompiler -fopenmp

# Run CUDA version on large dataset
# ./reference/nn_cuda reference/data/synthetic_convex_small.csv
# ./reference/nn_cuda reference/data/synthetic_convex_medium.csv
./reference/nn_cuda reference/data/synthetic_convex_large.csv