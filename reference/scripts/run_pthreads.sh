# Compile pthreads version
gcc -o reference/nn_pthreads reference/nn_pthreads.c -lm -pthread -pg -fopenmp

# Run pthreads version on large dataset
# ./reference/nn_pthreads reference/data/synthetic_convex_small.csv
# ./reference/nn_pthreads reference/data/synthetic_convex_medium.csv
./reference/nn_pthreads reference/data/synthetic_convex_large.csv