# Compile sequential version
gcc -o reference/nn reference/nn.c -lm -pg -fopenmp

# Run sequential version on large dataset
# ./reference/nn reference/data/synthetic_convex_small.csv
# ./reference/nn reference/data/synthetic_convex_medium.csv
./reference/nn reference/data/synthetic_convex_large.csv

# Run sequential version and log output
# ./reference/nn reference/data/synthetic_convex_large.csv > reference/log/log.txt