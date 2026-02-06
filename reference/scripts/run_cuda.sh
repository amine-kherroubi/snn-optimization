# Compile CUDA version
nvcc -o ../nn_cuda ../nn_cuda.cu  -Xcompiler -fopenmp \
 -gencode arch=compute_75,code=sm_75 \
 -Xptxas=-v && echo "Compiled nn_cuda.cu successfully"

# Run CUDA version on Small/Medium dataset

echo "--- Small Dataset (256 samples) ---"
../nn_cuda ../data/synthetic_convex_small.csv
echo 

echo "--- Medium Dataset (1024 samples) ---"
../nn_cuda ../data/synthetic_convex_medium.csv
echo

# echo "--- Large Dataset (4096 samples) ---"
# ../nn_cuda ../data/synthetic_convex_large.csv


rm -rf ../nn_cuda