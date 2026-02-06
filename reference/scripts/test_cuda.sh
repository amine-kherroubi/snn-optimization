# Compile and run test CUDA program
nvcc -o ../test_cuda ../test_cuda.cu
../test_cuda

rm -rf ../test_cuda