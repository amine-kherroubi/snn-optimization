// Optimized Parallel matrix multiplication using CUDA
// Improvements:
// - Pinned memory (cudaMallocHost) for faster transfers
// - CUDA events for timing
// - Transposed B matrix for coalesced memory access
// - CUDA streams for overlapping copy and compute (kernel fission)
// - Increased threads per block (32) for better occupancy
// - OpenMP parallelization for helper functions
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ! Network Parameters
#define INPUT_SIZE 32     // Number of input features
#define HIDDEN_SIZE 256   // Number of neurons in the hidden layer
#define OUTPUT_SIZE 1     // Number of output neurons
#define EPOCHS 100        // Number of training epochs
#define LOG_EVERY_EPOCH 1 // Log loss every n epochs
#define LEARNING_RATE 0.002
#define BATCH_SIZE 256 // Batch size for SGD
#define THREADS_PER_BLOCK 32 // Increased from 16 to 32 for better occupancy
#define NUM_STREAMS 2 // Number of CUDA streams for overlapping

// ! Data Structures
// Structure for matrix with optional pinned memory flag
typedef struct {
  int rows;
  int cols;
  float *data;
  int pinned; // Flag indicating if memory is pinned
} Matrix;

// ! Memory Management
// Function to allocate a matrix with pinned memory for faster GPU transfers
Matrix *allocate_matrix(int rows, int cols) {
  Matrix *m = (Matrix *)malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->pinned = 0;
  m->data = (float *)malloc(rows * cols * sizeof(float));
  return m;
}

// Function to allocate a matrix with pinned (page-locked) memory
Matrix *allocate_matrix_pinned(int rows, int cols) {
  Matrix *m = (Matrix *)malloc(sizeof(Matrix));
  m->rows = rows;
  m->cols = cols;
  m->pinned = 1;
  cudaError_t err = cudaMallocHost((void **)&m->data, rows * cols * sizeof(float));
  if (err != cudaSuccess) {
    printf("CUDA Host Alloc Error: %s\n", cudaGetErrorString(err));
    // Fallback to regular malloc
    m->data = (float *)malloc(rows * cols * sizeof(float));
    m->pinned = 0;
  }
  return m;
}

// Function to free a matrix
void free_matrix(Matrix *m) {
  if (m->pinned) {
    cudaFreeHost(m->data);
  } else {
    free(m->data);
  }
  free(m);
}

// ! Matrix Operations
// Function to initialize matrix with random values using He initialization
// Parallelized with OpenMP
void random_init(Matrix *m) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      // Thread-safe random initialization (each thread uses its own seed based on position)
      unsigned int seed = i * m->cols + j;
      m->data[i * m->cols + j] = (float)rand_r(&seed) / RAND_MAX;
    }
  }
}

// ! Matrix Operations (GPU version)
// Optimized kernel with coalesced memory access
// B is expected to be transposed (B_T) so we access B_T[col][k] = B_T[col * A_cols + k]
__global__ void mat_mult_kernel_transposed(float *A, float *B_T, float *C, 
                                           int A_rows, int A_cols, int B_cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < A_rows && col < B_cols) {
    float value = 0.0f;
    // Now both A and B_T are accessed in a more cache-friendly manner
    for (int k = 0; k < A_cols; k++) {
      value += A[row * A_cols + k] * B_T[col * A_cols + k];
    }
    C[row * B_cols + col] = value;
  }
}

// Standard kernel for cases where transpose is not beneficial
__global__ void mat_mult_kernel(float *A, float *B, float *C, int A_rows,
                                int A_cols, int B_cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < A_rows && col < B_cols) {
    float value = 0.0f;
    for (int k = 0; k < A_cols; k++) {
      value += A[row * A_cols + k] * B[k * B_cols + col];
    }
    C[row * B_cols + col] = value;
  }
}

// Kernel to transpose a matrix
__global__ void transpose_kernel(float *in, float *out, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < rows && col < cols) {
    out[col * rows + row] = in[row * cols + col];
  }
}

// GPU-accelerated matrix transpose
void transpose_gpu(float *d_in, float *d_out, int rows, int cols, cudaStream_t stream) {
  dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  dim3 numBlocks((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
  transpose_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_in, d_out, rows, cols);
}

// Persistent device memory for weight matrices to avoid repeated allocations
typedef struct {
  float *d_W1, *d_W2;
  float *d_W1_T, *d_W2_T; // Transposed versions for optimized access
  float *d_X, *d_Y;
  float *d_Z1, *d_Y_pred;
  float *d_temp1, *d_temp2; // Temporary buffers
  cudaStream_t streams[NUM_STREAMS];
  cudaEvent_t start, stop;
  int max_batch_size;
  int initialized;
} GPUContext;

GPUContext gpu_ctx = {0};

// Initialize GPU context with persistent memory
void init_gpu_context(int max_batch_size) {
  gpu_ctx.max_batch_size = max_batch_size;
  
  // Allocate weight matrices
  cudaMalloc(&gpu_ctx.d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&gpu_ctx.d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
  cudaMalloc(&gpu_ctx.d_W1_T, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
  cudaMalloc(&gpu_ctx.d_W2_T, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
  
  // Allocate batch buffers
  cudaMalloc(&gpu_ctx.d_X, max_batch_size * INPUT_SIZE * sizeof(float));
  cudaMalloc(&gpu_ctx.d_Y, max_batch_size * OUTPUT_SIZE * sizeof(float));
  cudaMalloc(&gpu_ctx.d_Z1, max_batch_size * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&gpu_ctx.d_Y_pred, max_batch_size * OUTPUT_SIZE * sizeof(float));
  
  // Temporary buffers for intermediate computations
  cudaMalloc(&gpu_ctx.d_temp1, max_batch_size * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&gpu_ctx.d_temp2, HIDDEN_SIZE * max_batch_size * sizeof(float));
  
  // Create streams for overlapping operations
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamCreate(&gpu_ctx.streams[i]);
  }
  
  // Create events for timing
  cudaEventCreate(&gpu_ctx.start);
  cudaEventCreate(&gpu_ctx.stop);
  
  gpu_ctx.initialized = 1;
}

// Cleanup GPU context
void cleanup_gpu_context() {
  if (!gpu_ctx.initialized) return;
  
  cudaFree(gpu_ctx.d_W1);
  cudaFree(gpu_ctx.d_W2);
  cudaFree(gpu_ctx.d_W1_T);
  cudaFree(gpu_ctx.d_W2_T);
  cudaFree(gpu_ctx.d_X);
  cudaFree(gpu_ctx.d_Y);
  cudaFree(gpu_ctx.d_Z1);
  cudaFree(gpu_ctx.d_Y_pred);
  cudaFree(gpu_ctx.d_temp1);
  cudaFree(gpu_ctx.d_temp2);
  
  for (int i = 0; i < NUM_STREAMS; i++) {
    cudaStreamDestroy(gpu_ctx.streams[i]);
  }
  
  cudaEventDestroy(gpu_ctx.start);
  cudaEventDestroy(gpu_ctx.stop);
  
  gpu_ctx.initialized = 0;
}

// Upload weights to GPU and compute transposed versions
void upload_weights_to_gpu(Matrix *W1, Matrix *W2) {
  cudaMemcpyAsync(gpu_ctx.d_W1, W1->data, INPUT_SIZE * HIDDEN_SIZE * sizeof(float),
                  cudaMemcpyHostToDevice, gpu_ctx.streams[0]);
  cudaMemcpyAsync(gpu_ctx.d_W2, W2->data, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float),
                  cudaMemcpyHostToDevice, gpu_ctx.streams[1]);
  
  // Compute transposed versions on GPU
  transpose_gpu(gpu_ctx.d_W1, gpu_ctx.d_W1_T, INPUT_SIZE, HIDDEN_SIZE, gpu_ctx.streams[0]);
  transpose_gpu(gpu_ctx.d_W2, gpu_ctx.d_W2_T, HIDDEN_SIZE, OUTPUT_SIZE, gpu_ctx.streams[1]);
  
  cudaDeviceSynchronize();
}

// Download weights from GPU
void download_weights_from_gpu(Matrix *W1, Matrix *W2) {
  cudaMemcpyAsync(W1->data, gpu_ctx.d_W1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float),
                  cudaMemcpyDeviceToHost, gpu_ctx.streams[0]);
  cudaMemcpyAsync(W2->data, gpu_ctx.d_W2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float),
                  cudaMemcpyDeviceToHost, gpu_ctx.streams[1]);
  cudaDeviceSynchronize();
}

// Function to multiply matrices on the GPU (standard version for general use)
Matrix *mat_mult(Matrix *A, Matrix *B) {
  if (A->cols != B->rows) {
    printf("Incompatible matrices for multiplication.\n");
    exit(1);
  }

  Matrix *C = allocate_matrix(A->rows, B->cols);

  float *d_A, *d_B, *d_C;
  size_t sizeA = A->rows * A->cols * sizeof(float);
  size_t sizeB = B->rows * B->cols * sizeof(float);
  size_t sizeC = C->rows * C->cols * sizeof(float);

  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, A->data, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B->data, sizeB, cudaMemcpyHostToDevice);

  dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  dim3 numBlocks((B->cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (A->rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

  mat_mult_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, A->rows,
                                                  A->cols, B->cols);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    exit(1);
  }

  cudaDeviceSynchronize();
  cudaMemcpy(C->data, d_C, sizeC, cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return C;
}

// Matrix subtraction: C = A - B (OpenMP parallelized)
Matrix *mat_sub(Matrix *A, Matrix *B) {
  if (A->rows != B->rows || A->cols != B->cols) {
    printf("Incompatible matrices for subtraction.\n");
    exit(1);
  }
  Matrix *C = allocate_matrix(A->rows, A->cols);
  
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < A->rows; i++)
    for (int j = 0; j < A->cols; j++)
      C->data[i * A->cols + j] =
          A->data[i * A->cols + j] - B->data[i * A->cols + j];
  return C;
}

// Matrix scalar multiplication: A = A * scalar (OpenMP parallelized)
void mat_scalar_mult(Matrix *A, float scalar) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < A->rows; i++)
    for (int j = 0; j < A->cols; j++)
      A->data[i * A->cols + j] *= scalar;
}

// ! Activation Functions
// Function to apply ReLU activation (OpenMP parallelized)
void relu(Matrix *m) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < m->rows; i++)
    for (int j = 0; j < m->cols; j++)
      m->data[i * m->cols + j] = fmaxf(0, m->data[i * m->cols + j]);
}

// CUDA kernel for ReLU
__global__ void relu_kernel(float *data, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    data[idx] = fmaxf(0.0f, data[idx]);
  }
}

// Function to compute derivative of ReLU (OpenMP parallelized)
Matrix *relu_derivative(Matrix *m) {
  Matrix *derivative = allocate_matrix(m->rows, m->cols);
  
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < m->rows; i++)
    for (int j = 0; j < m->cols; j++)
      derivative->data[i * m->cols + j] =
          (m->data[i * m->cols + j] > 0) ? 1.0f : 0.0f;
  return derivative;
}

// CUDA kernel for ReLU derivative (element-wise multiplication)
__global__ void relu_derivative_mult_kernel(float *dZ, float *Z, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dZ[idx] *= (Z[idx] > 0.0f) ? 1.0f : 0.0f;
  }
}

// ! Loss Functions
// Function to compute Mean Squared Error (OpenMP parallelized)
float mean_squared_error(Matrix *Y_pred, Matrix *Y_true) {
  float mse = 0.0f;
  
  #pragma omp parallel for collapse(2) reduction(+:mse)
  for (int i = 0; i < Y_pred->rows; i++)
    for (int j = 0; j < Y_pred->cols; j++)
      mse += powf(Y_pred->data[i * Y_pred->cols + j] -
                  Y_true->data[i * Y_true->cols + j], 2);
  return mse / Y_pred->rows;
}

// ! Optimization
// Function to update weights: W = W - learning_rate * grad (OpenMP parallelized)
void update_weights(Matrix *W, Matrix *grad, float learning_rate) {
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < W->rows; i++)
    for (int j = 0; j < W->cols; j++)
      W->data[i * W->cols + j] -=
          learning_rate * grad->data[i * grad->cols + j];
}

// CUDA kernel for weight update
__global__ void update_weights_kernel(float *W, float *grad, float learning_rate, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    W[idx] -= learning_rate * grad[idx];
  }
}

// CUDA kernel for matrix subtraction and scaling (dZ2 = (Y_pred - Y_batch) * scale)
__global__ void sub_and_scale_kernel(float *result, float *A, float *B, 
                                     float scale, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = (A[idx] - B[idx]) * scale;
  }
}

// GPU-optimized backpropagation using streams for overlapping operations
void backpropagation_gpu(Matrix *X_batch, Matrix *Y_batch, Matrix *Z1,
                         Matrix *Y_pred, Matrix *W1, Matrix *W2, int batch_size) {
  size_t size_Z1 = batch_size * HIDDEN_SIZE * sizeof(float);
  size_t size_Y = batch_size * OUTPUT_SIZE * sizeof(float);
  size_t size_X = batch_size * INPUT_SIZE * sizeof(float);
  
  float *d_dZ2, *d_dW2, *d_dZ1, *d_dW1;
  float *d_X_batch, *d_Y_batch, *d_Z1, *d_Y_pred;
  float *d_Z1_T, *d_X_batch_T;
  
  // Allocate temporary buffers
  cudaMalloc(&d_dZ2, size_Y);
  cudaMalloc(&d_dW2, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
  cudaMalloc(&d_dZ1, size_Z1);
  cudaMalloc(&d_dW1, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&d_X_batch, size_X);
  cudaMalloc(&d_Y_batch, size_Y);
  cudaMalloc(&d_Z1, size_Z1);
  cudaMalloc(&d_Y_pred, size_Y);
  cudaMalloc(&d_Z1_T, size_Z1);
  cudaMalloc(&d_X_batch_T, size_X);
  
  // Stream 0: Copy X_batch and Z1
  cudaMemcpyAsync(d_X_batch, X_batch->data, size_X, 
                  cudaMemcpyHostToDevice, gpu_ctx.streams[0]);
  cudaMemcpyAsync(d_Z1, Z1->data, size_Z1, 
                  cudaMemcpyHostToDevice, gpu_ctx.streams[0]);
  
  // Stream 1: Copy Y_batch and Y_pred
  cudaMemcpyAsync(d_Y_batch, Y_batch->data, size_Y, 
                  cudaMemcpyHostToDevice, gpu_ctx.streams[1]);
  cudaMemcpyAsync(d_Y_pred, Y_pred->data, size_Y, 
                  cudaMemcpyHostToDevice, gpu_ctx.streams[1]);
  
  cudaDeviceSynchronize();
  
  // Compute dZ2 = (Y_pred - Y_batch) * (2/batch_size)
  int threads = 256;
  int blocks_Y = (batch_size * OUTPUT_SIZE + threads - 1) / threads;
  sub_and_scale_kernel<<<blocks_Y, threads, 0, gpu_ctx.streams[0]>>>(
      d_dZ2, d_Y_pred, d_Y_batch, 2.0f / batch_size, batch_size * OUTPUT_SIZE);
  
  // Transpose Z1 for dW2 = Z1^T * dZ2
  transpose_gpu(d_Z1, d_Z1_T, batch_size, HIDDEN_SIZE, gpu_ctx.streams[0]);
  
  cudaStreamSynchronize(gpu_ctx.streams[0]);
  
  // Compute dW2 = Z1^T * dZ2
  dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  dim3 numBlocks_dW2((OUTPUT_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (HIDDEN_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
  mat_mult_kernel<<<numBlocks_dW2, threadsPerBlock, 0, gpu_ctx.streams[0]>>>(
      d_Z1_T, d_dZ2, d_dW2, HIDDEN_SIZE, batch_size, OUTPUT_SIZE);
  
  // Update W2
  int blocks_W2 = (HIDDEN_SIZE * OUTPUT_SIZE + threads - 1) / threads;
  cudaStreamSynchronize(gpu_ctx.streams[0]);
  update_weights_kernel<<<blocks_W2, threads, 0, gpu_ctx.streams[0]>>>(
      gpu_ctx.d_W2, d_dW2, LEARNING_RATE, HIDDEN_SIZE * OUTPUT_SIZE);
  
  // Compute dZ1 = dZ2 * W2^T (using pre-transposed W2_T)
  dim3 numBlocks_dZ1((HIDDEN_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
  mat_mult_kernel_transposed<<<numBlocks_dZ1, threadsPerBlock, 0, gpu_ctx.streams[1]>>>(
      d_dZ2, gpu_ctx.d_W2_T, d_dZ1, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
  
  // Apply ReLU derivative to dZ1
  int blocks_Z1 = (batch_size * HIDDEN_SIZE + threads - 1) / threads;
  cudaStreamSynchronize(gpu_ctx.streams[1]);
  relu_derivative_mult_kernel<<<blocks_Z1, threads, 0, gpu_ctx.streams[1]>>>(
      d_dZ1, d_Z1, batch_size * HIDDEN_SIZE);
  
  // Transpose X_batch for dW1 = X_batch^T * dZ1
  transpose_gpu(d_X_batch, d_X_batch_T, batch_size, INPUT_SIZE, gpu_ctx.streams[1]);
  
  cudaStreamSynchronize(gpu_ctx.streams[1]);
  
  // Compute dW1 = X_batch^T * dZ1
  dim3 numBlocks_dW1((HIDDEN_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (INPUT_SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);
  mat_mult_kernel<<<numBlocks_dW1, threadsPerBlock, 0, gpu_ctx.streams[1]>>>(
      d_X_batch_T, d_dZ1, d_dW1, INPUT_SIZE, batch_size, HIDDEN_SIZE);
  
  // Update W1
  int blocks_W1 = (INPUT_SIZE * HIDDEN_SIZE + threads - 1) / threads;
  cudaStreamSynchronize(gpu_ctx.streams[1]);
  update_weights_kernel<<<blocks_W1, threads, 0, gpu_ctx.streams[1]>>>(
      gpu_ctx.d_W1, d_dW1, LEARNING_RATE, INPUT_SIZE * HIDDEN_SIZE);
  
  // Update transposed weight matrices
  transpose_gpu(gpu_ctx.d_W1, gpu_ctx.d_W1_T, INPUT_SIZE, HIDDEN_SIZE, gpu_ctx.streams[0]);
  transpose_gpu(gpu_ctx.d_W2, gpu_ctx.d_W2_T, HIDDEN_SIZE, OUTPUT_SIZE, gpu_ctx.streams[1]);
  
  cudaDeviceSynchronize();
  
  // Free temporary buffers
  cudaFree(d_dZ2);
  cudaFree(d_dW2);
  cudaFree(d_dZ1);
  cudaFree(d_dW1);
  cudaFree(d_X_batch);
  cudaFree(d_Y_batch);
  cudaFree(d_Z1);
  cudaFree(d_Y_pred);
  cudaFree(d_Z1_T);
  cudaFree(d_X_batch_T);
}

// CPU fallback backpropagation (kept for reference/debugging)
void backpropagation(Matrix *X_batch, Matrix *Y_batch, Matrix *Z1,
                     Matrix *Y_pred, Matrix *W1, Matrix *W2, int batch_size) {
  // Compute dZ2 = Y_pred - Y_batch
  Matrix *dZ2 = mat_sub(Y_pred, Y_batch);
  mat_scalar_mult(dZ2, 2.0f / batch_size);

  // Compute dW2 = Z1^T * dZ2
  Matrix *Z1_T = allocate_matrix(Z1->cols, Z1->rows);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < Z1->rows; i++) {
    for (int j = 0; j < Z1->cols; j++) {
      Z1_T->data[j * Z1->rows + i] = Z1->data[i * Z1->cols + j];
    }
  }
  Matrix *dW2 = mat_mult(Z1_T, dZ2);
  update_weights(W2, dW2, LEARNING_RATE);
  free_matrix(dW2);
  free_matrix(Z1_T);

  // Compute dZ1 = dZ2 * W2^T
  Matrix *W2_T = allocate_matrix(W2->cols, W2->rows);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < W2->rows; i++) {
    for (int j = 0; j < W2->cols; j++) {
      W2_T->data[j * W2->rows + i] = W2->data[i * W2->cols + j];
    }
  }
  Matrix *dZ1 = mat_mult(dZ2, W2_T);

  // Apply ReLU derivative
  Matrix *dZ1_derivative = relu_derivative(Z1);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < dZ1->rows; i++) {
    for (int j = 0; j < dZ1->cols; j++) {
      dZ1->data[i * dZ1->cols + j] *=
          dZ1_derivative->data[i * dZ1_derivative->cols + j];
    }
  }
  free_matrix(dZ1_derivative);
  free_matrix(W2_T);

  // Compute dW1 = X_batch^T * dZ1
  Matrix *X_batch_T = allocate_matrix(X_batch->cols, X_batch->rows);
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < X_batch->rows; i++) {
    for (int j = 0; j < X_batch->cols; j++) {
      X_batch_T->data[j * X_batch->rows + i] =
          X_batch->data[i * X_batch->cols + j];
    }
  }
  Matrix *dW1 = mat_mult(X_batch_T, dZ1);
  update_weights(W1, dW1, LEARNING_RATE);
  free_matrix(dW1);
  free_matrix(X_batch_T);

  // Free allocated matrices
  free_matrix(dZ2);
  free_matrix(dZ1);
}

// ! Batch Processing
// Function to get a batch from the dataset (OpenMP parallelized)
void get_batch(Matrix *X, Matrix *Y, Matrix *X_batch, Matrix *Y_batch,
               int batch_start, int batch_size) {
  #pragma omp parallel for
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < INPUT_SIZE; j++)
      X_batch->data[i * INPUT_SIZE + j] =
          X->data[(batch_start + i) * INPUT_SIZE + j];
    Y_batch->data[i * OUTPUT_SIZE] = Y->data[(batch_start + i) * OUTPUT_SIZE];
  }
}

// GPU-optimized forward pass
void forward_pass_gpu(Matrix *X_batch, Matrix *W1, Matrix *W2, 
                      Matrix *Z1, Matrix *Y_pred, int batch_size) {
  size_t size_X = batch_size * INPUT_SIZE * sizeof(float);
  size_t size_Z1 = batch_size * HIDDEN_SIZE * sizeof(float);
  size_t size_Y = batch_size * OUTPUT_SIZE * sizeof(float);
  
  // Copy X_batch to GPU using stream
  cudaMemcpyAsync(gpu_ctx.d_X, X_batch->data, size_X, 
                  cudaMemcpyHostToDevice, gpu_ctx.streams[0]);
  
  // Wait for copy to complete
  cudaStreamSynchronize(gpu_ctx.streams[0]);
  
  // Z1 = X_batch * W1 (using transposed W1 for better coalescing)
  dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  dim3 numBlocks_Z1((HIDDEN_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
  
  mat_mult_kernel_transposed<<<numBlocks_Z1, threadsPerBlock, 0, gpu_ctx.streams[0]>>>(
      gpu_ctx.d_X, gpu_ctx.d_W1_T, gpu_ctx.d_Z1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
  
  // Apply ReLU
  int threads = 256;
  int blocks = (batch_size * HIDDEN_SIZE + threads - 1) / threads;
  relu_kernel<<<blocks, threads, 0, gpu_ctx.streams[0]>>>(
      gpu_ctx.d_Z1, batch_size * HIDDEN_SIZE);
  
  // Y_pred = Z1 * W2 (using transposed W2 for better coalescing)
  dim3 numBlocks_Y((OUTPUT_SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (batch_size + threadsPerBlock.y - 1) / threadsPerBlock.y);
  
  mat_mult_kernel_transposed<<<numBlocks_Y, threadsPerBlock, 0, gpu_ctx.streams[0]>>>(
      gpu_ctx.d_Z1, gpu_ctx.d_W2_T, gpu_ctx.d_Y_pred, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
  
  // Copy results back to host (overlapping with potential next batch copy)
  cudaMemcpyAsync(Z1->data, gpu_ctx.d_Z1, size_Z1, 
                  cudaMemcpyDeviceToHost, gpu_ctx.streams[0]);
  cudaMemcpyAsync(Y_pred->data, gpu_ctx.d_Y_pred, size_Y, 
                  cudaMemcpyDeviceToHost, gpu_ctx.streams[0]);
  
  cudaStreamSynchronize(gpu_ctx.streams[0]);
}

// ! Data Loading
// Function to load CSV and populate X and Y, Assuming the last column is Y
int load_csv(const char *filename, Matrix **X, Matrix **Y, int *num_samples) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    printf("Failed to open file.\n");
    return -1;
  }
  char line[1024];
  int count = 0;
  // First pass to count samples
  while (fgets(line, sizeof(line), file))
    count++;
  *num_samples = count;
  rewind(file);
  
  // Allocate X and Y with pinned memory for faster GPU transfers
  *X = allocate_matrix_pinned(count, INPUT_SIZE);
  *Y = allocate_matrix_pinned(count, OUTPUT_SIZE);
  
  int i = 0;
  while (fgets(line, sizeof(line), file)) {
    char *token = strtok(line, ",");
    int j = 0;
    while (token) {
      if (j < INPUT_SIZE) {
        (*X)->data[i * INPUT_SIZE + j] = atof(token);
      } else {
        (*Y)->data[i * OUTPUT_SIZE] = atof(token);
      }
      j++;
      token = strtok(NULL, ",");
    }
    i++;
  }
  fclose(file);
  return 0;
}

// Main function
int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <data.csv>\n", argv[0]);
    return -1;
  }

  // CUDA event timing variables
  cudaEvent_t start_event, stop_event;
  float milliseconds = 0;
  cudaEventCreate(&start_event);
  cudaEventCreate(&stop_event);

  Matrix *X, *Y;
  int num_samples;
  if (load_csv(argv[1], &X, &Y, &num_samples) != 0)
    return -1;

  // Initialize GPU context
  init_gpu_context(BATCH_SIZE);

  // Allocate and initialize weights with pinned memory
  Matrix *W1 = allocate_matrix_pinned(INPUT_SIZE, HIDDEN_SIZE);
  Matrix *W2 = allocate_matrix_pinned(HIDDEN_SIZE, OUTPUT_SIZE);
  random_init(W1);
  random_init(W2);

  // Upload initial weights to GPU
  upload_weights_to_gpu(W1, W2);

  // Start measuring time using CUDA events
  cudaEventRecord(start_event);

  // Training loop
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
    for (int batch_start = 0; batch_start < num_samples;
         batch_start += BATCH_SIZE) {
      int batch_end = fmin(batch_start + BATCH_SIZE, num_samples);
      int batch_size = batch_end - batch_start;

      // Extract batch using pinned memory
      Matrix *X_batch = allocate_matrix_pinned(batch_size, INPUT_SIZE);
      Matrix *Y_batch = allocate_matrix_pinned(batch_size, OUTPUT_SIZE);
      get_batch(X, Y, X_batch, Y_batch, batch_start, batch_size);

      // Allocate output matrices with pinned memory
      Matrix *Z1 = allocate_matrix_pinned(batch_size, HIDDEN_SIZE);
      Matrix *Y_pred = allocate_matrix_pinned(batch_size, OUTPUT_SIZE);

      // GPU-optimized forward pass
      forward_pass_gpu(X_batch, W1, W2, Z1, Y_pred, batch_size);

      // Compute loss (on CPU for simplicity)
      float loss = mean_squared_error(Y_pred, Y_batch);
      if ((batch_start == 0) && ((epoch % LOG_EVERY_EPOCH == 0 && epoch != 0) ||
                                 epoch == 1 || epoch == EPOCHS - 1))
        printf("Epoch %d, MSE: %f\n", epoch, loss);

      // GPU-optimized backward pass
      backpropagation_gpu(X_batch, Y_batch, Z1, Y_pred, W1, W2, batch_size);

      // Free allocated matrices
      free_matrix(Z1);
      free_matrix(Y_pred);
      free_matrix(X_batch);
      free_matrix(Y_batch);
    }
  }

  // Stop measuring time
  cudaEventRecord(stop_event);
  cudaEventSynchronize(stop_event);
  cudaEventElapsedTime(&milliseconds, start_event, stop_event);

  printf("Training time: %.4f seconds\n", milliseconds / 1000.0f);

  // Download final weights from GPU
  download_weights_from_gpu(W1, W2);

  // Cleanup
  cleanup_gpu_context();
  cudaEventDestroy(start_event);
  cudaEventDestroy(stop_event);
  
  free_matrix(W1);
  free_matrix(W2);
  free_matrix(X);
  free_matrix(Y);

  return 0;
}
