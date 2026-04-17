// Parallel matrix multiplication using CUDA.
// Optimized version: pre-allocate reusable matrices to minimize allocation
// overhead.
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Network parameters ---
#define INPUT_SIZE 32   // Number of input features
#define HIDDEN_SIZE 256 // Number of neurons in the hidden layer
#define OUTPUT_SIZE 1   // Number of output neurons
#define EPOCHS 100      // Number of training epochs
#define LEARNING_RATE 0.002
#define BATCH_SIZE 256 // Batch size for SGD.
#define THREADS_PER_BLOCK 16
#define TEST_RUN_COUNT 10 // Number of runs for averaging.
#define STREAM_COUNT 3
#define TILE_ROW_COUNT 128
#define PREALLOCATED_MATRIX_COUNT                                              \
  10 // Number of pre-allocated reusable matrices.

// --- Data structures ---
typedef struct {
  int row_count;
  int column_count;
  float *data;
  int uses_pinned_memory;
  int is_reusable; // 1 = part of pre-allocated pool, should not be individually
                   // freed.
} Matrix;

typedef struct {
  float *device_left_tiles[STREAM_COUNT];
  float *device_result_tiles[STREAM_COUNT];
  cudaStream_t streams[STREAM_COUNT];
  size_t left_tile_byte_count;
  size_t result_tile_byte_count;
  int maximum_left_column_count;
  int maximum_right_column_count;
  int initialized;
} GlobalGPUContext;

static GlobalGPUContext global_gpu_context;

typedef struct {
  Matrix X_batch;   // BATCH_SIZE x INPUT_SIZE
  Matrix Y_batch;   // BATCH_SIZE x OUTPUT_SIZE
  Matrix Z1;        // BATCH_SIZE x HIDDEN_SIZE
  Matrix Y_pred;    // BATCH_SIZE x OUTPUT_SIZE
  Matrix dZ2;       // BATCH_SIZE x OUTPUT_SIZE
  Matrix Z1_T;      // HIDDEN_SIZE x BATCH_SIZE
  Matrix dW2;       // HIDDEN_SIZE x OUTPUT_SIZE
  Matrix W2_T;      // OUTPUT_SIZE x HIDDEN_SIZE
  Matrix dZ1;       // BATCH_SIZE x HIDDEN_SIZE
  Matrix X_batch_T; // INPUT_SIZE x BATCH_SIZE

  float *dZ1_deriv_data; // BATCH_SIZE x HIDDEN_SIZE
  float *dW1_data;       // INPUT_SIZE x HIDDEN_SIZE
  int initialized;
} MatrixPool;

static MatrixPool matrix_pool;

static void initialize_pool_matrix(Matrix *matrix, int row_count,
                                   int column_count) {
  matrix->row_count = row_count;
  matrix->column_count = column_count;
  matrix->is_reusable = 1;
  matrix->uses_pinned_memory = 1;

  size_t byte_count = (size_t)row_count * (size_t)column_count * sizeof(float);
  cudaError_t error = cudaMallocHost((void **)&matrix->data, byte_count);
  if (error != cudaSuccess) {
    matrix->data = (float *)malloc(byte_count);
    matrix->uses_pinned_memory = 0;
  }
}

static void free_pool_matrix(Matrix *matrix) {
  if (matrix->uses_pinned_memory) {
    cudaFreeHost(matrix->data);
  } else {
    free(matrix->data);
  }
  matrix->data = NULL;
}

static void initialize_matrix_pool() {
  if (matrix_pool.initialized) {
    return;
  }

  initialize_pool_matrix(&matrix_pool.X_batch, BATCH_SIZE, INPUT_SIZE);
  initialize_pool_matrix(&matrix_pool.Y_batch, BATCH_SIZE, OUTPUT_SIZE);
  initialize_pool_matrix(&matrix_pool.Z1, BATCH_SIZE, HIDDEN_SIZE);
  initialize_pool_matrix(&matrix_pool.Y_pred, BATCH_SIZE, OUTPUT_SIZE);
  initialize_pool_matrix(&matrix_pool.dZ2, BATCH_SIZE, OUTPUT_SIZE);
  initialize_pool_matrix(&matrix_pool.Z1_T, HIDDEN_SIZE, BATCH_SIZE);
  initialize_pool_matrix(&matrix_pool.dW2, HIDDEN_SIZE, OUTPUT_SIZE);
  initialize_pool_matrix(&matrix_pool.W2_T, OUTPUT_SIZE, HIDDEN_SIZE);
  initialize_pool_matrix(&matrix_pool.dZ1, BATCH_SIZE, HIDDEN_SIZE);
  initialize_pool_matrix(&matrix_pool.X_batch_T, INPUT_SIZE, BATCH_SIZE);

  matrix_pool.dZ1_deriv_data =
      (float *)malloc((size_t)BATCH_SIZE * (size_t)HIDDEN_SIZE * sizeof(float));
  matrix_pool.dW1_data =
      (float *)malloc((size_t)INPUT_SIZE * (size_t)HIDDEN_SIZE * sizeof(float));

  matrix_pool.initialized = 1;
  printf("[Pool] Pre-allocated %d reusable matrices + 2 extra buffers\n",
         PREALLOCATED_MATRIX_COUNT);
}

static void cleanup_matrix_pool() {
  if (!matrix_pool.initialized) {
    return;
  }

  free_pool_matrix(&matrix_pool.X_batch);
  free_pool_matrix(&matrix_pool.Y_batch);
  free_pool_matrix(&matrix_pool.Z1);
  free_pool_matrix(&matrix_pool.Y_pred);
  free_pool_matrix(&matrix_pool.dZ2);
  free_pool_matrix(&matrix_pool.Z1_T);
  free_pool_matrix(&matrix_pool.dW2);
  free_pool_matrix(&matrix_pool.W2_T);
  free_pool_matrix(&matrix_pool.dZ1);
  free_pool_matrix(&matrix_pool.X_batch_T);

  free(matrix_pool.dZ1_deriv_data);
  free(matrix_pool.dW1_data);

  matrix_pool.initialized = 0;
  printf("[Pool] Freed all pre-allocated matrices\n");
}

static void ensure_tile_capacity(int left_column_count,
                                 int right_column_count) {
  size_t required_left_tile_byte_count =
      (size_t)TILE_ROW_COUNT * (size_t)left_column_count * sizeof(float);
  size_t required_result_tile_byte_count =
      (size_t)TILE_ROW_COUNT * (size_t)right_column_count * sizeof(float);

  int needs_reallocation = 0;

  if (required_left_tile_byte_count > global_gpu_context.left_tile_byte_count) {
    global_gpu_context.left_tile_byte_count = required_left_tile_byte_count;
    global_gpu_context.maximum_left_column_count = left_column_count;
    needs_reallocation = 1;
  }

  if (required_result_tile_byte_count >
      global_gpu_context.result_tile_byte_count) {
    global_gpu_context.result_tile_byte_count = required_result_tile_byte_count;
    global_gpu_context.maximum_right_column_count = right_column_count;
    needs_reallocation = 1;
  }

  if (needs_reallocation) {
    for (int stream_index = 0; stream_index < STREAM_COUNT; stream_index++) {
      if (global_gpu_context.device_left_tiles[stream_index] != NULL) {
        cudaFree(global_gpu_context.device_left_tiles[stream_index]);
      }
      if (global_gpu_context.device_result_tiles[stream_index] != NULL) {
        cudaFree(global_gpu_context.device_result_tiles[stream_index]);
      }

      cudaMalloc((void **)&global_gpu_context.device_left_tiles[stream_index],
                 global_gpu_context.left_tile_byte_count);
      cudaMalloc((void **)&global_gpu_context.device_result_tiles[stream_index],
                 global_gpu_context.result_tile_byte_count);
    }
  }
}

static void initialize_global_gpu_context() {
  if (global_gpu_context.initialized) {
    return;
  }

  global_gpu_context.left_tile_byte_count = 0;
  global_gpu_context.result_tile_byte_count = 0;
  global_gpu_context.maximum_left_column_count = 0;
  global_gpu_context.maximum_right_column_count = 0;

  for (int stream_index = 0; stream_index < STREAM_COUNT; stream_index++) {
    global_gpu_context.device_left_tiles[stream_index] = NULL;
    global_gpu_context.device_result_tiles[stream_index] = NULL;
    cudaStreamCreate(&global_gpu_context.streams[stream_index]);
  }

  global_gpu_context.initialized = 1;
}

static void cleanup_global_gpu_context() {
  if (!global_gpu_context.initialized) {
    return;
  }

  for (int stream_index = 0; stream_index < STREAM_COUNT; stream_index++) {
    cudaFree(global_gpu_context.device_left_tiles[stream_index]);
    cudaFree(global_gpu_context.device_result_tiles[stream_index]);
    cudaStreamDestroy(global_gpu_context.streams[stream_index]);
  }

  global_gpu_context.initialized = 0;
}

// --- Memory management ---
static Matrix *allocate_matrix(int row_count, int column_count) {
  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  matrix->row_count = row_count;
  matrix->column_count = column_count;
  matrix->is_reusable = 0;
  matrix->uses_pinned_memory = 1;

  size_t byte_count = (size_t)row_count * (size_t)column_count * sizeof(float);
  cudaError_t error = cudaMallocHost((void **)&matrix->data, byte_count);
  if (error != cudaSuccess) {
    matrix->data = (float *)malloc(byte_count);
    matrix->uses_pinned_memory = 0;
  }

  return matrix;
}

static void free_matrix(Matrix *matrix) {
  if (matrix->is_reusable) {
    return;
  }

  if (matrix->uses_pinned_memory) {
    cudaFreeHost(matrix->data);
  } else {
    free(matrix->data);
  }
  free(matrix);
}

// --- Matrix operations ---
static void random_initialize_matrix(Matrix *matrix) {
  float scale = sqrtf(2.0f / (float)matrix->row_count);
  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < matrix->column_count;
         column_index++) {
      float random_value = 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f;
      matrix->data[row_index * matrix->column_count + column_index] =
          random_value * scale;
    }
  }
}

// --- CUDA kernels ---
__global__ void matrix_multiply_kernel(const float *left_matrix,
                                       const float *right_matrix,
                                       float *result_matrix, int left_row_count,
                                       int left_column_count,
                                       int right_column_count) {
  int row_index = blockIdx.y * blockDim.y + threadIdx.y;
  int column_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (row_index < left_row_count && column_index < right_column_count) {
    float value = 0.0f;
    for (int inner_index = 0; inner_index < left_column_count; inner_index++) {
      value += left_matrix[row_index * left_column_count + inner_index] *
               right_matrix[inner_index * right_column_count + column_index];
    }
    result_matrix[row_index * right_column_count + column_index] = value;
  }
}

static void matrix_multiply_into(const Matrix *left_matrix,
                                 const Matrix *right_matrix,
                                 Matrix *result_matrix) {
  if (left_matrix->column_count != right_matrix->row_count) {
    printf("Incompatible matrices for multiplication.\n");
    exit(1);
  }

  result_matrix->row_count = left_matrix->row_count;
  result_matrix->column_count = right_matrix->column_count;

  size_t right_byte_count = (size_t)right_matrix->row_count *
                            (size_t)right_matrix->column_count * sizeof(float);

  if (!global_gpu_context.initialized) {
    initialize_global_gpu_context();
  }

  ensure_tile_capacity(left_matrix->column_count, right_matrix->column_count);

  float *device_right_matrix = NULL;
  cudaMalloc((void **)&device_right_matrix, right_byte_count);
  cudaMemcpy(device_right_matrix, right_matrix->data, right_byte_count,
             cudaMemcpyHostToDevice);

  dim3 threads_per_block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

  for (int row_start_index = 0, tile_index = 0;
       row_start_index < left_matrix->row_count;
       row_start_index += TILE_ROW_COUNT, tile_index++) {
    int tile_row_count =
        (row_start_index + TILE_ROW_COUNT <= left_matrix->row_count)
            ? TILE_ROW_COUNT
            : (left_matrix->row_count - row_start_index);
    int stream_index = tile_index % STREAM_COUNT;

    const float *left_tile_host =
        left_matrix->data + row_start_index * left_matrix->column_count;
    cudaMemcpyAsync(
        global_gpu_context.device_left_tiles[stream_index], left_tile_host,
        (size_t)tile_row_count * (size_t)left_matrix->column_count *
            sizeof(float),
        cudaMemcpyHostToDevice, global_gpu_context.streams[stream_index]);

    dim3 blocks_per_grid(
        (right_matrix->column_count + threads_per_block.x - 1) /
            threads_per_block.x,
        (tile_row_count + threads_per_block.y - 1) / threads_per_block.y);

    matrix_multiply_kernel<<<blocks_per_grid, threads_per_block, 0,
                             global_gpu_context.streams[stream_index]>>>(
        global_gpu_context.device_left_tiles[stream_index], device_right_matrix,
        global_gpu_context.device_result_tiles[stream_index], tile_row_count,
        left_matrix->column_count, right_matrix->column_count);

    float *result_tile_host =
        result_matrix->data + row_start_index * result_matrix->column_count;
    cudaMemcpyAsync(
        result_tile_host, global_gpu_context.device_result_tiles[stream_index],
        (size_t)tile_row_count * (size_t)right_matrix->column_count *
            sizeof(float),
        cudaMemcpyDeviceToHost, global_gpu_context.streams[stream_index]);
  }

  for (int stream_index = 0; stream_index < STREAM_COUNT; stream_index++) {
    cudaStreamSynchronize(global_gpu_context.streams[stream_index]);
  }

  cudaFree(device_right_matrix);
}

static void matrix_subtract_into(const Matrix *left_matrix,
                                 const Matrix *right_matrix,
                                 Matrix *result_matrix) {
  if (left_matrix->row_count != right_matrix->row_count ||
      left_matrix->column_count != right_matrix->column_count) {
    printf("Incompatible matrices for subtraction.\n");
    exit(1);
  }

  result_matrix->row_count = left_matrix->row_count;
  result_matrix->column_count = left_matrix->column_count;
  for (int row_index = 0; row_index < left_matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < left_matrix->column_count;
         column_index++) {
      int element_index = row_index * left_matrix->column_count + column_index;
      result_matrix->data[element_index] =
          left_matrix->data[element_index] - right_matrix->data[element_index];
    }
  }
}

static void matrix_scale_in_place(Matrix *matrix, float scalar) {
  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < matrix->column_count;
         column_index++) {
      matrix->data[row_index * matrix->column_count + column_index] *= scalar;
    }
  }
}

static void transpose_into(const Matrix *source_matrix,
                           Matrix *destination_matrix) {
  destination_matrix->row_count = source_matrix->column_count;
  destination_matrix->column_count = source_matrix->row_count;

  for (int row_index = 0; row_index < source_matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < source_matrix->column_count;
         column_index++) {
      destination_matrix
          ->data[column_index * source_matrix->row_count + row_index] =
          source_matrix
              ->data[row_index * source_matrix->column_count + column_index];
    }
  }
}

// --- Activation Functions ---
// Function to apply ReLU activation (in-place)
void relu(Matrix *m) {
  for (int i = 0; i < m->row_count; i++)
    for (int j = 0; j < m->column_count; j++)
      m->data[i * m->column_count + j] =
          fmaxf(0, m->data[i * m->column_count + j]);
}

// In-place ReLU derivative: writes derivative into out_data buffer
void relu_derivative_into(Matrix *m, float *out_data) {
  for (int i = 0; i < m->row_count; i++)
    for (int j = 0; j < m->column_count; j++)
      out_data[i * m->column_count + j] =
          (m->data[i * m->column_count + j] > 0) ? 1.0f : 0.0f;
}

// --- Loss Functions ---
// Function to compute Mean Squared Error
float mean_squared_error(Matrix *Y_pred, Matrix *Y_true) {
  float mse = 0.0f;
  for (int i = 0; i < Y_pred->row_count; i++)
    for (int j = 0; j < Y_pred->column_count; j++)
      mse += pow(Y_pred->data[i * Y_pred->column_count + j] -
                     Y_true->data[i * Y_true->column_count + j],
                 2);
  return mse / Y_pred->row_count;
}

// --- Optimization ---
// Function to update weights: W = W - learning_rate * grad
void update_weights(Matrix *W, Matrix *grad, float learning_rate) {
  for (int i = 0; i < W->row_count; i++)
    for (int j = 0; j < W->column_count; j++)
      W->data[i * W->column_count + j] -=
          learning_rate * grad->data[i * grad->column_count + j];
}

// Update weights from raw data buffer
void update_weights_raw(Matrix *W, float *grad_data, int grad_cols,
                        float learning_rate) {
  for (int i = 0; i < W->row_count; i++)
    for (int j = 0; j < W->column_count; j++)
      W->data[i * W->column_count + j] -=
          learning_rate * grad_data[i * grad_cols + j];
}

// --- Backpropagation ---
void backpropagation(Matrix *X_batch, Matrix *Y_batch, Matrix *Z1,
                     Matrix *Y_pred, Matrix *W1, Matrix *W2, int batch_size) {
  // Compute dZ2 = Y_pred - Y_batch (into pool dZ2)
  matrix_subtract_into(Y_pred, Y_batch, &matrix_pool.dZ2);
  matrix_scale_in_place(&matrix_pool.dZ2, 2.0f / batch_size);

  // Compute Z1^T (into pool Z1_T)
  transpose_into(Z1, &matrix_pool.Z1_T);

  // Compute dW2 = Z1^T * dZ2 (into pool dW2)
  matrix_multiply_into(&matrix_pool.Z1_T, &matrix_pool.dZ2, &matrix_pool.dW2);
  update_weights(W2, &matrix_pool.dW2, LEARNING_RATE);

  // Compute W2^T (into pool W2_T)
  transpose_into(W2, &matrix_pool.W2_T);

  // Compute dZ1 = dZ2 * W2^T (into pool dZ1)
  matrix_multiply_into(&matrix_pool.dZ2, &matrix_pool.W2_T, &matrix_pool.dZ1);

  // Apply ReLU derivative (into extra buffer)
  relu_derivative_into(Z1, matrix_pool.dZ1_deriv_data);
  for (int i = 0; i < matrix_pool.dZ1.row_count; i++) {
    for (int j = 0; j < matrix_pool.dZ1.column_count; j++) {
      matrix_pool.dZ1.data[i * matrix_pool.dZ1.column_count + j] *=
          matrix_pool.dZ1_deriv_data[i * matrix_pool.dZ1.column_count + j];
    }
  }

  // Compute X_batch^T (into pool X_batch_T)
  transpose_into(X_batch, &matrix_pool.X_batch_T);

  // Compute dW1 = X_batch^T * dZ1 (into extra dW1 buffer via temp Matrix
  // wrapper)
  Matrix dW1_wrapper;
  dW1_wrapper.row_count = INPUT_SIZE;
  dW1_wrapper.column_count = HIDDEN_SIZE;
  dW1_wrapper.data = matrix_pool.dW1_data;
  dW1_wrapper.uses_pinned_memory = 1;
  dW1_wrapper.is_reusable = 1;
  matrix_multiply_into(&matrix_pool.X_batch_T, &matrix_pool.dZ1, &dW1_wrapper);
  update_weights_raw(W1, matrix_pool.dW1_data, HIDDEN_SIZE, LEARNING_RATE);
}

// --- Batch Processing ---
// Function to get a batch from the dataset (writes into pre-allocated matrices)
void get_batch(Matrix *X, Matrix *Y, Matrix *X_batch, Matrix *Y_batch,
               int batch_start, int batch_size) {
  X_batch->row_count = batch_size;
  Y_batch->row_count = batch_size;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < INPUT_SIZE; j++)
      X_batch->data[i * INPUT_SIZE + j] =
          X->data[(batch_start + i) * INPUT_SIZE + j];
    Y_batch->data[i * OUTPUT_SIZE] = Y->data[(batch_start + i) * OUTPUT_SIZE];
  }
}

// --- Data Loading ---
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
  // Allocate X and Y
  *X = allocate_matrix(count, INPUT_SIZE);
  *Y = allocate_matrix(count, OUTPUT_SIZE);
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

  double start_time, end_time;
  double total_time = 0.0;
  float total_final_mse = 0.0f;

  Matrix *X, *Y;
  int num_samples;
  if (load_csv(argv[1], &X, &Y, &num_samples) != 0)
    return -1;

  // Pre-allocate the matrix pool once (10 matrices + 2 extra buffers)
  initialize_matrix_pool();

  // Run training multiple times
  for (int run = 0; run < TEST_RUN_COUNT; run++) {
    // Allocate and initialize weights (these change each run)
    Matrix *W1 = allocate_matrix(INPUT_SIZE, HIDDEN_SIZE);
    Matrix *W2 = allocate_matrix(HIDDEN_SIZE, OUTPUT_SIZE);
    random_initialize_matrix(W1);
    random_initialize_matrix(W2);

    // Start measuring time
    start_time = omp_get_wtime();

    float final_mse = 0.0f;

    // Training loop — NO malloc/free inside this loop!
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
      for (int batch_start = 0; batch_start < num_samples;
           batch_start += BATCH_SIZE) {
        int batch_end = fmin(batch_start + BATCH_SIZE, num_samples);
        int batch_size = batch_end - batch_start;

        // Extract batch into pre-allocated pool matrices
        get_batch(X, Y, &matrix_pool.X_batch, &matrix_pool.Y_batch, batch_start,
                  batch_size);

        // Forward pass: X -> Hidden Layer -> ReLU -> Output Layer
        matrix_multiply_into(&matrix_pool.X_batch, W1, &matrix_pool.Z1);
        relu(&matrix_pool.Z1);
        matrix_multiply_into(&matrix_pool.Z1, W2, &matrix_pool.Y_pred);

        // Compute loss
        final_mse =
            mean_squared_error(&matrix_pool.Y_pred, &matrix_pool.Y_batch);

        // Backward pass (all temporaries use pool matrices)
        backpropagation(&matrix_pool.X_batch, &matrix_pool.Y_batch,
                        &matrix_pool.Z1, &matrix_pool.Y_pred, W1, W2,
                        batch_size);
      }
    }

    // Stop measuring time
    end_time = omp_get_wtime();
    total_time += (end_time - start_time);
    total_final_mse += final_mse;

    // Cleanup weights for this run
    free_matrix(W1);
    free_matrix(W2);
  }

  // Print average training time and MSE
  printf("Average training time over %d runs: %.4f seconds | Average final "
         "MSE: %.6f\n",
         TEST_RUN_COUNT, total_time / TEST_RUN_COUNT,
         total_final_mse / TEST_RUN_COUNT);

  // Cleanup — all at once
  free_matrix(X);
  free_matrix(Y);
  cleanup_matrix_pool();
  cleanup_global_gpu_context();

  return 0;
}
