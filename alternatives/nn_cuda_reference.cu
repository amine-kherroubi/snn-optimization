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

// --- Data structures ---
typedef struct {
  int row_count;
  int column_count;
  float *data;
} Matrix;

// --- Memory management ---
static Matrix *allocate_matrix(int row_count, int column_count) {
  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  matrix->row_count = row_count;
  matrix->column_count = column_count;
  matrix->data =
      (float *)malloc((size_t)row_count * (size_t)column_count * sizeof(float));
  return matrix;
}

static void free_matrix(Matrix *matrix) {
  free(matrix->data);
  free(matrix);
}

// --- Matrix operations ---
static void random_initialize_matrix(Matrix *matrix) {
  // He initialization: scale by sqrt(2 / fan_in) for ReLU networks.
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

static void check_cuda(cudaError_t error, const char *message) {
  if (error == cudaSuccess) {
    return;
  }
  fprintf(stderr, "CUDA error (%s): %s\n", message, cudaGetErrorString(error));
  exit(1);
}

static Matrix *matrix_multiply(const Matrix *left_matrix,
                               const Matrix *right_matrix) {
  if (left_matrix->column_count != right_matrix->row_count) {
    printf("Incompatible matrices for multiplication.\n");
    exit(1);
  }

  Matrix *result_matrix =
      allocate_matrix(left_matrix->row_count, right_matrix->column_count);

  size_t left_byte_count = (size_t)left_matrix->row_count *
                           (size_t)left_matrix->column_count * sizeof(float);
  size_t right_byte_count = (size_t)right_matrix->row_count *
                            (size_t)right_matrix->column_count * sizeof(float);
  size_t result_byte_count = (size_t)result_matrix->row_count *
                             (size_t)result_matrix->column_count *
                             sizeof(float);

  float *device_left_matrix = NULL;
  float *device_right_matrix = NULL;
  float *device_result_matrix = NULL;

  check_cuda(cudaMalloc((void **)&device_left_matrix, left_byte_count),
             "cudaMalloc(left_matrix)");
  check_cuda(cudaMalloc((void **)&device_right_matrix, right_byte_count),
             "cudaMalloc(right_matrix)");
  check_cuda(cudaMalloc((void **)&device_result_matrix, result_byte_count),
             "cudaMalloc(result_matrix)");

  check_cuda(cudaMemcpy(device_left_matrix, left_matrix->data, left_byte_count,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(left_matrix)");
  check_cuda(cudaMemcpy(device_right_matrix, right_matrix->data,
                        right_byte_count, cudaMemcpyHostToDevice),
             "cudaMemcpy(right_matrix)");

  dim3 threads_per_block(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
  dim3 blocks_per_grid((right_matrix->column_count + threads_per_block.x - 1) /
                           threads_per_block.x,
                       (left_matrix->row_count + threads_per_block.y - 1) /
                           threads_per_block.y);

  matrix_multiply_kernel<<<blocks_per_grid, threads_per_block>>>(
      device_left_matrix, device_right_matrix, device_result_matrix,
      left_matrix->row_count, left_matrix->column_count,
      right_matrix->column_count);
  check_cuda(cudaGetLastError(), "matrix_multiply_kernel launch");
  check_cuda(cudaDeviceSynchronize(), "matrix_multiply_kernel synchronize");

  check_cuda(cudaMemcpy(result_matrix->data, device_result_matrix,
                        result_byte_count, cudaMemcpyDeviceToHost),
             "cudaMemcpy(result_matrix)");

  cudaFree(device_left_matrix);
  cudaFree(device_right_matrix);
  cudaFree(device_result_matrix);

  return result_matrix;
}

static Matrix *matrix_subtract(const Matrix *left_matrix,
                               const Matrix *right_matrix) {
  if (left_matrix->row_count != right_matrix->row_count ||
      left_matrix->column_count != right_matrix->column_count) {
    printf("Incompatible matrices for subtraction.\n");
    exit(1);
  }

  Matrix *result_matrix =
      allocate_matrix(left_matrix->row_count, left_matrix->column_count);
  for (int row_index = 0; row_index < left_matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < left_matrix->column_count;
         column_index++) {
      int element_index = row_index * left_matrix->column_count + column_index;
      result_matrix->data[element_index] =
          left_matrix->data[element_index] - right_matrix->data[element_index];
    }
  }

  return result_matrix;
}

static void matrix_scale_in_place(Matrix *matrix, float scalar) {
  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < matrix->column_count;
         column_index++) {
      matrix->data[row_index * matrix->column_count + column_index] *= scalar;
    }
  }
}

// --- Activation functions ---
static void apply_relu_in_place(Matrix *matrix) {
  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < matrix->column_count;
         column_index++) {
      int element_index = row_index * matrix->column_count + column_index;
      matrix->data[element_index] = fmaxf(0.0f, matrix->data[element_index]);
    }
  }
}

static Matrix *compute_relu_derivative(const Matrix *matrix) {
  Matrix *derivative = allocate_matrix(matrix->row_count, matrix->column_count);
  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < matrix->column_count;
         column_index++) {
      int element_index = row_index * matrix->column_count + column_index;
      derivative->data[element_index] =
          (matrix->data[element_index] > 0.0f) ? 1.0f : 0.0f;
    }
  }
  return derivative;
}

// --- Loss functions ---
static float compute_mean_squared_error(const Matrix *predicted_output,
                                        const Matrix *expected_output) {
  float mean_squared_error = 0.0f;
  for (int row_index = 0; row_index < predicted_output->row_count;
       row_index++) {
    for (int column_index = 0; column_index < predicted_output->column_count;
         column_index++) {
      int element_index =
          row_index * predicted_output->column_count + column_index;
      float error = predicted_output->data[element_index] -
                    expected_output->data[element_index];
      mean_squared_error += error * error;
    }
  }
  return mean_squared_error / (float)predicted_output->row_count;
}

// --- Optimization ---
static void apply_gradient_descent_update(Matrix *weights,
                                          const Matrix *gradient,
                                          float learning_rate) {
  for (int row_index = 0; row_index < weights->row_count; row_index++) {
    for (int column_index = 0; column_index < weights->column_count;
         column_index++) {
      int element_index = row_index * weights->column_count + column_index;
      weights->data[element_index] -=
          learning_rate * gradient->data[element_index];
    }
  }
}

static void backpropagate_and_update_weights(const Matrix *input_batch,
                                             const Matrix *target_batch,
                                             const Matrix *hidden_layer_output,
                                             const Matrix *predicted_output,
                                             Matrix *weights_input_to_hidden,
                                             Matrix *weights_hidden_to_output,
                                             int batch_size) {
  Matrix *output_error = matrix_subtract(predicted_output, target_batch);
  matrix_scale_in_place(output_error, 2.0f / (float)batch_size);

  Matrix *hidden_layer_output_transpose = allocate_matrix(
      hidden_layer_output->column_count, hidden_layer_output->row_count);
  for (int row_index = 0; row_index < hidden_layer_output->row_count;
       row_index++) {
    for (int column_index = 0; column_index < hidden_layer_output->column_count;
         column_index++) {
      hidden_layer_output_transpose
          ->data[column_index * hidden_layer_output->row_count + row_index] =
          hidden_layer_output
              ->data[row_index * hidden_layer_output->column_count +
                     column_index];
    }
  }

  Matrix *weights_hidden_to_output_gradient =
      matrix_multiply(hidden_layer_output_transpose, output_error);
  apply_gradient_descent_update(weights_hidden_to_output,
                                weights_hidden_to_output_gradient,
                                LEARNING_RATE);
  free_matrix(weights_hidden_to_output_gradient);
  free_matrix(hidden_layer_output_transpose);

  Matrix *weights_hidden_to_output_transpose =
      allocate_matrix(weights_hidden_to_output->column_count,
                      weights_hidden_to_output->row_count);
  for (int row_index = 0; row_index < weights_hidden_to_output->row_count;
       row_index++) {
    for (int column_index = 0;
         column_index < weights_hidden_to_output->column_count;
         column_index++) {
      weights_hidden_to_output_transpose
          ->data[column_index * weights_hidden_to_output->row_count +
                 row_index] =
          weights_hidden_to_output
              ->data[row_index * weights_hidden_to_output->column_count +
                     column_index];
    }
  }

  Matrix *hidden_layer_error =
      matrix_multiply(output_error, weights_hidden_to_output_transpose);

  Matrix *relu_derivative_matrix = compute_relu_derivative(hidden_layer_output);
  for (int row_index = 0; row_index < hidden_layer_error->row_count;
       row_index++) {
    for (int column_index = 0; column_index < hidden_layer_error->column_count;
         column_index++) {
      int element_index =
          row_index * hidden_layer_error->column_count + column_index;
      hidden_layer_error->data[element_index] *=
          relu_derivative_matrix->data[element_index];
    }
  }

  free_matrix(relu_derivative_matrix);
  free_matrix(weights_hidden_to_output_transpose);

  Matrix *input_batch_transpose =
      allocate_matrix(input_batch->column_count, input_batch->row_count);
  for (int row_index = 0; row_index < input_batch->row_count; row_index++) {
    for (int column_index = 0; column_index < input_batch->column_count;
         column_index++) {
      input_batch_transpose
          ->data[column_index * input_batch->row_count + row_index] =
          input_batch
              ->data[row_index * input_batch->column_count + column_index];
    }
  }

  Matrix *weights_input_to_hidden_gradient =
      matrix_multiply(input_batch_transpose, hidden_layer_error);
  apply_gradient_descent_update(
      weights_input_to_hidden, weights_input_to_hidden_gradient, LEARNING_RATE);
  free_matrix(weights_input_to_hidden_gradient);
  free_matrix(input_batch_transpose);

  free_matrix(output_error);
  free_matrix(hidden_layer_error);
}

// --- Dataset utilities ---
static void copy_mini_batch(const Matrix *input_dataset,
                            const Matrix *target_dataset, Matrix *input_batch,
                            Matrix *target_batch, int start_index,
                            int batch_size) {
  for (int batch_index = 0; batch_index < batch_size; batch_index++) {
    for (int feature_index = 0; feature_index < INPUT_SIZE; feature_index++) {
      input_batch
          ->data[batch_index * input_batch->column_count + feature_index] =
          input_dataset
              ->data[(start_index + batch_index) * input_dataset->column_count +
                     feature_index];
    }
    target_batch->data[batch_index * target_batch->column_count] =
        target_dataset
            ->data[(start_index + batch_index) * target_dataset->column_count];
  }
}

static int load_dataset_from_csv(const char *filename, Matrix **input_dataset,
                                 Matrix **target_dataset,
                                 int *number_of_samples) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    printf("Failed to open file: %s\n", filename);
    return -1;
  }

  char line_buffer[1024];
  int sample_count = 0;
  while (fgets(line_buffer, sizeof(line_buffer), file) != NULL) {
    sample_count++;
  }

  *number_of_samples = sample_count;
  rewind(file);

  *input_dataset = allocate_matrix(sample_count, INPUT_SIZE);
  *target_dataset = allocate_matrix(sample_count, OUTPUT_SIZE);

  for (int sample_index = 0;
       sample_index < sample_count &&
       fgets(line_buffer, sizeof(line_buffer), file) != NULL;
       sample_index++) {
    char *token = strtok(line_buffer, ",");
    int feature_index = 0;

    while (token != NULL) {
      if (feature_index < INPUT_SIZE) {
        (*input_dataset)
            ->data[sample_index * (*input_dataset)->column_count +
                   feature_index] = atof(token);
      } else {
        (*target_dataset)
            ->data[sample_index * (*target_dataset)->column_count] =
            atof(token);
      }
      feature_index++;
      token = strtok(NULL, ",");
    }
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

  double start_time;
  double end_time;
  double total_time = 0.0;
  float total_final_mse = 0.0f;

  Matrix *input_dataset = NULL;
  Matrix *target_dataset = NULL;
  int number_of_samples = 0;

  if (load_dataset_from_csv(argv[1], &input_dataset, &target_dataset,
                            &number_of_samples) != 0) {
    return -1;
  }

  for (int run_index = 0; run_index < TEST_RUN_COUNT; run_index++) {
    Matrix *weights_input_to_hidden = allocate_matrix(INPUT_SIZE, HIDDEN_SIZE);
    Matrix *weights_hidden_to_output =
        allocate_matrix(HIDDEN_SIZE, OUTPUT_SIZE);
    random_initialize_matrix(weights_input_to_hidden);
    random_initialize_matrix(weights_hidden_to_output);

    start_time = omp_get_wtime();

    float final_mse = 0.0f;

    for (int epoch_index = 0; epoch_index < EPOCHS; epoch_index++) {
      for (int batch_start_index = 0; batch_start_index < number_of_samples;
           batch_start_index += BATCH_SIZE) {
        int batch_end_index = batch_start_index + BATCH_SIZE;
        if (batch_end_index > number_of_samples) {
          batch_end_index = number_of_samples;
        }

        int batch_size = batch_end_index - batch_start_index;

        Matrix *input_batch = allocate_matrix(batch_size, INPUT_SIZE);
        Matrix *target_batch = allocate_matrix(batch_size, OUTPUT_SIZE);
        copy_mini_batch(input_dataset, target_dataset, input_batch,
                        target_batch, batch_start_index, batch_size);

        Matrix *hidden_layer_output =
            matrix_multiply(input_batch, weights_input_to_hidden);
        apply_relu_in_place(hidden_layer_output);
        Matrix *predicted_output =
            matrix_multiply(hidden_layer_output, weights_hidden_to_output);

        final_mse = compute_mean_squared_error(predicted_output, target_batch);

        backpropagate_and_update_weights(
            input_batch, target_batch, hidden_layer_output, predicted_output,
            weights_input_to_hidden, weights_hidden_to_output, batch_size);

        free_matrix(hidden_layer_output);
        free_matrix(predicted_output);
        free_matrix(input_batch);
        free_matrix(target_batch);
      }
    }

    end_time = omp_get_wtime();
    total_time += (end_time - start_time);
    total_final_mse += final_mse;

    free_matrix(weights_input_to_hidden);
    free_matrix(weights_hidden_to_output);
  }

  printf("Average training time over %d runs: %.4f seconds | Average final "
         "MSE: %.6f\n",
         TEST_RUN_COUNT, total_time / TEST_RUN_COUNT,
         total_final_mse / TEST_RUN_COUNT);

  free_matrix(input_dataset);
  free_matrix(target_dataset);

  return 0;
}
