#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Network parameters ---
#define INPUT_SIZE 32     // Number of input features
#define HIDDEN_SIZE 256   // Number of neurons in the hidden layer
#define OUTPUT_SIZE 1     // Number of output neurons
#define EPOCHS 100        // Number of training epochs
#define LOG_EVERY_EPOCH 1 // Log loss every n epochs
#define LEARNING_RATE 0.002
#define BATCH_SIZE 256 // Batch size for SGD.

// --- Data structures ---
typedef struct {
  int row_count;
  int column_count;
  float **data;
} Matrix;

// --- Memory management ---
static Matrix *allocate_matrix(int row_count, int column_count) {
  Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
  matrix->row_count = row_count;
  matrix->column_count = column_count;
  matrix->data = (float **)malloc((size_t)row_count * sizeof(float *));

  for (int row_index = 0; row_index < row_count; row_index++) {
    matrix->data[row_index] =
        (float *)calloc((size_t)column_count, sizeof(float));
  }

  return matrix;
}

static void free_matrix(Matrix *matrix) {
  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    free(matrix->data[row_index]);
  }
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
      matrix->data[row_index][column_index] = random_value * scale;
    }
  }
}

static Matrix *matrix_multiply(const Matrix *left_matrix,
                               const Matrix *right_matrix) {
  if (left_matrix->column_count != right_matrix->row_count) {
    printf("Incompatible matrices for multiplication.\n");
    exit(1);
  }

  Matrix *result_matrix =
      allocate_matrix(left_matrix->row_count, right_matrix->column_count);

  for (int row_index = 0; row_index < left_matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < right_matrix->column_count;
         column_index++) {
      float sum = 0.0f;

      for (int inner_index = 0; inner_index < left_matrix->column_count;
           inner_index++) {
        sum += left_matrix->data[row_index][inner_index] *
               right_matrix->data[inner_index][column_index];
      }

      result_matrix->data[row_index][column_index] = sum;
    }
  }

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
      result_matrix->data[row_index][column_index] =
          left_matrix->data[row_index][column_index] -
          right_matrix->data[row_index][column_index];
    }
  }

  return result_matrix;
}

static void matrix_scale_in_place(Matrix *matrix, float scalar) {
  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < matrix->column_count;
         column_index++) {
      matrix->data[row_index][column_index] *= scalar;
    }
  }
}

// --- Activation functions ---
static void apply_relu_in_place(Matrix *matrix) {
  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < matrix->column_count;
         column_index++) {
      matrix->data[row_index][column_index] =
          fmaxf(0.0f, matrix->data[row_index][column_index]);
    }
  }
}

static Matrix *compute_relu_derivative(const Matrix *matrix) {
  Matrix *derivative = allocate_matrix(matrix->row_count, matrix->column_count);

  for (int row_index = 0; row_index < matrix->row_count; row_index++) {
    for (int column_index = 0; column_index < matrix->column_count;
         column_index++) {
      derivative->data[row_index][column_index] =
          (matrix->data[row_index][column_index] > 0.0f) ? 1.0f : 0.0f;
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
      float error = predicted_output->data[row_index][column_index] -
                    expected_output->data[row_index][column_index];
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
      weights->data[row_index][column_index] -=
          learning_rate * gradient->data[row_index][column_index];
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
      hidden_layer_output_transpose->data[column_index][row_index] =
          hidden_layer_output->data[row_index][column_index];
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
      weights_hidden_to_output_transpose->data[column_index][row_index] =
          weights_hidden_to_output->data[row_index][column_index];
    }
  }

  Matrix *hidden_layer_error =
      matrix_multiply(output_error, weights_hidden_to_output_transpose);

  Matrix *relu_derivative_matrix = compute_relu_derivative(hidden_layer_output);
  for (int row_index = 0; row_index < hidden_layer_error->row_count;
       row_index++) {
    for (int column_index = 0; column_index < hidden_layer_error->column_count;
         column_index++) {
      hidden_layer_error->data[row_index][column_index] *=
          relu_derivative_matrix->data[row_index][column_index];
    }
  }

  free_matrix(relu_derivative_matrix);
  free_matrix(weights_hidden_to_output_transpose);

  Matrix *input_batch_transpose =
      allocate_matrix(input_batch->column_count, input_batch->row_count);
  for (int row_index = 0; row_index < input_batch->row_count; row_index++) {
    for (int column_index = 0; column_index < input_batch->column_count;
         column_index++) {
      input_batch_transpose->data[column_index][row_index] =
          input_batch->data[row_index][column_index];
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
      input_batch->data[batch_index][feature_index] =
          input_dataset->data[start_index + batch_index][feature_index];
    }
    target_batch->data[batch_index][0] =
        target_dataset->data[start_index + batch_index][0];
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

    for (int feature_index = 0; feature_index < INPUT_SIZE; feature_index++) {
      if (token == NULL) {
        break;
      }
      (*input_dataset)->data[sample_index][feature_index] = atof(token);
      token = strtok(NULL, ",");
    }

    if (token != NULL) {
      (*target_dataset)->data[sample_index][0] = atof(token);
    }
  }

  fclose(file);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <data.csv>\n", argv[0]);
    return -1;
  }

  double start_time;
  double end_time;

  Matrix *input_dataset = NULL;
  Matrix *target_dataset = NULL;
  int number_of_samples = 0;

  if (load_dataset_from_csv(argv[1], &input_dataset, &target_dataset,
                            &number_of_samples) != 0) {
    return -1;
  }

  Matrix *weights_input_to_hidden = allocate_matrix(INPUT_SIZE, HIDDEN_SIZE);
  Matrix *weights_hidden_to_output = allocate_matrix(HIDDEN_SIZE, OUTPUT_SIZE);
  random_initialize_matrix(weights_input_to_hidden);
  random_initialize_matrix(weights_hidden_to_output);

  start_time = omp_get_wtime();

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
      copy_mini_batch(input_dataset, target_dataset, input_batch, target_batch,
                      batch_start_index, batch_size);

      Matrix *hidden_layer_output =
          matrix_multiply(input_batch, weights_input_to_hidden);
      apply_relu_in_place(hidden_layer_output);
      Matrix *predicted_output =
          matrix_multiply(hidden_layer_output, weights_hidden_to_output);

      float mean_squared_error =
          compute_mean_squared_error(predicted_output, target_batch);
      if (epoch_index % LOG_EVERY_EPOCH == 0 && batch_start_index == 0) {
        printf("Epoch %d, MSE: %f\n", epoch_index, mean_squared_error);
      }

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

  printf("Training time: %.4f seconds\n", end_time - start_time);

  free_matrix(weights_input_to_hidden);
  free_matrix(weights_hidden_to_output);
  free_matrix(input_dataset);
  free_matrix(target_dataset);

  return 0;
}
