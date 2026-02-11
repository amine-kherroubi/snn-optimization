// Parallel matrix multiplication using CUDA
// Optimized version: pre-allocate reusable matrices to minimize malloc/free
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ! Network Parameters
#define INPUT_SIZE 32   // Number of input features
#define HIDDEN_SIZE 256 // Number of neurons in the hidden layer
#define OUTPUT_SIZE 1   // Number of output neurons
#define EPOCHS 100      // Number of training epochs
#define LEARNING_RATE 0.002
#define BATCH_SIZE 256 // Batch size for SGD
#define THREADS_PER_BLOCK 16
#define NUM_TEST_RUNS 10 // Number of times to run training for averaging
#define NUM_STREAMS 3
#define TILE_ROWS 128
#define NUM_PREALLOCATED 10 // Number of pre-allocated reusable matrices

// ! Data Structures
typedef struct
{
    int rows;
    int cols;
    float *data;
    int pinned;
    int reusable; // 1 = part of pre-allocated pool, should not be individually freed
} Matrix;

// Global GPU context to avoid repeated allocations
typedef struct
{
    float *d_A_tiles[NUM_STREAMS];
    float *d_C_tiles[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    size_t tile_bytes_A;
    size_t tile_bytes_C;
    int max_A_cols;  // Add: Track max A columns for tile allocation
    int max_B_cols;  // Add: Track max B columns for tile allocation
    int initialized;
} GlobalGPUContext;

static GlobalGPUContext g_gpu_ctx = {0};

// Pre-allocated matrix pool for training loop temporaries
typedef struct
{
    Matrix X_batch;   // 0: BATCH_SIZE x INPUT_SIZE
    Matrix Y_batch;   // 1: BATCH_SIZE x OUTPUT_SIZE
    Matrix Z1;        // 2: BATCH_SIZE x HIDDEN_SIZE
    Matrix Y_pred;    // 3: BATCH_SIZE x OUTPUT_SIZE
    Matrix dZ2;       // 4: BATCH_SIZE x OUTPUT_SIZE
    Matrix Z1_T;      // 5: HIDDEN_SIZE x BATCH_SIZE
    Matrix dW2;       // 6: HIDDEN_SIZE x OUTPUT_SIZE
    Matrix W2_T;      // 7: OUTPUT_SIZE x HIDDEN_SIZE
    Matrix dZ1;       // 8: BATCH_SIZE x HIDDEN_SIZE
    Matrix X_batch_T; // 9: INPUT_SIZE x BATCH_SIZE
    // dZ1_derivative reuses dW1 memory since they don't overlap in lifetime
    // dW1 is computed after dZ1_derivative is consumed
    float *dZ1_deriv_data; // Extra buffer: BATCH_SIZE x HIDDEN_SIZE
    float *dW1_data;       // Extra buffer: INPUT_SIZE x HIDDEN_SIZE
    int initialized;
} MatrixPool;

static MatrixPool g_pool = {0};

void init_pool_matrix(Matrix *m, int rows, int cols)
{
    m->rows = rows;
    m->cols = cols;
    m->reusable = 1;
    m->pinned = 1;
    cudaError_t err =
        cudaMallocHost((void **)&m->data, rows * cols * sizeof(float));
    if (err != cudaSuccess)
    {
        m->data = (float *)malloc(rows * cols * sizeof(float));
        m->pinned = 0;
    }
}

void free_pool_matrix(Matrix *m)
{
    if (m->pinned)
        cudaFreeHost(m->data);
    else
        free(m->data);
    m->data = NULL;
}

void init_matrix_pool()
{
    if (g_pool.initialized)
        return;

    init_pool_matrix(&g_pool.X_batch, BATCH_SIZE, INPUT_SIZE);
    init_pool_matrix(&g_pool.Y_batch, BATCH_SIZE, OUTPUT_SIZE);
    init_pool_matrix(&g_pool.Z1, BATCH_SIZE, HIDDEN_SIZE);
    init_pool_matrix(&g_pool.Y_pred, BATCH_SIZE, OUTPUT_SIZE);
    init_pool_matrix(&g_pool.dZ2, BATCH_SIZE, OUTPUT_SIZE);
    init_pool_matrix(&g_pool.Z1_T, HIDDEN_SIZE, BATCH_SIZE);
    init_pool_matrix(&g_pool.dW2, HIDDEN_SIZE, OUTPUT_SIZE);
    init_pool_matrix(&g_pool.W2_T, OUTPUT_SIZE, HIDDEN_SIZE);
    init_pool_matrix(&g_pool.dZ1, BATCH_SIZE, HIDDEN_SIZE);
    init_pool_matrix(&g_pool.X_batch_T, INPUT_SIZE, BATCH_SIZE);

    // Extra buffers (not wrapped in Matrix, used via raw pointer)
    cudaError_t err;
    err = cudaMallocHost((void **)&g_pool.dZ1_deriv_data,
                         BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    if (err != cudaSuccess)
        g_pool.dZ1_deriv_data =
            (float *)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));

    err = cudaMallocHost((void **)&g_pool.dW1_data,
                         INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    if (err != cudaSuccess)
        g_pool.dW1_data =
            (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));

    g_pool.initialized = 1;
    printf("[Pool] Pre-allocated %d reusable matrices + 2 extra buffers\n",
           NUM_PREALLOCATED);
}

void cleanup_matrix_pool()
{
    if (!g_pool.initialized)
        return;

    free_pool_matrix(&g_pool.X_batch);
    free_pool_matrix(&g_pool.Y_batch);
    free_pool_matrix(&g_pool.Z1);
    free_pool_matrix(&g_pool.Y_pred);
    free_pool_matrix(&g_pool.dZ2);
    free_pool_matrix(&g_pool.Z1_T);
    free_pool_matrix(&g_pool.dW2);
    free_pool_matrix(&g_pool.W2_T);
    free_pool_matrix(&g_pool.dZ1);
    free_pool_matrix(&g_pool.X_batch_T);

    // Free extra buffers (try pinned first)
    cudaFreeHost(g_pool.dZ1_deriv_data);
    cudaFreeHost(g_pool.dW1_data);

    g_pool.initialized = 0;
    printf("[Pool] Freed all pre-allocated matrices\n");
}

// Resize a pool matrix for a smaller batch (just adjust rows, no realloc)
void pool_matrix_set_rows(Matrix *m, int rows) { m->rows = rows; }

void ensure_tile_capacity(int A_cols, int B_cols)
{
    size_t required_A = TILE_ROWS * A_cols * sizeof(float);
    size_t required_C = TILE_ROWS * B_cols * sizeof(float);

    int needs_realloc = 0;

    // Check if we need larger A tiles
    if (required_A > g_gpu_ctx.tile_bytes_A)
    {
        g_gpu_ctx.tile_bytes_A = required_A;
        g_gpu_ctx.max_A_cols = A_cols;
        needs_realloc = 1;
    }

    // Check if we need larger C tiles
    if (required_C > g_gpu_ctx.tile_bytes_C)
    {
        g_gpu_ctx.tile_bytes_C = required_C;
        g_gpu_ctx.max_B_cols = B_cols;
        needs_realloc = 1;
    }

    if (needs_realloc)
    {
        // Reallocate tiles for all streams
        for (int s = 0; s < NUM_STREAMS; s++)
        {
            if (g_gpu_ctx.d_A_tiles[s]) cudaFree(g_gpu_ctx.d_A_tiles[s]);
            if (g_gpu_ctx.d_C_tiles[s]) cudaFree(g_gpu_ctx.d_C_tiles[s]);

            cudaMalloc((void **)&g_gpu_ctx.d_A_tiles[s], g_gpu_ctx.tile_bytes_A);
            cudaMalloc((void **)&g_gpu_ctx.d_C_tiles[s], g_gpu_ctx.tile_bytes_C);
        }
    }
}

void init_global_gpu_context()
{
    if (g_gpu_ctx.initialized)
        return;

    g_gpu_ctx.tile_bytes_A = 0;
    g_gpu_ctx.tile_bytes_C = 0;
    g_gpu_ctx.max_A_cols = 0;
    g_gpu_ctx.max_B_cols = 0;

    for (int s = 0; s < NUM_STREAMS; s++)
    {
        g_gpu_ctx.d_A_tiles[s] = NULL;
        g_gpu_ctx.d_C_tiles[s] = NULL;
        cudaStreamCreate(&g_gpu_ctx.streams[s]);
    }

    g_gpu_ctx.initialized = 1;
}

void cleanup_global_gpu_context()
{
    if (!g_gpu_ctx.initialized)
        return;

    for (int s = 0; s < NUM_STREAMS; s++)
    {
        cudaFree(g_gpu_ctx.d_A_tiles[s]);
        cudaFree(g_gpu_ctx.d_C_tiles[s]);
        cudaStreamDestroy(g_gpu_ctx.streams[s]);
    }

    g_gpu_ctx.initialized = 0;
}

// ! Memory Management
// Function to allocate a matrix (only used for non-pool matrices: W1, W2, X, Y)
Matrix *allocate_matrix(int rows, int cols)
{
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    m->rows = rows;
    m->cols = cols;
    m->reusable = 0;
    m->pinned = 1;
    cudaError_t err =
        cudaMallocHost((void **)&m->data, rows * cols * sizeof(float));
    if (err != cudaSuccess)
    {
        m->data = (float *)malloc(rows * cols * sizeof(float));
        m->pinned = 0;
    }
    return m;
}

// Function to free a matrix (skips pool matrices)
void free_matrix(Matrix *m)
{
    if (m->reusable)
        return; // Pool matrix — do not free individually
    if (m->pinned)
        cudaFreeHost(m->data);
    else
        free(m->data);
    free(m);
}

// ! Matrix Operations
// Function to initialize matrix with random values
void random_init(Matrix *m)
{
    for (int i = 0; i < m->rows; i++)
    {
        for (int j = 0; j < m->cols; j++)
        {
            m->data[i * m->cols + j] = (float)rand() / RAND_MAX;
        }
    }
}

// ! Matrix Operations (GPU version)
__global__ void mat_mult_kernel(float *A, float *B, float *C, int A_rows,
                                int A_cols, int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols)
    {
        float value = 0.0f;
        for (int k = 0; k < A_cols; k++)
        {
            value += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = value;
    }
}

// In-place matrix multiply: writes result into pre-allocated C
void mat_mult_into(Matrix *A, Matrix *B, Matrix *C)
{
    if (A->cols != B->rows)
    {
        printf("Incompatible matrices for multiplication.\n");
        exit(1);
    }

    C->rows = A->rows;
    C->cols = B->cols;

    size_t sizeB = B->rows * B->cols * sizeof(float);

    // Initialize GPU context on first call
    if (!g_gpu_ctx.initialized)
    {
        init_global_gpu_context();
    }

    // Ensure tile buffers are large enough for this operation
    ensure_tile_capacity(A->cols, B->cols);

    // Allocate B for this operation (NO CACHING - prevents stale data bugs)
    float *d_B;
    cudaMalloc((void **)&d_B, sizeB);
    cudaMemcpy(d_B, B->data, sizeB, cudaMemcpyHostToDevice);

    // Tile by rows of A/C with triple buffering and async copies
    for (int row_start = 0, tile_idx = 0; row_start < A->rows;
         row_start += TILE_ROWS, tile_idx++)
    {
        int tile_rows =
            (row_start + TILE_ROWS <= A->rows) ? TILE_ROWS : (A->rows - row_start);
        int s = tile_idx % NUM_STREAMS;

        const float *A_tile_host = A->data + row_start * A->cols;
        cudaMemcpyAsync(g_gpu_ctx.d_A_tiles[s], A_tile_host,
                        tile_rows * A->cols * sizeof(float), cudaMemcpyHostToDevice,
                        g_gpu_ctx.streams[s]);

        dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
        dim3 numBlocks((B->cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (tile_rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

        mat_mult_kernel<<<numBlocks, threadsPerBlock, 0, g_gpu_ctx.streams[s]>>>(
            g_gpu_ctx.d_A_tiles[s], d_B, g_gpu_ctx.d_C_tiles[s],
            tile_rows, A->cols, B->cols);

        float *C_tile_host = C->data + row_start * C->cols;
        cudaMemcpyAsync(C_tile_host, g_gpu_ctx.d_C_tiles[s],
                        tile_rows * B->cols * sizeof(float), cudaMemcpyDeviceToHost,
                        g_gpu_ctx.streams[s]);
    }

    // Synchronize all streams
    for (int s = 0; s < NUM_STREAMS; s++)
    {
        cudaStreamSynchronize(g_gpu_ctx.streams[s]);
    }

    // Free B for this operation
    cudaFree(d_B);
}

// In-place matrix subtraction: C = A - B (writes into C)
void mat_sub_into(Matrix *A, Matrix *B, Matrix *C)
{
    if (A->rows != B->rows || A->cols != B->cols)
    {
        printf("Incompatible matrices for subtraction.\n");
        exit(1);
    }
    C->rows = A->rows;
    C->cols = A->cols;
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            C->data[i * A->cols + j] =
                A->data[i * A->cols + j] - B->data[i * A->cols + j];
}

// Matrix scalar multiplication: A = A * scalar
void mat_scalar_mult(Matrix *A, float scalar)
{
    for (int i = 0; i < A->rows; i++)
        for (int j = 0; j < A->cols; j++)
            A->data[i * A->cols + j] *= scalar;
}

// In-place transpose: writes transpose of src into dst
void transpose_into(Matrix *src, Matrix *dst)
{
    dst->rows = src->cols;
    dst->cols = src->rows;
    for (int i = 0; i < src->rows; i++)
        for (int j = 0; j < src->cols; j++)
            dst->data[j * src->rows + i] = src->data[i * src->cols + j];
}

// ! Activation Functions
// Function to apply ReLU activation (in-place)
void relu(Matrix *m)
{
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            m->data[i * m->cols + j] = fmaxf(0, m->data[i * m->cols + j]);
}

// In-place ReLU derivative: writes derivative into out_data buffer
void relu_derivative_into(Matrix *m, float *out_data)
{
    for (int i = 0; i < m->rows; i++)
        for (int j = 0; j < m->cols; j++)
            out_data[i * m->cols + j] = (m->data[i * m->cols + j] > 0) ? 1.0f : 0.0f;
}

// ! Loss Functions
// Function to compute Mean Squared Error
float mean_squared_error(Matrix *Y_pred, Matrix *Y_true)
{
    float mse = 0.0f;
    for (int i = 0; i < Y_pred->rows; i++)
        for (int j = 0; j < Y_pred->cols; j++)
            mse += pow(Y_pred->data[i * Y_pred->cols + j] -
                           Y_true->data[i * Y_true->cols + j],
                       2);
    return mse / Y_pred->rows;
}

// ! Optimization
// Function to update weights: W = W - learning_rate * grad
void update_weights(Matrix *W, Matrix *grad, float learning_rate)
{
    for (int i = 0; i < W->rows; i++)
        for (int j = 0; j < W->cols; j++)
            W->data[i * W->cols + j] -=
                learning_rate * grad->data[i * grad->cols + j];
}

// Update weights from raw data buffer
void update_weights_raw(Matrix *W, float *grad_data, int grad_cols,
                        float learning_rate)
{
    for (int i = 0; i < W->rows; i++)
        for (int j = 0; j < W->cols; j++)
            W->data[i * W->cols + j] -= learning_rate * grad_data[i * grad_cols + j];
}

// ! Backpropagation (uses pre-allocated pool matrices)
void backpropagation(Matrix *X_batch, Matrix *Y_batch, Matrix *Z1,
                     Matrix *Y_pred, Matrix *W1, Matrix *W2, int batch_size)
{
    // Compute dZ2 = Y_pred - Y_batch (into pool dZ2)
    mat_sub_into(Y_pred, Y_batch, &g_pool.dZ2);
    mat_scalar_mult(&g_pool.dZ2, 2.0f / batch_size);

    // Compute Z1^T (into pool Z1_T)
    transpose_into(Z1, &g_pool.Z1_T);

    // Compute dW2 = Z1^T * dZ2 (into pool dW2)
    mat_mult_into(&g_pool.Z1_T, &g_pool.dZ2, &g_pool.dW2);
    update_weights(W2, &g_pool.dW2, LEARNING_RATE);

    // Compute W2^T (into pool W2_T)
    transpose_into(W2, &g_pool.W2_T);

    // Compute dZ1 = dZ2 * W2^T (into pool dZ1)
    mat_mult_into(&g_pool.dZ2, &g_pool.W2_T, &g_pool.dZ1);

    // Apply ReLU derivative (into extra buffer)
    relu_derivative_into(Z1, g_pool.dZ1_deriv_data);
    for (int i = 0; i < g_pool.dZ1.rows; i++)
    {
        for (int j = 0; j < g_pool.dZ1.cols; j++)
        {
            g_pool.dZ1.data[i * g_pool.dZ1.cols + j] *=
                g_pool.dZ1_deriv_data[i * g_pool.dZ1.cols + j];
        }
    }

    // Compute X_batch^T (into pool X_batch_T)
    transpose_into(X_batch, &g_pool.X_batch_T);

    // Compute dW1 = X_batch^T * dZ1 (into extra dW1 buffer via temp Matrix wrapper)
    Matrix dW1_wrapper;
    dW1_wrapper.rows = INPUT_SIZE;
    dW1_wrapper.cols = HIDDEN_SIZE;
    dW1_wrapper.data = g_pool.dW1_data;
    dW1_wrapper.pinned = 1;
    dW1_wrapper.reusable = 1;
    mat_mult_into(&g_pool.X_batch_T, &g_pool.dZ1, &dW1_wrapper);
    update_weights_raw(W1, g_pool.dW1_data, HIDDEN_SIZE, LEARNING_RATE);
}

// ! Batch Processing
// Function to get a batch from the dataset (writes into pre-allocated matrices)
void get_batch(Matrix *X, Matrix *Y, Matrix *X_batch, Matrix *Y_batch,
               int batch_start, int batch_size)
{
    X_batch->rows = batch_size;
    Y_batch->rows = batch_size;
    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
            X_batch->data[i * INPUT_SIZE + j] =
                X->data[(batch_start + i) * INPUT_SIZE + j];
        Y_batch->data[i * OUTPUT_SIZE] = Y->data[(batch_start + i) * OUTPUT_SIZE];
    }
}

// ! Data Loading
// Function to load CSV and populate X and Y, Assuming the last column is Y
int load_csv(const char *filename, Matrix **X, Matrix **Y, int *num_samples)
{
    FILE *file = fopen(filename, "r");
    if (!file)
    {
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
    while (fgets(line, sizeof(line), file))
    {
        char *token = strtok(line, ",");
        int j = 0;
        while (token)
        {
            if (j < INPUT_SIZE)
            {
                (*X)->data[i * INPUT_SIZE + j] = atof(token);
            }
            else
            {
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
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
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
    init_matrix_pool();

    // Run training multiple times
    for (int run = 0; run < NUM_TEST_RUNS; run++)
    {
        // Allocate and initialize weights (these change each run)
        Matrix *W1 = allocate_matrix(INPUT_SIZE, HIDDEN_SIZE);
        Matrix *W2 = allocate_matrix(HIDDEN_SIZE, OUTPUT_SIZE);
        random_init(W1);
        random_init(W2);

        // Start measuring time
        start_time = omp_get_wtime();

        float final_mse = 0.0f;

        // Training loop — NO malloc/free inside this loop!
        for (int epoch = 0; epoch < EPOCHS; epoch++)
        {
            for (int batch_start = 0; batch_start < num_samples;
                 batch_start += BATCH_SIZE)
            {
                int batch_end = fmin(batch_start + BATCH_SIZE, num_samples);
                int batch_size = batch_end - batch_start;

                // Extract batch into pre-allocated pool matrices
                get_batch(X, Y, &g_pool.X_batch, &g_pool.Y_batch, batch_start,
                          batch_size);

                // Forward pass: X -> Hidden Layer -> ReLU -> Output Layer
                mat_mult_into(&g_pool.X_batch, W1, &g_pool.Z1);
                relu(&g_pool.Z1);
                mat_mult_into(&g_pool.Z1, W2, &g_pool.Y_pred);

                // Compute loss
                final_mse = mean_squared_error(&g_pool.Y_pred, &g_pool.Y_batch);

                // Backward pass (all temporaries use pool matrices)
                backpropagation(&g_pool.X_batch, &g_pool.Y_batch, &g_pool.Z1,
                                &g_pool.Y_pred, W1, W2, batch_size);
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
    printf("Average training time over %d runs: %.4f seconds | Average final MSE: %.6f\n", NUM_TEST_RUNS,
           total_time / NUM_TEST_RUNS, total_final_mse / NUM_TEST_RUNS);

    // Cleanup — all at once
    free_matrix(X);
    free_matrix(Y);
    cleanup_matrix_pool();
    cleanup_global_gpu_context();

    return 0;
}
