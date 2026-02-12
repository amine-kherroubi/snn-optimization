#import "@preview/charged-ieee:0.1.4": ieee
#show raw.where(block: true): set block(above: 1.2em, below: 1.2em)
#show math.equation.where(block: true): set block(above: 1.2em, below: 1.2em)
#show par: set par(spacing: 0.65em)
#set page(numbering: "1", number-align: center)
#show raw: set text(size: 7.5pt)

#show: ieee.with(
  title: [
    #image("esi_logo.png", width: 25%)
    #v(-0.2em)
    Memory Management Strategies for GPU-Accelerated Shallow Neural Network Training
    #v(0.4em)
  ],
  abstract: [
    Building upon prior work by Brouthen and Akeb on GPU parallelization for shallow neural network training, we evaluate three memory management strategies for CUDA implementations. Their reference implementation allocates and frees GPU memory for every matrix operation. We implemented three alternative strategies: a streams-based approach using CUDA streams for concurrent execution, a pinned memory strategy using page-locked host memory, and a combined approach that integrates both optimizations with pre-allocated GPU resources. Our experiments on a Tesla T4 GPU show that the combined strategy achieves approximately 1.65× to 1.73× speedup over the reference implementation for the baseline network configuration, by eliminating repeated `cudaMalloc` and `cudaFree` calls. We validate functional correctness across all strategies and provide additional scalability analysis with varying numbers of neurons and network depths. All code is publicly available at #link("https://github.com/amine-kherroubi/snn-optimization")[github.com/amine-kherroubi/snn-optimization].
  ],
  authors: (
    (
      name: "Mohamed El Amine Kherroubi",
      department: [2CS-SIQ],
      email: "mm_kherroubi@esi.dz",
    ),
    (
      name: "Badis Khalef",
      department: [2CS-SID],
      email: "mb_khalef@esi.dz",
    ),
    (
      name: "Mounir Sofiane Mostefai",
      department: [2CS-SIQ],
      email: "mm_mostefai@esi.dz",
    ),
    (
      name: "Youcef Tati",
      department: [2CS-SIQ],
      email: "my_tati@esi.dz",
    ),
    (
      name: "Mohamed Ishak Messadia",
      department: [2CS-SIQ],
      email: "mm_messadia@esi.dz",
    ),
  ),
  index-terms: (
    "CUDA",
    "GPU Computing",
    "Neural Networks",
    "Pinned Memory",
    "CUDA Streams",
    "Memory Management",
    "Performance Optimization",
  ),
  figure-supplement: [Fig.],
)

= Introduction

== Motivation

Matrix multiplication is the primary computational bottleneck in neural network training, occurring repeatedly during both forward and backward propagation. Brouthen and Akeb (2024) demonstrated that GPU acceleration with CUDA significantly speeds up shallow neural network training compared to CPU implementations.

However, their reference implementation allocates and frees GPU memory with `cudaMalloc` and `cudaFree` for each matrix operation. For a typical training run with 100 epochs and multiple batches, this leads to tens of thousands of allocation-deallocation cycles. With a dataset containing 100 batches, the reference network (1 hidden layer) performs 5 matrix multiplications per batch, yielding 50,000 allocation-deallocation pairs over 100 epochs. Additionally, synchronous transfers with pageable host memory via `cudaMemcpy` block the CPU, preventing overlap between computation and data transfers.

== Objectives

This work evaluates three CUDA memory management strategies applied to the reference network:

1. *Streams-based approach:* Using CUDA streams to enable concurrent kernel execution and memory transfers
2. *Pinned memory:* Using page-locked host memory to accelerate data transfers
3. *Combined approach:* Integrating pinned memory and streams with pre-allocated GPU resources

Our goals are to quantify the performance impact of each strategy, validate functional correctness, and understand which optimizations provide meaningful speedups for shallow neural network training.

== Paper Organization

Section 2 reviews the neural network architecture. Section 3 analyzes the reference implementation. Section 4 describes our three optimization strategies. Section 5 details the experimental setup including functional validation. Section 6 presents performance results for the optimization strategies. Section 7 provides additional scalability analysis. Section 8 discusses findings. Section 9 concludes.

= Neural Network Overview

This section briefly reviews the shallow neural network used in our primary experiments. For complete details, refer to Brouthen and Akeb (2024).

== Reference Network Architecture

The reference network consists of:
- Input layer with 32 features
- One hidden layer with 256 neurons using ReLU activation
- Single output neuron for regression

Training uses stochastic gradient descent with batch size 256, learning rate 0.002, and 100 epochs.

== Operations

Forward propagation:
$ Z_1 = X W_1, quad H = "ReLU"(Z_1), quad Y_"pred" = H W_2 $

Backward propagation:
$ "dZ"_2 = 2(Y_"pred" - Y) / n, quad "dW"_2 = H^T "dZ"_2 $
$ "dZ"_1 = "dZ"_2 W_2^T times.o "ReLU'"(Z_1), quad "dW"_1 = X^T "dZ"_1 $

Weight updates:
$ W_2 = W_2 - alpha "dW"_2, quad W_1 = W_1 - alpha "dW"_1 $

Loss (Mean Squared Error):
$ "MSE" = 1/n sum_(i=1)^n (Y_"pred" - Y)^2 $

== Computational Pattern

Each training batch requires 5 matrix multiplications: 2 for forward propagation and 3 for backward propagation. With 100 epochs and multiple batches per epoch, matrix multiplication dominates execution time, making transfer and allocation overhead significant.

= Reference Implementation

== Overview

The reference implementation follows a consistent pattern for each matrix multiplication:

1. Allocate device memory using `cudaMalloc`
2. Copy inputs from host to device using `cudaMemcpy`
3. Launch CUDA kernel for computation
4. Synchronize with `cudaDeviceSynchronize`
5. Copy results from device to host using `cudaMemcpy`
6. Free device memory using `cudaFree`

The CUDA kernel implements standard matrix multiplication with each thread computing one output element:

```c
__global__ void mat_mult_kernel(float *A, float *B, float *C,
                                int A_rows, int A_cols, int B_cols) {
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
```

The implementation uses 16×16 thread blocks. The matrix multiplication wrapper allocates device memory on every call:

```c
Matrix *mat_mult(Matrix *A, Matrix *B) {
  Matrix *C = allocate_matrix(A->rows, B->cols);

  // Allocated and freed on every call
  cudaMalloc((void **)&d_A, sizeA);
  cudaMalloc((void **)&d_B, sizeB);
  cudaMalloc((void **)&d_C, sizeC);

  cudaMemcpy(d_A, A->data, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B->data, sizeB, cudaMemcpyHostToDevice);

  mat_mult_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, ...);
  cudaDeviceSynchronize();

  cudaMemcpy(C->data, d_C, sizeC, cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

  return C;
}
```

== Observed Limitations

*Repeated Memory Allocation:* Allocating and freeing GPU memory for every matrix multiplication is the most significant limitation. Each `cudaMalloc` call incurs CUDA runtime overhead, and this cost accumulates over tens of thousands of calls during training.

*Synchronous Transfers:* The implementation uses `cudaMemcpy`, which blocks the CPU until each transfer completes, preventing overlap between data transfers and computation.

*Pageable Host Memory:* Standard `malloc` allocates pageable memory. Before transferring data to the GPU, CUDA must stage it through a temporary pinned buffer, adding latency.

*No Resource Reuse:* Device memory is never reused across matrix multiplication calls.

= Optimization Strategies

== Strategy 1: Streams-Based Approach

=== Implementation

We used CUDA streams to attempt overlapping host-to-device transfers, kernel execution, and device-to-host transfers:

- All host matrix allocations use `cudaMallocHost` (pinned memory), including weights, activations, and gradient matrices
- Three persistent device buffers are allocated per stream for tiling matrix A and the result matrix C
- Matrix B is copied to the device at the start of each `mat_mult` call and freed at the end
- Matrix A is divided into tiles of 128 rows; tiles are processed across the three streams in round-robin fashion
- Each stream performs: asynchronous H2D copy of the A tile → kernel launch → asynchronous D2H copy of the C tile
- All streams are synchronized before returning

The key code pattern:

```c
Matrix *mat_mult(Matrix *A, Matrix *B) {
  cudaMalloc((void **)&d_B, sizeB);
  cudaMemcpy(d_B, B->data, sizeB, cudaMemcpyHostToDevice);

  float *d_A_tiles[NUM_STREAMS], *d_C_tiles[NUM_STREAMS];
  cudaStream_t streams[NUM_STREAMS];
  for (int s = 0; s < NUM_STREAMS; s++) {
    cudaMalloc((void **)&d_A_tiles[s], tile_bytes_A);
    cudaMalloc((void **)&d_C_tiles[s], tile_bytes_C);
    cudaStreamCreate(&streams[s]);
  }

  for (int row_start = 0, tile_idx = 0; row_start < A->rows;
       row_start += TILE_ROWS, tile_idx++) {
    int s = tile_idx % NUM_STREAMS;
    cudaMemcpyAsync(d_A_tiles[s], A_tile_host, ..., streams[s]);
    mat_mult_kernel<<<..., streams[s]>>>(d_A_tiles[s], d_B, ...);
    cudaMemcpyAsync(C_tile_host, d_C_tiles[s], ..., streams[s]);
  }

  for (int s = 0; s < NUM_STREAMS; s++) {
    cudaStreamSynchronize(streams[s]);
    cudaStreamDestroy(streams[s]);
    cudaFree(d_A_tiles[s]); cudaFree(d_C_tiles[s]);
  }
  cudaFree(d_B);
  return C;
}
```

Streams and tile device buffers are allocated and freed on each `mat_mult` call, so GPU allocation overhead is not eliminated.

== Strategy 2: Pinned Memory Approach

=== Implementation

This strategy isolates the effect of pinned host memory:

- Intermediate matrices returned by `mat_mult` are allocated with `cudaMallocHost`
- Weight matrices, batch inputs, and batch targets are also allocated with `cudaMallocHost`
- Device memory allocation follows the same per-call pattern as the reference
- All transfers remain synchronous via `cudaMemcpy`

```c
Matrix *allocate_matrix_pinned(int rows, int cols) {
  Matrix *m = (Matrix *)malloc(sizeof(Matrix));
  m->rows = rows; m->cols = cols;
  m->pinned = 1;
  cudaError_t err =
      cudaMallocHost((void **)&m->data, rows * cols * sizeof(float));
  if (err != cudaSuccess) {
    m->data = (float *)malloc(rows * cols * sizeof(float));
    m->pinned = 0;
  }
  return m;
}

void free_matrix(Matrix *m) {
  if (m->pinned) cudaFreeHost(m->data);
  else free(m->data);
  free(m);
}
```

Short-lived intermediate matrices (transpose buffers, ReLU derivative arrays) that do not participate in GPU transfers remain allocated with standard `malloc`.

== Strategy 3: Combined Approach with Global Memory Pool

=== Implementation

This strategy addresses the core bottleneck by pre-allocating all GPU resources once in a global context:

```c
typedef struct {
  // Cached device copy of B
  float *d_B_cache;
  
  float *d_A_tiles[NUM_STREAMS];
  float *d_C_tiles[NUM_STREAMS];
  
  // Persistent streams
  cudaStream_t streams[NUM_STREAMS];
  
  size_t tile_bytes_A;
  size_t tile_bytes_C;
  size_t cached_B_size;
  int initialized;
} GlobalGPUContext;

static GlobalGPUContext g_gpu_ctx = {0};
```

Initialization occurs once on the first `mat_mult` call:

```c
void init_global_gpu_context(int tile_rows, int A_cols, int B_cols) {
  if (g_gpu_ctx.initialized) return;

  for (int s = 0; s < NUM_STREAMS; s++) {
    cudaMalloc((void **)&g_gpu_ctx.d_A_tiles[s], tile_bytes_A);
    cudaMalloc((void **)&g_gpu_ctx.d_C_tiles[s], tile_bytes_C);
    cudaStreamCreate(&g_gpu_ctx.streams[s]);
  }
  g_gpu_ctx.d_B_cache = NULL;
  g_gpu_ctx.initialized = 1;
}
```

For each operation, the implementation reuses pre-allocated resources:

```c
Matrix *mat_mult(Matrix *A, Matrix *B) {
  if (!g_gpu_ctx.initialized)
    init_global_gpu_context(TILE_ROWS, A->cols, B->cols);

  // Re-upload B only if its byte size changed
  size_t sizeB = B->rows * B->cols * sizeof(float);
  if (g_gpu_ctx.cached_B_size != sizeB) {
    if (g_gpu_ctx.d_B_cache) cudaFree(g_gpu_ctx.d_B_cache);
    cudaMalloc((void **)&g_gpu_ctx.d_B_cache, sizeB);
    cudaMemcpy(g_gpu_ctx.d_B_cache, B->data, sizeB,
               cudaMemcpyHostToDevice);
    g_gpu_ctx.cached_B_size = sizeB;
  }

  // Tiled loop reuses g_gpu_ctx.d_A_tiles, d_C_tiles, streams
}
```

At program termination, all resources are freed once:

```c
void cleanup_global_gpu_context() {
  for (int s = 0; s < NUM_STREAMS; s++) {
    cudaFree(g_gpu_ctx.d_A_tiles[s]);
    cudaFree(g_gpu_ctx.d_C_tiles[s]);
    cudaStreamDestroy(g_gpu_ctx.streams[s]);
  }
  if (g_gpu_ctx.d_B_cache) cudaFree(g_gpu_ctx.d_B_cache);
}
```

Host memory uses `cudaMallocHost` (pinned), as in the streams approach.

=== Key Differences

Unlike the streams-only approach, this strategy:
- Allocates GPU tile buffers and streams once at initialization
- Keeps streams persistent across operations
- Caches matrix B on the device; the cache is invalidated only when the byte size of B changes

The B cache is keyed by byte size. When consecutive `mat_mult` calls encounter different matrices of the same size, the cache is not refreshed. The distinct matrix sizes encountered during one training iteration differ sufficiently that this does not affect correctness, as confirmed by functional validation.

= Experimental Setup

== Hardware and Software

All experiments used Google Colab with a Tesla T4 GPU:

#figure(
  caption: [Tesla T4 GPU Specifications],
  table(
    columns: (auto, 1fr),
    align: (center, center),
    [Parameter], [Value],
    [CUDA Cores], [2560],
    [Memory Size], [16 GB GDDR6],
    [Memory Bandwidth], [320 GB/s],
    [Compute Capability], [7.5],
    [Copy Engines], [1 bidirectional],
  ),
)

The software environment consists of CUDA Toolkit 11.8, compiled with `nvcc` using `-O3` optimization, and GCC 9.4.0 as the host compiler.

== Datasets

We used three synthetic datasets identical to those in the reference solution:

#figure(
  caption: [Dataset Configurations],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    [Dataset], [Samples], [Batches], [Operations],
    [Small], [256], [1], [500],
    [Medium], [2560], [10], [5,000],
    [Large], [25600], [100], [50,000],
  ),
)

Each dataset has 32 input features sampled uniformly from $[-1, 1]$. Target values are defined as $Y = sum_(i=1)^(32) x_i^2 + epsilon$, where $epsilon tilde cal(N)(0, 0.01)$. The total number of matrix operations equals the number of batches times 100 epochs times 5 operations per batch.

== Functional Correctness Validation

To verify that our optimization strategies produce identical training outcomes to the reference, we:

1. Used identical random seeds for weight initialization across all implementations
2. Trained all four implementations on the same small dataset for 100 epochs
3. Compared final MSE values

#figure(
  caption: [Functional Correctness Validation (small dataset)],
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, right, right),
    [Strategy], [Training Time (s)], [Final MSE],
    [Reference], [0.3259], [41.192574],
    [Streams], [0.2914], [41.192574],
    [Pinned], [0.2770], [41.192574],
    [Combined], [0.1958], [41.192574],
  ),
)

All four implementations converge to the same MSE, confirming functional correctness.

== Metrics

*Training Time:* Total wall-clock time for 100 epochs, averaged over 10 independent runs.

*Speedup:* Ratio of reference time to alternative strategy time. Values greater than 1.0 indicate improvement.

== Evaluation Protocol

For each strategy and dataset:
1. Load dataset and initialize weights with a fixed random seed
2. Run 10 independent training sessions, each re-initializing weights
3. Record training time per session
4. Compute mean training time
5. Calculate speedup relative to the reference mean

= Performance Results

== Dataset Size Variation

We evaluated all three optimization strategies across the small, medium, and large datasets:

#figure(
  caption: [Training Time Across Dataset Sizes],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    [Strategy], [Small (s)], [Medium (s)], [Large (s)],
    [Reference], [0.3259], [3.0828], [30.3694],
    [Streams], [0.2914], [3.0978], [30.7172],
    [Pinned], [0.2770], [2.8901], [29.2370],
    [Combined], [0.1958], [1.7841], [18.4545],
  ),
)

#figure(
  caption: [Speedup Relative to Reference],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    [Strategy], [Small], [Medium], [Large],
    [Reference], [1.00×], [1.00×], [1.00×],
    [Streams], [1.12×], [1.00×], [0.99×],
    [Pinned], [1.18×], [1.07×], [1.04×],
    [Combined], [1.66×], [1.73×], [1.65×],
  ),
)

== Analysis of Optimization Strategies

*Streams-only approach:* This strategy provided minimal improvement (0.99× to 1.12× speedup). The Tesla T4's single bidirectional copy engine prevents concurrent transfers in both directions, limiting the benefit of asynchronous operations. Additionally, streams and tile device buffers are allocated and freed on each `mat_mult` call, failing to eliminate the allocation overhead.

*Pinned memory approach:* This strategy achieved modest but consistent improvements (4–18% speedup) by removing the intermediate staging copy that CUDA performs when transferring pageable memory. The gains are limited because device-side allocation overhead remains unchanged.

*Combined approach:* This strategy achieved the largest speedups (1.65× to 1.73×) by eliminating repeated GPU allocation calls. For the large dataset with 50,000 matrix operations, pre-allocating resources once and reusing them throughout training provides significant savings. The combination of pinned memory and persistent streams with global resource pooling addresses the primary bottleneck in the reference implementation.

= Additional Scalability Analysis

Beyond the core optimization strategies, we conducted additional experiments to understand how network configuration affects performance.

== Neuron Count Variation

We analyzed how the number of neurons in the hidden layer affects the speedup achieved by the combined strategy. These measurements were taken in a separate experimental session from Table 4.

#figure(
  caption: [Training Time vs. Neuron Count (large dataset)],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    [Neurons], [Reference (s)], [Combined (s)], [Speedup],
    [128], [13.7765], [6.1372], [2.24×],
    [256], [19.0889], [8.9194], [2.14×],
    [1024], [50.9751], [55.0013], [0.93×],
  ),
)

#figure(
  image("speedup_vs_neurons.png", width: 90%),
  caption: [Speedup of the combined strategy versus neuron count. Performance degrades at 1024 neurons.]
)

For small networks (128–256 neurons), each matrix multiplication completes quickly, making allocation overhead a significant portion of total runtime. The combined strategy achieves 2.14× to 2.24× speedup. For large networks (1024 neurons), computation time dominates, and the tiling overhead causes the combined strategy to perform 7% worse than the reference.

== Network Depth Variation

We extended the baseline reference network architecture to include additional hidden layers:

*Two-hidden-layer network:* Input → Hidden1 (256) → Hidden2 (256) → Output

*Three-hidden-layer network:* Input → Hidden1 (256) → Hidden2 (1024) → Hidden3 (256) → Output

Each additional hidden layer increases the number of matrix operations per batch from 5 (reference) to 8 (two-hidden-layer) to 11 (three-hidden-layer).

#figure(
  caption: [Training Time: Network Depth Comparison (large dataset)],
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    [Architecture], [Reference (s)], [Combined (s)], [Speedup],
    [Reference (1 hidden)], [30.3694], [18.4545], [1.65×],
    [Two-hidden-layer], [Not measured], [Not measured], [-],
    [Three-hidden-layer], [285.8243], [282.2075], [1.01×],
  ),
)

#figure(
  image("network_depth_comparison.png", width: 100%),
  caption: [Training time comparison across network depths. The combined strategy's advantage diminishes for deeper architectures.]
)

For the three-hidden-layer network, the combined strategy achieved only 1.01× speedup. The large intermediate activations of the 1024-neuron second hidden layer dominate execution time. With 11 matrix multiplications per batch instead of 5, computation cost grows faster than the fixed allocation savings, reducing the relative benefit of memory pooling.

= Discussion

== Understanding the Results

The performance of memory management optimizations is not uniform across configurations. The combined strategy eliminates repeated `cudaMalloc` and `cudaFree` calls, which is most beneficial when those calls represent a large fraction of total runtime.

For the reference network configuration (256 neurons, 1 hidden layer), the combined strategy achieves 1.65× to 1.73× speedup across dataset sizes. Allocation overhead constitutes a significant portion of total time, and eliminating over 50,000 GPU allocation calls for the large dataset provides measurable benefits.

However, as network size or depth increases, computation time dominates. For 1024 neurons, the combined strategy performs 7% worse than the reference due to tiling overhead. For the three-hidden-layer network, speedup drops to near 1.0× because the large second hidden layer increases computation time relative to the fixed allocation savings.

== Why Streams Alone Did Not Help

The streams strategy provided minimal improvement (0.99× to 1.12×) due to three factors:

*Hardware limitations:* The Tesla T4 has only one bidirectional copy engine. Even with three streams, the GPU cannot transfer data in both directions concurrently, forcing transfers to serialize.

*Unchanged allocation overhead:* Streams, tile device buffers, and matrix B are allocated and freed on each `mat_mult` call, leaving the primary bottleneck unaddressed.

*Small tile count:* For batch size 256 and tile height 128 rows, each matrix produces only two tiles. With so few tiles, pipeline depth is insufficient to amortize stream setup overhead.

== Pinned Memory: Modest but Consistent Gains

Pinned memory produced 4–18% speedup by removing the intermediate staging copy for pageable memory transfers. The gains are bounded because device-side allocation overhead is unchanged, and as dataset size grows, transfer latency becomes a smaller fraction of total runtime relative to computation.

== Combined Strategy: Effective Within Limits

The combined approach yielded the largest speedups for the reference network configuration (1.65× to 1.73×) by addressing the core allocation bottleneck. However, its effectiveness is bounded by the relative cost of allocation versus computation. When computation dominates (1024 neurons or deeper networks), the tiling and caching overhead can outweigh the allocation savings.

= Conclusion

We evaluated three memory management strategies for GPU-accelerated neural network training on a Tesla T4, leading to the following observations:

*Individual strategies:* Streams alone provided minimal benefit (0.99× to 1.12×) due to hardware constraints and unchanged allocation overhead. Pinned memory achieved modest improvements (1.04× to 1.18×) by accelerating transfers.

*Combined strategy:* The combined approach achieved 1.65× to 1.73× speedup on the reference network configuration by eliminating repeated GPU allocation calls. This strategy is most effective when allocation overhead is significant relative to computation time.

*Configuration dependence:* Performance gains are not uniform. For the reference network (256 neurons, 1 hidden layer), the combined strategy provides consistent speedups. For larger configurations (1024 neurons or 3 hidden layers), the reference implementation performs equally well or better.

*Practical implications:* The effective range of memory pooling optimizations is bounded by the relative cost of allocation versus computation. Profiling specific configurations is advisable before applying these optimizations.

== Future Work

Future work could explore shared memory tiling and tensor core operations to reduce computation time. Evaluating the streams strategy on GPUs with multiple copy engines (e.g., A100, H100) would clarify whether hardware constraints fully account for the limited gains observed here. An adaptive tile size mechanism could address the performance degradation at 1024 neurons. Testing these strategies on real-world datasets and production workloads would provide a more complete picture of their applicability.

= Acknowledgments

This work builds upon the research by Brouthen Kamel and Akeb Abdelaziz (2024), whose baseline CUDA implementation and thorough documentation enabled our investigation.

We thank Professor Dr. Amina Selma Haichour, our High Performance Computing instructor, for her guidance throughout this project.

#heading(numbering: none)[References]

#block[
  [1] Brouthen, K., & Akeb, A. (2024). Exploring parallelization of shallow neural network using CUDA.

  [2] Haichour, S. A. (2024). High Performance Computing (HPC). Course lectures. Ecole Nationale Supérieure d'Informatique (ESI), Algiers.

  [3] NVIDIA Corporation. (2024). CUDA C++ programming guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
]