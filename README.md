<img width="1628" height="774" alt="project" src="https://github.com/user-attachments/assets/34366123-9a66-4998-9b01-729da7ad79e8" />

# GPU Memory Management for Shallow Neural Network Training

A complete CUDA research case study on memory management optimizations for a shallow neural network. This repository compares a reference GPU implementation against optimized variants that reduce allocation overhead, improve memory transfer, and use CUDA streams.

## 1. What is this project?

1. A performance study of memory management in CUDA-based shallow neural network training.
2. A comparison between a reference GPU implementation and three optimization strategies:
   - **Streams** — concurrent kernel and transfer execution using CUDA streams.
   - **Pinned memory** — page-locked host buffers for faster host-device transfers.
   - **Combined** — both streams and pinned memory with pre-allocated GPU buffers.
3. Experimental evaluation of scaling behavior across dataset sizes and network depth.
4. Educational deliverables for a CUDA research project at École Nationale Supérieure d'Informatique (ESI).

## 2. Documents

- [Project Specification](project_specification[FRENCH].pdf) - Original project requirements and guidelines in French.
- [Final Report](project_report[ENGLISH].pdf) - Complete research article with experimental results and analysis.
- [Reference Report](reference/report/reference_report[ENGLISH].pdf) - Baseline implementation and report by Brouthen and Akeb.

## 3. Why this matters

GPU memory management is often the hidden bottleneck in neural network training. When device allocations and host transfers are repeated for every matrix operation, the overhead can dominate runtime. This project shows how alternative memory strategies can substantially improve performance for shallow network training while preserving correctness.

## 4. Key technical facts

- Input dimension: **32 features**
- Hidden layer size: **256 neurons**
- Output dimension: **1 neuron**
- Training epochs: **100**
- Batch size: **256**
- Metrics: **average runtime** and **final MSE** over multiple runs
- Data: synthetic convex datasets in `reference/data/`
- GPU test target: **NVIDIA T4** with compute capability **7.5**

## 5. Requirements

### Hardware
- NVIDIA GPU with compute capability **7.5+**

### Software
- CUDA Toolkit **11.8+**
- GNU GCC **9.4+**
- `nvcc` available in `PATH`
- Python **3.10+** for optional analysis scripts
- `python3 -m venv` support

### Optional Python packages
- `numpy`
- `pandas`
- `matplotlib`

Install optional Python dependencies with:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python_requirements.txt
```

## 6. Install and build

### 6.1 Clone the repository

```bash
git clone <repo-url>
cd <repo-name>
```

### 6.2 Build the reference implementations

```bash
cd reference

gcc -O3 -o nn nn.c -lm -fopenmp

gcc -O3 -o nn_pthreads nn_pthreads.c -lm -pthread -fopenmp

nvcc -O3 -o nn_cuda nn_cuda.cu -Xcompiler -fopenmp -gencode arch=compute_75,code=sm_75
```

### 6.3 Build the optimized CUDA variants

From the repository root:

```bash
cd alternatives
./run_all.sh
```

This script compiles the main CUDA variants and runs each of them on the three synthetic datasets.

### 6.4 Build a single optimized variant manually

```bash
cd alternatives
nvcc -O3 -Xcompiler -fopenmp -gencode arch=compute_75,code=sm_75 nn_cuda_combined.cu -o nn_cuda_combined
```

## 7. Run the code

### 7.1 Run the CUDA baseline

From `reference/`:

```bash
./nn_cuda ../reference/data/synthetic_convex_small.csv
```

### 6.2 Run a combined optimization variant

From `alternatives/`:

```bash
./nn_cuda_combined ../reference/data/synthetic_convex_small.csv
```

### 6.3 Run the sequential and pthread versions

From `reference/`:

```bash
./nn ../reference/data/synthetic_convex_small.csv
./nn_pthreads ../reference/data/synthetic_convex_small.csv
```

### 6.4 Run all variants automatically

```bash
cd alternatives
./run_all.sh
```

This command:
1. compiles `nn_cuda_reference`, `nn_cuda_streams`, `nn_cuda_pinned`, and `nn_cuda_combined`
2. runs each on `small`, `medium`, and `large`
3. prints timing output for each dataset

## 7. Directory structure

1. `alternatives/`
   - Optimized CUDA implementations and evaluation scripts.
   - Variants include `nn_cuda_reference.cu`, `nn_cuda_streams.cu`, `nn_cuda_pinned.cu`, and `nn_cuda_combined.cu`.
   - Depth variants: `*_two_layers.cu`, `*_three_layers.cu`.
2. `reference/`
   - Baseline code and datasets.
   - `nn.c`, `nn_pthreads.c`, `nn_cuda.cu`, `test_cuda.cu`.
   - `data/` contains `synthetic_convex_small.csv`, `synthetic_convex_medium.csv`, and `synthetic_convex_large.csv`.
3. `report/`
   - Final article source and report materials.
4. `run_full_project.ipynb`
   - Notebook for analysis and visualization.
5. `python_requirements.txt`
   - Python dependencies for optional analysis.

## 8. Experimental findings

- The **combined** strategy is the best-performing optimization in this study.
- **Streams-only** offers small improvements when compute and transfer overlap is already limited.
- **Pinned memory** improves transfer throughput and reduces host-device overhead.
- Performance gains depend strongly on network size, batch count, and layer depth.
- Depth variants show that adding more hidden layers reduces the benefit of memory-only optimizations.

## 10. Authors

Mohamed El Amine Kherroubi, Badis Khalef, Mounir Sofiane Mostefai, Youcef Tati, Mohamed Ishak Messadia
2CS-SIQ/SID, École Nationale Supérieure d'Informatique (ESI), Algiers

## 11. References

[1] Brouthen, K., & Akeb, A. (2024). Exploring parallelization of shallow neural network using CUDA.
