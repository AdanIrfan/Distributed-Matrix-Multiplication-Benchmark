# PDC CCP Matrix Multiplication Project Report

## 1. Project Overview
This project implements and compares multiple matrix multiplication approaches for parallel and distributed computing. The goal is to evaluate performance and correctness across sequential CPU, GPU, shared-memory, and distributed-memory implementations.

Implemented versions:
- Sequential CPU baseline (`src/cpu_baseline.cpp`)
- CUDA naïve GPU implementation (`src/cuda_naive.cu`)
- CUDA tiled shared-memory GPU implementation (`src/cuda_tiled.cu`)
- OpenMP directive-based CPU parallel implementation (`src/openmp_parallel.cpp`)
- MPI distributed-memory implementation (`src/mpi_matrix_mul.cpp`)

The project was developed and tested on Google Colab using Tesla T4 GPU, CUDA/NVCC, GCC/G++, OpenMP, and MPI.

## 2. Objectives
- Implement matrix multiplication using multiple parallelization strategies.
- Compare execution time and speedup for different implementations.
- Validate correctness across implementations.
- Provide containerization support using Docker.
- Document the build/run process and results.

## 3. Implementation Details

### 3.1 CPU Baseline
- File: `src/cpu_baseline.cpp`
- Simple triple-loop matrix multiplication.
- Used as the reference baseline for speedup calculations.

### 3.2 CUDA Naïve Implementation
- File: `src/cuda_naive.cu`
- Each GPU thread computes one output element.
- No shared-memory tiling or advanced memory optimizations.

### 3.3 CUDA Tiled Shared-Memory Implementation
- File: `src/cuda_tiled.cu`
- Uses shared memory for blocks of size 16x16.
- Each thread block cooperatively loads sub-blocks of matrices A and B.
- Uses boundary checks to support matrix sizes not divisible by tile size, including 250x250.

### 3.4 OpenMP Parallel Implementation
- File: `src/openmp_parallel.cpp`
- Uses OpenMP `#pragma omp parallel for` to parallelize the outer loop.
- Designed for multi-core CPU execution.

### 3.5 MPI Distributed Implementation
- File: `src/mpi_matrix_mul.cpp`
- Uses MPI to distribute rows of matrix A across processes.
- Broadcasts matrix B to all processes and gathers partial results.
- Requires matrix size to be divisible by the number of processes in the current implementation.

## 4. Results and Analysis

### 4.1 Correctness
Correctness validation was performed using checksum comparison across all implementations. For matrix sizes 250x250, 256x256, and 512x512, all implementations produced matching checksums.

### 4.2 Performance Summary
The project contains results files in `results/`:
- `timing_results.csv`
- `speedup_results.csv`
- `execution_time_comparison.png`
- `speedup_comparison.png`

Important observations:
- CUDA performance improves significantly with larger matrices.
- Small matrix sizes (e.g., 256x256) may not benefit from GPU execution due to overhead.
- The tiled CUDA implementation is optimized for GPU memory reuse and shows strong performance for larger sizes.
- MPI performance depends on problem size and communication overhead; small problems may not show speedup.

### 4.3 Notable Experimental Findings
- `cuda_naive` and `cuda_tiled` show a large speedup compared to the CPU baseline for 512x512.
- The CPU OpenMP version provides an improvement over the sequential baseline, but additional profiling is needed to determine thread scaling efficiency.
- MPI distributed execution is implemented and functional, but communication overhead can dominate for moderate matrix sizes.

## 5. Containerization
Docker support is provided in the `docker/` folder:
- `docker/Dockerfile`
- `docker/docker_commands.txt`

The Docker image includes required build tools, CUDA support, and MPI dependencies.

## 6. Usage Instructions
Basic compilation and execution commands are documented in `README.md`.

### Compile and run examples
```bash
cd pdc_matrix_project

# CPU baseline
g++ -O2 src/cpu_baseline.cpp -o cpu_baseline
./cpu_baseline 512

# CUDA naive
nvcc -O2 src/cuda_naive.cu -o cuda_naive
./cuda_naive 512

# CUDA tiled
nvcc -O2 src/cuda_tiled.cu -o cuda_tiled
./cuda_tiled 512

# OpenMP
g++ -O2 -fopenmp src/openmp_parallel.cpp -o openmp_parallel
./openmp_parallel 512

# MPI
mpic++ -O2 src/mpi_matrix_mul.cpp -o mpi_matrix_mul
mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_matrix_mul 512
```

## 7. Gaps and Recommendations
### 7.1 Missing or incomplete components
- A complete technical report with formal analysis, architecture, and recommendations is still needed.
- Additional robustness testing for non-square matrices and larger input sizes is recommended.

### 7.2 Recommended next steps
- Add an OpenMP target/offload or OpenACC GPU implementation.
- Extend MPI support to handle matrix sizes not divisible by process count.
- Add a cloud deployment experiment and scaling study.
- Add more extensive performance metrics and profiling output.
- Document results in a final technical report and presentation.

## 8. Conclusion
This project demonstrates a strong multi-approach implementation of matrix multiplication for parallel and distributed computing. It already includes working CPU, CUDA, OpenMP, and MPI implementations plus containerization. To complete the project fully, add GPU directive-based offload support, cloud-scale experiment evidence, and a formal technical report summarizing methodology and findings.
