# PDC CCP Project: Distributed Matrix Multiplication Framework

## Project Overview
This project implements and compares multiple matrix multiplication approaches for Parallel and Distributed Computing.

Implemented versions:
1. Sequential CPU baseline
2. CUDA naive GPU implementation
3. CUDA tiled shared-memory GPU implementation
4. OpenMP directive-based CPU parallel implementation
5. MPI distributed-memory implementation

## Platform Used
Development and testing were performed on Google Colab using:
- Tesla T4 GPU
- CUDA / NVCC
- GCC/G++
- OpenMP
- MPI

## Source Files
- `src/cpu_baseline.cpp` - Sequential CPU version
- `src/cuda_naive.cu` - CUDA naive GPU version
- `src/cuda_tiled.cu` - CUDA tiled shared-memory GPU version
- `src/openmp_parallel.cpp` - OpenMP directive-based version
- `src/mpi_matrix_mul.cpp` - MPI distributed-memory version

## Results
Result files are stored in the `results/` folder:
- `timing_results.csv`
- `speedup_results.csv`
- `speedup_comparison.png`
- `execution_time_comparison.png`

## Correctness Check
Correctness was verified using checksum comparison. For each matrix size, all implementations produced matching checksums.

## Matrix Sizes Tested
- 250 x 250 edge case, not divisible by tile size
- 256 x 256
- 512 x 512

## Docker
Docker configuration is available in the `docker/` folder:
- `Dockerfile`
- `docker_commands.txt`

## Google Colab Setup and Run Commands
This project was developed and tested on Google Colab with Tesla T4 GPU. To run on Colab:

1. Create a new Colab notebook and set runtime to GPU (Runtime > Change runtime type > Hardware accelerator > GPU).
2. Upload the `src/` folder or clone the repo:
   ```
   !git clone <your-repo-url>  # If on GitHub
   %cd <repo-folder>
   ```
   or unzip
   !unzip "PDC_CCP_Matrix_Project_Final-1.zip"
   
3. Install MPI:
   ```
   !apt-get update && apt-get install -y mpich
   ```
   Navigate:
   %cd /content/PDC_CCP_Matrix_Project_Final-1/pdc_matrix_project
   !ls

4. Compile the programs:
   ```
   !g++ -O2 src/cpu_baseline.cpp -o cpu_baseline
   !nvcc -O2 src/cuda_naive.cu -o cuda_naive
   !nvcc -O2 src/cuda_tiled.cu -o cuda_tiled
   !g++ -O2 -fopenmp src/openmp_parallel.cpp -o openmp_parallel
   !mpic++ -O2 src/mpi_matrix_mul.cpp -o mpi_matrix_mul
   ```
If you want to try OpenMP target offload, add these examples (host/compiler dependent):
```bash
# If your compiler supports OpenMP offload:
g++ -O2 -fopenmp src/openmp_parallel.cpp -o openmp_parallel
./openmp_parallel 512 gpu

# Recommended compilers for real offload:
# NVIDIA HPC:
nvc++ -O2 -mp src/openmp_parallel.cpp -o openmp_parallel
./openmp_parallel 512 gpu
```
5. Run the implementations (replace 512 with desired matrix size):
   ```
   !./cpu_baseline 512
   !./cuda_naive 512
   !./cuda_tiled 512
   !./openmp_parallel 512
   !mpirun --allow-run-as-root --oversubscribe -np 2 ./mpi_matrix_mul 512
   ```

Notes: GPU runtime required for CUDA. Outputs print to console; modify code to save results if needed.

## Basic Run Commands

Compile CPU:
g++ -O2 src/cpu_baseline.cpp -o src/cpu_baseline

Run CPU:
./src/cpu_baseline 512

Compile CUDA naive:
nvcc -O2 src/cuda_naive.cu -o src/cuda_naive

Run CUDA naive:
./src/cuda_naive 512

Compile CUDA tiled:
nvcc -O2 src/cuda_tiled.cu -o src/cuda_tiled

Run CUDA tiled:
./src/cuda_tiled 512

Compile OpenMP:
g++ -O2 -fopenmp src/openmp_parallel.cpp -o src/openmp_parallel

Run OpenMP:
./src/openmp_parallel 512

Compile MPI:
mpic++ -O2 src/mpi_matrix_mul.cpp -o src/mpi_matrix_mul

Run MPI:
mpirun --allow-run-as-root --oversubscribe -np 2 ./src/mpi_matrix_mul 512

## Docker Run Commands
Build the Docker image from inside the `pdc_matrix_project` folder:
cd /mnt/c/Users/"Kashif irfan"/Desktop/"ML project"/PDC_CCP_Matrix_Project_Final-1/pdc_matrix_project
```bash
docker build -f docker/Dockerfile -t pdc-matrix-project .
```
Run the container with GPU support:
```bash
docker run --rm -it --gpus all pdc-matrix-project
```
Run the container without GPU support:
```bash
docker run --rm -it pdc-matrix-project
```
Inside the running container, execute:
```bash
./cpu_baseline 512
./cuda_naive 512
./cuda_tiled 512
./openmp_parallel 512
mpirun -np 2 ./mpi_matrix_mul 512
```

## Notes
The CUDA tiled implementation uses shared memory and boundary checks to support matrix sizes that are not exact multiples of tile size.
