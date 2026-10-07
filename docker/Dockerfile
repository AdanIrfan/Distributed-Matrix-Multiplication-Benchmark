FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    mpich \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY src/ /app/src/

RUN g++ -O2 /app/src/cpu_baseline.cpp -o /app/cpu_baseline
RUN nvcc -O2 /app/src/cuda_naive.cu -o /app/cuda_naive
RUN nvcc -O2 /app/src/cuda_tiled.cu -o /app/cuda_tiled
RUN g++ -O2 -fopenmp /app/src/openmp_parallel.cpp -o /app/openmp_parallel
RUN mpic++ -O2 /app/src/mpi_matrix_mul.cpp -o /app/mpi_matrix_mul

CMD ["/bin/bash"]
