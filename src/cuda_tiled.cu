#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

using namespace std;

#define TILE_SIZE 16

__global__ void matMulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tiledColA = t * TILE_SIZE + threadIdx.x;
        int tiledRowB = t * TILE_SIZE + threadIdx.y;

        if (row < N && tiledColA < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + tiledColA];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiledRowB < N && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[tiledRowB * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void initializeMatrix(float* M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = static_cast<float>((i % 100) + 1);
    }
}

double checksum(const float* M, int N) {
    double sum = 0.0;

    for (int i = 0; i < N * N; i++) {
        sum += M[i];
    }

    return sum;
}

int main(int argc, char* argv[]) {
    int N = 512;

    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N <= 0) {
        cerr << "Error: Matrix size must be positive." << endl;
        return 1;
    }

    cout << "CUDA Tiled Shared-Memory Matrix Multiplication" << endl;
    cout << "Matrix size: " << N << " x " << N << endl;

    size_t bytes = N * N * sizeof(float);

    float *A, *B, *C;

    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    initializeMatrix(A, N);
    initializeMatrix(B, N);

    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (N + TILE_SIZE - 1) / TILE_SIZE
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matMulTiled<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaDeviceSynchronize();

    cout << "Execution time (ms): " << milliseconds << endl;
    cout << "Checksum: " << checksum(C, N) << endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
