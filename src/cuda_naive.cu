#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>

using namespace std;

__global__ void matMulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }

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

    cout << "CUDA Naive Matrix Multiplication" << endl;
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

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matMulNaive<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

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
