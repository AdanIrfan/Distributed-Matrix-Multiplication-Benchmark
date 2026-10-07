#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <string>

using namespace std;

// Initialize matrix
void initializeMatrix(vector<float>& M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = static_cast<float>((i % 100) + 1);
    }
}

// CPU OpenMP version
void multiplyOpenMP(const vector<float>& A,
                    const vector<float>& B,
                    vector<float>& C,
                    int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// GPU OpenMP target offload version
void multiplyOpenMPTarget(const vector<float>& A,
                          const vector<float>& B,
                          vector<float>& C,
                          int N) {

    #pragma omp target data map(to: A[0:N*N], B[0:N*N]) map(from: C[0:N*N])
    {
        #pragma omp target teams distribute parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;

                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * N + j];
                }

                C[i * N + j] = sum;
            }
        }
    }
}

// Simple checksum
double checksum(const vector<float>& M) {
    double sum = 0.0;
    for (float v : M) sum += v;
    return sum;
}

int main(int argc, char* argv[]) {

    int N = 512;
    if (argc > 1) N = atoi(argv[1]);

    bool useGPU = false;
    if (argc > 2) {
        string mode = argv[2];
        if (mode == "gpu") useGPU = true;
    }

    cout << "Matrix size: " << N << " x " << N << endl;

    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N, 0.0f);

    initializeMatrix(A, N);
    initializeMatrix(B, N);

    auto start = chrono::high_resolution_clock::now();

    if (useGPU) {
        cout << "Running OpenMP GPU offload version" << endl;
        multiplyOpenMPTarget(A, B, C, N);
    } else {
        cout << "Running OpenMP CPU version" << endl;
        multiplyOpenMP(A, B, C, N);
    }

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsed = end - start;

    cout << "Execution time (ms): " << elapsed.count() << endl;
    cout << "Checksum: " << checksum(C) << endl;

    return 0;
}
