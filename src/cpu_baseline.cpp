#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cmath>

using namespace std;

void initializeMatrix(vector<float>& M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = static_cast<float>((i % 100) + 1);
    }
}

void multiplyCPU(const vector<float>& A, const vector<float>& B, vector<float>& C, int N) {
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

double checksum(const vector<float>& M) {
    double sum = 0.0;
    for (float value : M) {
        sum += value;
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

    cout << "CPU Baseline Matrix Multiplication" << endl;
    cout << "Matrix size: " << N << " x " << N << endl;

    vector<float> A(N * N);
    vector<float> B(N * N);
    vector<float> C(N * N, 0.0f);

    initializeMatrix(A, N);
    initializeMatrix(B, N);

    auto start = chrono::high_resolution_clock::now();

    multiplyCPU(A, B, C, N);

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsed = end - start;

    cout << "Execution time (ms): " << elapsed.count() << endl;
    cout << "Checksum: " << checksum(C) << endl;

    return 0;
}
