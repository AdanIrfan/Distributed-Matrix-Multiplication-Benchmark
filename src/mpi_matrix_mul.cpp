#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

void initializeMatrix(vector<float>& M, int N) {
    for (int i = 0; i < N * N; i++) {
        M[i] = static_cast<float>((i % 100) + 1);
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 256;
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    if (N <= 0) {
        if (rank == 0) {
            cerr << "Error: Matrix size must be positive." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (N % size != 0) {
        if (rank == 0) {
            cerr << "Error: For this first MPI version, N must be divisible by number of MPI processes." << endl;
            cerr << "N = " << N << ", processes = " << size << endl;
        }
        MPI_Finalize();
        return 1;
    }

    int rowsPerProcess = N / size;

    vector<float> A;
    vector<float> B(N * N);
    vector<float> C;

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N, 0.0f);

        initializeMatrix(A, N);
        initializeMatrix(B, N);

        cout << "MPI Distributed Matrix Multiplication" << endl;
        cout << "Matrix size: " << N << " x " << N << endl;
        cout << "MPI processes: " << size << endl;
    }

    vector<float> localA(rowsPerProcess * N);
    vector<float> localC(rowsPerProcess * N, 0.0f);

    MPI_Bcast(B.data(), N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Scatter(
        A.data(),
        rowsPerProcess * N,
        MPI_FLOAT,
        localA.data(),
        rowsPerProcess * N,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int i = 0; i < rowsPerProcess; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;

            for (int k = 0; k < N; k++) {
                sum += localA[i * N + k] * B[k * N + j];
            }

            localC[i * N + j] = sum;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    MPI_Gather(
        localC.data(),
        rowsPerProcess * N,
        MPI_FLOAT,
        C.data(),
        rowsPerProcess * N,
        MPI_FLOAT,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        double elapsed_ms = (end - start) * 1000.0;

        cout << "Execution time (ms): " << elapsed_ms << endl;
        cout << "Checksum: " << checksum(C) << endl;
    }

    MPI_Finalize();
    return 0;
}
