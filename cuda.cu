#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono;

// ---------------------- VECTOR ADDITION --------------------------
__global__ void vectorAddCUDA(int *a, int *b, int *c, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

void vectorAddCPU(int *a, int *b, int *c, int n)
{
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
    // for (int i = 0; i < n; i++)
    //     cout << c[i] << " ";
    // cout << endl;
}

// ---------------------- MATRIX MULTIPLICATION ---------------------
__global__ void matrixMulCUDA(int *a, int *b, int *c, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
            sum += a[row * N + k] * b[k * N + col];
        c[row * N + col] = sum;
    }
}

void matrixMulCPU(int *a, int *b, int *c, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            int sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i * N + k] * b[k * N + j];
            c[i * N + j] = sum;
        }

    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < N; j++)
    //     {
    //         cout << c[i * N + j] << " ";
    //     }
    //     cout << endl;
    // }
}

// ------------------------ MAIN FUNCTION ---------------------------
int main()
{
    const int vecSize = 1 << 24; // ~16 million
    const int matrixSize = 1024; // 1024x1024 matrix

    // ------------------ Vector Addition ------------------
    int *h_a = new int[vecSize];
    int *h_b = new int[vecSize];
    int *h_c_cpu = new int[vecSize];
    int *h_c_gpu = new int[vecSize];

    for (int i = 0; i < vecSize; ++i)
    {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    auto start = high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c_cpu, vecSize);
    auto end = high_resolution_clock::now();
    cout << "[Vector Addition - CPU] Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, vecSize * sizeof(int));
    cudaMalloc(&d_b, vecSize * sizeof(int));
    cudaMalloc(&d_c, vecSize * sizeof(int));

    cudaMemcpy(d_a, h_a, vecSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, vecSize * sizeof(int), cudaMemcpyHostToDevice);

    start = high_resolution_clock::now();
    vectorAddCUDA<<<(vecSize + 255) / 256, 256>>>(d_a, d_b, d_c, vecSize);
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();

    cudaMemcpy(h_c_gpu, d_c, vecSize * sizeof(int), cudaMemcpyDeviceToHost);
    cout << "[Vector Addition - GPU] Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n\n";

    // ------------------ Matrix Multiplication ------------------
    int *matA = new int[matrixSize * matrixSize];
    int *matB = new int[matrixSize * matrixSize];
    int *matC_cpu = new int[matrixSize * matrixSize];
    int *matC_gpu = new int[matrixSize * matrixSize];

    for (int i = 0; i < matrixSize * matrixSize; ++i)
    {
        matA[i] = rand() % 100;
        matB[i] = rand() % 100;
    }

    start = high_resolution_clock::now();
    matrixMulCPU(matA, matB, matC_cpu, matrixSize);
    end = high_resolution_clock::now();
    cout << "[Matrix Multiplication - CPU] Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    int *d_matA, *d_matB, *d_matC;
    size_t bytes = matrixSize * matrixSize * sizeof(int);
    cudaMalloc(&d_matA, bytes);
    cudaMalloc(&d_matB, bytes);
    cudaMalloc(&d_matC, bytes);

    cudaMemcpy(d_matA, matA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((matrixSize + 15) / 16, (matrixSize + 15) / 16);

    start = high_resolution_clock::now();
    matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_matA, d_matB, d_matC, matrixSize);
    cudaDeviceSynchronize();
    end = high_resolution_clock::now();
    cudaMemcpy(matC_gpu, d_matC, bytes, cudaMemcpyDeviceToHost);

    cout << "[Matrix Multiplication - GPU] Time: " << duration_cast<milliseconds>(end - start).count() << " ms\n";

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    delete[] matA;
    delete[] matB;
    delete[] matC_cpu;
    delete[] matC_gpu;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    return 0;
}



// Commands to run
// nvcc Assignment4.cu -o output
// ./output



// This CUDA C++ program benchmarks parallel (GPU) vs. sequential (CPU) 
// computation for two fundamental tasks: vector addition and matrix multiplication. 
// It first allocates large arrays—~16 million elements for vectors and 1024×1024 matrices—using dynamic memory. 
// In the vector addition section, it performs element-wise addition of vectors a and b to get c, 
// first using a CPU function and then a CUDA kernel. The GPU kernel calculates the global thread 
// index and performs the addition only if the index is within bounds. CUDA memory allocation (cudaMalloc) 
// and data transfers (cudaMemcpy) are used to move data between host (CPU) and device (GPU). For matrix multiplication, 
// it computes each cell of the output matrix c as the dot product of the corresponding row and column 
// from input matrices a and b. The GPU kernel leverages 2D thread and block indexing to handle rows 
// and columns, significantly accelerating the operation by exploiting massive parallelism. 
// High-resolution timers measure execution times, highlighting how GPU computation—despite overhead 
// from memory transfers—can outperform the CPU, especially for compute-heavy tasks like matrix multiplication. 
// Finally, all dynamically allocated memory is cleaned up to prevent leaks. The code effectively demonstrates 
// how CUDA can accelerate numerical tasks by parallelizing them across hundreds or thousands of threads.