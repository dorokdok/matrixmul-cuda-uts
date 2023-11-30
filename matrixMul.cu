#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <omp.h>

#define N 8

void fillMatrix(float *matrix, int size) {
    srand(1301204001);
    for (int i = 0; i < size; ++i) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

__global__ void matrixMulGPU(float *a, float *b, float *c) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += a[row * N + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }

}

void matrixMulCPU(float *a, float *b, float *c) {
    omp_set_num_threads(6);
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

int main() {
    float *h_a, *h_b, *h_c;
    size_t size = N * N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    fillMatrix(h_a, N * N);
    fillMatrix(h_b, N * N);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadPerSumbu = 16;

    dim3 threadsPerBlock(threadPerSumbu, threadPerSumbu);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto startGPU = std::chrono::high_resolution_clock::now();
    matrixMulGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    auto stopGPU = std::chrono::high_resolution_clock::now();
    auto durationGPU = std::chrono::duration_cast<std::chrono::microseconds>(stopGPU - startGPU);

    std::cout << "Number of Blocks: " << (((N + threadsPerBlock.x - 1) / threadsPerBlock.x) * ((N + threadsPerBlock.y - 1) / threadsPerBlock.y)) << ", Number of Threads per Block: " << threadPerSumbu * threadPerSumbu << std::endl;

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    float *h_c_cpu = (float*)malloc(size);

    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixMulCPU(h_a, h_b, h_c_cpu);
    auto stopCPU = std::chrono::high_resolution_clock::now();
    auto durationCPU = std::chrono::duration_cast<std::chrono::microseconds>(stopCPU - startCPU);

    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_a, N, N);

    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_b, N, N);

    std::cout << "Result :" << std::endl;
    printMatrix(h_c, N, N);


    std::cout << "GPU Execution Time: " << durationGPU.count() << " microseconds" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);

    system("pause"); 
    return 0;
}
