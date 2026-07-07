// tiling+shared memory
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <random>
#include "cuda_runtime.h"

#define TILE_SIZE 32
using numType = float;

template <typename T>
void compareMatrix(T *V1, T *V2, int W, int H)
{
    for (int i = 0; i < W * H; i++)
    {
        std::cout << "V2[i] = " << V1[i] << ", V2[i]=" << V2[i] << std::endl;
    }
}

template <typename T>
void initMatrix(T *V, int W, int H)
{
    // std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, 100);

    for (int i = 0; i < W * H; i++)
    {
        V[i] = T(dis(gen));
    }
}

template <typename T>
void matrixMultiplyCpu(T *A, T *B, T *C, int M, int K, int N)
{
    for (int m = 0; m < M; m++)
    {
        for(int n = 0; n < N; n++) {
            T sum = T(0);

            for(int k = 0; k < K; k++) {
                sum += A[K * m + k] * B[N * k + n];
            }

            C[m * N + n] = sum;
        }
    }
}

template <typename T>
__global__ void matrixMultiplyGpu(T *A, T *B, T *C, int M, int K, int N)
{
    __shared__ T sA[TILE_SIZE][TILE_SIZE];
    __shared__ T sB[TILE_SIZE][TILE_SIZE];

    int n = blockDim.x * blockIdx.x + threadIdx.x;
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    // int globalId = gridDim.x * blockDim.x * m + n; // N * m + n

    int numTiles = K / TILE_SIZE;
    T sum = 0;

    for(int i = 0; i < numTiles; i++) {
        sA[threadIdx.y][threadIdx.x] = A[K * m + TILE_SIZE * i + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[N * (TILE_SIZE * i + threadIdx.y) + n];
        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }


    C[N * m + n] = sum;
}

int main()
{
    int M = 1024, K = 2048, N = 512;

    numType *hA, *hB, *hR1, *hR2;
    hA = (numType *)malloc(sizeof(numType) * M * K);
    hB = (numType *)malloc(sizeof(numType) * K * N);
    hR1 = (numType *)malloc(sizeof(numType) * M * N);
    hR2 = (numType *)malloc(sizeof(numType) * M * N);
    initMatrix<numType>(hA, M, K);
    initMatrix<numType>(hB, K, N);

    // ========== gpu ==========
    numType *dA, *dB, *dR;
    cudaMalloc((void **)&dA, sizeof(numType) * M * K);
    cudaMalloc((void **)&dB, sizeof(numType) * K * N);
    cudaMalloc((void **)&dR, sizeof(numType) * M * N);
    cudaMemcpy(dA, hA, sizeof(numType) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(numType) * K * N, cudaMemcpyHostToDevice);

    cudaEvent_t gpuStart, gpuStop;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);

    dim3 blockShape(32, 32, 1);
    dim3 gridShape((N + blockShape.x - 1) / blockShape.x, (M + blockShape.y - 1) / blockShape.y, 1);

    cudaEventRecord(gpuStart, 0);
    matrixMultiplyGpu<numType><<<gridShape, blockShape, 0, 0>>>(dA, dB, dR, M, K, N);
    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaMemcpy(hR1, dR,  sizeof(numType) * M * N, cudaMemcpyDeviceToHost);

    float gpu_duration = 0.0;
    cudaEventElapsedTime(&gpu_duration, gpuStart, gpuStop);

    printf("gpu add finished, kernel duration is %f ms. \n", gpu_duration);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dR);

    // ========== cpu ==========
    auto cpuStart = std::chrono::high_resolution_clock::now();
    matrixMultiplyCpu<numType>(hA, hB, hR2, M, K, N);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpuStop - cpuStart;
    printf("cpu add finished, kernel duration is %lf ms. \n", cpu_duration.count());

    // ========== result compare ==========
    std::cout << "cpu result: " << hR1[0] << ", gpu result: " << hR2[0] << std::endl;

    free(hA);
    free(hB);
    free(hR1);
    free(hR2);
}