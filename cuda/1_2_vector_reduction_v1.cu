// V2: 顺序寻址优化Sequential Addressing + 加载时首次相加 First Add During Load
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <random>
#include "cuda_runtime.h"

using numType = int;

template <typename T>
void initMatrix(T *V, int N)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    for (int i = 0; i < N; i++)
    {
        V[i] = T(1);
        // V[i] = T(dis(gen));
    }
}

template <typename T>
void vectorReductionCpu(T *V, int N)
{
    for (int i = 1; i < N; i++)
    {
        V[0] += V[i];
    }
}

template <typename T>
__global__ void vectorReductionGpu(T *A, int N)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * 2 * blockIdx.x + threadIdx.x;

    if(gid + blockDim.x < N) {
        A[gid] += A[gid + blockDim.x];
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        // 易错点： 必须加gid + stride < N，否则如果上一个kernel循环后剩下20个元素，N就为20，但是blockDim.x还是128
        // 不加这个判断就会把这20个都加上随机值了
        if (tid < stride && gid + stride < N) 
        {
            A[gid] += A[gid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        A[blockIdx.x] = A[gid];
    }
}

int main()
{
    int N = 16 * 1024 * 1024;
    numType cpuR, gpuR;

    numType *hA;
    hA = (numType *)malloc(sizeof(numType) * N);
    initMatrix<numType>(hA, N);

    // ========== gpu ==========
    numType *dA;
    cudaMalloc((void **)&dA, sizeof(numType) * N);
    cudaMemcpy(dA, hA, sizeof(numType) * N, cudaMemcpyHostToDevice);

    cudaEvent_t gpuStart, gpuStop;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);

    int M = N;
    dim3 blockShape(256, 1, 1);
    dim3 gridShape((M + 2 * blockShape.x - 1) / (2 * blockShape.x), 1, 1);

    cudaEventRecord(gpuStart, 0);
    while (gridShape.x > 1)
    {
        vectorReductionGpu<numType><<<gridShape, blockShape, 0, 0>>>(dA, M);
        cudaDeviceSynchronize();
        M = gridShape.x;
        gridShape.x = (M + 2 * blockShape.x - 1) / (2 * blockShape.x);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        }
    }
    vectorReductionGpu<numType><<<gridShape, blockShape, 0, 0>>>(dA, M);
    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaMemcpy(&gpuR, dA, sizeof(numType), cudaMemcpyDeviceToHost);

    float gpu_duration = 0.0;
    cudaEventElapsedTime(&gpu_duration, gpuStart, gpuStop);

    printf("gpu add finished, kernel duration is %f ms. \n", gpu_duration);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);
    cudaFree(dA);

    // ========== cpu ==========
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vectorReductionCpu<numType>(hA, N);
    cpuR = hA[0];
    auto cpuStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpuStop - cpuStart;
    printf("cpu add finished, kernel duration is %lf ms. \n", cpu_duration.count());

    // ========== result compare ==========
    std::cout << "cpu result: " << cpuR << ", gpu result: " << gpuR << std::endl;

    free(hA);
}
