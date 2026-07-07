// V3: 顺序寻址优化Sequential Addressing 
// + 加载时首次相加 First Add During Load 
// + shared memory 
// + 循环展开 loop unrolling
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
__device__ void warpReduce(volatile T* sdata, int tid)
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
} 

template <typename T>
__global__ void vectorReductionGpu(T *A, int N)
{   
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int gid = blockDim.x * 2 * blockIdx.x + threadIdx.x;

    sdata[tid] = (gid < N ? A[gid] : 0) + (gid + blockDim.x < N ? A[gid + blockDim.x] : T(0));
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 32; stride >>= 1)
    {
        if (tid < stride) 
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if(tid < 32) warpReduce(sdata, tid); // 线程数==32时，一个warp就能执行完成

    if (tid == 0)
    {
        A[blockIdx.x] = sdata[tid];
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
        vectorReductionGpu<numType><<<gridShape, blockShape, sizeof(numType) * blockShape.x, 0>>>(dA, M);
        cudaDeviceSynchronize();
        M = gridShape.x;
        gridShape.x = (M + 2 * blockShape.x - 1) / (2 * blockShape.x);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        }
    }
    vectorReductionGpu<numType><<<gridShape, blockShape, sizeof(numType) * blockShape.x, 0>>>(dA, M);
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
