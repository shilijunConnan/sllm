#include <stdio.h>
#include <chrono>
#include <iostream>
#include "cuda_runtime.h"

constexpr int N = 1024 * 1024;
using numType = float;

void getDeviceInfo()
{
    int deviceId = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);

    printf("========= GPU 设备信息: %s =========\n", prop.name);
    int maxBlocksPerSM = 0;
    cudaDeviceGetAttribute(&maxBlocksPerSM, cudaDevAttrMaxBlocksPerMultiprocessor, deviceId);
    printf("[硬件硬限制] 每个 SM 最大支持的 Block 数量: %d\n", maxBlocksPerSM);
    printf("[硬件硬限制] 每个 SM 最大支持的线程数量: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("[硬件硬限制] 每个 SM 最大支持的 warp 数量: %d\n", prop.maxThreadsPerMultiProcessor / 32);

    printf("========= 影响 Block 数量的 SM 资源上限 =========\n");
    int regsPerSM = 0;
    cudaDeviceGetAttribute(&regsPerSM, cudaDevAttrMaxRegistersPerMultiprocessor, deviceId);
    printf("[-] 每个 SM 的寄存器总量 (32-bit): %d, 每个 Block 允许的最大寄存器数量: %d\n", regsPerSM, prop.regsPerBlock);

    int sharedMemPerSM = 0;
    cudaDeviceGetAttribute(&sharedMemPerSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, deviceId);
    printf("[-] 每个 SM 的共享内存总量: %.2f KB, 每个 Block 默认最大共享内存: %.2f KB\n", sharedMemPerSM / 1024.0, prop.sharedMemPerBlock / 1024.0);

    printf("========= GPU 架构信息 =========\n");
    printf("[架构信息] 计算能力 (Compute Capability): %d.%d\n", prop.major, prop.minor);
    printf("[架构信息] SM (Multiprocessors) 总数: %d\n\n", prop.multiProcessorCount);
}

template <typename T>
void compareVector(T *V1, T *V2, int N)
{
    for (int i = 0; i < N; i++)
    {
        printf("V2[i] = %d, V2[i]=%d", V1[i], V2[i]);
    }
}

template <typename T>
void initVector(T *V, int N)
{
    for (int i = 0; i < N; i++)
    {
        V[i] = T(i);
    }
}

template <typename T>
void vectorAddCpu(T *A, T *B, T *C, int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
__global__ void vectorAddGpu(T *A, T *B, T *C, int N)
{
    int threadId = blockDim.x * blockIdx.x + threadIdx.x;
    if (threadId < N)
        C[threadId] = A[threadId] + B[threadId];
}

int main()
{
    getDeviceInfo();
    size_t size = sizeof(numType) * N;

    numType *hA, *hB, *hR1, *hR2;
    hA = (numType *)malloc(size);
    hB = (numType *)malloc(size);
    hR1 = (numType *)malloc(size);
    hR2 = (numType *)malloc(size);
    initVector<numType>(hA, N);
    initVector<numType>(hB, N);

    // ========== gpu add ==========
    numType *dA, *dB, *dR;
    cudaMalloc((void **)&dA, size);
    cudaMalloc((void **)&dB, size);
    cudaMalloc((void **)&dR, size);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEvent_t gpuStart, gpuStop;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);

    int blockSize = 32;
    dim3 gridShape((N + blockSize - 1) / blockSize, 1, 1);
    dim3 blockShape(blockSize, 1, 1);

    cudaEventRecord(gpuStart, 0);
    vectorAddGpu<numType><<<gridShape, blockShape, 0, 0>>>(dA, dB, dR, N);
    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaMemcpy(hR1, dR, size, cudaMemcpyDeviceToHost);

    float gpu_duration = 0.0;
    cudaEventElapsedTime(&gpu_duration, gpuStart, gpuStop);

    printf("gpu add finished, kernel duration is %f ms. \n", gpu_duration);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dR);

    // ========== cpu add ==========
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vectorAddCpu<numType>(hA, hB, hR2, N);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpuStop - cpuStart;
    printf("cpu add finished, kernel duration is %lf ms. \n", cpu_duration.count());

    // ========== result compare ==========
    std::cout << "cpu result: " << hR1[N - 1] << ", gpu result: " << hR2[N - 1] << std::endl;

    free(hA);
    free(hB);
    free(hR1);
    free(hR2);
}