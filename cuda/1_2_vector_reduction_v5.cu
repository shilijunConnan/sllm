// V4: 顺序寻址优化Sequential Addressing + 加载时首次相加 First Add During Load + shared memory + warp shuffle
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
T vectorReductionCpu(T *V, int N)
{
    T _max = 0;
    for (int i = 0; i < N; i++)
    {
        _max += V[i];
    }
    return _max;
}

template <typename T>
__global__ void vectorReductionGpu(T *A, T* _sum, int N)
{   
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    T sum = 0;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += A[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid+stride];
        }
        __syncthreads();
    }

    if(tid == 0) {
        atomicAdd(_sum, sdata[0]);
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
    numType *dA, *sumGpu;
    cudaMalloc((void **)&dA, sizeof(numType) * N);
    cudaMalloc((void **)&sumGpu, sizeof(numType));
    cudaMemset(sumGpu, 0, sizeof(numType)); // 必须清零！
    cudaMemcpy(dA, hA, sizeof(numType) * N, cudaMemcpyHostToDevice);

    cudaEvent_t gpuStart, gpuStop;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);

    dim3 blockShape(256, 1, 1);
    dim3 gridShape(128, 1, 1);
    cudaEventRecord(gpuStart, 0);


    vectorReductionGpu<numType><<<gridShape, blockShape, sizeof(numType) * blockShape.x, 0>>>(dA, sumGpu, N);
    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaMemcpy(&gpuR, sumGpu, sizeof(numType), cudaMemcpyDeviceToHost);

    float gpu_duration = 0.0;
    cudaEventElapsedTime(&gpu_duration, gpuStart, gpuStop);

    printf("gpu add finished, kernel duration is %f ms. \n", gpu_duration);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);
    cudaFree(dA);

    // ========== cpu ==========
    auto cpuStart = std::chrono::high_resolution_clock::now();
    cpuR = vectorReductionCpu(hA, N);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpuStop - cpuStart;
    printf("cpu add finished, kernel duration is %lf ms. \n", cpu_duration.count());

    // ========== result compare ==========
    std::cout << "cpu result: " << cpuR << ", gpu result: " << gpuR << std::endl;

    free(hA);
}
