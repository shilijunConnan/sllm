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
        V[0] += V[i]; // 易错点1: 原地修改v[0]，循环从i=1开始
    }
}

template <typename T>
__global__ void vectorReductionGpu(T *A, int N)
{
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    for (int stride = 1; stride < blockDim.x; stride *= 2) // 易错点2: stride需要小于当前block的线程数
    {
        __syncthreads();
        if (tid % (stride * 2) == 0 && (gid + stride) < N) // 易错点3: 需要判断 gid + stride 是否越界
        {
            A[gid] += A[gid + stride];
        }
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
    dim3 gridShape((M + blockShape.x - 1) / blockShape.x, 1, 1);

    cudaEventRecord(gpuStart, 0);
    while (gridShape.x > 1)
    {
        vectorReductionGpu<numType><<<gridShape, blockShape, 0, 0>>>(dA, M); // 易错点4: kernel中N要及时更新为M
        M = gridShape.x;
        gridShape.x = (M + blockShape.x - 1) / blockShape.x;
    }
    vectorReductionGpu<numType><<<gridShape, blockShape, 0, 0>>>(dA, M); // 易错点5: 当gridShape.x == 1时退出了循环，但并不代表此时元素都计算完了，可能还有1-128个元素的情况，所以最后还要计算一次
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
