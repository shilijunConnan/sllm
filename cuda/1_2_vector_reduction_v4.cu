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
    extern __shared__ T sdata[];
    T sum = T(0);

    int tid = threadIdx.x;
    int gid = blockDim.x * 2 * blockIdx.x + threadIdx.x;

    // 1. 进来先做一个大reduce，把超过2 * blocksize的元素reduce到blocksize
    sum = (gid < N ? A[gid] : T(0)) + (gid + blockDim.x < N ? A[gid + blockDim.x] : T(0));

    // 2. 对每个warp进行shuffle，获取warp中所有和到第一个线程
    for (int stride = 32 >> 1; stride > 0; stride >>= 1)
    {   
        sum += __shfl_down_sync(0xffffffff, sum, stride);
    }
    // 3. 将每个warp中的总和存到shared memory中，共享内存没有同步操作，所以同步线程
    if((tid & 31) == 0) {
        sdata[tid >> 5] = sum;
    }
    __syncthreads(); 

    // 4.再对sdata进行shuffle（由于一个block最大线程数为1024，warp有32线程，所以sdata最多32个数据），直接复用前32个线程继续计算
    if(tid < (blockDim.x >> 5)) {
        sum = sdata[tid];
        for(int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (tid == 0)
    {
        A[blockIdx.x] = sum;
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
        vectorReductionGpu<numType><<<gridShape, blockShape, sizeof(numType) * blockShape.x / 32, 0>>>(dA, M);
        cudaDeviceSynchronize();
        M = gridShape.x;
        gridShape.x = (M + 2 * blockShape.x - 1) / (2 * blockShape.x);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        }
    }
    vectorReductionGpu<numType><<<gridShape, blockShape, sizeof(numType) * blockShape.x / 32, 0>>>(dA, M);
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
