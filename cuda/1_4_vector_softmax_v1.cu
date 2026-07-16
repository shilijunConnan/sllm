// online softmax
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>
#include "cuda_runtime.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

using numType = float;

template <typename T>
void initMatrix(T *V, int N)
{
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, 100);

    for (int i = 0; i < N; i++)
    {
        V[i] = T(1);
        // V[i] = T(dis(gen));
    }
}

template <typename T>
void vectorSoftmaxCpu(T *V, int N)
{
    // 1.计算最大值
    T max = V[0];
    for (int i = 0; i < N; i++)
    {
        max = V[i] > max ? V[i] : max;
    }
    // 2.计算每个e^xi
    T sum = 0;
    for (int i = 0; i < N; i++)
    {
        V[i] = std::exp(V[i] - max);
        sum += V[i];
    }
    // 3.softmax
    for (int i = 0; i < N; i++)
    {
        V[i] /= sum;
    }
}

template <typename T>
__global__ void step1(T *A, T* out_max, T* out_sum, int N)
{
    extern __shared__ char s_mem[];
    T* smax = (T*)s_mem;
    T* ssum = (T*)(s_mem + blockDim.x * sizeof(T));
    int tid = threadIdx.x;
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    T _max = A[0];
    T _sum = 0;

    for(int i = gid; i < N; i += blockDim.x * gridDim.x) {
        if (A[i] > _max) {
            _sum = _sum * exp(A[i] - _max);
            _max = A[i];
        }
        _sum += exp(A[i] - _max);
    }

    smax[tid] = _max;
    ssum[tid] = _sum;
    __syncthreads();

    _max = smax[0];
    _sum = 0;
    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if(tid < stride) {
            _max = fmaxf(smax[tid], smax[tid+stride]);
            _sum = ssum[tid] * exp(smax[tid] - _max) + ssum[tid+stride] * exp(smax[tid + stride] - _max);
            smax[tid] = _max;
            ssum[tid] = _sum;
        }
        __syncthreads();
    }

    if(tid == 0) {
        atomicMax((int*)(out_max), __float_as_int(_max));
        atomicAdd(out_sum, _sum);
    }
}

template <typename T>
__global__ void step2(T *A, T *out, T *out_max, T* out_sum, int N)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;

    for(int i = gid; i < N; i += blockDim.x * gridDim.x) {
        out[i] = std::exp(A[i] - out_max[0]) / out_sum[0];
    }
}
int main()
{
    int N = 1024 * 1024;
    numType cpuR, gpuR;

    numType *hA;
    hA = (numType *)malloc(sizeof(numType) * N);
    initMatrix<numType>(hA, N);

    // ========== gpu ==========
    numType *dA, *out, *out_max, *out_sum; 
    cudaMalloc((void **)&dA, sizeof(numType) * N);
    cudaMalloc((void **)&out, sizeof(numType) * N);
    cudaMalloc((void **)&out_max, sizeof(numType));
    cudaMalloc((void **)&out_sum, sizeof(numType));

    cudaMemcpy(dA, hA, sizeof(numType) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(out_max, &hA[0],sizeof(numType),cudaMemcpyHostToDevice);
    cudaMemset(out_sum, 0, sizeof(numType));

    cudaEvent_t gpuStart, gpuStop;
    cudaEventCreate(&gpuStart);
    cudaEventCreate(&gpuStop);

    dim3 blockShape(256, 1, 1);
    dim3 gridShape(128, 1, 1);
    cudaEventRecord(gpuStart, 0);
    step1<<<gridShape, blockShape, 2 * sizeof(numType) * blockShape.x>>>(dA, out_max, out_sum, N);
    step2<<<gridShape, blockShape, sizeof(numType) * blockShape.x>>>(dA, out, out_max, out_sum, N);
    cudaEventRecord(gpuStop, 0);
    cudaEventSynchronize(gpuStop);
    cudaMemcpy(&gpuR, out, sizeof(numType), cudaMemcpyDeviceToHost);

    float gpu_duration = 0.0;
    cudaEventElapsedTime(&gpu_duration, gpuStart, gpuStop);

    printf("gpu add finished, kernel duration is %f ms. \n", gpu_duration);
    cudaEventDestroy(gpuStart);
    cudaEventDestroy(gpuStop);

    // ========== cpu ==========
    auto cpuStart = std::chrono::high_resolution_clock::now();
    vectorSoftmaxCpu<numType>(hA, N);
    cpuR = hA[0];
    auto cpuStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpuStop - cpuStart;
    printf("cpu add finished, kernel duration is %lf ms. \n", cpu_duration.count());

    // ========== result compare ==========
    std::cout << "cpu result: " << cpuR << ", gpu result: " << gpuR << std::endl;

    free(hA);
    cudaFree(dA);
    cudaFree(out);
    cudaFree(out_sum);
    cudaFree(out_max);

}
