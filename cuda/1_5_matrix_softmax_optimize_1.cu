#include <stdio.h>
#include <chrono>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>
#include <vector>
#include <cfloat>
#include "cuda_runtime.h"

using numType = float;

template <typename T>
void initMatrix(T *V, int W, int H)
{
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_int_distribution<> dis(1, 100);

    for (int i = 0; i < W * H; i++)
    {
        // V[i] = T(1);
        V[i] = T(dis(gen));
    }
}

template <typename T>
void printMatrix(T *V, int W, int H)
{
    for (int i = 0; i < W * H; i++)
    {
        std::cout << V[i] << " ";
        if (i % W == 0) {
            std::cout << std::endl;
        }
    }
}

template <typename T>
void matrixSoftmaxCpu(T *V, T* O, int W, int H)
{
    // 1.计算最大值
    std::vector<T> row_max(H, 0);
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            int gid = i * W + j;
            row_max[i] = std::max(row_max[i], V[gid]);
        }
    }
    // 2.计算每个e^xi
    std::vector<T> row_sum(H, 0);
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            int gid = i * W + j;
            O[gid] = std::exp(V[gid] - row_max[i]);
            row_sum[i] += O[gid];
        }
    }
    // 3.softmax
    for(int i = 0; i < H; i++) {
        for(int j = 0; j < W; j++) {
            int gid = i * W + j;
            O[gid] /= row_sum[i];
        }
    }
}
void matrixSoftmaxCpuLaunch(numType *V, int W, int H, size_t size) {
    numType *out;
    out = (numType*)malloc(size);
    auto cpuStart = std::chrono::high_resolution_clock::now();
    matrixSoftmaxCpu<numType>(V, out, W, H);
    auto cpuStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = cpuStop - cpuStart;
    printf("cpu softmax finished, kernel duration is %lf ms. \n", duration.count());
    std::cout << "cpu softmax result[0]: " << out[0] << std::endl;

    free(out);
}

template <typename T>
__global__ void matrixSoftmaxGpuKernel(T *V, T* out, int W, int H) {
    extern __shared__ char s_mem[];
    T* tmax = (T*)s_mem;
    T* tsum = (T*)(s_mem + sizeof(T) * blockDim.x);

    int tid = threadIdx.x;
    int memoryStart = blockIdx.x * W + tid;
    tmax[tid] = -FLT_MAX;
    tsum[tid] = 0;
    __syncthreads();
    // 1. 计算max, 更新sum
    for(int stride = 0; stride < W; stride += blockDim.x) {
        if (tid + stride < W) {
            T old = tmax[tid];
            tmax[tid] = fmaxf(tmax[tid], V[memoryStart + stride]);
            if(tmax[tid] > old) {
                tsum[tid] = tsum[tid] *__expf(old - tmax[tid]);
            }
            tsum[tid] += __expf(V[memoryStart + stride] - tmax[tid]);
        }
        __syncthreads();
    }

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if(tid < stride) {
            T old = tmax[tid];
            tmax[tid] = fmaxf(tmax[tid], tmax[tid + stride]);
            tsum[tid] = tsum[tid] * __expf(old - tmax[tid]) + tsum[tid + stride] * __expf(tmax[tid + stride] - tmax[tid]);
        }
        __syncthreads();
    }
    // 2. 计算softmax
    for(int stride = 0; stride < W; stride += blockDim.x) {
        if (tid + stride < W) {
            out[memoryStart + stride] = __expf(V[memoryStart + stride] - tmax[0]) / tsum[0];
        }
        __syncthreads();
    }
    
}

void matrixSoftmaxGpuLaunch(numType*V, int W, int H, size_t size) {
    numType *dIn, *dOut;
    cudaMalloc((void**)&dIn, size);
    cudaMalloc((void**)&dOut, size);

    cudaMemcpy(dIn, V, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 blockShape(128, 1, 1);
    dim3 gridShape(H, 1, 1);
    size_t shared_memory = sizeof(numType) * blockShape.x * 2;

    cudaEventRecord(start);
    matrixSoftmaxGpuKernel<<<gridShape, blockShape, shared_memory>>>(dIn, dOut, W, H);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    numType *hOut;
    hOut = (numType*)malloc(size);
    cudaMemcpy(hOut, dOut, size, cudaMemcpyDeviceToHost);
    float duration = 0.0f;
    cudaEventElapsedTime(&duration, start, stop);
    printf("gpu softmax finished, kernel duration is %f ms. \n", duration);
    std::cout << "gpu softmax result[0]: " << hOut[0] << std::endl;

    free(hOut);
    cudaFree(dIn);
    cudaFree(dOut);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}

int main()
{
    int W = 1024 * 256 , H = 1024;
    size_t size = sizeof(numType) * W * H;

    numType *V;
    V = (numType *)malloc(size);
    initMatrix<numType>(V, W, H);

    std::cout << "==========cpu softmax==========\n";
    matrixSoftmaxCpuLaunch(V, W, H, size);
    std::cout << "\n==========gpu softmax==========\n";
    matrixSoftmaxGpuLaunch(V, W, H, size);
    std::cout << "\n==========all done==========\n";
    free(V);
}
