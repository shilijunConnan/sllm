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
    std::vector<T> row_max(H, -FLT_MAX);
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
    T* warpMax = (T*)s_mem;
    T* warpSum = (T*)(s_mem + sizeof(T) * (blockDim.x >> 5));

    int tid = threadIdx.x;
    int memoryStart = blockIdx.x * W + tid;
    int laneId = tid & 31;
    
    // 1. 计算max, 更新sum
    T tmax = -FLT_MAX;
    T tsum = 0;
    
    for(int stride = 0; stride < W; stride += blockDim.x) {
        if (tid + stride < W) {
            T old = tmax;
            tmax = fmaxf(tmax, V[memoryStart + stride]);
            if(tmax > old) {
                tsum = tsum *__expf(old - tmax);
            }
            tsum += __expf(V[memoryStart + stride] - tmax);
        }
    }

    
    for(int stride = 16; stride > 0; stride >>= 1) {
        T old1 = tmax;
        T old2 = __shfl_down_sync(0xffffffff, tmax, stride);
        T tsum2 = __shfl_down_sync(0xffffffff, tsum, stride);
        if (laneId < stride) {
            tmax = fmaxf(old1, old2);
            tsum = tsum * __expf(old1 - tmax) +  tsum2 * __expf(old2 - tmax);
        }
    }

    if (laneId == 0) {
        warpMax[tid >> 5] = tmax;
        warpSum[tid >> 5] = tsum;
    }
    __syncthreads();

    int warp_num = (blockDim.x + 31) / 32;
    if(tid < warp_num){
        tmax = warpMax[tid];
        tsum = warpSum[tid];
    }else{
        tmax = -FLT_MAX;
        tsum = 0;
    }
    for(int stride = 16; stride > 0; stride >>= 1) {
        T old1 = tmax;
        T old2 = __shfl_down_sync(0xffffffff, tmax, stride);
        T tsum2 = __shfl_down_sync(0xffffffff, tsum, stride);
        tmax = fmaxf(old1, old2);
        tsum = tsum * __expf(old1 - tmax) +  tsum2 * __expf(old2 - tmax);
    }
    if(tid == 0) {
        warpMax[0] = tmax;
        warpSum[0] = tsum;
    }
    __syncthreads();
    // 2. 计算softmax
    tmax = warpMax[0];
    tsum = warpSum[0];
    for(int stride = 0; stride < W; stride += blockDim.x) {
        if (tid + stride < W) {
            out[memoryStart + stride] = __expf(V[memoryStart + stride] - tmax) / tsum;
        }
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
    size_t shared_memory = sizeof(numType) * blockShape.x * 2 >> 5;

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
    int W = 1024, H = 1024;
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
