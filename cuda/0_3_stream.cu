#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 128;
    size_t size = N * sizeof(float);

    // 1. 分配 Host 锁页内存（多流异步必备大前提！）
    float *h_A, *h_B, *h_C1, *h_C2;
    cudaMallocHost(&h_A, size);
    cudaMallocHost(&h_B, size);
    cudaMallocHost(&h_C1, size);
    cudaMallocHost(&h_C2, size);

    // 初始化数据
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 分配 Device（显存）内存
    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;
    cudaMalloc(&d_A1, size); cudaMalloc(&d_B1, size); cudaMalloc(&d_C1, size);
    cudaMalloc(&d_A2, size); cudaMalloc(&d_B2, size); cudaMalloc(&d_C2, size);

    // ==========================================
    // 核心第一步：创建两个独立的流
    // ==========================================
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // ==========================================
    // 核心第二步：向两个流中异步派发任务
    // ==========================================
    
    // ---- 编排 流 1 的流水线 ----
    cudaMemcpyAsync(d_A1, h_A, size, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(d_B1, h_B, size, cudaMemcpyHostToDevice, stream1);
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A1, d_B1, d_C1, N);
    cudaMemcpyAsync(h_C1, d_C1, size, cudaMemcpyDeviceToHost, stream1);

    // ---- 编排 流 2 的流水线 ----
    // 当 CPU 在发布这些命令时，流 1 可能已经在 GPU 里开跑了，两条线互不干扰
    cudaMemcpyAsync(d_A2, h_A, size, cudaMemcpyHostToDevice, stream2);
    cudaMemcpyAsync(d_B2, h_B, size, cudaMemcpyHostToDevice, stream2);
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A2, d_B2, d_C2, N);
    cudaMemcpyAsync(h_C2, d_C2, size, cudaMemcpyDeviceToHost, stream2);

    // 此时，CPU 已经瞬间执行到了这里（因为上面的 Async 函数和 Kernel 启动全都不阻塞 CPU）
    std::cout << "CPU 任务派发完毕，GPU 正在双轨并行计算中..." << std::endl;

    // ==========================================
    // 核心第三步：精准同步各个流（收网取结果）
    // ==========================================
    cudaStreamSynchronize(stream1); // 强行等待流 1 彻底做完
    cudaStreamSynchronize(stream2); // 强行等待流 2 彻底做完

    std::cout << "所有流计算结束！流 1 结果示例: " << h_C1[0] << ", 流 2 结果示例: " << h_C2[0] << std::endl;

    // ==========================================
    // 核心第四步：销毁流并释放资源
    // ==========================================
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaFree(d_A1); cudaFree(d_B1); cudaFree(d_C1);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C1); cudaFreeHost(h_C2);

    return 0;
}