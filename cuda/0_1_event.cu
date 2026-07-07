#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
        exit(result);                             \
    }                                             \
} while(0)

__global__ void simpleTest() {
    int threadId = threadIdx.x;
    printf("threadId is %d", threadId);
}

int main() {
    // ==========================================
    // 第一步：声明并创建事件对象
    // ==========================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==========================================
    // 第二步：在默认流（0号流）中记录起始事件(这会在 GPU 的工作队列中插入一个“标记”)
    // ==========================================
    cudaEventRecord(start, 0); // 0表示默认流

    // 启动 Kernel（异步执行）
    simpleTest<<<1,1>>>();

    // ==========================================
    // 第三步：在同一个流中记录结束事件(当 GPU 执行完上面的 Kernel 后，就会触发这个结束标记)
    // ==========================================
    cudaEventRecord(stop, 0);

    // ==========================================
    // 第四步：同步主机，等待结束事件被触发(因为 Kernel 是异步的，CPU 必须等待 GPU 真正执行到 stop 事件才能计算时间)
    // ==========================================
    cudaEventSynchronize(stop);

    // ==========================================
    // 第五步：计算两个事件之间的时间差
    // ==========================================
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("kernel duration is %f", milliseconds);

    // ==========================================
    // 第六步：销毁事件，释放资源
    // ==========================================
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}