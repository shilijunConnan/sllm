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

// 一个简单的 CUDA 核函数：每个线程打印自己的 ID
__global__ void helloFromGPU() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId < 5) { // 只让前5个线程打印，避免刷屏
        printf("Hello from GPU thread %d!\n", threadId);
    }
}

int main() {
    // ==========================================
    // 1. cudaGetDeviceCount 示例
    // ==========================================
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "系统中检测到的 GPU 数量: " << deviceCount << std::endl;

    if (deviceCount == 0) {
        std::cout << "未检测到支持 CUDA 的 GPU，程序退出。" << std::endl;
        return 0;
    }

    // ==========================================
    // 2. cudaSetDevice 示例
    // ==========================================
    // 假设我们选择使用编号为 0 的第一块 GPU
    int targetDevice = 0; 
    CUDA_CHECK(cudaSetDevice(targetDevice));
    std::cout << "成功切换并绑定到 GPU 设备 " << targetDevice << std::endl;

    // ==========================================
    // 3. cudaGetDeviceProperties 示例
    // ==========================================
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, targetDevice));
    
    std::cout << "\n------ 设备 " << targetDevice << " 的硬件属性 ------" << std::endl;
    std::cout << "设备名称:         " << prop.name << std::endl;
    std::cout << "计算能力 (Compute):" << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM 数量 (SMs):     " << prop.multiProcessorCount << std::endl;
    std::cout << "每个常规块的最大共享内存: " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "--------------------------------\n" << std::endl;

    // ==========================================
    // 4. cudaDeviceSynchronize 示例
    // ==========================================
    std::cout << "【CPU】准备启动 GPU 核函数..." << std::endl;
    
    // 启动核函数（这步操作对 CPU 来说是异步的，CPU 发出指令后会立刻往下走）
    helloFromGPU<<<1, 128>>>(); 
    
    std::cout << "【CPU】核函数已启动，正在调用 cudaDeviceSynchronize() 等待 GPU 完成..." << std::endl;
    
    // 阻塞 CPU，直到 GPU 的 helloFromGPU 彻底执行完
    CUDA_CHECK(cudaDeviceSynchronize()); 
    
    std::cout << "【CPU】GPU 任务已全部结束，CPU 继续向下执行并退出程序。" << std::endl;

    return 0;
}