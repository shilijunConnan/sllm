#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// 1. 自定义pin allocator
template <typename T>
struct PinnedAllocator {
    using value_type = T;

    PinnedAllocator() noexcept {}
    template <typename U> PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

    // 关键点 1：使用 cudaHostAlloc 代替普通的 malloc/new
    T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        T* ptr = nullptr;
        cudaError_t err = cudaHostAlloc((void**)&ptr, n * sizeof(T), cudaHostAllocDefault);
        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    // 关键点 2：使用 cudaFreeHost 代替普通的 free/delete
    void deallocate(T* p, std::size_t) noexcept {
        cudaFreeHost(p);
    }

    // C++ 标准所必需的比较操作符
    template <typename U> bool operator==(const PinnedAllocator<U>&) const noexcept { return true; }
    template <typename U> bool operator!=(const PinnedAllocator<U>&) const noexcept { return false; }
};

// 2. 使用各种stl
// 起一个优雅的别名
template <typename T>
using cuda_vector = std::vector<T, PinnedAllocator<T>>;

int main() {
    size_t N = 1 << 20; // 1M 个元素

    // 1. 使用自定义分配器创建 vector
    // 此时它底层的物理内存被安全地“钉”在内存条中，绝对不会被交换到硬盘
    cuda_vector<float> h_pinned_vec(N);

    // 2. 像操作普通 vector 一样操作它（支持 STL 算法、直接按索引读写）
    std::fill(h_pinned_vec.begin(), h_pinned_vec.end(), 3.14f);
    h_pinned_vec[0] = 42.0f; 

    // 3. 在 GPU 上分配显存
    float* d_ptr;
    cudaMalloc((void**)&d_ptr, N * sizeof(float));

    // 4. 创建流进行异步传输
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 5. 安全地传入 cudaMemcpyAsync！
    // h_pinned_vec.data() 获取的直接就是锁页内存的裸指针
    cudaMemcpyAsync(d_ptr, h_pinned_vec.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream);

    // 执行 GPU 核函数...
    // MyKernel<<<grid, block, 0, stream>>>(d_ptr, ...);

    // 同步与释放
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(d_ptr);

    return 0; // h_pinned_vec 离开作用域，自动调用 deallocate，触发 cudaFreeHost，绝对没有内存泄漏！
}