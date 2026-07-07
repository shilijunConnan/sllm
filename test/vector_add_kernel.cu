#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add_cuda(
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor c)
{
    const int n = a.numel();

    constexpr int THREADS = 256;
    const int BLOCKS = (n + THREADS - 1) / THREADS;

    vector_add_kernel<<<BLOCKS, THREADS>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
}