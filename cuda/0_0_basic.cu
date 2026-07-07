#include <stdio.h>
#include "cuda_runtime.h"

__global__ void testIdx_1D()
{
    int globalId = blockDim.x * blockIdx.x + threadIdx.x;
    printf("blockDim.x = %d, blockIdx.x = %d, threadIdx.x = %d, globalId = %d\n", blockDim.x, blockIdx.x, threadIdx.x, globalId);
}

void test1D() {
    dim3 grid1D(4);
    dim3 block1D(4);

    testIdx_1D<<<grid1D, block1D>>>();
    cudaDeviceSynchronize();
}

__global__ void testIdx_2D()
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = bx * blockDim.x + tx;
    int gy = by * blockDim.y + ty;

    int rowFirstId = gy * (gridDim.x * blockDim.x) + gx;
    int colFirstId = gx * (gridDim.y * blockDim.x) + gy;
    printf("rowFirstId %d, colFirstId = %d\n", rowFirstId, colFirstId);
}

void test2D() {
    dim3 grid2D(2,2);
    dim3 block2D(2,2);

    testIdx_2D<<<grid2D, block2D>>>();
    cudaDeviceSynchronize();
}

__global__ void testIdx_3D()
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int gx = bx * blockDim.x + tx;
    int gy = by * blockDim.y + ty;
    int gz = bz * blockDim.z + tz;

    int width = gridDim.x * blockDim.x
    int height = gridDim.y * blockDim.y
    int depth = gridDim.z * blockDim.z

    int globalId = gz * (width * height) + gy * width + gx;
    printf("globalId %d\n", globalId);
}

void test3D() {
    dim3 grid3D(2,2,2);
    dim3 block3D(2,2,2);

    testIdx_3D<<<grid3D, block3D>>>();
    cudaDeviceSynchronize();
}

int main() {
    test1D();
    test2D();
    test3D();
    return 0;
}