#include "paged_attention.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <iostream>

#define MAX_SEQ_LEN 4096

__global__ void score_softmax_kernel(
    float *score,
    const int batch,
    const int num_heads,
    const int seq_cur_len,
    const int seq_pre_len
) 
{
    extern __shared__ char s_mem[]; 
    float *tmax = (float*)s_mem;
    float *tsum = (float*)(s_mem + sizeof(float) * blockDim.x);

    int tid = threadIdx.x;

    int batch_id = blockIdx.x / (num_heads * seq_cur_len);
    int remain = blockIdx.x % (num_heads * seq_cur_len);
    int num_heads_id = remain / seq_cur_len;
    int seq_cur_len_id = remain % seq_cur_len;

    int memory_start_id = batch_id * (num_heads * seq_cur_len * seq_pre_len) 
                        + num_heads_id * (seq_cur_len * seq_pre_len)     
                        + seq_cur_len_id * (seq_pre_len);

    tmax[tid] = -FLT_MAX;
    tsum[tid] = 0;

    // 计算max
    for(int stride = tid; stride < seq_pre_len; stride += blockDim.x) {
        tmax[tid] = fmaxf(tmax[tid], score[memory_start_id + stride]);
    }
    __syncthreads();


    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tmax[tid] = fmaxf(tmax[tid], tmax[tid + stride]);
        }
        __syncthreads();
    }

    // 计算sum
    for(int stride = tid; stride < seq_pre_len; stride += blockDim.x) {
        tsum[tid] += __expf(score[memory_start_id + stride] - tmax[0]);
    }
    __syncthreads();

    for(int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            tsum[tid] += tsum[tid + stride];
        }
        __syncthreads();
    }

    // 计算概率
    for(int stride = tid; stride < seq_pre_len; stride += blockDim.x) {
        score[memory_start_id + stride] = __expf(score[memory_start_id + stride] - tmax[0]) / tsum[0];
    }
}

__global__ void q_matmul_k_kernel(
    const float* q,
    const float* k_cache,
    const int* block_table,
    float* score,

    float scale,
    int seq_len,
    int q_heads,
    int group_size,
    int max_num_blocks,

    int layer,
    int num_blocks,
    int k_heads,
    int block_size,
    int head_dim)
{
    int tid = threadIdx.x;
    // q [batch,q_heads, 1, head_dim]
    // k [num_layers, num_blocks, k_heads, block_size, head_dim]
    // score [batch, 1_heads, 1, seq_len]
    int batch_id = blockIdx.x / q_heads;
    int q_head_id = blockIdx.x % q_heads;

    int q_memory_start = (batch_id * q_heads + q_head_id)* head_dim;
    int score_memory_start = (batch_id * q_heads + q_head_id)* seq_len;

    // 第一版一个线程计算一个score
    for (int offset = tid; offset < seq_len; offset += blockDim.x) {
        int block_id = block_table[batch_id * max_num_blocks + seq_offset / block_size];
        int seq_id = offset % block_size;
        int k_head_id = q_head_id / group_size;

        int k_memory_start = (((layer * num_blocks + block_id) * k_heads + k_head_id) * block_size + seq_id) * head_dim;

        float tsum = 0;
        for(int i = 0; i < head_dim; i++) {
            tsum += (q[q_memory_start + i] * k_cache[k_memory_start + i]);
        }
        score[score_memory_start + offset] = tsum * scale;
    }
}

__global__ void score_matmul_v_kernel(
    const float* score,
    const float* v_cache,
    const int* block_table,
    float* output,

    int seq_len,
    int q_heads,
    int group_size,
    int max_num_blocks,

    int layer,
    int num_blocks,
    int k_heads,
    int block_size,
    int head_dim)
{
    int tid = threadIdx.x;
    // score [batch, 1_heads, 1, seq_len]
    // v [num_layers, num_blocks, k_heads, block_size, head_dim]
    // out [batch,q_heads, 1, head_dim]
    int batch_id = blockIdx.x / q_heads;
    int out_head_id = blockIdx.x % q_heads;

    int out_memory_start = (batch_id * q_heads + out_head_id)* head_dim;
    int score_memory_start = (batch_id * q_heads + out_head_id)* seq_len;

    // 第一版一个线程计算一个head_dim
    for (int dim_offset = tid; dim_offset < head_dim; dim_offset += blockDim.x) {
        float tsum = 0;
        for(int seq_offset = 0; seq_offset < seq_len; seq_offset++) {
            int block_id = block_table[batch_id * max_num_blocks + seq_offset / block_size];
            int seq_id = seq_offset % block_size;
            int v_head_id = out_head_id / group_size;
            int v_memory_id = (((layer * num_blocks + block_id) * k_heads + v_head_id) * block_size + seq_id) * head_dim + dim_offset;

            tsum += (score[score_memory_start + seq_offset] * v_cache[v_memory_id]);
        }
        output[out_memory_start + dim_offset] = tsum;
    }
}

torch::Tensor paged_attention_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    int64_t seq_len,
    int64_t layer)
{

    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(block_table);

    CHECK_FLOAT(q);
    CHECK_FLOAT(k_cache);
    CHECK_FLOAT(v_cache);
    CHECK_INT(block_table);

    const int num_layers = k_cache.size(0);
    const int num_blocks = k_cache.size(1);
    const int block_size = k_cache.size(2);
    const int k_heads  = k_cache.size(3);
    const int head_dim   = k_cache.size(4);

    const int batch = q.size(0);
    const int q_heads = q.size(1);
    const int seq_cur_len = q.size(2);
    const int group_size = q_heads / k_heads;
    const int max_num_blocks = block_table.size(1);

    TORCH_CHECK(layer >= 0 && layer < num_layers);

    auto output = torch::zeros_like(q);


    // 1. Q@K.T 
    const float scale = 1 / std::sqrt(head_dim);
    auto score = torch::zeros({batch, q_heads, seq_cur_len, seq_len}, q.options());

    dim3 gridShape1(batch * q_heads, 1, 1);
    dim3 blockShape1(128);
    q_matmul_k_kernel<<<gridShape1, blockShape1>>>(
        q.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        block_table.data_ptr<int>(),
        score.data_ptr<float>(),

        static_cast<float>(scale),
        static_cast<int>(seq_len),
        q_heads,
        group_size,
        max_num_blocks,

        static_cast<int>(layer),
        num_blocks,
        k_heads,
        block_size,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "Softmax Kernel Launch Failed: ",
        cudaGetErrorString(err));

    // 2. softmax
    dim3 gridShape2(batch * q_heads * seq_cur_len, 1, 1);
    dim3 blockShape2(128, 1, 1);
    size_t smem = sizeof(float) * blockShape2.x * 2;
    score_softmax_kernel<<<gridShape2, blockShape2, smem>>>(score, batch, q_heads, seq_cur_len, seq_len);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "Softmax Kernel Launch Failed: ",
        cudaGetErrorString(err));

    // 3. output 
    dim3 gridShape3(batch * q_heads, 1, 1);
    dim3 blockShape3(128);
    score_matmul_v_kernel<<<gridShape3, blockShape3>>>(
        score.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        block_table.data_ptr<int>(),
        output.data_ptr<float>(),

        static_cast<int>(seq_len),
        q_heads,
        group_size,
        max_num_blocks,

        static_cast<int>(layer),
        num_blocks,
        k_heads,
        block_size,
        head_dim
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(
        err == cudaSuccess,
        "Softmax Kernel Launch Failed: ",
        cudaGetErrorString(err));
    cudaDeviceSynchronize();
    return output;
}
