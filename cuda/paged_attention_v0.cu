#include "paged_attention.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#define MAX_SEQ_LEN 4096

//=====================================================
// CUDA Kernel Declaration
//=====================================================

__global__ void paged_attention_kernel(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    const int* block_table,
    float* output,

    int seq_len,
    int layer,

    int num_blocks,
    int block_size,

    int num_heads,
    int head_dim);

//=====================================================
// Host Launcher
//=====================================================

torch::Tensor paged_attention_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    int64_t seq_len,
    int64_t layer)
{
    //-------------------------------------------------
    // Tensor Check
    //-------------------------------------------------

    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(block_table);

    CHECK_FLOAT(q);
    CHECK_FLOAT(k_cache);
    CHECK_FLOAT(v_cache);
    CHECK_INT(block_table);

    //-------------------------------------------------
    // Shape
    //-------------------------------------------------

    const int num_layers = k_cache.size(0);
    const int num_blocks = k_cache.size(1);
    const int block_size = k_cache.size(2);
    const int num_heads  = k_cache.size(3);
    const int head_dim   = k_cache.size(4);

    TORCH_CHECK(layer >= 0 && layer < num_layers);

    //-------------------------------------------------
    // Output
    //-------------------------------------------------

    auto output = torch::zeros_like(q);

    //-------------------------------------------------
    // Launch Config
    //-------------------------------------------------

    dim3 grid(num_heads);
    dim3 block(128);

    //-------------------------------------------------
    // Launch
    //-------------------------------------------------

    paged_attention_kernel<<<grid, block>>>(
        q.data_ptr<float>(),
        k_cache.data_ptr<float>(),
        v_cache.data_ptr<float>(),
        block_table.data_ptr<int>(),
        output.data_ptr<float>(),

        static_cast<int>(seq_len),
        static_cast<int>(layer),

        num_blocks,
        block_size,

        num_heads,
        head_dim);

    //-------------------------------------------------
    // CUDA Error
    //-------------------------------------------------

    cudaError_t err = cudaGetLastError();

    TORCH_CHECK(
        err == cudaSuccess,
        "PagedAttention Kernel Launch Failed: ",
        cudaGetErrorString(err));

    return output;
}

//=====================================================
// CUDA Kernel
//=====================================================

__global__ void paged_attention_kernel(
    const float* q,
    const float* k_cache,
    const float* v_cache,
    const int* block_table,
    float* output,

    int seq_len,
    int layer,

    int num_blocks,
    int block_size,

    int num_heads,
    int head_dim)
{
    //-------------------------------------------------
    // 当前 Attention Head
    //-------------------------------------------------


    int head = blockIdx.x;
    int tid = threadIdx.x;
    //-------------------------------------------------
    // 第一版：只有 thread0 工作
    //-------------------------------------------------

    if (threadIdx.x != 0)
        return;

    //-------------------------------------------------
    // q shape:
    // [1, num_heads, 1, head_dim]
    //-------------------------------------------------

    const float* q_ptr =
        q + head * head_dim;


    //-------------------------------------------------
    // output
    //-------------------------------------------------

    float* out_ptr =
        output + head * head_dim;

    //-------------------------------------------------
    // score buffer
    //-------------------------------------------------

    float score[MAX_SEQ_LEN];

    //-------------------------------------------------
    // QK
    //-------------------------------------------------

    for (int token = 0; token < seq_len; token++)
    {
        //---------------------------------------------
        // logical block
        //---------------------------------------------

        int logical_block =
            token / block_size;

        //---------------------------------------------
        // token offset
        //---------------------------------------------

        int offset =
            token % block_size;

        //---------------------------------------------
        // physical block
        //---------------------------------------------

        int physical_block =
            block_table[logical_block];

        //---------------------------------------------
        // k address
        //
        // shape:
        // [layer,
        //  block,
        //  token,
        //  head,
        //  dim]
        //---------------------------------------------

        size_t index =
            ((((layer * num_blocks
            + physical_block)

            * block_size
            + offset)

            * num_heads
            + head)

            * head_dim);

        const float* k_ptr =
            k_cache + index;

        //---------------------------------------------
        // dot(Q,K)
        //---------------------------------------------

        float local_dot = 0.f;

        for(int i=tid;
                i<head_dim;
                i+=blockDim.x)
        {
            local_dot +=
                q_ptr[i] * k_ptr[i];
        }

        score[token] =
            dot / sqrtf((float)head_dim);
    }

        //-------------------------------------------------
    // Softmax
    //-------------------------------------------------

    float max_score = -FLT_MAX;

    for (int i = 0; i < seq_len; i++)
    {
        if (score[i] > max_score)
            max_score = score[i];
    }

    float sum = 0.f;

    for (int i = 0; i < seq_len; i++)
    {
        score[i] = expf(score[i] - max_score);
        sum += score[i];
    }

    const float inv_sum = 1.f / sum;

    for (int i = 0; i < seq_len; i++)
    {
        score[i] *= inv_sum;
    }

    //-------------------------------------------------
    // output = score @ V
    //-------------------------------------------------

    for (int dim = 0; dim < head_dim; dim++)
    {
        out_ptr[dim] = 0.f;
    }

    for (int token = 0; token < seq_len; token++)
    {
        int logical_block = token / block_size;
        int offset = token % block_size;
        int physical_block = block_table[logical_block];

        size_t index =
            ((((layer * num_blocks
            + physical_block)

            * block_size
            + offset)

            * num_heads
            + head)

            * head_dim);

        const float* v_ptr =
            v_cache + index;

        float weight = score[token];

        for (int dim = 0; dim < head_dim; dim++)
        {
            out_ptr[dim] +=
                weight * v_ptr[dim];
        }
    }
}
