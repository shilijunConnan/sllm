#include "paged_attention.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cmath>

namespace {

template <typename scalar_t>
__global__ void paged_attention_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,
    const scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    scalar_t* __restrict__ output,
    int layer,
    int batch_size,
    int max_num_blocks_per_seq,
    int total_num_blocks,
    int block_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim) {
    const int q_head = blockIdx.x;
    const int batch = blockIdx.y;
    const int tid = threadIdx.x;
    const int kv_head = q_head / (num_q_heads / num_kv_heads);
    const int seq_len = seq_lens[batch];

    extern __shared__ float shared[];
    float* reductions = shared;

    const scalar_t* q_ptr = q + ((batch * num_q_heads + q_head) * head_dim);

    float thread_max = -FLT_MAX;
    for (int token = tid; token < seq_len; token += blockDim.x) {
        const int logical_block = token / block_size;
        const int block_offset = token % block_size;
        const int physical_block = block_tables[batch * max_num_blocks_per_seq + logical_block];

        if (physical_block >= 0 && physical_block < total_num_blocks) {
            const scalar_t* k_ptr =
                k_cache + (((((layer * total_num_blocks + physical_block) * block_size + block_offset) *
                             num_kv_heads + kv_head) *
                            head_dim));
            float dot = 0.0f;
            for (int dim = 0; dim < head_dim; ++dim) {
                dot += static_cast<float>(q_ptr[dim]) * static_cast<float>(k_ptr[dim]);
            }
            thread_max = fmaxf(thread_max, dot * rsqrtf(static_cast<float>(head_dim)));
        }
    }
    reductions[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reductions[tid] = fmaxf(reductions[tid], reductions[tid + stride]);
        }
        __syncthreads();
    }
    const float max_logit = reductions[0];

    float thread_sum = 0.0f;
    for (int token = tid; token < seq_len; token += blockDim.x) {
        const int logical_block = token / block_size;
        const int block_offset = token % block_size;
        const int physical_block = block_tables[batch * max_num_blocks_per_seq + logical_block];
        if (physical_block < 0 || physical_block >= total_num_blocks) {
            continue;
        }

        const scalar_t* k_ptr =
            k_cache + (((((layer * total_num_blocks + physical_block) * block_size + block_offset) *
                         num_kv_heads + kv_head) *
                        head_dim));
        float dot = 0.0f;
        for (int q_dim = 0; q_dim < head_dim; ++q_dim) {
            dot += static_cast<float>(q_ptr[q_dim]) * static_cast<float>(k_ptr[q_dim]);
        }
        thread_sum += expf(dot * rsqrtf(static_cast<float>(head_dim)) - max_logit);
    }
    reductions[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reductions[tid] += reductions[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reductions[0];

    scalar_t* out_ptr = output + ((batch * num_q_heads + q_head) * head_dim);
    for (int dim = tid; dim < head_dim; dim += blockDim.x) {
        float acc = 0.0f;
        for (int token = 0; token < seq_len; ++token) {
            const int logical_block = token / block_size;
            const int block_offset = token % block_size;
            const int physical_block = block_tables[batch * max_num_blocks_per_seq + logical_block];
            if (physical_block < 0 || physical_block >= total_num_blocks) {
                continue;
            }

            const scalar_t* v_ptr =
                v_cache + (((((layer * total_num_blocks + physical_block) * block_size + block_offset) *
                             num_kv_heads + kv_head) *
                            head_dim));
            const scalar_t* k_ptr =
                k_cache + (((((layer * total_num_blocks + physical_block) * block_size + block_offset) *
                             num_kv_heads + kv_head) *
                            head_dim));
            float dot = 0.0f;
            for (int q_dim = 0; q_dim < head_dim; ++q_dim) {
                dot += static_cast<float>(q_ptr[q_dim]) * static_cast<float>(k_ptr[q_dim]);
            }
            const float weight = expf(dot * rsqrtf(static_cast<float>(head_dim)) - max_logit) * inv_sum;
            acc += weight * static_cast<float>(v_ptr[dim]);
        }
        out_ptr[dim] = static_cast<scalar_t>(acc);
    }
}

}  // namespace

torch::Tensor paged_attention_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t layer) {
    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(block_tables);
    CHECK_INPUT(seq_lens);

    TORCH_CHECK(q.dim() == 4, "q must have shape [batch, num_q_heads, 1, head_dim]");
    TORCH_CHECK(k_cache.dim() == 5, "k_cache must have shape [layers, blocks, block_size, num_kv_heads, head_dim]");
    TORCH_CHECK(v_cache.sizes() == k_cache.sizes(), "v_cache must have the same shape as k_cache");
    TORCH_CHECK(q.size(2) == 1, "decode paged attention expects q sequence length to be 1");
    TORCH_CHECK(q.scalar_type() == k_cache.scalar_type(), "q and k_cache must have the same dtype");
    TORCH_CHECK(q.scalar_type() == v_cache.scalar_type(), "q and v_cache must have the same dtype");
    TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
    TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");

    if (block_tables.dim() == 1) {
        block_tables = block_tables.view({1, block_tables.size(0)});
    }
    if (seq_lens.dim() == 0) {
        seq_lens = seq_lens.view({1});
    }

    const int batch_size = q.size(0);
    const int num_q_heads = q.size(1);
    const int head_dim = q.size(3);
    const int num_layers = k_cache.size(0);
    const int total_num_blocks = k_cache.size(1);
    const int block_size = k_cache.size(2);
    const int num_kv_heads = k_cache.size(3);
    const int max_num_blocks_per_seq = block_tables.size(1);

    TORCH_CHECK(layer >= 0 && layer < num_layers, "layer is out of range");
    TORCH_CHECK(k_cache.size(4) == head_dim, "q and k_cache head_dim must match");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    TORCH_CHECK(block_tables.size(0) == batch_size, "block_tables batch dimension must match q");
    TORCH_CHECK(seq_lens.numel() == batch_size, "seq_lens must contain one length per batch element");

    const auto max_seq_len = seq_lens.max().item<int32_t>();
    TORCH_CHECK(max_seq_len > 0, "seq_lens must be positive");
    TORCH_CHECK((max_seq_len + block_size - 1) / block_size <= max_num_blocks_per_seq,
                "block_tables does not contain enough blocks for seq_lens");

    auto output = torch::empty_like(q);
    constexpr int threads = 128;
    const int shared_bytes = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, q.scalar_type(), "paged_attention_forward", [&] {
        paged_attention_kernel<scalar_t><<<
            dim3(num_q_heads, batch_size),
            threads,
            shared_bytes,
            at::cuda::getCurrentCUDAStream()>>>(
            q.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            block_tables.data_ptr<int32_t>(),
            seq_lens.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(),
            static_cast<int>(layer),
            batch_size,
            max_num_blocks_per_seq,
            total_num_blocks,
            block_size,
            num_q_heads,
            num_kv_heads,
            head_dim);
    });

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
