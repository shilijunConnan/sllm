#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>
#include <stdexcept>

//=====================================================
// Tensor Check
//=====================================================

#define CHECK_CUDA(x)                                                   \
    TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x)                                             \
    TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define CHECK_FLOAT(x)                                                  \
    TORCH_CHECK((x).scalar_type() == at::kFloat,                        \
                #x " must be float32")

#define CHECK_INT(x)                                                    \
    TORCH_CHECK((x).scalar_type() == at::kInt,                          \
                #x " must be int32")

#define CHECK_INPUT(x)                                                  \
    CHECK_CUDA(x);                                                      \
    CHECK_CONTIGUOUS(x)

//=====================================================
// Forward
//=====================================================

torch::Tensor paged_attention_forward(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    int64_t seq_len,
    int64_t layer);