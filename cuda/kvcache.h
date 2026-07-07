#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

class PhysicalKVCache {
private:
    torch::Tensor k_cache;
    torch::Tensor v_cache;
public:
    PhysicalKVCache(int num_layers, int num_blocks, int block_size, int num_heads, int head_dim, torch::Device device = torch::kCUDA) {
        std::vector<int64_t> shape = {num_layers, num_blocks, block_size, num_heads, head_dim};
        this->k_cache = torch::zeros(shape, torch::TensorOptions().device(device));
        this->v_cache = torch::zeros(shape, torch::TensorOptions().device(device));
    }

    void write_block(int layer_id, int block_id, int offset, const torch::Tensor& k, const torch::Tensor& v) {
        namespace BI = torch::indexing;
        // 使用 Tensor 的 index_put_ 方法进行原地赋值
        // Python: self.k_cache[layer_id, block_id, offset] = k
        this->k_cache.index_put_({layer_id, block_id, offset}, k);
        this->v_cache.index_put_({layer_id, block_id, offset}, v);
    }

    std::tuple<torch::Tensor, torch::Tensor> read_block(int layer_id, int block_id, int offset) {
        namespace BI = torch::indexing;
        // Python: :offset 对应 C++ 中的 BI::Slice(BI::None, offset)
        auto k_res = this->k_cache.index({layer_id, block_id, BI::Slice(BI::None, offset)});
        auto v_res = this->v_cache.index({layer_id, block_id, BI::Slice(BI::None, offset)});

        return std::make_tuple(k_res, v_res);
    }}
