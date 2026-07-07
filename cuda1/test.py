import torch

import paged_attention_cuda1


def main():
    torch.manual_seed(0)

    num_layers = 2
    num_blocks = 8
    block_size = 4
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 16
    seq_len = 11
    layer = 1

    k_cache = torch.randn(
        num_layers,
        num_blocks,
        block_size,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    v_cache = torch.randn_like(k_cache)
    block_table = torch.tensor([[3, 1, 6]], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], device="cuda", dtype=torch.int32)
    q = torch.randn(1, num_q_heads, 1, head_dim, device="cuda", dtype=torch.float32)

    cuda_out = paged_attention_cuda1.forward(q, k_cache, v_cache, block_table, seq_lens, layer)

    ref = torch.zeros_like(cuda_out)
    repeat = num_q_heads // num_kv_heads
    for q_head in range(num_q_heads):
        kv_head = q_head // repeat
        keys = []
        values = []
        for token in range(seq_len):
            physical = block_table[0, token // block_size].item()
            offset = token % block_size
            keys.append(k_cache[layer, physical, offset, kv_head])
            values.append(v_cache[layer, physical, offset, kv_head])
        keys = torch.stack(keys)
        values = torch.stack(values)
        scores = (q[0, q_head, 0] @ keys.T) / (head_dim**0.5)
        ref[0, q_head, 0] = torch.softmax(scores, dim=-1) @ values

    max_error = (cuda_out - ref).abs().max().item()
    print("max_error:", max_error)
    assert torch.allclose(cuda_out, ref, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    main()
