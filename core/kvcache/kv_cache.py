"""
kv cache v2: one request one KVCache
"""
import torch


class KVCache:
    def __init__(self, num_layers: int):
        self.k_values = [None] * num_layers
        self.v_values = [None] * num_layers

    def update_cache(self, idx: int, k: torch.Tensor, v: torch.Tensor):
        # [batch, n_heads, seq_len, head_dim]
        if self.k_values[idx] is None:
            self.k_values[idx] = k
            self.v_values[idx] = v
        else:
            self.k_values[idx] = torch.cat((self.k_values[idx], k[:,:,-1:,:]), dim=-2)
            self.v_values[idx] = torch.cat((self.v_values[idx], v[:,:,-1:,:]), dim=-2)

    def get_cache(self, idx: int):
        return (self.k_values[idx], self.v_values[idx])
