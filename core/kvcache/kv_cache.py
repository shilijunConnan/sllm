"""
kv cache v3: kv cache python block + cuda paged attention
"""
from dataclasses import dataclass
from collections import deque
from typing import Tuple

import torch

class PhysicalKVCache:
    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        device="cuda",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        shape = (num_layers, num_blocks, block_size, num_heads, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)

    def write_block(self, layer_id: int, block_id: int, offset: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k_cache[layer_id, block_id, offset] = k
        self.v_cache[layer_id, block_id, offset] = v

    def read_block(self, layer_id: int, block_id: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[layer_id, block_id, :offset], self.v_cache[layer_id, block_id, :offset]


@dataclass
class BlockState:
    ref_count: int = 0


class KVBlockManager:
    def __init__(self,
                 num_layers: int,
                 num_blocks: int,
                 block_size: int,
                 num_heads: int,
                 head_dim: int,
                 device="cuda",
                 dtype: torch.dtype = torch.float32,
                 ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.kv = PhysicalKVCache(num_layers, num_blocks, block_size, num_heads, head_dim, device, dtype)
        from sllm.core.kvcache.request import requestContext

        requestContext.set_kv(self.kv)

        self.free_blocks = deque(range(num_blocks))
        self.block_states = {
            i: BlockState() for i in range(num_blocks)
        }
        # optional: prefix cache
        # self.hash_to_block = {}

    def allocate_block(self) -> int:
        if not self.free_blocks:
            raise RuntimeError(f"KVBlockManager.allocate_block called on {self} failed, no free blocks.")

        block_id = self.free_blocks.popleft()
        state = self.block_states[block_id]
        state.ref_count = 1
        return block_id

    def free_block(self, block_id: int) -> None:
        state = self.block_states[block_id]
        state.ref_count -= 1

        if state.ref_count == 0:
            state.seq_id = -1
            self.free_blocks.append(block_id)

    def increase_ref_count(self, block_id: int) -> None:
        self.block_states[block_id].ref_count += 1

# 这个例子记录一下，张量更新时，只要对应的shape一样就ok
# a = torch.zeros((1, 2, 3, 4, 4))
# cur = torch.ones((4, 2, 4))
# print(cur)
# a[0, 0, 0] = cur[:, 0, :]
# a[0, 0, 1] = cur[:, 1, :]
# # print(a[0, 0, :2])
# print(a)
