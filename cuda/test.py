import torch
import paged_attention

torch.manual_seed(0)

###############################################
# Config
###############################################

num_layers = 1
num_blocks = 4
block_size = 4

num_heads = 2
head_dim = 8

seq_len = 10
layer = 0

###############################################
# Build KV Cache
###############################################

k_cache = torch.randn(
    num_layers,
    num_blocks,
    block_size,
    num_heads,
    head_dim,
    device="cuda",
    dtype=torch.float32,
)

v_cache = torch.randn_like(k_cache)

###############################################
# block table
#
# logical:
# block0 -> physical1
# block1 -> physical3
# block2 -> physical0
###############################################

block_table = torch.tensor(
    [1, 3, 0],
    device="cuda",
    dtype=torch.int32,
)

###############################################
# Query
#
# shape:
# [1, H, 1, D]
###############################################

q = torch.randn(
    1,
    num_heads,
    1,
    head_dim,
    device="cuda",
)

###############################################
# CUDA
###############################################

cuda_out = paged_attention.forward(
    q,
    k_cache,
    v_cache,
    block_table,
    seq_len,
    layer,
)

###############################################
# Python Reference
###############################################

python_out = torch.zeros_like(cuda_out)

for head in range(num_heads):

    q_vec = q[0, head, 0]

    ks = []
    vs = []

    for token in range(seq_len):

        logical = token // block_size
        offset = token % block_size

        physical = block_table[logical].item()

        ks.append(
            k_cache[layer, physical, offset, head]
        )

        vs.append(
            v_cache[layer, physical, offset, head]
        )

    K = torch.stack(ks, dim=0)
    V = torch.stack(vs, dim=0)

    score = torch.matmul(
        q_vec,
        K.T,
    )

    score /= head_dim ** 0.5

    score = torch.softmax(score, dim=-1)

    out = torch.matmul(score, V)

    python_out[0, head, 0] = out

###############################################
# Compare
###############################################

max_error = (cuda_out - python_out).abs().max()

print("===================================")
print("CUDA Output")
print(cuda_out)

print("===================================")
print("Python Output")
print(python_out)

print("===================================")
print("Max Error:", max_error.item())

assert torch.allclose(
    cuda_out,
    python_out,
    atol=1e-4,
    rtol=1e-4,
)

print("Paged Attention Test Passed!")