# cuda1 paged attention

Build and smoke test:

```bash
cd cuda1
python setup.py build_ext --inplace
python test.py
```

The extension exposes:

```python
import paged_attention_cuda1

out = paged_attention_cuda1.forward(
    q,            # [batch, num_q_heads, 1, head_dim]
    k_cache,      # [layers, blocks, block_size, num_kv_heads, head_dim]
    v_cache,      # same shape as k_cache
    block_tables, # int32 [batch, max_blocks] or [max_blocks]
    seq_lens,     # int32 [batch] or scalar
    layer,        # int
)
```
