"""
v1版本：kv cache V1版本
"""
from typing import List, Optional

import torch
import torch.nn as nn

from sllm.utils.mappings import ACT2CLS
from sllm.utils.config import ModelConfig
from sllm.utils.pretrained import ModelPretrained
from sllm.core.kvcache.kv_cache import KVCache


class Embedding(nn.Embedding):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int) -> None:
        super().__init__(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids)

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states @ self.weight.t()


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, rms_norm_eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.rms_norm_eps = rms_norm_eps

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_dtype = input_tensor.dtype
        input_tensor = input_tensor.to(torch.float32)
        rms = torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)
        input_tensor = input_tensor * rms
        return self.weight * input_tensor.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.rms_norm_eps}"


class RotaryEmbedding(nn.Module):
    def __init__(self, rope_theta: int, rope_scaling: float, head_dim: int) -> None:
        super().__init__()
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim
        theta = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("theta", theta, persistent=False)

    def _rotary_half(self, input_tensor: torch.Tensor) -> torch.Tensor:
        x = input_tensor[..., :input_tensor.shape[-1] // 2]
        y = input_tensor[..., input_tensor.shape[-1] // 2:]
        return torch.cat((-y, x), dim=-1)

    def _get_emb_sin_cos(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ntheta = position_ids[:, :, None] * self.theta[None, None, :]
        emb = torch.cat((ntheta, ntheta), dim=-1)
        sin = emb.sin().unsqueeze(1)
        cos = emb.cos().unsqueeze(1)
        return sin, cos

    def apply_rotary_pos_emb(self, input_tensor: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        sin, cos = self._get_emb_sin_cos(position_ids)
        return input_tensor * cos + self._rotary_half(input_tensor) * sin


class SelfAttention(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 num_attention_heads: int,
                 num_key_value_heads: int,
                 head_dim: int,
                 attention_dropout: float,
                 rms_norm_eps: float = 1e-6) -> None:
        super().__init__()
        assert num_attention_heads % num_key_value_heads == 0

        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.repeat_times = num_attention_heads // num_key_value_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, rms_norm_eps)

    def forward(self,
                input_tensor: torch.Tensor,
                rotary_embed: RotaryEmbedding,
                position_ids: torch.Tensor,
                past_key_values: KVCache,
                idx: int,
                attention_mask: torch.Tensor = None,
                kv_pos: Optional[List[int]] = None,
                is_decode: bool = False
                ) -> torch.Tensor:
        input_shape = input_tensor.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_norm(self.q_proj(input_tensor).view(hidden_shape).transpose(1, 2))
        k = self.k_norm(self.k_proj(input_tensor).view(hidden_shape).transpose(1, 2))
        v = self.v_proj(input_tensor).view(hidden_shape).transpose(1, 2)

        q = rotary_embed.apply_rotary_pos_emb(q, position_ids)
        k = rotary_embed.apply_rotary_pos_emb(k, position_ids)

        if kv_pos is None:
            past_key_values.update_cache(idx, k, v)
        else:
            for i, pos in enumerate(kv_pos):
                past_key_values.v_values[idx][:, :, pos, :] = v[i, :, 0, :]
                past_key_values.k_values[idx][:, :, pos, :] = k[i, :, 0, :]

        k, v = past_key_values.get_cache(idx)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask_value = torch.finfo(score.dtype).min


        if is_decode:
            pass
        else:
            seq_len = score.shape[-1]
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=score.device, dtype=torch.bool),
                diagonal=1,
            )
            score = score.masked_fill(causal_mask[None, None, :, :], mask_value)

        mask = (1 - attention_mask[:, None, None, :]) * mask_value
        score += mask

        score = nn.functional.dropout(torch.softmax(score, dim=-1), p=self.attention_dropout, training=self.training)
        output = (score @ v).transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(output)

    def _repeat_kv(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.repeat_times == 1:
            return input_tensor
        batch, num_heads, seq_len, head_dim = input_tensor.shape
        input_tensor = input_tensor[:, :, None, :, :].expand(batch, num_heads, self.repeat_times, seq_len, head_dim)
        return input_tensor.reshape(batch, -1, seq_len, head_dim)


class MLP(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2CLS[hidden_act]

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(input_tensor)) * self.up_proj(input_tensor))


class DecoderLayer(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.self_attn = SelfAttention(model_config.hidden_size,
                                       model_config.num_attention_heads,
                                       model_config.num_key_value_heads,
                                       model_config.head_dim,
                                       model_config.attention_dropout,
                                       model_config.rms_norm_eps
                                       )
        self.mlp = MLP(model_config.hidden_size, model_config.intermediate_size, model_config.hidden_act)
        self.input_layernorm = RMSNorm(model_config.hidden_size, model_config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(model_config.hidden_size, model_config.rms_norm_eps)

    def forward(self,
                input_tensor: torch.Tensor,
                rotary_embed: RotaryEmbedding,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                past_key_values: KVCache,
                idx: int,
                kv_pos: Optional[List[int]] = None,
                is_decode: bool = False) -> torch.Tensor:
        x = input_tensor + self.self_attn(self.input_layernorm(input_tensor), rotary_embed, position_ids,
                                          past_key_values, idx, attention_mask, kv_pos, is_decode=is_decode)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.embed_tokens = Embedding(model_config.vocab_size, model_config.hidden_size)
        self.layers = nn.ModuleList([
            DecoderLayer(model_config)
            for _ in range(model_config.num_hidden_layers)
        ])
        self.norm = RMSNorm(model_config.hidden_size, model_config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(model_config.rope_theta, model_config.rope_scaling, model_config.head_dim)

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                past_key_values: KVCache,
                kv_pos: Optional[List[int]] = None,
                is_decode: bool = False) -> torch.Tensor:
        input_tensor = self.embed_tokens(input_ids)

        for idx, layer in enumerate(self.layers):
            input_tensor = layer(input_tensor,
                                 self.rotary_emb,
                                 position_ids,
                                 attention_mask,
                                 past_key_values,
                                 idx,
                                 kv_pos,
                                 is_decode)

        return self.norm(input_tensor)


class Qwen3ForCausalLM(nn.Module, ModelPretrained):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model = Qwen3Model(model_config)
        self.lm_head = nn.Linear(
            model_config.hidden_size,
            model_config.vocab_size,
            bias=False
        )
        if model_config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                past_key_values: KVCache,
                kv_pos: Optional[List[int]] = None,
                is_decode: bool = False) -> torch.Tensor:
        hidden_states = self.model(input_ids, position_ids, attention_mask, past_key_values, kv_pos, is_decode)
        output = self.lm_head(hidden_states)
        return output
