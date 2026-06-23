import torch
import torch.nn.functional as F

from core.kvcache.kv_cache import KVCache
from sllm.core.scheduler.request import RequestState


class Batch:
    def __init__(self, input_ids, attention_mask, position_ids, pask_keys_values, kv_pos=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.pask_keys_values = pask_keys_values
        self.kv_pos = kv_pos


class BatchBuilder:
    def __init__(self, pad_token_id, device):
        self.pad_token_id = pad_token_id
        self.device = device

    def prefill_build(self, requests: list[RequestState], num_layers: int) -> Batch:
        batch_size = len(requests)
        max_len = max([r.input_ids.shape[-1] for r in requests])
        input_ids = torch.empty((batch_size, max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.empty((batch_size, max_len), dtype=torch.long, device=self.device)
        position_ids = torch.empty((batch_size, max_len), dtype=torch.long, device=self.device)
        pask_keys_values = KVCache(num_layers)
        for i, request in enumerate(requests):
            req_ids = request.input_ids.to(self.device)  # [1, cur_len]
            cur_len = req_ids.shape[-1]
            pad_len = max_len - cur_len

            if pad_len > 0:
                padded_ids = F.pad(req_ids, (pad_len, 0), value=self.pad_token_id)
            else:
                padded_ids = req_ids
            input_ids[i] = padded_ids.squeeze(0)

            attention_mask[i, :pad_len] = 0
            attention_mask[i, pad_len:] = 1

            position_ids[i, :pad_len] = 0
            position_ids[i, pad_len:] = torch.arange(0, cur_len, device=self.device)

        return Batch(input_ids, attention_mask, position_ids, pask_keys_values)

    def decode_build(self, requests: list[RequestState], num_layers: int):
        batch_size = len(requests)
        sum_len = sum([r.get_cur_position() for r in requests])
        input_ids = torch.empty((batch_size,1), dtype=torch.long, device=self.device)
        attention_mask = torch.empty((batch_size, sum_len), dtype=torch.long, device=self.device)
        position_ids = torch.empty((batch_size,1), dtype=torch.long, device=self.device)
        pask_keys_values = KVCache(num_layers)
        kv_pos = []
        pos = 0
        for i, request in enumerate(requests):
            if request.generated_ids:
                token_id = torch.tensor(request.generated_ids[-1], dtype=torch.long, device=self.device)
            else:
                token_id = request.input_ids[0][-1].to(self.device)
            input_ids[i,0] = token_id

            position_ids[i,0] = request.get_cur_position() - 1

            cur_len = request.get_cur_position()

            attention_mask[i, :] = 0
            attention_mask[i, pos:pos + cur_len] = 1
            kv_pos.append(pos + cur_len - 1)
            pos = pos + cur_len
            for i in range(num_layers):
                k = request.kv_cache.k_values[i]
                v = request.kv_cache.v_values[i]

                slot = torch.zeros_like(
                    k[:, :, :1, :]
                )
                if pask_keys_values.k_values[i] is not None:
                    pask_keys_values.k_values[i] = torch.cat((pask_keys_values.k_values[i], k, slot), dim=-2).to(self.device)
                    pask_keys_values.v_values[i] = torch.cat((pask_keys_values.v_values[i], v, slot), dim=-2).to(self.device)
                else:
                    pask_keys_values.k_values[i] = torch.cat((k, slot), dim=-2).to(self.device)
                    pask_keys_values.v_values[i] = torch.cat((v, slot), dim=-2).to(self.device)

        return Batch(input_ids, attention_mask, position_ids, pask_keys_values, kv_pos)
