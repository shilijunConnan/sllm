from typing import List

import torch
import torch.nn as nn

from sllm.core.kvcache.kv_cache import KVBlockManager
from sllm.core.kvcache.request import RequestState


class ModelRunner:
    def __init__(self, model: nn.Module, kv_manager: KVBlockManager) -> None:
        self.model = model
        self.kv_manager = kv_manager
        self.model.eval()

    def execute(self, requests: List[RequestState], is_prefill: bool = False):
        results = []
        try:
            with torch.no_grad():
                if is_prefill:
                    for req in requests:
                        input_ids = req.input_ids
                        attention_mask = req.attention_mask
                        position_ids = req.position_ids
                        outputs, k_list, v_list = self.model(input_ids=input_ids,
                                                             position_ids=position_ids,
                                                             attention_mask=attention_mask)
                        results.append(outputs[:, -1, :])
                        req.write_block(k_list, v_list)
                else:
                    for req in requests:
                        device = req.input_ids.device
                        input_ids = torch.LongTensor([[req.generated_ids[-1]]]).to(device)
                        attention_mask = torch.ones((1, req.get_sum_len())).to(device)
                        position_ids = torch.LongTensor([[req.get_sum_len() - 1]]).to(device)

                        past_keys = [None] * self.kv_manager.num_layers
                        past_values = [None] * self.kv_manager.num_layers
                        last_block_id = req.get_last_block_id()
                        last_block_offset = req.last_block_offset

                        layer_k_blocks = [[] for _ in range(self.kv_manager.num_layers)]
                        layer_v_blocks = [[] for _ in range(self.kv_manager.num_layers)]
                        for table_id in req.kv_block_table:
                            if table_id == last_block_id:
                                offset = last_block_offset
                            else:
                                offset = self.kv_manager.block_size
                            for layer_id in range(self.kv_manager.num_layers):
                                block_keys, block_values = self.kv_manager.kv.read_block(layer_id, table_id, offset)
                                layer_k_blocks[layer_id].append(block_keys)
                                layer_v_blocks[layer_id].append(block_values)
                        for layer_id in range(self.kv_manager.num_layers):
                            past_keys[layer_id] = torch.cat(layer_k_blocks[layer_id], dim=0).transpose(0, 1).unsqueeze(0).contiguous().to(device)
                            past_values[layer_id] = torch.cat(layer_v_blocks[layer_id], dim=0).transpose(0, 1).unsqueeze(0).contiguous().to(device)
                            past_keys[layer_id] = past_keys[layer_id]

                        outputs, k_list, v_list = self.model(input_ids=input_ids,
                                                             position_ids=position_ids,
                                                             attention_mask=attention_mask,
                                                             past_keys=past_keys,
                                                             past_values=past_values, )
                        results.append(outputs[:, -1, :])
                        req.write_block(k_list, v_list)
        except Exception as e:
            import traceback
            traceback.print_exc()
        return results
