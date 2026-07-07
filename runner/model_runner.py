from typing import List

import torch
import torch.nn as nn

from sllm.core.kvcache.kv_cache import KVBlockManager
from sllm.core.kvcache.request import RequestState, requestContext


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
                        requestContext.set(req, "prefill")
                        input_ids = req.input_ids
                        attention_mask = req.attention_mask
                        position_ids = req.position_ids
                        outputs = self.model(input_ids=input_ids,
                                                             position_ids=position_ids,
                                                             attention_mask=attention_mask)
                        results.append(outputs[:, -1, :])
                else:
                    for req in requests:
                        requestContext.set(req, "decode")
                        device = req.input_ids.device
                        input_ids = torch.LongTensor([[req.generated_ids[-1]]]).to(device)
                        attention_mask = torch.ones((1, req.get_sum_len())).to(device)
                        position_ids = torch.LongTensor([[req.get_sum_len() - 1]]).to(device)

                        outputs = self.model(input_ids=input_ids,
                                                             position_ids=position_ids,
                                                             attention_mask=attention_mask,
                                                            is_prefill=False)
                        results.append(outputs[:, -1, :])
        except Exception as e:
            import traceback
            traceback.print_exc()
        return results
