from typing import Optional, List

import torch
import torch.nn as nn


class ModelRunner:
    def __init__(self, model: nn.Module, is_prefill: bool = True) -> None:
        self.model = model
        self.model.eval()

    def execute(self,
                input_ids: torch.Tensor,
                position_ids: Optional[torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                past_key_values=None,
                kv_pos: Optional[List[int]] = None,
                is_decode: bool = False,):
        with torch.no_grad():
            outputs = self.model(input_ids,
                                 position_ids=position_ids,
                                 attention_mask=attention_mask,
                                 past_key_values=past_key_values,
                                 kv_pos=kv_pos,
                                 is_decode=is_decode
                                 )
        return outputs
