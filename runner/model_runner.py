from typing import Optional

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
                past_key_values=None):
        with torch.no_grad():
            outputs = self.model(input_ids,
                                 position_ids=position_ids,
                                 attention_mask=attention_mask,
                                 past_key_values=past_key_values)
        return outputs
