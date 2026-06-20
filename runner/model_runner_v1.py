import torch
import torch.nn as nn


class ModelRunner:
    def __init__(self, model: nn.Module, is_prefill: bool = True) -> None:
        self.model = model
        self.model.eval()
        self.is_prefill = is_prefill

    def execute(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                past_key_values=None):
        if not self.is_prefill:
            assert input_ids.shape[1] == 1
        with torch.no_grad():
            outputs = self.model(input_ids,
                                 position_ids=position_ids,
                                 attention_mask=attention_mask,
                                 past_key_values=past_key_values)
        return outputs
