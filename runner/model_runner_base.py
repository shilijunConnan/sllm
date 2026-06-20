import torch
import torch.nn as nn


class ModelRunner:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def execute(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        with torch.no_grad():
            outputs = self.model(input_ids, position_ids=position_ids, attention_mask=attention_mask)

        return outputs
