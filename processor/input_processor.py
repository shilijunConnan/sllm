from typing import List, Dict

import torch
from transformers import AutoTokenizer

from utils.config import SllmConfig


class InputProcessor:
    def __init__(self, model_path: str, sllm_config: SllmConfig) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        self.sllm_config = sllm_config

    def _apply_chat_template(self, text: List[dict]) -> str:
        text = self.tokenizer.apply_chat_template(text, add_generation_prompt=True, tokenize=False)
        return text

    def encode(self, text: List[dict]) -> Dict[str, torch.Tensor]:
        text = self._apply_chat_template(text)
        inputs = self.tokenizer(text,
                              return_tensors="pt",
                              max_length=self.sllm_config.max_input_len,
                              padding=True,
                              truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = torch.arange(0, input_ids.shape[1]).expand(input_ids.shape[0], input_ids.shape[1])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

