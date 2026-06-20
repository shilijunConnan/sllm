from typing import List, Tuple, Union

import torch

from processor.output_processor import OutputProcessor
from sllm.utils.config import ModelConfig, GenerationConfig, SllmConfig
from sllm.processor.input_processor import InputProcessor
from sllm.runner.model_runner_base import ModelRunner
from sllm.models import get_model_class


class LlmEngineBase:
    def __init__(self, model_path: str) -> None:
        # 0.1 init configs
        self.model_config = ModelConfig(model_path)
        self.generation_config = GenerationConfig(model_path)
        self.sllm_config = SllmConfig()
        self.device = self._select_device()
        print(f"[INFO] Using device: {self.device}")

        # 0.2 init input_processor
        self.input_processor = InputProcessor(model_path=model_path, sllm_config=self.sllm_config)

        # 0.3 init model & model_runner
        module = get_model_class(self.model_config.architectures[0])
        model = module.from_pretrained(model_path=model_path,
                                       model_config=self.model_config,
                                       tie_word_embeddings=self.model_config.tie_word_embeddings,
                                       device=self.device)
        self.model_runner = ModelRunner(model)

        # 0.3 init output_processor
        self.output_processor = OutputProcessor(model_path=model_path, sllm_config=self.sllm_config)

    def _select_device(self) -> torch.device:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _prepare_inputs(self, messages: List[any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.input_processor.encode(messages)
        input_ids = inputs["input_ids"].to(self.device)
        position_ids = inputs["position_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        return input_ids, position_ids, attention_mask

    def _build_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len, device=input_ids.device).expand(batch_size, seq_len)

    def _is_eos(self, token_id: int) -> bool:
        eos_token_id = self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            return token_id == eos_token_id
        return token_id in eos_token_id

    def step(self, messages: List[dict]) -> Union[str, List[str]]:
        # 1.1 transform messgaes to input data
        input_ids, position_ids, attention_mask = self._prepare_inputs(messages)
        batch_size = input_ids.shape[0]
        # 1.2 array to keep output tokenid
        generated_ids = [[] for _ in range(batch_size)]
        # 1.3 tensor to keep status
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
        pad_token_id = self.generation_config.pad_token_id
        # 1.4 loop until max_output_len
        for i in range(self.sllm_config.max_output_len):
            logits = self.model_runner.execute(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
            # take the last one
            next_token_logits = logits[:, -1, :] # here size from [batch, seq, vocab] to [batch, vocab]
            next_token = self.output_processor.sample(next_token_logits, input_ids)
            next_attention_mask = torch.ones_like(next_token)
            print(i)
            # loop each batch_idx
            for batch_idx in range(batch_size):
                # already finished request, set next token as pad, and mask as 0
                if finished[batch_idx]:
                    next_token[batch_idx, 0] = pad_token_id
                    next_attention_mask[batch_idx, 0] = 0
                    continue

                next_token_id = next_token[batch_idx, 0].item()
                # update finish status
                if self._is_eos(next_token_id):
                    finished[batch_idx] = True
                    next_attention_mask[batch_idx, 0] = 0
                    continue
                # update result
                generated_ids[batch_idx].append(next_token_id)

            # every request in the batch appeared eos token, stop this batch
            if finished.all():
                break

            # not every request finished, continue the loop, add new part to the old data
            # all data transfer to the loop again
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            attention_mask = torch.cat([attention_mask, next_attention_mask], dim=-1)
            position_ids = self._build_position_ids(input_ids)

        # gather all data in the output
        outputs = [self.input_processor.decode(token_ids) for token_ids in generated_ids]
        if batch_size == 1:
            return outputs[0]
        return outputs


