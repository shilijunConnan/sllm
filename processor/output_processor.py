import torch

from utils.config import GenerationConfig, SllmConfig


class OutputProcessor:
    def __init__(self, model_path:str, sllm_config:SllmConfig):
        self.generation_config = GenerationConfig(model_path)
        self.sllm_config = sllm_config

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = getattr(self.generation_config, "temperature", 1.0)
        if temperature is None or temperature <= 0:
            return logits
        return logits / temperature

    def _apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        repetition_penalty = getattr(self.generation_config, "repetition_penalty", 1.0)
        if input_ids is None or repetition_penalty is None or repetition_penalty == 1.0:
            return logits

        logits = logits.clone()
        for batch_idx in range(logits.shape[0]):
            token_ids = torch.unique(input_ids[batch_idx])
            token_logits = logits[batch_idx, token_ids]
            logits[batch_idx, token_ids] = torch.where(
                token_logits < 0,
                token_logits * repetition_penalty,
                token_logits / repetition_penalty,
            )
        return logits

    def _apply_no_repeat_ngram(self, logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        no_repeat_ngram_size = getattr(self.generation_config, "no_repeat_ngram_size", 0)
        if (
            input_ids is None
            or no_repeat_ngram_size is None
            or no_repeat_ngram_size <= 0
            or input_ids.shape[-1] < no_repeat_ngram_size - 1
        ):
            return logits

        logits = logits.clone()
        min_value = torch.finfo(logits.dtype).min
        prefix_size = no_repeat_ngram_size - 1

        for batch_idx in range(input_ids.shape[0]):
            token_ids = input_ids[batch_idx].tolist()
            prefix = token_ids[-prefix_size:] if prefix_size > 0 else []
            banned_tokens = []

            for start_idx in range(len(token_ids) - no_repeat_ngram_size + 1):
                ngram = token_ids[start_idx:start_idx + no_repeat_ngram_size]
                if prefix_size == 0 or ngram[:-1] == prefix:
                    banned_tokens.append(ngram[-1])

            if banned_tokens:
                logits[batch_idx, banned_tokens] = min_value
        return logits

    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        top_k = getattr(self.generation_config, "top_k", 0)
        if top_k is None or top_k <= 0:
            return logits

        top_k = min(top_k, logits.shape[-1])
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k_values = top_k_values[:, -1, None]
        return logits.masked_fill(logits < min_top_k_values, torch.finfo(logits.dtype).min)

    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        top_p = getattr(self.generation_config, "top_p", 1.0)
        if top_p is None or top_p >= 1.0:
            return logits
        if top_p <= 0:
            raise ValueError("top_p must be greater than 0")

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)

    def sample(self, logits: torch.Tensor, input_ids: torch.Tensor = None) -> torch.Tensor:
        do_sample = getattr(self.generation_config, "do_sample", True)
        if not do_sample:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = self._apply_repetition_penalty(logits, input_ids)
        logits = self._apply_no_repeat_ngram(logits, input_ids)
        logits = self._apply_temperature(logits)
        logits = self._apply_top_k(logits)
        logits = self._apply_top_p(logits)
        probability = torch.softmax(logits, dim=-1)
        probability = torch.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0)

        invalid_rows = probability.sum(dim=-1) <= 0
        if invalid_rows.any():
            greedy_tokens = torch.argmax(logits, dim=-1, keepdim=True)
            sampled_tokens = torch.multinomial(probability.masked_fill(invalid_rows[:, None], 1.0), 1)
            sampled_tokens[invalid_rows] = greedy_tokens[invalid_rows]
            return sampled_tokens

        return torch.multinomial(probability, 1)
