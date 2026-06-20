import asyncio
from typing import Iterator, List, Tuple, Union

import torch

from models import get_model_class
from processor.output_processor import OutputProcessor
from sllm.utils.config import ModelConfig, GenerationConfig, SllmConfig
from sllm.processor.input_processor import InputProcessor
from sllm.core.kvcache.kv_cache_v1 import KVCache
from sllm.runner.model_runner_v1 import ModelRunner
from sllm.models.qwen3_v1 import Qwen3ForCausalLM


class LlmEngineV1:
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

    def _prepare_kv_cache(self):
        return KVCache(self.model_config.num_hidden_layers)

    def _is_eos(self, token_ids: torch.Tensor) -> torch.Tensor:
        eos_token_id = self.generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            return token_ids == eos_token_id
        else:
            eos_tensor = torch.tensor(eos_token_id, device=token_ids.device)
            # token_ids: [batch,1], eos_tensor: [num_eos] -> broadcasting [batch, num_eos]
            is_eos = (token_ids == eos_tensor.unsqueeze(0)).any(dim=1, keepdim=True)  # [batch,1]
            return is_eos

    # benchmark测试batch对比
    def step(self, messages: List[dict]) -> Union[str, List[str]]:
        # 1.1 transform messgaes to input data
        input_ids, position_ids, attention_mask = self._prepare_inputs(messages)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # 1.2 array to keep output tokenid
        max_seq_len = self.sllm_config.max_output_len
        pad_token_id = self.generation_config.pad_token_id
        generated_ids = torch.full(size=(batch_size, max_seq_len), fill_value=pad_token_id, device=input_ids.device)
        # 1.3 tensor to keep status
        finished = torch.zeros(batch_size, dtype=torch.int, device=input_ids.device)
        past_key_values = self._prepare_kv_cache()
        # 1.4 loop until max_output_len
        for step_idx in range(self.sllm_config.max_output_len):
            logits = self.model_runner.execute(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            print(step_idx)
            # take the last one
            next_token_logits = logits[:, -1, :]  # here size from [batch, seq, vocab] to [batch, vocab]
            next_token_ids = self.output_processor.sample(next_token_logits, input_ids)  # [batch, 1]
            next_attention_mask = torch.ones_like(next_token_ids)
            # according to finish status, update next_token_ids, finish status, update mask
            finished_mask = finished.unsqueeze(1)
            next_token_ids = finished_mask * pad_token_id + (1 - finished_mask) * next_token_ids

            finished = finished | self._is_eos(next_token_ids).squeeze(1)
            next_attention_mask = next_attention_mask * (1 - finished).unsqueeze(1)
            # update result
            generated_ids[:, step_idx] = next_token_ids.squeeze(1)

            # every request in the batch appeared eos token, stop this batch
            if finished.all():
                break

            input_ids = next_token_ids
            attention_mask = torch.cat([attention_mask, next_attention_mask], dim=-1)
            position_ids = torch.full(size=(batch_size, 1), fill_value=(seq_len + step_idx), device=input_ids.device)

        # gather all data in the output
        outputs = [self.input_processor.decode(token_ids) for token_ids in generated_ids]
        if batch_size == 1:
            return outputs[0]
        return outputs

    async def batch_stream(self, messages: List[dict], sample_params: List[dict]) -> Iterator[List[str]]:
        input_ids, position_ids, attention_mask = self._prepare_inputs(messages)
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        generated_ids_tensor = torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
        decoded_text_list = [""] * batch_size
        past_key_values = self._prepare_kv_cache()

        for step_idx in range(self.sllm_config.max_output_len):
            logits = self.model_runner.execute(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )
            next_token_logits = logits[:, -1, :]
            next_token_ids = self.output_processor.sample(next_token_logits, input_ids)

            if self._is_eos(next_token_ids).all():
                break
            generated_ids_tensor = torch.cat([generated_ids_tensor, next_token_ids], dim=-1)
            next_text = self.input_processor.decode(generated_ids_tensor)
            yield_data = [""] * batch_size
            for i, text in enumerate(next_text):
                yield_data[i] = text[len(decoded_text_list[i]):]
                decoded_text_list[i] = text
            yield yield_data

            input_ids = next_token_ids
            next_attention_mask = torch.ones_like(next_token_ids)
            attention_mask = torch.cat([attention_mask, next_attention_mask], dim=-1)
            position_ids = torch.full(size=(batch_size, 1), fill_value=(seq_len + step_idx), device=input_ids.device)
            await asyncio.sleep(0)

    def close(self) -> None:
        print("[INFO] Shutting down LlmEngineV1 and clearing GPU memory...")

        # 1. 释放 model 和 model_runner 的引用
        if hasattr(self, 'model_runner') and self.model_runner is not None:
            # 如果你的 ModelRunner 内部也持有了 model，可以尝试先把它移到 cpu
            # 这一步能更彻底地强制释放 CUDA 显存
            if hasattr(self.model_runner, 'model') and self.model_runner.model is not None:
                try:
                    self.model_runner.model.to("cpu")
                except Exception:
                    pass
                self.model_runner.model = None

            self.model_runner = None

        # 2. 释放其他可能持有 Tensor 缓存的组件
        self.input_processor = None
        self.output_processor = None

        # 3. 强制进行垃圾回收
        import gc
        gc.collect()

        # 4. 清空 PyTorch CUDA/MPS 显存缓存
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # 清理进程间通信的残留显存（推荐）
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        print("[INFO] LlmEngineV1 shutdown complete.")
