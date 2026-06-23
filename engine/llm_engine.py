import asyncio
from typing import Iterator, List, Tuple, Union

import torch

from sllm.core.scheduler.batch import BatchBuilder
from models import get_model_class
from processor.output_processor import OutputProcessor
from sllm.utils.config import ModelConfig, GenerationConfig, SllmConfig
from sllm.processor.input_processor import InputProcessor
from sllm.core.kvcache.kv_cache import KVCache
from sllm.runner.model_runner import ModelRunner
from sllm.core.scheduler.scheduler import Scheduler
from sllm.utils.request_tools import ChatCompletionRequest
from sllm.core.scheduler.request import RequestState, RequestStatus


class LlmEngine:
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

        # 0.4 init scheduler
        self.scheduler = Scheduler(max_batch_size=self.sllm_config.max_batch_size)
        self.batch_builder = BatchBuilder(pad_token_id=self.input_processor.get_pad_token_id(), device=self.device)

    def add_request(self, request_id: str, request: ChatCompletionRequest) -> RequestState:
        input_ids, attention_mask, position_ids = self._prepare_inputs(request.messages)
        req = RequestState(request_id=request_id, request=request, sllm_config=self.sllm_config)
        req.status = RequestStatus.PREFILL_WAITING
        req.input_ids = input_ids.to(self.device)
        req.inputs_token_num = input_ids.shape[-1]
        req.attention_mask = attention_mask.to(self.device)
        req.position_ids = position_ids.to(self.device)
        req.kv_cache = KVCache(self.model_config.num_hidden_layers)
        self.scheduler.add_request(req)
        return req

    async def prefill_background_loop(self):
        while True:
            await self.scheduler.prefill_event.wait()
            prefill_requests: List[RequestState] = self.scheduler.get_prefill_batch()
            if not self.scheduler.prefill_waiting_queue:
                self.scheduler.prefill_event.clear()

            if prefill_requests:
                try:
                    batch = self.batch_builder.prefill_build(prefill_requests, self.model_config.num_hidden_layers)
                    logits = self.model_runner.execute(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        position_ids=batch.position_ids,
                        past_key_values=batch.pask_keys_values,
                        is_decode=False
                    )

                    next_token_logits = logits[:, -1:, :]

                    for idx, req in enumerate(prefill_requests):
                        # pass sample params here

                        next_token_id = self.output_processor.sample(next_token_logits[idx, :, :],
                                                                     batch.input_ids[idx, :])

                        if self._is_eos(next_token_id) or req.max_tokens <= 1:
                            req.status = RequestStatus.FINISHED
                            req.words_queue.put_nowait(None)
                            continue
                        req.generated_ids.append(next_token_id.item())
                        req.generated_token_num += 1
                        req.generated_text = self.input_processor.decode(req.generated_ids)
                        if req.generated_text != "":
                            req.words_queue.put_nowait(req.generated_text)
                        req.attention_mask = None
                        req.position_ids = None
                        req.status = RequestStatus.DECODER_WAITING

                        for i in range(self.model_config.num_hidden_layers):
                            k_values = batch.pask_keys_values.k_values[i][idx, :, -req.inputs_token_num:, :]
                            v_values = batch.pask_keys_values.v_values[i][idx, :, -req.inputs_token_num:, :]
                            req.kv_cache.update_cache(i, k_values.unsqueeze(0), v_values.unsqueeze(0))
                    self.scheduler.remove_prefilled_requests()
                    await asyncio.sleep(0)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            else:
                await asyncio.sleep(0.1)
                continue

    async def decode_background_loop(self):
        while True:
            await self.scheduler.decode_event.wait()
            decode_requests = self.scheduler.get_decode_batch()

            if decode_requests:
                try:
                    batch = self.batch_builder.decode_build(decode_requests, self.model_config.num_hidden_layers)
                    logits = self.model_runner.execute(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        position_ids=batch.position_ids,
                        past_key_values=batch.pask_keys_values,
                        kv_pos=batch.kv_pos,
                        is_decode=True
                    )
                    for idx, req in enumerate(decode_requests):
                        pre_token = torch.cat(
                            (req.input_ids, torch.tensor(req.generated_ids, device=self.device).unsqueeze(0)), dim=-1)
                        next_token_id = self.output_processor.sample(logits[idx, :, :], pre_token)

                        if self._is_eos(next_token_id) or req.max_tokens <= req.generated_token_num + 1:
                            req.status = RequestStatus.FINISHED
                            req.words_queue.put_nowait(None)
                            continue
                        req.generated_ids.append(next_token_id.item())
                        req.generated_token_num += 1

                        cur_text = self.input_processor.decode(req.generated_ids)
                        one_token = cur_text[0][len(req.generated_text[0]):]
                        req.generated_text = cur_text
                        if one_token != "":
                            req.words_queue.put_nowait(one_token)
                        req.attention_mask = None
                        req.position_ids = None

                        for i in range(self.model_config.num_hidden_layers):
                            req.kv_cache.update_cache(
                                i,
                                batch.pask_keys_values.k_values[i][:, :, batch.kv_pos[idx]: batch.kv_pos[idx] + 1, :],
                                batch.pask_keys_values.v_values[i][:, :, batch.kv_pos[idx]: batch.kv_pos[idx] + 1, :]
                            )

                    self.scheduler.remove_finished_requests()
                    if len(self.scheduler.decode_queue) == 0:
                        self.scheduler.decode_event.clear()
                    await asyncio.sleep(0)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    break
            else:
                await asyncio.sleep(0.1)
                continue

    def _select_device(self) -> torch.device:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _prepare_inputs(self, messages: List[any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.input_processor.encode(messages)
        return (inputs["input_ids"], inputs["attention_mask"], inputs["position_ids"])

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
