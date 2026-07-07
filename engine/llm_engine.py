import asyncio
from typing import List, Tuple

import torch

from sllm.models import get_model_class
from sllm.processor.output_processor import OutputProcessor
from sllm.utils.config import ModelConfig, GenerationConfig, SllmConfig
from sllm.processor.input_processor import InputProcessor
from sllm.core.kvcache.kv_cache import KVBlockManager
from sllm.runner.model_runner import ModelRunner
from sllm.core.scheduler.scheduler import Scheduler
from sllm.utils.request_tools import ChatCompletionRequest
from sllm.core.kvcache.request import RequestState, RequestStatus


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
        model_dtype = next(model.parameters()).dtype
        self.kv_manager = KVBlockManager(
            self.model_config.num_hidden_layers,
            self._calculate_max_kv_block_num(),
            self.sllm_config.block_size,
            self.model_config.num_key_value_heads,
            self.model_config.head_dim,
            device=self.device,
            dtype=model_dtype,
        )
        self.model_runner = ModelRunner(model, self.kv_manager)

        # 0.3 init output_processor
        self.output_processor = OutputProcessor(model_path=model_path, sllm_config=self.sllm_config)

        # 0.4 init scheduler
        self.scheduler = Scheduler(self.sllm_config, self.model_config)

    def _calculate_max_kv_block_num(self) -> int:
        assert self.sllm_config.max_input_len + self.sllm_config.max_output_len == self.sllm_config.max_seq_len, "max_input_len add max_output_len not equal to max_seq_len"
        max_seq_len = self.sllm_config.max_seq_len
        max_batch_size = self.sllm_config.max_batch_size
        block_size = self.sllm_config.block_size

        block_per_req = (max_seq_len + block_size - 1) // block_size
        return block_per_req * max_batch_size

    def add_request(self, request_id: str, request: ChatCompletionRequest) -> RequestState:
        input_ids, attention_mask, position_ids = self._prepare_inputs(request.messages)
        req = RequestState(
            request_id=request_id,
            request=request,
            sllm_config=self.sllm_config,
            kv_manager=self.kv_manager,
            device=self.device,
        )
        req.status = RequestStatus.PREFILL_WAITING
        req.input_ids = input_ids.to(self.device)
        req.inputs_token_num = input_ids.shape[-1]
        req.attention_mask = attention_mask.to(self.device)
        req.position_ids = position_ids.to(self.device)
        self.scheduler.add_request(req)
        return req

    async def prefill_background_loop(self):
        while True:
            await self.scheduler.prefill_event.wait()
            prefill_requests = self.scheduler.get_prefill_batch()
            if not self.scheduler.prefill_waiting_queue:
                self.scheduler.prefill_event.clear()

            if prefill_requests:
                try:
                    logits = self.model_runner.execute(prefill_requests, is_prefill=True)
                    for req_idx, req in enumerate(prefill_requests):
                        next_token_logits = logits[req_idx]
                        next_token_ids = self.output_processor.sample(next_token_logits, req.input_ids)

                        if self._is_eos(next_token_ids).all() or req.max_tokens <= 1:
                            req.status = RequestStatus.FINISHED
                            req.words_queue.put_nowait(None)
                            self._free_req_blocks(req)
                            continue
                        req.generated_ids.append(next_token_ids.item())
                        req.generated_token_num += 1
                        req.generated_text = self.input_processor.decode(req.generated_ids)
                        if req.generated_text != "":
                            req.words_queue.put_nowait(req.generated_text)
                        req.attention_mask = None
                        req.position_ids = None
                        req.status = RequestStatus.DECODER_WAITING
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
                logits = self.model_runner.execute(decode_requests, is_prefill=False)

                for req_idx, req in enumerate(decode_requests):
                    next_token_logits = logits[req_idx]
                    next_token_ids = self.output_processor.sample(next_token_logits, req.input_ids)

                    if self._is_eos(next_token_ids).all() or req.max_tokens <= req.generated_token_num + 1:
                        req.status = RequestStatus.FINISHED
                        req.words_queue.put_nowait(None)
                        self._free_req_blocks(req)
                        continue

                    req.generated_ids.append(next_token_ids.item())
                    req.generated_token_num += 1

                    cur_text = self.input_processor.decode(next_token_ids)[0]
                    req.words_queue.put_nowait(cur_text)
                    req.attention_mask = None
                    req.position_ids = None
                self.scheduler.remove_finished_requests()
                if len(self.scheduler.decode_queue) == 0:
                    self.scheduler.decode_event.clear()
                await asyncio.sleep(0)
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

    def _is_eos(self, token_ids: torch.Tensor) -> torch.Tensor:
        eos_token_id = self.generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            return token_ids == eos_token_id
        else:
            eos_tensor = torch.tensor(eos_token_id, device=token_ids.device)
            # token_ids: [batch,1], eos_tensor: [num_eos] -> broadcasting [batch, num_eos]
            is_eos = (token_ids == eos_tensor.unsqueeze(0)).any(dim=1, keepdim=True)  # [batch,1]
            return is_eos
    def _free_req_blocks(self, request: RequestState):
        for table_id in request.kv_block_table:
            block_id = int(table_id.item())
            if block_id >= 0:
                self.kv_manager.free_block(block_id)


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
