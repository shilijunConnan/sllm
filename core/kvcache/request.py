import asyncio
import enum
from typing import List, Tuple

import torch

from sllm.utils.request_tools import ChatCompletionRequest
from sllm.utils.config import SllmConfig
from sllm.core.kvcache.kv_cache import KVBlockManager


class RequestStatus(enum.Enum):
    PREFILL_WAITING = 0
    PREFILL_RUNNING = 1
    DECODER_WAITING = 2
    DECODER_RUNNING = 3
    FINISHED = 4
    FAILED = 5


class RequestState:
    def __init__(self, request_id: str, request: ChatCompletionRequest, sllm_config: SllmConfig,
                 kv_manager: KVBlockManager):
        self.kv_manager = kv_manager
        self.request_id = request_id
        self.status = RequestStatus.PREFILL_WAITING
        self.served_model_name = request.model

        self.input_ids = None
        self.generated_ids = []
        self.generated_text = ""
        self.words_queue = asyncio.Queue()

        self.attention_mask = None
        self.position_ids = None
        self.kv_block_table = []
        self.last_block_offset = 0

        self.inputs_token_num = 0
        self.generated_token_num = 0
        self.max_tokens = request.max_tokens if request.max_tokens <= sllm_config.max_output_len else sllm_config.max_output_len

        self.sample_params = self._get_sample_params(request)

    @staticmethod
    def _get_sample_params(request_params: ChatCompletionRequest) -> dict:
        sample_params = {}
        if hasattr(request_params, "temperature"):
            sample_params["temperature"] = request_params.temperature

        return sample_params

    def get_sum_len(self) -> int:
        return self.inputs_token_num + self.generated_token_num

    def get_last_block_id(self) -> int:
        return self.kv_block_table[-1]

    def write_block(self, k_list: List[torch.Tensor], v_list: List[torch.Tensor]) -> None:
        """
        k_list: [num_layers, 1, num_heads, seq_len, head_dim]
        """
        num_layers = len(k_list)
        seq_len = k_list[0].shape[-2]
        bs = self.kv_manager.block_size

        for seq_id in range(seq_len):
            if not self.kv_block_table or self.last_block_offset >= bs:
                block_id = self.kv_manager.allocate_block()
                self.kv_block_table.append(block_id)
                self.last_block_offset = 0

            block_id = self.kv_block_table[-1]

            for layer_id in range(num_layers):
                k = k_list[layer_id][0, :, seq_id, :]
                v = v_list[layer_id][0, :, seq_id, :]
                self.kv_manager.kv.write_block(layer_id, block_id, self.last_block_offset, k, v)
            self.last_block_offset += 1

    def __repr__(self):
        msg = (f"RequestStatus("
               f"request_id: {self.request_id}, \n"
               f"status: {self.status}, \n"
               f"model: {self.served_model_name})\n"
               f"input_ids: {self.input_ids}, \n"
               f"generated_ids: {self.generated_ids}, \n"
               f"generated_text: {self.generated_text}, \n"
               f"attention_mask: {self.attention_mask}, \n"
               f"position_ids: {self.position_ids}, \n"
               f"kv_cache: {self.kv_block_table}, \n"
               f"inputs_token_num: {self.inputs_token_num}, \n"
               f"generated_token_num: {self.generated_token_num}, \n"
               f"max_tokens: {self.max_tokens}, \n"
               f"sample_params: {self.sample_params})\n")
        return msg
