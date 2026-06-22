import asyncio
import enum

import torch

from sllm.utils.request_tools import ChatCompletionRequest
from sllm.core.kvcache.kv_cache import KVCache
from utils.config import SllmConfig


class RequestStatus(enum.Enum):
    PREFILL_WAITING = 0
    PREFILL_RUNNING = 1
    DECODER_WAITING = 2
    DECODER_RUNNING = 3
    FINISHED = 4
    FAILED = 5


class RequestState:
    def __init__(self, request_id: str, request: ChatCompletionRequest, sllm_config: SllmConfig):
        self.request_id = request_id
        self.status = RequestStatus.PREFILL_WAITING
        self.served_model_name = request.model

        self.input_ids = None
        self.generated_ids = []
        self.generated_text = ""
        self.words_queue = asyncio.Queue()

        self.attention_mask = None
        self.position_ids = None
        self.kv_cache: KVCache | None = None
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

    def get_cur_position(self) -> int:
        return self.inputs_token_num + self.generated_token_num

    def __repr__(self):
        msg = (f"RequestStatus("
               f"request_id: {self.request_id}, "
               f"status: {self.status}, "
               f"model: {self.served_model_name})"
               f"input_ids: {self.input_ids}, "
               f"generated_ids: {self.generated_ids}, "
               f"generated_text: {self.generated_text}, "
               f"attention_mask: {self.attention_mask}, "
               f"position_ids: {self.position_ids}, "
               f"kv_cache: {self.kv_cache}, "
               f"inputs_token_num: {self.inputs_token_num}, "
               f"generated_token_num: {self.generated_token_num}, "
               f"max_tokens: {self.max_tokens}, "
               f"sample_params: {self.sample_params})")
        return msg
