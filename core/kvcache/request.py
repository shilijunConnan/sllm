import asyncio
import enum
from typing import TYPE_CHECKING

import torch

from sllm.utils.request_tools import ChatCompletionRequest
from sllm.utils.config import SllmConfig

if TYPE_CHECKING:
    from sllm.core.kvcache.kv_cache import KVBlockManager, PhysicalKVCache


class RequestStatus(enum.Enum):
    PREFILL_WAITING = 0
    PREFILL_RUNNING = 1
    DECODER_WAITING = 2
    DECODER_RUNNING = 3
    FINISHED = 4
    FAILED = 5


class RequestState:
    def __init__(self, request_id: str, request: ChatCompletionRequest, sllm_config: SllmConfig,
                 kv_manager: "KVBlockManager", device: torch.device):
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
        self.kv_block_table = torch.full(
            ((sllm_config.max_seq_len + sllm_config.block_size - 1) // sllm_config.block_size,),
            -1,
            device=device,
            dtype=torch.int32,
        )

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

    def write_block(self, layer_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        支持 k/v 形状为: [1, num_heads, seq_len, head_dim]
        """

        seq_len = k.size(2)
        current_start = self.get_sum_len() - seq_len

        for i in range(seq_len):
            total_offset = current_start + i
            logical_block_id = total_offset // self.kv_manager.block_size
            block_offset = total_offset % self.kv_manager.block_size
            physical_block_id = int(self.kv_block_table[logical_block_id].item())
            if physical_block_id < 0:
                physical_block_id = self.kv_manager.allocate_block()
                self.kv_block_table[logical_block_id] = physical_block_id

            self.kv_manager.kv.write_block(
                layer_id, physical_block_id, block_offset, k[0,:, i, :], v[0,:, i, :]
            )

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

class RequestContext:
    def __init__(self):
        self.kv: "PhysicalKVCache | None" = None
        self.prefill_request = None
        self.decode_request = None
        self.prefill_request_list = []
        self.decode_request_list = []

    def set_kv(self, kv: "PhysicalKVCache"):
        self.kv = kv

    def get(self, type: str):
        if type == "prefill":
            return self.prefill_request
        if type == "decode":
            return self.decode_request
        return None

    def set(self, data: RequestState, type: str):
        if type == "prefill":
            self.prefill_request = data
        if type == "decode":
            self.decode_request = data

    def reset(self, type: str):
        if type == "prefill":
            self.prefill_request = []
        if type == "decode":
            self.decode_request = []


    # def set(self, data: List[RequestState], type: str):
    #     if type == "prefill":
    #         self.prefill_request_list = data
    #     if type == "decode":
    #         self.decode_request_list = data
    #
    # def reset(self, type: str):
    #     if type == "prefill":
    #         self.prefill_request_list = []
    #     if type == "decode":
    #         self.decode_request_list = []

    def __repr__(self):
        return f"requestContext(prefill={self.prefill_request_list}, decode={self.decode_request_list})"

requestContext = RequestContext()
