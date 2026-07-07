import asyncio
from collections import deque
from typing import List

from sllm.core.kvcache.request import RequestState, RequestStatus
from sllm.utils.config import SllmConfig, ModelConfig


class Scheduler:
    def __init__(self, sllm_config: SllmConfig, model_config: ModelConfig) -> None:
        self.sllm_config = sllm_config
        self.model_config = model_config

        self.prefill_waiting_queue = deque()
        self.prefill_queue: List[RequestState] = []
        self.decode_waiting_queue = deque()
        self.decode_queue: List[RequestState] = []

        self.prefill_event = asyncio.Event()
        self.decode_event = asyncio.Event()

    def add_request(self, request: RequestState) -> None:
        self.prefill_waiting_queue.append(request)
        self.prefill_event.set()

    def get_prefill_batch(self):
        while len(self.prefill_queue) < self.sllm_config.max_batch_size and self.prefill_waiting_queue:
            request: RequestState = self.prefill_waiting_queue.popleft()
            request.status = RequestStatus.PREFILL_RUNNING
            self.prefill_queue.append(request)
        return self.prefill_queue

    def get_decode_batch(self):
        while len(self.decode_queue) < self.sllm_config.max_batch_size and self.decode_waiting_queue:
            request = self.decode_waiting_queue.popleft()
            request.status = RequestStatus.DECODER_RUNNING
            self.decode_queue.append(request)
        return list(self.decode_queue)

    def remove_prefilled_requests(self):
        for r in self.prefill_queue:
            if r.status == RequestStatus.DECODER_WAITING:
                self.decode_waiting_queue.append(r)
        self.prefill_queue = []
        self.decode_event.set()

    def remove_finished_requests(self):
        self.decode_queue = [
            r
            for r in self.decode_queue
            if r.status != RequestStatus.FINISHED
        ]
