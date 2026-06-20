import os
import asyncio
import uuid
from typing import Optional

from sllm.engine.llm_engine_v1 import LlmEngineV1
from sllm.utils.request_tools import ChatCompletionRequest, RequestContext


class AsyncLLM:
    def __init__(self, model_path: str, served_model_name: Optional[str]) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name or os.path.basename(os.path.abspath(model_path))
        self.engine = None

        self.request_queue = asyncio.Queue()
        self.result_dict = {}
        self.wait_time = 0.05
        self.max_batch_size = 16

        self.is_shutdown = False

    def init_engine(self):
        self.engine = LlmEngineV1(model_path=self.model_path)

    def shutdown(self):
        self.engine.shutdown()
        self.engine = None
        self.request_queue.put_nowait(None)
        self.result_dict.clear()
        self.is_shutdown = True

    async def scheduler(self):
        while not self.is_shutdown:
            batch_ids = []
            batch_messages = []
            batch_sample_params = []
            try:
                request_id, messages, sample_params = await asyncio.wait_for(self.request_queue.get(),
                                                                             timeout=self.wait_time)
                batch_ids.append(request_id)
                batch_messages.append(messages)
                batch_sample_params.append(sample_params)
            except asyncio.TimeoutError:
                continue

            while len(batch_ids) < self.max_batch_size:
                try:
                    request_id, messages, sample_params = self.request_queue.get_nowait()
                    batch_ids.append(request_id)
                    batch_messages.append(messages)
                    batch_sample_params.append(sample_params)
                except asyncio.QueueEmpty:
                    break
            asyncio.create_task(self.run_batch_stream(batch_ids, batch_messages, batch_sample_params))

    def add_request(self, request: ChatCompletionRequest):
        request_id = f"sllm-{uuid.uuid4().hex}"
        ctx = RequestContext()
        self.result_dict[request_id] = ctx
        task_params = (request_id,
                       request.messages,
                       self._get_sample_params(request))
        self.request_queue.put_nowait(task_params)
        return request_id, ctx

    async def run_batch_stream(self, batch_ids, batch_messages, batch_sample_params):
        async for batch_token in self.engine.batch_stream(batch_messages, batch_sample_params):
            for i, token in enumerate(batch_token):
                delta = {
                    "content": token
                }
                await self.result_dict[batch_ids[i]].send(delta)

        for request_id in batch_ids:
            ctx = self.result_dict.pop(request_id)
            if ctx:
                await ctx.close()


    def _get_sample_params(self, request_params: ChatCompletionRequest) -> dict:
        sample_params = {}
        if hasattr(request_params, "temperature"):
            sample_params["temperature"] = request_params.temperature

        return sample_params
