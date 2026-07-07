import asyncio
import os
from typing import Optional

from sllm.core.kvcache.request import RequestState
from sllm.engine.llm_engine import LlmEngine
from sllm.utils.request_tools import ChatCompletionRequest


class AsyncLLM:
    def __init__(self, model_path: str, served_model_name: Optional[str]) -> None:
        self.model_path = model_path
        self.served_model_name = served_model_name or os.path.basename(os.path.abspath(model_path))
        self.engine = None

        self.is_shutdown = False

    def init_engine(self):
        self.engine = LlmEngine(model_path=self.model_path)
        self.prefill_task = asyncio.create_task(self.engine.prefill_background_loop())
        self.decode_task = asyncio.create_task(self.engine.decode_background_loop())

    def shutdown(self):
        self.engine.close()
        self.engine = None
        self.is_shutdown = True

    async def async_generate(self, request_id, request: ChatCompletionRequest):
        if not self.is_shutdown:
            req: RequestState = self.engine.add_request(
                request_id=request_id,
                request=request,
            )
            while True:
                token = await req.words_queue.get()
                if token is None:
                    break
                yield token
