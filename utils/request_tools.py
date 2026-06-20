import asyncio
import json
from typing import Iterator, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="sllm")
    messages: List[ChatMessage]
    stream: bool = Field(default=True)
    temperature: float = Field(default=1.0)


def openai_sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def build_chunk(
        request_id: str,
        model: str,
        created: int,
        delta: dict,
        finish_reason: Optional[str] = None,
) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }

class RequestContext:
    def __init__(self):
        self.queue = asyncio.Queue()

    async def send(self, token: str):
        await self.queue.put(token)

    async def close(self):
        await self.queue.put(None)

    async def __aiter__(self):
        while True:
            x = await self.queue.get()
            if x is None:
                break
            yield x

async def stream_generator(request_id: str, ctx: RequestContext, served_model_name: str, created_time: int):
    async for delta in ctx:
        data = build_chunk(
            request_id=request_id,
            model=served_model_name,
            created=created_time,
            delta=delta,
        )
        yield openai_sse(data)
    yield openai_sse(build_chunk(request_id, served_model_name, created_time, {}, finish_reason="stop"))
    yield "data: [DONE]\n\n"
