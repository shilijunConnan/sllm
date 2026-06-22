import asyncio
import json
from typing import List, Literal, Optional, AsyncGenerator
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="sllm")
    messages: List[ChatMessage]
    stream: bool = Field(default=True)
    max_tokens: int = Field(default=100)
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

async def stream_generator(request_id: str, ctx: AsyncGenerator , served_model_name: str, created_time: int):
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
