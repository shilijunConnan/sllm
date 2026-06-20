import argparse
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from sllm.utils.request_tools import *
from async_llm_v1 import AsyncLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sllm OpenAI-compatible streaming API server.")
    parser.add_argument("--model-path",
                        default=os.getenv("SLLM_MODEL_PATH", "/Users/shilijun-air/shilijun/huggingface/Qwen3-0.6B"),
                        help="Path to model directory.")
    parser.add_argument("--served-model-name", default=os.getenv("SLLM_MODEL_NAME", "sllm"),
                        help="Model name returned in responses.")
    parser.add_argument("--host", default=os.getenv("SLLM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SLLM_PORT", "8000")))
    return parser.parse_args()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global async_llm, scheduler_task, args

    args = parse_args()
    async_llm = AsyncLLM(args.model_path, args.served_model_name)
    async_llm.init_engine()
    scheduler_task = asyncio.create_task(async_llm.scheduler())
    app.host(args.host, args.port)
    print("scheduler started")

    yield

    print("shutting down scheduler")
    scheduler_task.cancel()
    async_llm.shutdown()
    try:
        await scheduler_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    lifespan=lifespan,
    title="sllm OpenAI-compatible stream API",
    version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    global async_llm
    request_id, ctx = async_llm.add_request(request)
    created = int(time.time())
    return StreamingResponse(
        stream_generator(request_id, ctx, request.model, created),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
