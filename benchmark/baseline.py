from sllm.benchmark.common.data_loader import BenchmarkDataLoader
from sllm.utils.timer import Timer
from sllm.engine.llm_engine_base import LlmEngineBase

model_path = "/Users/shilijun-air/shilijun/huggingface/Qwen3-0.6B"

with Timer().timing("init dataloader"):
    dataloader = BenchmarkDataLoader(total_size=32, batch_size=32).get_dataloader()

with Timer().timing("init model"):
    llm_engine = LlmEngineBase(model_path)

with Timer().timing("batch generation"):
    for batch in dataloader:
        output = llm_engine.step(batch)
        print(output)
