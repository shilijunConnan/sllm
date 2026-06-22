import asyncio
import json
import time
from typing import List

import httpx
from transformers import AutoTokenizer

from common.data_loader import StandardLLMDataset
# ================= 配置区域 =================
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "sllm"
CONCURRENT_USERS = 16  # 并发用户数（压测线索数）
TOTAL_REQUESTS = 32  # 总共测试的请求数

# 测试用的 Prompt
dataset_path = "/Users/shilijun-air/shilijun/huggingface/GSM8K_zh.json"
tokenizer_path = "/Users/shilijun-air/shilijun/huggingface/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# ============================================

class RequestMetrics:
    def __init__(self):
        self.success = False
        self.start_time = 0.0
        self.ttft = 0.0  # 首字延迟 (Time to First Token)
        self.total_time = 0.0  # 单个请求总耗时
        self.token_count = 0  # 吐出的 token 数量


async def send_stream_request(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, message: List[dict]) -> RequestMetrics:
    global tokenizer
    metrics = RequestMetrics()

    payload = {
        "model": MODEL_NAME,
        "messages": message,
        "stream": True
    }
    output_text = ""
    # 使用信号量控制最大并发用户数
    async with semaphore:
        metrics.start_time = time.perf_counter()
        try:
            async with client.stream("POST", API_URL, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    return metrics

                # 循环读取流式响应
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # 记录首 Token 时间
                    if metrics.ttft == 0.0:
                        metrics.ttft = time.perf_counter() - metrics.start_time

                    # 处理 OpenAI 标准的 data: 行
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            # 提取 delta.content，如果有内容则 token 计数 +1
                            delta = data_json["choices"][0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                output_text = output_text + delta["content"]
                        except Exception:
                            pass

                metrics.total_time = time.perf_counter() - metrics.start_time
                metrics.success = True
        except Exception as e:
            print(f"\n[ERROR] Request failed: {e}")
    metrics.token_count = len(tokenizer.encode(output_text))
    return metrics


async def main():
    print(f"=== 开始压测 {API_URL} ===")
    print(f"并发用户数 (Concurrency): {CONCURRENT_USERS}")
    print(f"目标总请求数 (Total Requests): {TOTAL_REQUESTS}")

    semaphore = asyncio.Semaphore(CONCURRENT_USERS)

    # 限制连接池大小，防止连接复用受限
    limits = httpx.Limits(max_keepalive_connections=CONCURRENT_USERS, max_connections=CONCURRENT_USERS * 2)

    start_benchmark_time = time.perf_counter()
    dataset = StandardLLMDataset(file_path=dataset_path, size=TOTAL_REQUESTS, is_train=False)
    async with httpx.AsyncClient(limits=limits) as client:
        # 创建所有请求任务
        tasks = [send_stream_request(client, semaphore, dataset[i]) for i in range(TOTAL_REQUESTS)]
        # 并发执行并等待完成
        results: List[RequestMetrics] = await asyncio.gather(*tasks)

    end_benchmark_time = time.perf_counter()
    total_wall_time = end_benchmark_time - start_benchmark_time

    # ================= 统计指标 =================
    successful_results = [r for r in results if r.success]
    failed_count = TOTAL_REQUESTS - len(successful_results)

    if not successful_results:
        print("❌ 所有请求均失败，请检查服务是否正常启动或接口字段是否对齐。")
        return

    total_tokens = sum(r.token_count for r in successful_results)
    avg_ttft = sum(r.ttft for r in successful_results) / len(successful_results)
    avg_latency = sum(r.total_time for r in successful_results) / len(successful_results)

    rps = len(successful_results) / total_wall_time
    tps = total_tokens / total_wall_time

    # ================= 打印报告 =================
    print("=" * 40)
    print("               压测报告               ")
    print("=" * 40)
    print(f"总压测耗时 (Total Time):     {total_wall_time:.2f} s")
    print(f"成功请求数 (Successful Req): {len(successful_results)}")
    print(f"失败请求数 (Failed Req):     {failed_count}")
    print(f"总输出 Token 数 (Total Tokens): {total_tokens}")
    print("-" * 40)
    print(f"每秒请求数 (RPS):            {rps:.2f} req/s  <-- 吞吐量(请求)")
    print(f"每秒吞吐 Token 数 (TPS):     {tps:.2f} tokens/s <-- 核心性能(显卡能力)")
    print("-" * 40)
    print(f"平均首字延迟 (Avg TTFT):      {avg_ttft * 1000:.2f} ms <-- 用户视觉快慢")
    print(f"平均每单总耗时 (Avg Latency):  {avg_latency:.2f} s")
    print("=" * 40)


if __name__ == "__main__":
    asyncio.run(main())