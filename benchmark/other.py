import torch
from transformers import AutoTokenizer
model_path = "/Users/shilijun-air/shilijun/huggingface/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
a = torch.tensor([[10024,18025],[10124,18025]])
print(a)
w = tokenizer.batch_decode(a)
print(w)


