import sys
try:
    from sllm.cuda import paged_attention
    print("paged_attention module loaded successfully")
    print(dir(paged_attention))  # 查看可用的函数
except ImportError as e:
    print(f"Failed to import paged_attention: {e}")
    paged_attention = None