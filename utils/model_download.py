import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

# 1. 定义模型 ID 和本地保存路径
model_id = "Qwen/Qwen3-0.6B"  # 如果是 0.6B 请替换为准确的仓库名
local_dir = "/voyager/huggingface/Qwen3-0.6B"

print(f"正在开始下载模型 {model_id} 到本地路径: {local_dir} ...")

# 2. 执行下载
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False, # 直接下载物理文件，方便后续拷贝
        ignore_patterns=["*.msgpack", "*.h5"], # 忽略非 PyTorch/Safetensors 格式（可选，省空间）
        resume_download=True          # 开启断点续传
    )
    print("\n🎉 下载成功！模型已完整保存。")
except Exception as e:
    print(f"\n❌ 下载失败，错误信息: {e}")