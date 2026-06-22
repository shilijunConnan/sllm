# from .qwen3 import Qwen3ForCausalLM
from .qwen3 import Qwen3ForCausalLM
# 建立架构名称与具体类的映射关系
_MODELS_REGISTRY = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}

def get_model_class(arch_name: str):
    if arch_name not in _MODELS_REGISTRY:
        raise ValueError(f"Model architecture {arch_name} 不支持！")
    return _MODELS_REGISTRY[arch_name]