import json
from pathlib import Path

class SllmConfig:
    max_seq_len = 40960
    max_input_len = 1024
    max_output_len = 1024
    max_batch_size = 16
    block_size = 16
    max_kv_block_num = 1024
    kv_block_size = 16
    def __init__(self):
        sllm_config_json_path = Path(__file__).resolve().parents[1] / "server" / "sllm_config.json"
        if not sllm_config_json_path.is_file():
            raise FileNotFoundError(f"Config file {sllm_config_json_path} not found")
        with open(sllm_config_json_path, "r", encoding="utf-8") as f:
            sllm_config = json.load(f)
        for k, v in sllm_config.items():
            setattr(self, k, v)



class ModelConfig:
    architectures = []
    attention_bias = False
    attention_dropout = 0.0
    bos_token_id = 151643
    eos_token_id = 151645
    head_dim = 128
    hidden_act = "silu"
    hidden_size = 1024
    initializer_range = 0.02
    intermediate_size = 3072
    max_position_embeddings = 40960
    max_window_layers = 28
    model_type = "qwen3"
    num_attention_heads = 16
    num_hidden_layers = 28
    num_key_value_heads = 8
    rms_norm_eps = 1e-06
    rope_scaling = None
    rope_theta = 1000000
    sliding_window = None
    tie_word_embeddings = True
    torch_dtype = "bfloat16"
    transformers_version = "4.51.0"
    use_cache = True
    use_sliding_window = True
    vocab_size = 151936

    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        if model_path.is_dir():
            model_config_json_path = Path(model_path) / "config.json"
        else:
            model_config_json_path = model_path
        if not model_config_json_path.exists():
            raise FileNotFoundError(f"Config file {model_config_json_path} not found")

        with open(model_config_json_path, "r") as f:
            model_config = json.load(f)
        self.model_config = model_config
        for key, value in model_config.items():
            setattr(self, key, value)


    def get_config(self) -> dict:
        return self.model_config


class GenerationConfig:
    bos_token_id = 151643
    eos_token_id = [151645, 151643]
    pad_token_id = 151643
    do_sample = True
    temperature = 0.6
    top_k = 20
    top_p = 0.95
    repetition_penalty = 1.1
    no_repeat_ngram_size = 0

    def __init__(self, model_path: str):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        if model_path.is_dir():
            generation_config_json_path = Path(model_path) / "generation_config.json"
        else:
            generation_config_json_path = model_path
        if not generation_config_json_path.exists():
            raise FileNotFoundError(f"Generation config file {generation_config_json_path} not found")

        with open(generation_config_json_path, "r") as f:
            generation_config = json.load(f)
        self.generation_config = generation_config
        for key, value in generation_config.items():
            setattr(self, key, value)

    def get_config(self) -> dict:
        return self.generation_config
