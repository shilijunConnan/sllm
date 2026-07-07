import os

from pathlib import Path
from safetensors.torch import load_file, save_file

from sllm.utils.config import ModelConfig


class ModelPretrained:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls,
                        model_path: str,
                        tie_word_embeddings: bool = True,
                        model_config: ModelConfig = None,
                        device="cpu"):

        state_path = Path(model_path) / "model.safetensors"
        if not state_path.exists():
            raise FileNotFoundError(model_path)
        state_dict = load_file(state_path)

        model = cls(model_config)
        model.to(device)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(f"[WARN] Missing keys: {len(load_result.missing_keys)}")
            print(f"[WARN] Unexpected keys: {len(load_result.unexpected_keys)}")
            if load_result.missing_keys[:10]:
                print(f"[WARN] Missing keys sample: {load_result.missing_keys[:10]}")
            if load_result.unexpected_keys[:10]:
                print(f"[WARN] Unexpected keys sample: {load_result.unexpected_keys[:10]}")

        if tie_word_embeddings and hasattr(model, "embed_tokens") and hasattr(model, "lm_head"):
            model.lm_head.weight = model.model.embed_tokens.weight

        return model

    def save_pretrained(self, save_dir: str, tie_word_embeddings: bool = True):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = self.state_dict()

        if tie_word_embeddings and hasattr(self, "lm_head") and hasattr(self.model, "embed_tokens"):
            state_dict.pop("lm_head.weight", None)

        file_path = Path(save_dir) / "model.safetensors"
        save_file(state_dict, str(file_path))
        print(f"[INFO] Model saved to {file_path} (tie_word_embeddings={tie_word_embeddings})")
