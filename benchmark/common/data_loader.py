import json
from typing import List

from torch.utils.data import Dataset, DataLoader


class StandardLLMDataset(Dataset):
    def __init__(self, file_path, size, is_train=True) -> None:
        with open(file_path, 'r', encoding="utf-8") as f:
            self.data = json.load(f)
        self.data = self.data[:size]
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> List[dict]:
        sample = self.data[idx]
        if self.is_train:
            message = [
                {"role": "user", "content": sample["question"]},
                {"role": "assistant", "content": sample["answer"]}
            ]
        else:
            message = [
                {"role": "user", "content": sample["question"]},
            ]
        return message


class BenchmarkDataLoader:
    def __init__(self, json_path: str = "/Users/shilijun-air/shilijun/huggingface/GSM8K_zh.json",
                 total_size: int = 32,
                 batch_size: int = 32,
                 is_train=False,
                 is_shuffle=False, ):
        self.dataset = StandardLLMDataset(json_path, total_size, is_train=is_train)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=is_shuffle,
                                     collate_fn=self._collector_fn)

    def _collector_fn(self, batch) -> List[dict]:
        return batch

    def get_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_dataset(self) -> Dataset:
        return self.dataset
