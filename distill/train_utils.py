"""Training utilities for optional SFT distillation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("PyTorch is required for distillation utilities.") from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class PairRecord:
    prompt: str
    response: str


def load_pairs_jsonl(path: str) -> List[PairRecord]:
    records: List[PairRecord] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt", "")
            response = payload.get("response", "")
            if isinstance(prompt, str) and isinstance(response, str):
                if prompt.strip() and response.strip():
                    records.append(PairRecord(prompt=prompt, response=response))
    return records


class PromptCompletionDataset(Dataset):
    def __init__(self, records: List[PairRecord], tokenizer: Any, max_length: int = 2048) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        prompt_ids = self.tokenizer.encode(rec.prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(rec.response, add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id or 0

        input_ids = (prompt_ids + response_ids + [eos_id])[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids + [eos_id])[: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class DataCollatorForCausalLM:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].shape[0] for item in batch)
        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            length = item["input_ids"].shape[0]
            pad = max_len - length
            input_ids.append(
                torch.cat([item["input_ids"], torch.full((pad,), self.pad_token_id, dtype=torch.long)])
            )
            labels.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
            attention_mask.append(torch.cat([item["attention_mask"], torch.zeros((pad,), dtype=torch.long)]))

        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "labels": torch.stack(labels, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
        }
