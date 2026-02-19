"""Publish distilled student checkpoint to Hugging Face Hub."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push distilled student checkpoint to Hugging Face")
    parser.add_argument("--repo_id", required=True, help="HF repo id, e.g. org/student-llm")
    parser.add_argument("--model_dir", required=True, help="Local directory with model/tokenizer files")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--base_model", default="unknown")
    parser.add_argument("--dataset_note", default="Teacher-generated prompt/response pairs")
    parser.add_argument("--commit_message", default="Upload distilled student model")
    return parser.parse_args()


def build_model_card(repo_id: str, base_model: str, dataset_note: str) -> str:
    return f"""---
library_name: transformers
tags:
  - distillation
  - causal-lm
license: apache-2.0
---

# {repo_id}

Distilled student model checkpoint.

## Training summary

- Base model: `{base_model}`
- Objective: supervised fine-tuning distillation (SFT)
- Data: {dataset_note}

## Limitations

- Quality depends on teacher coverage and prompt distribution.
- May underperform experts on rare edge cases not represented in distillation data.
"""


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=["*.tmp", "__pycache__/*", "*.pyc"],
    )

    card = build_model_card(args.repo_id, args.base_model, args.dataset_note)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(card)
        tmp_path = tmp.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add student model card",
    )
    print(f"Uploaded distilled student to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
