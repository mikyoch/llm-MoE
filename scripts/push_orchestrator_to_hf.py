"""Publish orchestrator package artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push orchestrator artifact to Hugging Face model repo")
    parser.add_argument("--repo_id", required=True, help="HF repo id, e.g. org/unified-router-llm")
    parser.add_argument("--source_dir", default=".", help="Local folder to upload")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--commit_message", default="Upload orchestrator artifact")
    return parser.parse_args()


def build_model_card(repo_id: str) -> str:
    return f"""---
library_name: transformers
tags:
  - llm-router
  - ensemble
  - orchestration
license: apache-2.0
---

# {repo_id}

This repository contains a **single-process LLM orchestrator** package.

## Important

- This artifact is a router ensemble product, not a single merged checkpoint.
- It routes prompts to specialist experts and exposes one unified API.
- Experts may use different tokenizers and architectures.

## Contents

- `unified_llm/` package source
- `examples/config.yaml`
- serving and CLI utilities
- optional distillation scripts

## Usage

Install package and run:

```bash
pip install -e .
unified-llm run --config examples/config.yaml --prompt "Hello"
```
"""


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_folder(
        folder_path=str(source_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        ignore_patterns=[
            ".git/*",
            ".pytest_cache/*",
            "__pycache__/*",
            "*.pyc",
            ".venv/*",
            "outputs/*",
        ],
    )

    card = build_model_card(args.repo_id)
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(card)
        tmp_path = tmp.name

    api.upload_file(
        path_or_fileobj=tmp_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="Add orchestrator model card",
    )
    print(f"Uploaded orchestrator artifact to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
