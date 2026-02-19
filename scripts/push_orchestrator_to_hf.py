"""Publish HF-style orchestrator artifact to Hugging Face Hub."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from unified_llm.hf_export import export_hf_router_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push orchestrator artifact to Hugging Face model repo")
    parser.add_argument("--repo_id", required=True, help="HF repo id, e.g. org/unified-router-llm")
    parser.add_argument("--router_config", default="examples/config.yaml", help="Router YAML config path")
    parser.add_argument(
        "--artifact_dir",
        default=None,
        help="Optional output folder for HF-style artifact (default: temporary directory)",
    )
    parser.add_argument(
        "--source_dir",
        default=None,
        help="Upload raw source directory instead of HF-style artifact (not recommended)",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None)
    parser.add_argument("--commit_message", default="Upload orchestrator artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_dir:
        upload_dir = Path(args.source_dir).resolve()
        if not upload_dir.exists():
            raise FileNotFoundError(f"source_dir does not exist: {upload_dir}")
    else:
        if args.artifact_dir:
            artifact_dir = Path(args.artifact_dir).resolve()
            artifact_dir.mkdir(parents=True, exist_ok=True)
        else:
            artifact_dir = Path(tempfile.mkdtemp(prefix="unified_router_hf_"))
        export_hf_router_artifact(
            router_config_path=args.router_config,
            output_dir=str(artifact_dir),
            artifact_name=args.repo_id,
        )
        upload_dir = artifact_dir

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api.upload_folder(
        folder_path=str(upload_dir),
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
    print(f"Uploaded orchestrator artifact from {upload_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
