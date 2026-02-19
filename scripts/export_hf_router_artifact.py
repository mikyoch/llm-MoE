"""Build an HF-style model artifact folder for UnifiedRouterLLM."""

from __future__ import annotations

import argparse

from unified_llm.hf_export import export_hf_router_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export HF-style UnifiedRouterLLM artifact")
    parser.add_argument("--router_config", required=True, help="Path to router YAML config")
    parser.add_argument("--output_dir", required=True, help="Output folder for HF-style artifact")
    parser.add_argument("--artifact_name", default="UnifiedRouterLLM")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = export_hf_router_artifact(
        router_config_path=args.router_config,
        output_dir=args.output_dir,
        artifact_name=args.artifact_name,
    )
    print(f"Exported HF-style artifact to {out}")


if __name__ == "__main__":
    main()
