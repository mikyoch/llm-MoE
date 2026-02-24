"""Command line interface for UnifiedRouterLLM."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

from unified_llm.unified import UnifiedRouterLLM


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unified-llm", description="Unified multi-expert LLM orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run one prompt")
    run.add_argument("--config", required=True, help="Path to YAML config")
    run.add_argument("--prompt", required=True, help="Prompt text")
    run.add_argument("--mode", default=None, help="Override mode for this request")
    run.add_argument("--temperature", type=float, default=None)
    run.add_argument("--top_p", type=float, default=None)
    run.add_argument("--max_new_tokens", type=int, default=None)
    run.add_argument("--verbose", action="store_true")

    serve = sub.add_parser("serve", help="Start OpenAI-compatible HTTP server")
    serve.add_argument("--config", required=True, help="Path to YAML config")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def _generation_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    if args.temperature is not None:
        payload["temperature"] = args.temperature
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = args.max_new_tokens
    return payload


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        model = UnifiedRouterLLM.from_config(args.config)
        result = model.generate(
            prompt=args.prompt,
            return_metadata=args.verbose,
            mode=args.mode,
            **_generation_kwargs(args),
        )
        if args.verbose:
            print(json.dumps(result, indent=2, ensure_ascii=True))
        else:
            print(result)
        return

    if args.command == "serve":
        try:
            import uvicorn
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("uvicorn is required for serving mode.") from exc
        from unified_llm.server import create_app

        app = create_app(args.config)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
