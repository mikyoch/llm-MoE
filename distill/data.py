"""Data helpers for optional distillation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from unified_llm.unified import UnifiedRouterLLM


def load_prompts_jsonl(path: str | Path) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            prompt = payload.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt)
    return prompts


def write_teacher_pairs_jsonl(
    router_model: UnifiedRouterLLM,
    prompts: Iterable[str],
    output_path: str | Path,
    mode: str = "run_all_select",
    max_samples: Optional[int] = None,
) -> int:
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for prompt in prompts:
            result = router_model.generate(prompt, return_metadata=True, mode=mode)
            row: Dict[str, object] = {
                "prompt": prompt,
                "response": result["text"],
                "metadata": result["metadata"],
            }
            out.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
            if max_samples is not None and count >= max_samples:
                break
    return count


def generate_teacher_pairs_from_config(
    router_config: str,
    prompts_jsonl: str,
    output_jsonl: str,
    mode: str = "run_all_select",
    max_samples: Optional[int] = None,
) -> int:
    model = UnifiedRouterLLM.from_config(router_config)
    prompts = load_prompts_jsonl(prompts_jsonl)
    return write_teacher_pairs_jsonl(
        router_model=model,
        prompts=prompts,
        output_path=output_jsonl,
        mode=mode,
        max_samples=max_samples,
    )
