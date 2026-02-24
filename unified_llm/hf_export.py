"""Utilities to export UnifiedRouterLLM as an HF-style model artifact."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

from unified_llm.config import load_config


CONFIGURATION_PY = '''from transformers import PretrainedConfig


class UnifiedRouterConfig(PretrainedConfig):
    model_type = "unified_router"

    def __init__(
        self,
        router_config_path="router_config.yaml",
        routing_mode="route_top1",
        expert_count=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.router_config_path = router_config_path
        self.routing_mode = routing_mode
        self.expert_count = expert_count
'''


MODELING_PY = '''import os
from typing import Any

import torch
from transformers import PreTrainedModel

from configuration_unified_router import UnifiedRouterConfig


class UnifiedRouterForCausalLM(PreTrainedModel):
    config_class = UnifiedRouterConfig
    base_model_prefix = "unified_router"

    def __init__(self, config: UnifiedRouterConfig):
        super().__init__(config)
        self._runtime = None

    def _ensure_runtime(self) -> None:
        if self._runtime is not None:
            return
        try:
            from unified_llm import UnifiedRouterLLM
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "unified-router-llm package is required. Install it with `pip install unified-router-llm`."
            ) from exc

        model_dir = self.name_or_path if hasattr(self, "name_or_path") else "."
        router_path = os.path.join(model_dir, self.config.router_config_path)
        self._runtime = UnifiedRouterLLM.from_config(router_path)

    def forward(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        raise NotImplementedError(
            "UnifiedRouterForCausalLM does not implement token-level forward(). "
            "Use generate(prompt=...) with text prompts."
        )

    @torch.no_grad()
    def generate(self, prompt: str, **gen_kwargs: Any) -> str:  # type: ignore[override]
        self._ensure_runtime()
        return self._runtime.generate(prompt=prompt, **gen_kwargs)
'''


MODEL_CARD_TEMPLATE = """---
library_name: transformers
tags:
  - llm-router
  - ensemble
  - unified-llm
license: apache-2.0
---

# {artifact_name}

HF-style UnifiedRouterLLM artifact.

## What this is

This repository is a **single-process router ensemble model package** that
routes prompts across specialist experts and exposes one generation API.

## Important

- Not a single merged dense checkpoint.
- Supports heterogeneous expert tokenizers.
- Use with `trust_remote_code=True`.

## Example

```python
from transformers import AutoModelForCausalLM, AutoConfig

config = AutoConfig.from_pretrained("{artifact_name}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("{artifact_name}", trust_remote_code=True)
text = model.generate("Hello world", max_new_tokens=128)
print(text)
```
"""


def _build_config_json(router_mode: str, expert_count: int) -> Dict[str, Any]:
    return {
        "model_type": "unified_router",
        "architectures": ["UnifiedRouterForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_unified_router.UnifiedRouterConfig",
            "AutoModelForCausalLM": "modeling_unified_router.UnifiedRouterForCausalLM",
        },
        "router_config_path": "router_config.yaml",
        "routing_mode": router_mode,
        "expert_count": expert_count,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.41.0",
    }


def _build_generation_config() -> Dict[str, Any]:
    return {
        "max_new_tokens": 256,
        "temperature": 0.2,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "do_sample": True,
    }


def export_hf_router_artifact(
    router_config_path: str,
    output_dir: str,
    artifact_name: str = "UnifiedRouterLLM",
) -> str:
    """
    Export an HF-style model folder (similar layout to standard model repos)
    for the UnifiedRouterLLM product.
    """
    cfg = load_config(router_config_path)
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    with open(router_config_path, "r", encoding="utf-8") as f:
        raw_router_cfg = yaml.safe_load(f) or {}

    # Normalize dummy classifier path for artifact-local usage.
    if cfg.router.type == "dummy_classifier" and cfg.router.dummy_model_path:
        config_dir = Path(router_config_path).resolve().parent
        src = Path(cfg.router.dummy_model_path)
        if not src.is_absolute():
            config_relative = (config_dir / src).resolve()
            cwd_relative = src.resolve()
            src = config_relative if config_relative.exists() else cwd_relative
        if src.exists():
            shutil.copy2(src, out / "dummy_classifier_model.json")
            raw_router_cfg.setdefault("router", {})
            raw_router_cfg["router"]["dummy_model_path"] = "dummy_classifier_model.json"

    (out / "router_config.yaml").write_text(yaml.safe_dump(raw_router_cfg, sort_keys=False), encoding="utf-8")

    config_json = _build_config_json(router_mode=cfg.unified_llm.mode, expert_count=len(cfg.experts))
    (out / "config.json").write_text(json.dumps(config_json, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out / "generation_config.json").write_text(
        json.dumps(_build_generation_config(), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (out / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "AutoTokenizer",
                "model_max_length": 32768,
                "note": "UnifiedRouterLLM routes text; each expert uses its own tokenizer.",
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "special_tokens_map.json").write_text(json.dumps({}, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (out / "__init__.py").write_text("", encoding="utf-8")
    (out / "configuration_unified_router.py").write_text(CONFIGURATION_PY, encoding="utf-8")
    (out / "modeling_unified_router.py").write_text(MODELING_PY, encoding="utf-8")
    (out / "README.md").write_text(MODEL_CARD_TEMPLATE.format(artifact_name=artifact_name), encoding="utf-8")

    return str(out)
