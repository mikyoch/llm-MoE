from __future__ import annotations

from pathlib import Path

import pytest

from unified_llm.config import AppConfig, load_config


def test_load_config_from_yaml(tmp_path: Path) -> None:
    config_text = """
unified_llm:
  mode: route_top1
  max_parallel: 2
  seed: 1
  logging:
    level: INFO
    json: true
router:
  type: python_module
  python_module: "tests.mock_router_module:route"
  confidence_threshold: 0.0
experts:
  - name: e1
    hf_model_id: model/e1
    tasks: [A]
  - name: e2
    hf_model_id: model/e2
    tasks: [B]
selection:
  judge:
    type: heuristic_judge
"""
    path = tmp_path / "cfg.yaml"
    path.write_text(config_text, encoding="utf-8")

    cfg = load_config(path)
    assert cfg.unified_llm.mode == "route_top1"
    assert cfg.unified_llm.max_parallel == 2
    assert len(cfg.experts) == 2
    assert cfg.task_to_experts()["A"] == ["e1"]


def test_invalid_duplicate_expert_name() -> None:
    raw = {
        "unified_llm": {"mode": "route_top1", "max_parallel": 1},
        "router": {"type": "python_module", "python_module": "tests.mock_router_module:route"},
        "experts": [
            {"name": "dup", "hf_model_id": "x", "tasks": ["A"]},
            {"name": "dup", "hf_model_id": "y", "tasks": ["B"]},
        ],
    }
    with pytest.raises(ValueError, match="Duplicate expert"):
        AppConfig.from_dict(raw)
