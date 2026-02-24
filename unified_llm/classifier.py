"""Dummy classifier model used for integer expert routing labels."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DummyRule:
    index: int
    keywords: List[str] = field(default_factory=list)


@dataclass
class DummyClassifierConfig:
    rules: List[DummyRule] = field(default_factory=list)
    default_index: int = 0
    confidence: float = 1.0
    num_experts_hint: int = 1


class DummyClassifierModel:
    """
    Lightweight rule-based classifier model that returns integer labels:
    "0", "1", "2", ...
    """

    def __init__(self, config: DummyClassifierConfig) -> None:
        self.config = config

    @classmethod
    def from_json(cls, path: str | Path, num_experts_hint: Optional[int] = None) -> "DummyClassifierModel":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        rules = []
        for item in raw.get("rules", []):
            rules.append(DummyRule(index=int(item["index"]), keywords=[str(x) for x in item.get("keywords", [])]))

        cfg = DummyClassifierConfig(
            rules=rules,
            default_index=int(raw.get("default_index", 0)),
            confidence=float(raw.get("confidence", 1.0)),
            num_experts_hint=int(raw.get("num_experts_hint", num_experts_hint or 1)),
        )
        if num_experts_hint is not None:
            cfg.num_experts_hint = int(num_experts_hint)
        return cls(cfg)

    @classmethod
    def default(cls, num_experts_hint: int = 1) -> "DummyClassifierModel":
        cfg = DummyClassifierConfig(
            rules=[
                DummyRule(index=0, keywords=["general", "overview", "summary"]),
                DummyRule(index=1, keywords=["code", "debug", "stacktrace", "bug"]),
                DummyRule(index=2, keywords=["math", "reasoning", "proof"]),
                DummyRule(index=3, keywords=["policy", "legal", "compliance", "safety"]),
            ],
            default_index=0,
            confidence=1.0,
            num_experts_hint=max(1, int(num_experts_hint)),
        )
        return cls(cfg)

    def predict(self, prompt: str) -> Dict[str, object]:
        low = prompt.lower()
        for rule in self.config.rules:
            for kw in rule.keywords:
                if kw.lower() in low:
                    return {"label": str(rule.index), "confidence": float(self.config.confidence)}

        hashed = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        hashed_int = int(hashed[:8], 16)
        upper = max(1, int(self.config.num_experts_hint))
        fallback_idx = hashed_int % upper

        if self.config.default_index >= 0:
            fallback_idx = min(self.config.default_index, upper - 1)

        return {"label": str(fallback_idx), "confidence": float(self.config.confidence)}

    def save_json(self, path: str | Path) -> None:
        payload = {
            "rules": [{"index": r.index, "keywords": r.keywords} for r in self.config.rules],
            "default_index": self.config.default_index,
            "confidence": self.config.confidence,
            "num_experts_hint": self.config.num_experts_hint,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
