"""Example rule-based router callable used by examples/config.yaml."""

from __future__ import annotations

from typing import Dict


def route(prompt: str) -> Dict[str, object]:
    low = prompt.lower()
    if any(k in low for k in ["sql", "analytics", "forecast"]):
        return {"label": "A", "confidence": 1.0}
    if any(k in low for k in ["code", "bug", "debug", "stacktrace"]):
        return {"label": "B", "confidence": 1.0}
    if any(k in low for k in ["reason", "math", "logic"]):
        return {"label": "C", "confidence": 1.0}
    if any(k in low for k in ["ops", "deployment", "infra"]):
        return {"label": "D", "confidence": 1.0}
    if any(k in low for k in ["policy", "compliance", "legal"]):
        return {"label": "E", "confidence": 1.0}
    return {"label": "F", "confidence": 1.0}
