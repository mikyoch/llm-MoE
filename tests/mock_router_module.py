"""Mock router module for unit tests."""

from __future__ import annotations


def route(prompt: str):  # noqa: ANN001
    if "alpha" in prompt.lower():
        return {"label": "A", "confidence": 1.0}
    return {"label": "B", "confidence": 0.9}
