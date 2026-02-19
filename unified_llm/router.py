"""Routing backends for task classification."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from unified_llm.config import RouterConfig


@dataclass
class RouteDecision:
    label: str
    confidence: float
    raw: Dict[str, Any] = field(default_factory=dict)


class BaseRouter:
    def route(self, prompt: str) -> RouteDecision:
        raise NotImplementedError


def _normalize_route_output(payload: Any) -> RouteDecision:
    if isinstance(payload, RouteDecision):
        return payload
    if isinstance(payload, str):
        return RouteDecision(label=payload, confidence=1.0, raw={"label": payload, "confidence": 1.0})
    if isinstance(payload, dict):
        if "label" not in payload:
            raise ValueError("Router output dict must include key 'label'")
        confidence = float(payload.get("confidence", 1.0))
        return RouteDecision(label=str(payload["label"]), confidence=confidence, raw=dict(payload))
    raise TypeError(f"Unsupported router output type: {type(payload)!r}")


def load_python_callable(path_spec: str) -> Callable[[str], Any]:
    """
    Load a callable from "module:function" path.
    """
    if ":" not in path_spec:
        raise ValueError(f"Expected python callable path 'module:function', got {path_spec!r}")
    module_name, func_name = path_spec.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ValueError(f"Resolved object is not callable: {path_spec!r}")
    return fn


class CallableRouter(BaseRouter):
    def __init__(self, fn: Callable[[str], Any], confidence_threshold: float = 0.0) -> None:
        self.fn = fn
        self.confidence_threshold = confidence_threshold

    def route(self, prompt: str) -> RouteDecision:
        decision = _normalize_route_output(self.fn(prompt))
        if decision.confidence < self.confidence_threshold:
            decision.raw["below_threshold"] = True
        return decision


class PythonModuleRouter(CallableRouter):
    def __init__(self, python_path: str, confidence_threshold: float = 0.0) -> None:
        super().__init__(load_python_callable(python_path), confidence_threshold=confidence_threshold)
        self.python_path = python_path


class LocalModelRouter(BaseRouter):
    """
    Local classifier backend. Expects a Hugging Face text-classification model path.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.0) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        try:
            from transformers import pipeline
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "transformers is required for LocalModelRouter. "
                "Install dependencies and ensure model path is valid."
            ) from exc

        self._pipe = pipeline("text-classification", model=model_path, tokenizer=model_path)

    def route(self, prompt: str) -> RouteDecision:
        output = self._pipe(prompt)
        if isinstance(output, list) and output:
            item = output[0]
        elif isinstance(output, dict):
            item = output
        else:
            raise ValueError(f"Unexpected classifier output from local model router: {output!r}")

        label = str(item.get("label"))
        confidence = float(item.get("score", 1.0))
        decision = RouteDecision(label=label, confidence=confidence, raw={"model_output": item})
        if decision.confidence < self.confidence_threshold:
            decision.raw["below_threshold"] = True
        return decision


def build_router(config: RouterConfig, callable_router: Optional[Callable[[str], Any]] = None) -> BaseRouter:
    if config.type == "python_module":
        return PythonModuleRouter(config.python_module or "", confidence_threshold=config.confidence_threshold)
    if config.type == "local_model":
        return LocalModelRouter(config.local_model or "", confidence_threshold=config.confidence_threshold)
    if config.type == "callable":
        if callable_router is None:
            raise ValueError("callable_router must be provided when router.type=callable")
        return CallableRouter(callable_router, confidence_threshold=config.confidence_threshold)
    raise ValueError(f"Unsupported router type: {config.type!r}")
