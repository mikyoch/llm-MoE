"""
Single-process dynamic router for one base model with multiple LoRA adapters.

Use this when you want one base checkpoint plus task-specialized adapters,
and route by classifier output without running separate services.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger("dynamic_adapter_router")
logger.setLevel(logging.INFO)


@dataclass
class ClassifierOutput:
    label: str
    confidence: float
    label_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class AdapterSpec:
    adapter_id: str
    adapter_path: str
    supported_tasks: List[str] = field(default_factory=list)
    generation_defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterRoutingPolicy:
    task_to_adapters: Dict[str, List[str]]
    fallback_adapters: List[str]
    confidence_threshold: float = 0.95
    low_confidence_mode: str = "first_fallback"  # "first_fallback" or "same_route"

    def validate(self) -> None:
        if self.low_confidence_mode not in {"first_fallback", "same_route"}:
            raise ValueError("low_confidence_mode must be 'first_fallback' or 'same_route'")


class DynamicAdapterRoutedLLM:
    """
    Dynamic adapter router with a single generate entrypoint.
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_specs: Iterable[AdapterSpec],
        policy: AdapterRoutingPolicy,
        tokenizer_path: Optional[str] = None,
        classifier_fn: Optional[Callable[[str], ClassifierOutput]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> None:
        policy.validate()
        self.classifier_fn = classifier_fn
        self.policy = policy
        self.adapter_specs: Dict[str, AdapterSpec] = {}
        for spec in adapter_specs:
            if spec.adapter_id in self.adapter_specs:
                raise ValueError(f"Duplicate adapter_id found: {spec.adapter_id}")
            self.adapter_specs[spec.adapter_id] = spec

        self._validate_policy_refs()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

        ids = list(self.adapter_specs.keys())
        if not ids:
            raise ValueError("At least one adapter is required.")

        first_id = ids[0]
        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_specs[first_id].adapter_path,
            adapter_name=first_id,
        )
        for adapter_id in ids[1:]:
            self.model.load_adapter(self.adapter_specs[adapter_id].adapter_path, adapter_name=adapter_id)

        self.model.eval()
        self._lock = threading.Lock()

    @classmethod
    def from_json_config(
        cls,
        path: str,
        classifier_fn: Optional[Callable[[str], ClassifierOutput]] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> "DynamicAdapterRoutedLLM":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        adapter_specs = [AdapterSpec(**item) for item in cfg["adapters"]]
        policy = AdapterRoutingPolicy(**cfg["policy"])
        return cls(
            base_model_path=cfg["base_model_path"],
            adapter_specs=adapter_specs,
            policy=policy,
            tokenizer_path=cfg.get("tokenizer_path"),
            classifier_fn=classifier_fn,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    def _validate_policy_refs(self) -> None:
        known = set(self.adapter_specs.keys())
        for task, ids in self.policy.task_to_adapters.items():
            for adapter_id in ids:
                if adapter_id not in known:
                    raise ValueError(f"Task '{task}' references unknown adapter '{adapter_id}'")
        for adapter_id in self.policy.fallback_adapters:
            if adapter_id not in known:
                raise ValueError(f"Fallback references unknown adapter '{adapter_id}'")

    def _route(self, clf: ClassifierOutput) -> str:
        ranked = self.policy.task_to_adapters.get(clf.label, [])
        if not ranked:
            ranked = list(self.policy.fallback_adapters)
        if not ranked:
            raise RuntimeError(f"No route configured for label='{clf.label}' and no fallback_adapters.")

        if clf.confidence >= self.policy.confidence_threshold:
            return ranked[0]
        if self.policy.low_confidence_mode == "same_route":
            return ranked[0]
        return self.policy.fallback_adapters[0]

    def _log_event(self, payload: Dict[str, Any]) -> None:
        logger.info(json.dumps(payload, ensure_ascii=True))

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        classifier_output: Optional[ClassifierOutput] = None,
        request_id: str = "unknown_request",
        generation_overrides: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ) -> Any:
        if classifier_output is None:
            if self.classifier_fn is None:
                raise ValueError("classifier_output is required when classifier_fn is not configured.")
            classifier_output = self.classifier_fn(prompt)

        adapter_id = self._route(classifier_output)
        defaults = dict(self.adapter_specs[adapter_id].generation_defaults)
        if generation_overrides:
            defaults.update(generation_overrides)

        self._log_event(
            {
                "event": "routing_decision",
                "request_id": request_id,
                "label": classifier_output.label,
                "confidence": float(classifier_output.confidence),
                "selected_adapter": adapter_id,
            }
        )

        started = time.time()
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Adapter switching mutates model state, so guard it for multi-threaded servers.
        with self._lock:
            self.model.set_adapter(adapter_id)
            output_ids = self.model.generate(**encoded, **defaults)

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        latency_ms = 1000.0 * (time.time() - started)

        self._log_event(
            {
                "event": "routing_result",
                "request_id": request_id,
                "selected_adapter": adapter_id,
                "latency_ms": latency_ms,
            }
        )

        if return_metadata:
            return {
                "text": text,
                "selected_adapter": adapter_id,
                "latency_ms": latency_ms,
            }
        return text


def build_static_classifier(label: str, confidence: float = 1.0) -> Callable[[str], ClassifierOutput]:
    def _fn(_: str) -> ClassifierOutput:
        return ClassifierOutput(label=label, confidence=confidence, label_scores={label: confidence})

    return _fn

