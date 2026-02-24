"""
Dynamic single-process multi-expert router for heterogeneous LLMs.

This module is designed for the practical case where:
1) Expert count and task coverage change over time.
2) Experts may use different tokenizers and even different architectures.
3) You want one runtime entrypoint (`generate`) without mandatory retraining.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger("dynamic_llm_router")
logger.setLevel(logging.INFO)


@dataclass
class ClassifierOutput:
    """Classifier decision for routing."""

    label: str
    confidence: float
    label_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExpertSpec:
    """Configuration of a single expert."""

    expert_id: str
    model_path: str
    tokenizer_path: Optional[str] = None
    supported_tasks: List[str] = field(default_factory=list)
    generation_defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingPolicy:
    """
    Routing policy:
    - If classifier confidence >= confidence_threshold, run top-1 expert only.
    - If confidence is lower, use low_confidence_mode.
    """

    task_to_experts: Dict[str, List[str]]
    fallback_experts: List[str]
    confidence_threshold: float = 0.95
    low_confidence_mode: str = "parallel"  # "parallel" or "single"
    max_parallel_experts: int = 3

    def validate(self) -> None:
        if self.low_confidence_mode not in {"parallel", "single"}:
            raise ValueError("low_confidence_mode must be 'parallel' or 'single'")
        if self.max_parallel_experts < 1:
            raise ValueError("max_parallel_experts must be >= 1")


@dataclass
class CandidateResponse:
    expert_id: str
    text: str
    latency_ms: float
    error: Optional[str] = None


@dataclass
class RouteDecision:
    request_id: str
    task_label: str
    confidence: float
    mode: str  # "single" or "parallel"
    candidate_expert_ids: List[str]


class ResponseSelector(Protocol):
    """
    Decides the final response if multiple experts were executed.
    """

    def select(self, decision: RouteDecision, candidates: List[CandidateResponse]) -> CandidateResponse:
        ...


class FirstSuccessfulSelector:
    """
    Stable selector:
    - Returns the first successful candidate by router order.
    - Useful when task->expert order is curated from offline evals.
    """

    def select(self, decision: RouteDecision, candidates: List[CandidateResponse]) -> CandidateResponse:
        by_id = {c.expert_id: c for c in candidates}
        for expert_id in decision.candidate_expert_ids:
            candidate = by_id.get(expert_id)
            if candidate and not candidate.error and candidate.text.strip():
                return candidate
        errors = [f"{c.expert_id}: {c.error}" for c in candidates if c.error]
        raise RuntimeError(f"No successful candidate found. Errors={errors}")


class HeuristicQualitySelector:
    """
    Lightweight quality selector that avoids extra training:
    - Rejects empty or errored outputs.
    - Scores candidates by non-repetition and useful length.
    """

    @staticmethod
    def _score(text: str) -> float:
        stripped = text.strip()
        if not stripped:
            return -1e9
        words = stripped.split()
        unique_ratio = len(set(words)) / max(1, len(words))
        length_bonus = min(len(words), 400) / 400.0
        return unique_ratio + length_bonus

    def select(self, decision: RouteDecision, candidates: List[CandidateResponse]) -> CandidateResponse:
        valid = [c for c in candidates if not c.error and c.text.strip()]
        if not valid:
            errors = [f"{c.expert_id}: {c.error}" for c in candidates if c.error]
            raise RuntimeError(f"No valid candidates to select. Errors={errors}")

        ranked = sorted(valid, key=lambda c: self._score(c.text), reverse=True)
        return ranked[0]


class ExpertRuntime:
    """
    Expert runtime with native model + tokenizer.
    This allows heterogeneous tokenizers without forcing one shared vocab.
    """

    def __init__(
        self,
        spec: ExpertSpec,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> None:
        self.spec = spec
        tok_path = spec.tokenizer_path or spec.model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, generation_overrides: Optional[Dict[str, Any]] = None) -> CandidateResponse:
        started = time.time()
        try:
            generate_kwargs = dict(self.spec.generation_defaults)
            if generation_overrides:
                generate_kwargs.update(generation_overrides)

            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            out_ids = self.model.generate(**encoded, **generate_kwargs)
            text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            return CandidateResponse(
                expert_id=self.spec.expert_id,
                text=text,
                latency_ms=1000.0 * (time.time() - started),
            )
        except Exception as exc:  # noqa: BLE001
            return CandidateResponse(
                expert_id=self.spec.expert_id,
                text="",
                latency_ms=1000.0 * (time.time() - started),
                error=str(exc),
            )


class DynamicRoutedLLM:
    """
    One-entrypoint router that supports:
    - Dynamic task->experts mappings
    - Dynamic expert count
    - Heterogeneous tokenizers/architectures
    """

    def __init__(
        self,
        expert_specs: Iterable[ExpertSpec],
        policy: RoutingPolicy,
        classifier_fn: Optional[Callable[[str], ClassifierOutput]] = None,
        selector: Optional[ResponseSelector] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> None:
        policy.validate()
        self.classifier_fn = classifier_fn
        self.selector = selector or FirstSuccessfulSelector()
        self.policy = policy
        self.torch_dtype = torch_dtype
        self.device_map = device_map

        self._specs: Dict[str, ExpertSpec] = {}
        self._runtimes: Dict[str, ExpertRuntime] = {}

        for spec in expert_specs:
            if spec.expert_id in self._specs:
                raise ValueError(f"Duplicate expert_id found: {spec.expert_id}")
            self._specs[spec.expert_id] = spec

        self._validate_policy_refs()

    @classmethod
    def from_json_config(
        cls,
        path: str,
        classifier_fn: Optional[Callable[[str], ClassifierOutput]] = None,
        selector: Optional[ResponseSelector] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> "DynamicRoutedLLM":
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        experts = [ExpertSpec(**item) for item in cfg["experts"]]
        policy = RoutingPolicy(**cfg["policy"])
        return cls(
            expert_specs=experts,
            policy=policy,
            classifier_fn=classifier_fn,
            selector=selector,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    def _validate_policy_refs(self) -> None:
        known = set(self._specs.keys())
        for task, ids in self.policy.task_to_experts.items():
            for expert_id in ids:
                if expert_id not in known:
                    raise ValueError(f"Policy task '{task}' references unknown expert '{expert_id}'")
        for expert_id in self.policy.fallback_experts:
            if expert_id not in known:
                raise ValueError(f"Policy fallback references unknown expert '{expert_id}'")

    def _runtime(self, expert_id: str) -> ExpertRuntime:
        runtime = self._runtimes.get(expert_id)
        if runtime is None:
            runtime = ExpertRuntime(
                self._specs[expert_id],
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
            )
            self._runtimes[expert_id] = runtime
        return runtime

    def _route(self, clf: ClassifierOutput, request_id: str) -> RouteDecision:
        ordered = self.policy.task_to_experts.get(clf.label, [])
        if not ordered:
            ordered = list(self.policy.fallback_experts)
        if not ordered:
            raise RuntimeError(f"No route configured for label='{clf.label}' and no fallbacks provided.")

        if clf.confidence >= self.policy.confidence_threshold:
            mode = "single"
            chosen = [ordered[0]]
        elif self.policy.low_confidence_mode == "single":
            mode = "single"
            chosen = [ordered[0]]
        else:
            mode = "parallel"
            chosen = ordered[: self.policy.max_parallel_experts]

        return RouteDecision(
            request_id=request_id,
            task_label=clf.label,
            confidence=clf.confidence,
            mode=mode,
            candidate_expert_ids=chosen,
        )

    def _log_event(self, payload: Dict[str, Any]) -> None:
        logger.info(json.dumps(payload, ensure_ascii=True))

    def _run_single(
        self,
        expert_id: str,
        prompt: str,
        generation_overrides: Optional[Dict[str, Any]],
    ) -> CandidateResponse:
        runtime = self._runtime(expert_id)
        return runtime.generate(prompt, generation_overrides=generation_overrides)

    def _run_parallel(
        self,
        expert_ids: List[str],
        prompt: str,
        generation_overrides: Optional[Dict[str, Any]],
    ) -> List[CandidateResponse]:
        results: List[CandidateResponse] = []
        with ThreadPoolExecutor(max_workers=len(expert_ids)) as pool:
            futures = {
                pool.submit(self._run_single, expert_id, prompt, generation_overrides): expert_id
                for expert_id in expert_ids
            }
            for future in as_completed(futures):
                results.append(future.result())
        return results

    def generate(
        self,
        prompt: str,
        classifier_output: Optional[ClassifierOutput] = None,
        request_id: str = "unknown_request",
        generation_overrides: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ) -> Any:
        """
        Unified generate endpoint:
        - Accepts external classifier output (recommended for production).
        - Or uses classifier_fn if configured.
        """

        if classifier_output is None:
            if self.classifier_fn is None:
                raise ValueError("classifier_output is required when classifier_fn is not configured.")
            classifier_output = self.classifier_fn(prompt)

        decision = self._route(classifier_output, request_id=request_id)
        self._log_event(
            {
                "event": "routing_decision",
                "request_id": request_id,
                "label": classifier_output.label,
                "confidence": float(classifier_output.confidence),
                "mode": decision.mode,
                "candidates": decision.candidate_expert_ids,
            }
        )

        if decision.mode == "single":
            candidate = self._run_single(decision.candidate_expert_ids[0], prompt, generation_overrides)
            selected = candidate
            candidates = [candidate]
        else:
            candidates = self._run_parallel(decision.candidate_expert_ids, prompt, generation_overrides)
            selected = self.selector.select(decision, candidates)

        self._log_event(
            {
                "event": "routing_result",
                "request_id": request_id,
                "selected_expert": selected.expert_id,
                "selected_latency_ms": selected.latency_ms,
                "candidate_count": len(candidates),
                "candidate_errors": {c.expert_id: c.error for c in candidates if c.error},
            }
        )

        if return_metadata:
            return {
                "text": selected.text,
                "selected_expert": selected.expert_id,
                "decision": decision,
                "candidates": candidates,
            }
        return selected.text


def build_static_classifier(label: str, confidence: float = 1.0) -> Callable[[str], ClassifierOutput]:
    """
    Helper for testing/integration before wiring a real classifier.
    """

    def _fn(_: str) -> ClassifierOutput:
        return ClassifierOutput(label=label, confidence=confidence, label_scores={label: confidence})

    return _fn

