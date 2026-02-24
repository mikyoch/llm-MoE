"""UnifiedRouterLLM core orchestration runtime."""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from unified_llm.config import AppConfig, LoggingConfig, load_config
from unified_llm.experts import ExpertPool, ExpertResult
from unified_llm.judges import BaseJudge, HeuristicJudge, JudgeResult, LLMJudge
from unified_llm.router import BaseRouter, RouteDecision, build_router

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None


LOGGER = logging.getLogger("unified_llm")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(payload, ensure_ascii=True)


def setup_logging(config: LoggingConfig) -> None:
    level = getattr(logging, config.level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        root.addHandler(handler)
    else:
        handler = root.handlers[0]
    if config.json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@dataclass
class GenerationMetadata:
    request_id: str
    mode: str
    label: str
    confidence: float
    selected_expert: str
    judge_type: str
    judge_rationale: str
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    candidate_scores: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    candidates: List[str] = field(default_factory=list)
    fallback_used: bool = False


class UnifiedRouterLLM:
    """
    Public entrypoint for single-process multi-expert LLM orchestration.
    """

    def __init__(
        self,
        config: AppConfig,
        router: BaseRouter,
        expert_pool: ExpertPool,
        judge: BaseJudge,
    ) -> None:
        self.config = config
        self.router = router
        self.expert_pool = expert_pool
        self.judge = judge
        self.heuristic_judge = HeuristicJudge()

    @classmethod
    def from_config(
        cls,
        config_path: str,
        router_callable: Optional[Any] = None,
    ) -> "UnifiedRouterLLM":
        config = load_config(config_path)
        config_dir = Path(config_path).resolve().parent
        if config.router.dummy_model_path:
            config.router.dummy_model_path = cls._resolve_reference_path(config.router.dummy_model_path, config_dir)
        if config.router.local_model:
            config.router.local_model = cls._resolve_reference_path(config.router.local_model, config_dir)
        setup_logging(config.unified_llm.logging)
        set_global_seed(config.unified_llm.seed)

        router = build_router(
            config.router,
            callable_router=router_callable,
            num_experts_hint=len(config.experts),
        )
        expert_pool = ExpertPool(config.experts)
        if config.selection.judge.type == "llm_judge":
            judge = LLMJudge(
                judge_expert_name=config.selection.judge.llm_judge_expert or "",
                expert_pool=expert_pool,
                rubric=config.selection.judge.rubric,
                fallback_judge=HeuristicJudge(),
            )
        else:
            judge = HeuristicJudge()
        return cls(config=config, router=router, expert_pool=expert_pool, judge=judge)

    @staticmethod
    def _resolve_reference_path(raw_path: str, config_dir: Path) -> str:
        path_obj = Path(raw_path)
        if path_obj.is_absolute():
            return str(path_obj)
        config_relative = (config_dir / path_obj).resolve()
        if config_relative.exists():
            return str(config_relative)
        cwd_relative = path_obj.resolve()
        if cwd_relative.exists():
            return str(cwd_relative)
        return str(config_relative)

    def generate(
        self,
        prompt: str,
        return_metadata: bool = False,
        mode: Optional[str] = None,
        request_id: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> Any:
        req_id = request_id or str(uuid.uuid4())
        route = self.router.route(prompt)
        selected_mode = mode or self.config.unified_llm.mode

        self._log_event(
            {
                "event": "route",
                "request_id": req_id,
                "label": route.label,
                "confidence": route.confidence,
                "mode": selected_mode,
            }
        )

        if selected_mode == "route_top1":
            text, meta = self._mode_route_top1(prompt, route, req_id, gen_kwargs)
        elif selected_mode == "run_all_select":
            text, meta = self._mode_run_all_select(prompt, route, req_id, gen_kwargs)
        elif selected_mode == "route_topk_then_judge":
            text, meta = self._mode_route_topk_then_judge(prompt, route, req_id, gen_kwargs)
        elif selected_mode == "cascade_refine":
            text, meta = self._mode_cascade_refine(prompt, route, req_id, gen_kwargs)
        else:
            raise ValueError(f"Unsupported mode={selected_mode!r}")

        if return_metadata:
            return {"text": text, "metadata": asdict(meta)}
        return text

    def generate_batch(
        self,
        prompts: List[str],
        return_metadata: bool = False,
        mode: Optional[str] = None,
        **gen_kwargs: Any,
    ) -> List[Any]:
        outputs: List[Any] = []
        for prompt in prompts:
            outputs.append(
                self.generate(
                    prompt=prompt,
                    return_metadata=return_metadata,
                    mode=mode,
                    **gen_kwargs,
                )
            )
        return outputs

    def stream(self, prompt: str, mode: Optional[str] = None, **gen_kwargs: Any) -> Iterator[str]:
        """
        Best-effort stream: yields word chunks from final selected text.
        """
        text = self.generate(prompt, mode=mode, **gen_kwargs)
        parts = text.split(" ")
        for idx, part in enumerate(parts):
            if idx == len(parts) - 1:
                yield part
            else:
                yield part + " "

    def _mode_route_top1(
        self,
        prompt: str,
        route: RouteDecision,
        request_id: str,
        gen_kwargs: Dict[str, Any],
    ) -> Tuple[str, GenerationMetadata]:
        candidates = self._experts_for_label(route.label)
        if not candidates:
            candidates = self.expert_pool.all_expert_names()
        chosen = candidates[0]

        result = self.expert_pool.generate_one(chosen, prompt, generation_overrides=gen_kwargs)
        self._log_candidate_results(request_id, [result])

        if result.error:
            remaining = [name for name in self.expert_pool.all_expert_names() if name != chosen]
            if not remaining:
                raise RuntimeError(f"Top-1 expert failed and no fallback available: {result.error}")
            text, meta = self._run_and_judge(
                prompt=prompt,
                route=route,
                request_id=request_id,
                candidate_experts=remaining,
                gen_kwargs=gen_kwargs,
                mode_name="route_top1_fallback_run_all_select",
                fallback_used=True,
            )
            return text, meta

        meta = GenerationMetadata(
            request_id=request_id,
            mode="route_top1",
            label=route.label,
            confidence=route.confidence,
            selected_expert=result.expert_name,
            judge_type="none",
            judge_rationale="top1 routing selected single expert",
            latencies_ms={result.expert_name: result.latency_ms},
            candidate_scores={result.expert_name: 1.0},
            errors={},
            candidates=[result.expert_name],
            fallback_used=False,
        )
        return result.text, meta

    def _mode_run_all_select(
        self,
        prompt: str,
        route: RouteDecision,
        request_id: str,
        gen_kwargs: Dict[str, Any],
    ) -> Tuple[str, GenerationMetadata]:
        return self._run_and_judge(
            prompt=prompt,
            route=route,
            request_id=request_id,
            candidate_experts=self.expert_pool.all_expert_names(),
            gen_kwargs=gen_kwargs,
            mode_name="run_all_select",
            fallback_used=False,
        )

    def _mode_route_topk_then_judge(
        self,
        prompt: str,
        route: RouteDecision,
        request_id: str,
        gen_kwargs: Dict[str, Any],
    ) -> Tuple[str, GenerationMetadata]:
        per_task = self.config.selection.topk.per_task.get(route.label, [])
        candidates = list(per_task) if per_task else self._experts_for_label(route.label)
        if self.config.selection.topk.include_generalist and self.config.selection.topk.generalist_expert:
            candidates.append(self.config.selection.topk.generalist_expert)
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            candidates = self.expert_pool.all_expert_names()

        return self._run_and_judge(
            prompt=prompt,
            route=route,
            request_id=request_id,
            candidate_experts=candidates,
            gen_kwargs=gen_kwargs,
            mode_name="route_topk_then_judge",
            fallback_used=False,
        )

    def _mode_cascade_refine(
        self,
        prompt: str,
        route: RouteDecision,
        request_id: str,
        gen_kwargs: Dict[str, Any],
    ) -> Tuple[str, GenerationMetadata]:
        if not self.config.refine.enabled:
            return self._mode_route_top1(prompt, route, request_id, gen_kwargs)
        if route.label not in self.config.refine.per_task:
            return self._mode_route_top1(prompt, route, request_id, gen_kwargs)

        draft_expert, refiner_expert = self.config.refine.per_task[route.label]
        draft = self.expert_pool.generate_one(draft_expert, prompt, generation_overrides=gen_kwargs)
        self._log_candidate_results(request_id, [draft])
        if draft.error:
            return self._run_and_judge(
                prompt=prompt,
                route=route,
                request_id=request_id,
                candidate_experts=[name for name in self.expert_pool.all_expert_names() if name != draft_expert],
                gen_kwargs=gen_kwargs,
                mode_name="cascade_refine_fallback",
                fallback_used=True,
            )

        refine_prompt = self._build_refine_prompt(prompt, draft.text)
        refined = self.expert_pool.generate_one(refiner_expert, refine_prompt, generation_overrides=gen_kwargs)
        self._log_candidate_results(request_id, [refined])
        if refined.error:
            meta = GenerationMetadata(
                request_id=request_id,
                mode="cascade_refine",
                label=route.label,
                confidence=route.confidence,
                selected_expert=draft.expert_name,
                judge_type="none",
                judge_rationale="refiner failed; returning draft",
                latencies_ms={
                    draft.expert_name: draft.latency_ms,
                    refined.expert_name: refined.latency_ms,
                },
                candidate_scores={draft.expert_name: 1.0},
                errors={refined.expert_name: refined.error or "unknown_refiner_error"},
                candidates=[draft.expert_name, refined.expert_name],
            )
            return draft.text, meta

        if self.config.refine.always_take_refined:
            meta = GenerationMetadata(
                request_id=request_id,
                mode="cascade_refine",
                label=route.label,
                confidence=route.confidence,
                selected_expert=refined.expert_name,
                judge_type="none",
                judge_rationale="always_take_refined=true",
                latencies_ms={
                    draft.expert_name: draft.latency_ms,
                    refined.expert_name: refined.latency_ms,
                },
                candidate_scores={refined.expert_name: 1.0, draft.expert_name: 0.0},
                errors={},
                candidates=[draft.expert_name, refined.expert_name],
            )
            return refined.text, meta

        # Judge between draft and refined text.
        draft_named = ExpertResult(
            expert_name=f"draft:{draft.expert_name}",
            text=draft.text,
            latency_ms=draft.latency_ms,
            error=draft.error,
        )
        refined_named = ExpertResult(
            expert_name=f"refined:{refined.expert_name}",
            text=refined.text,
            latency_ms=refined.latency_ms,
            error=refined.error,
        )
        judge_result = self.judge.evaluate(prompt, [draft_named, refined_named], requirements=None)
        selected = draft_named if judge_result.winner == draft_named.expert_name else refined_named
        text = selected.text

        meta = GenerationMetadata(
            request_id=request_id,
            mode="cascade_refine",
            label=route.label,
            confidence=route.confidence,
            selected_expert=selected.expert_name,
            judge_type=judge_result.judge_type,
            judge_rationale=judge_result.rationale,
            latencies_ms={
                draft_named.expert_name: draft_named.latency_ms,
                refined_named.expert_name: refined_named.latency_ms,
            },
            candidate_scores=judge_result.scores,
            errors={},
            candidates=[draft_named.expert_name, refined_named.expert_name],
        )
        self._log_event(
            {
                "event": "judge_decision",
                "request_id": request_id,
                "winner": selected.expert_name,
                "scores": judge_result.scores,
                "judge_type": judge_result.judge_type,
            }
        )
        return text, meta

    def _run_and_judge(
        self,
        prompt: str,
        route: RouteDecision,
        request_id: str,
        candidate_experts: List[str],
        gen_kwargs: Dict[str, Any],
        mode_name: str,
        fallback_used: bool,
    ) -> Tuple[str, GenerationMetadata]:
        results = self.expert_pool.generate_many(
            candidate_experts,
            prompt,
            max_parallel=self.config.unified_llm.max_parallel,
            generation_overrides=gen_kwargs,
        )
        self._log_candidate_results(request_id, results)
        valid = [item for item in results if not item.error and item.text.strip()]
        if not valid:
            errors = {item.expert_name: (item.error or "empty_output") for item in results}
            raise RuntimeError(f"No valid expert outputs. Errors={errors}")

        judge_result = self.judge.evaluate(prompt, results, requirements=None)
        selected = self._find_by_name(valid, judge_result.winner)
        if selected is None:
            # Robust fallback if judge returns an unknown winner.
            fallback = self.heuristic_judge.evaluate(prompt, results, requirements=None)
            selected = self._find_by_name(valid, fallback.winner) or valid[0]
            judge_result = fallback

        meta = GenerationMetadata(
            request_id=request_id,
            mode=mode_name,
            label=route.label,
            confidence=route.confidence,
            selected_expert=selected.expert_name,
            judge_type=judge_result.judge_type,
            judge_rationale=judge_result.rationale,
            latencies_ms={item.expert_name: item.latency_ms for item in results},
            candidate_scores=judge_result.scores,
            errors={item.expert_name: item.error for item in results if item.error},
            candidates=[item.expert_name for item in results],
            fallback_used=fallback_used,
        )
        self._log_event(
            {
                "event": "judge_decision",
                "request_id": request_id,
                "winner": selected.expert_name,
                "scores": judge_result.scores,
                "judge_type": judge_result.judge_type,
            }
        )
        return selected.text, meta

    def _build_refine_prompt(self, original_prompt: str, draft_answer: str) -> str:
        return (
            "You are a response refiner.\n"
            "Improve the draft answer for correctness, instruction-following, and clarity.\n"
            "Preserve facts unless clearly wrong, and do not add unsupported claims.\n\n"
            f"Original user prompt:\n{original_prompt}\n\n"
            f"Draft answer:\n{draft_answer}\n\n"
            "Return only the improved final answer."
        )

    def _experts_for_label(self, label: str) -> List[str]:
        # Integer labels (0,1,2,...) are interpreted as expert index routing.
        idx = self._maybe_parse_index(label)
        if idx is not None:
            index_map = self.config.expert_name_by_index()
            if idx in index_map:
                return [index_map[idx]]

        names = self.expert_pool.experts_for_task(label)
        if not names:
            names = self.expert_pool.all_expert_names()
        return names

    def _maybe_parse_index(self, label: str) -> Optional[int]:
        stripped = str(label).strip()
        if stripped.isdigit():
            return int(stripped)
        return None

    def _find_by_name(self, items: Iterable[ExpertResult], name: str) -> Optional[ExpertResult]:
        for item in items:
            if item.expert_name == name:
                return item
        return None

    def _log_candidate_results(self, request_id: str, results: List[ExpertResult]) -> None:
        for item in results:
            self._log_event(
                {
                    "event": "expert_result",
                    "request_id": request_id,
                    "expert": item.expert_name,
                    "latency_ms": item.latency_ms,
                    "error": item.error,
                }
            )

    def _log_event(self, payload: Dict[str, Any]) -> None:
        LOGGER.info(json.dumps(payload, ensure_ascii=True))
