"""Configuration schema and loader for UnifiedRouterLLM."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


VALID_MODES = {
    "route_top1",
    "run_all_select",
    "route_topk_then_judge",
    "cascade_refine",
}
VALID_ROUTER_TYPES = {"python_module", "local_model", "callable", "dummy_classifier"}
VALID_JUDGE_TYPES = {"heuristic_judge", "llm_judge"}
VALID_LOAD_STRATEGIES = {"eager", "lazy"}


@dataclass
class LoggingConfig:
    level: str = "INFO"
    json: bool = True


@dataclass
class UnifiedLLMSection:
    mode: str = "route_top1"
    max_parallel: int = 1
    seed: int = 1234
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class RouterConfig:
    type: str = "python_module"
    python_module: Optional[str] = None
    local_model: Optional[str] = None
    dummy_model_path: Optional[str] = None
    dummy_num_experts_hint: Optional[int] = None
    confidence_threshold: float = 0.0


@dataclass
class ExpertConfig:
    name: str
    hf_model_id: str
    index: Optional[int] = None
    tasks: List[str] = field(default_factory=list)
    dtype: str = "bfloat16"
    device: str = "auto"
    load_strategy: str = "lazy"
    trust_remote_code: bool = False
    offload_to_cpu: bool = False
    gen_defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JudgeConfig:
    type: str = "heuristic_judge"
    llm_judge_expert: Optional[str] = None
    rubric: str = "accuracy, instruction_following, safety, format, concision"


@dataclass
class TopKConfig:
    include_generalist: bool = False
    generalist_expert: Optional[str] = None
    per_task: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SelectionConfig:
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    topk: TopKConfig = field(default_factory=TopKConfig)


@dataclass
class RefineConfig:
    enabled: bool = False
    per_task: Dict[str, List[str]] = field(default_factory=dict)
    always_take_refined: bool = False


@dataclass
class AppConfig:
    unified_llm: UnifiedLLMSection
    router: RouterConfig
    experts: List[ExpertConfig]
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "AppConfig":
        if "unified_llm" not in raw:
            raise ValueError("Missing required section: unified_llm")
        if "router" not in raw:
            raise ValueError("Missing required section: router")
        if "experts" not in raw:
            raise ValueError("Missing required section: experts")

        unified_raw = raw.get("unified_llm", {})
        logging_raw = unified_raw.get("logging", {})
        unified = UnifiedLLMSection(
            mode=unified_raw.get("mode", "route_top1"),
            max_parallel=int(unified_raw.get("max_parallel", 1)),
            seed=int(unified_raw.get("seed", 1234)),
            logging=LoggingConfig(
                level=logging_raw.get("level", "INFO"),
                json=bool(logging_raw.get("json", True)),
            ),
        )

        router_raw = raw.get("router", {})
        router = RouterConfig(
            type=router_raw.get("type", "python_module"),
            python_module=router_raw.get("python_module"),
            local_model=router_raw.get("local_model"),
            dummy_model_path=router_raw.get("dummy_model_path"),
            dummy_num_experts_hint=router_raw.get("dummy_num_experts_hint"),
            confidence_threshold=float(router_raw.get("confidence_threshold", 0.0)),
        )

        experts = []
        for item in raw.get("experts", []):
            experts.append(
                ExpertConfig(
                    name=item["name"],
                    hf_model_id=item["hf_model_id"],
                    index=item.get("index"),
                    tasks=list(item.get("tasks", [])),
                    dtype=item.get("dtype", "bfloat16"),
                    device=item.get("device", "auto"),
                    load_strategy=item.get("load_strategy", "lazy"),
                    trust_remote_code=bool(item.get("trust_remote_code", False)),
                    offload_to_cpu=bool(item.get("offload_to_cpu", False)),
                    gen_defaults=dict(item.get("gen_defaults", {})),
                )
            )

        selection_raw = raw.get("selection", {})
        judge_raw = selection_raw.get("judge", {})
        topk_raw = selection_raw.get("topk", {})
        selection = SelectionConfig(
            judge=JudgeConfig(
                type=judge_raw.get("type", "heuristic_judge"),
                llm_judge_expert=judge_raw.get("llm_judge_expert"),
                rubric=judge_raw.get("rubric", "accuracy, instruction_following, safety, format, concision"),
            ),
            topk=TopKConfig(
                include_generalist=bool(topk_raw.get("include_generalist", False)),
                generalist_expert=topk_raw.get("generalist_expert"),
                per_task=dict(topk_raw.get("per_task", {})),
            ),
        )

        refine_raw = raw.get("refine", {})
        refine = RefineConfig(
            enabled=bool(refine_raw.get("enabled", False)),
            per_task=dict(refine_raw.get("per_task", {})),
            always_take_refined=bool(refine_raw.get("always_take_refined", False)),
        )

        cfg = AppConfig(
            unified_llm=unified,
            router=router,
            experts=experts,
            selection=selection,
            refine=refine,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.unified_llm.mode not in VALID_MODES:
            raise ValueError(f"Invalid unified_llm.mode={self.unified_llm.mode!r}, expected one of {sorted(VALID_MODES)}")
        if self.unified_llm.max_parallel < 1:
            raise ValueError("unified_llm.max_parallel must be >= 1")

        if self.router.type not in VALID_ROUTER_TYPES:
            raise ValueError(f"Invalid router.type={self.router.type!r}, expected one of {sorted(VALID_ROUTER_TYPES)}")
        if self.router.type == "python_module" and not self.router.python_module:
            raise ValueError("router.python_module is required when router.type=python_module")
        if self.router.type == "local_model" and not self.router.local_model:
            raise ValueError("router.local_model is required when router.type=local_model")
        if self.router.type == "dummy_classifier":
            if self.router.dummy_num_experts_hint is not None and int(self.router.dummy_num_experts_hint) < 1:
                raise ValueError("router.dummy_num_experts_hint must be >= 1 when provided")
        if not (0.0 <= self.router.confidence_threshold <= 1.0):
            raise ValueError("router.confidence_threshold must be between 0.0 and 1.0")

        if not self.experts:
            raise ValueError("At least one expert must be configured")

        seen = set()
        seen_indices = set()
        for expert in self.experts:
            if expert.name in seen:
                raise ValueError(f"Duplicate expert name={expert.name!r}")
            seen.add(expert.name)
            if expert.index is not None:
                idx = int(expert.index)
                if idx < 0:
                    raise ValueError(f"Expert index must be >= 0 for expert={expert.name!r}")
                if idx in seen_indices:
                    raise ValueError(f"Duplicate expert index={idx}")
                seen_indices.add(idx)
            if expert.load_strategy not in VALID_LOAD_STRATEGIES:
                raise ValueError(
                    f"Invalid load_strategy for expert={expert.name!r}: {expert.load_strategy!r}. "
                    f"Expected one of {sorted(VALID_LOAD_STRATEGIES)}"
                )

        expert_names = {e.name for e in self.experts}
        if self.selection.judge.type not in VALID_JUDGE_TYPES:
            raise ValueError(
                f"Invalid selection.judge.type={self.selection.judge.type!r}, expected one of {sorted(VALID_JUDGE_TYPES)}"
            )
        if self.selection.judge.type == "llm_judge":
            judge_name = self.selection.judge.llm_judge_expert
            if not judge_name:
                raise ValueError("selection.judge.llm_judge_expert is required when judge.type=llm_judge")
            if judge_name not in expert_names:
                raise ValueError(f"selection.judge.llm_judge_expert={judge_name!r} is not a configured expert")

        if self.selection.topk.include_generalist:
            generalist = self.selection.topk.generalist_expert
            if not generalist:
                raise ValueError("selection.topk.generalist_expert is required when include_generalist=true")
            if generalist not in expert_names:
                raise ValueError(f"selection.topk.generalist_expert={generalist!r} is not a configured expert")

        for task, names in self.selection.topk.per_task.items():
            if not isinstance(names, list) or not names:
                raise ValueError(f"selection.topk.per_task[{task!r}] must be a non-empty list of expert names")
            for name in names:
                if name not in expert_names:
                    raise ValueError(f"selection.topk.per_task[{task!r}] references unknown expert {name!r}")

        for task, pair in self.refine.per_task.items():
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"refine.per_task[{task!r}] must be a list of [draft_expert, refiner_expert]")
            draft_name, refiner_name = pair
            if draft_name not in expert_names:
                raise ValueError(f"refine.per_task[{task!r}] draft expert {draft_name!r} is unknown")
            if refiner_name not in expert_names:
                raise ValueError(f"refine.per_task[{task!r}] refiner expert {refiner_name!r} is unknown")

    def task_to_experts(self) -> Dict[str, List[str]]:
        """
        Build a dynamic task->experts map from expert declarations.
        Order follows the experts list order in config.
        """
        mapping: Dict[str, List[str]] = {}
        for expert in self.experts:
            for task in expert.tasks:
                mapping.setdefault(task, []).append(expert.name)
        return mapping

    def expert_name_by_index(self) -> Dict[int, str]:
        """
        Build index->expert mapping.
        If index is omitted for some experts, fallback to list order index.
        """
        mapping: Dict[int, str] = {}
        for pos, expert in enumerate(self.experts):
            idx = int(expert.index) if expert.index is not None else pos
            if idx not in mapping:
                mapping[idx] = expert.name
        return mapping


def load_config(path: str | Path) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.from_dict(raw)
