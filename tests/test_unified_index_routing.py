from __future__ import annotations

from unified_llm.config import AppConfig
from unified_llm.experts import ExpertResult
from unified_llm.judges import HeuristicJudge
from unified_llm.router import RouteDecision
from unified_llm.unified import UnifiedRouterLLM


class FakeRouter:
    def route(self, prompt: str) -> RouteDecision:  # noqa: ARG002
        return RouteDecision(label="1", confidence=1.0, raw={})


class FakeExpertPool:
    def __init__(self) -> None:
        self.last_called = None

    def all_expert_names(self):  # noqa: ANN201
        return ["e0", "e1"]

    def experts_for_task(self, task_label: str):  # noqa: ARG002, ANN201
        return []

    def generate_one(self, expert_name: str, prompt: str, generation_overrides=None):  # noqa: ANN001, ANN201
        self.last_called = expert_name
        return ExpertResult(expert_name=expert_name, text=f"answer from {expert_name}", latency_ms=1.0)

    def generate_many(self, expert_names, prompt, max_parallel, generation_overrides=None):  # noqa: ANN001, ANN201
        return [self.generate_one(name, prompt, generation_overrides) for name in expert_names]


def test_route_top1_uses_index_label_to_select_expert() -> None:
    cfg = AppConfig.from_dict(
        {
            "unified_llm": {"mode": "route_top1", "max_parallel": 1},
            "router": {"type": "dummy_classifier"},
            "experts": [
                {"name": "e0", "hf_model_id": "m0", "index": 0},
                {"name": "e1", "hf_model_id": "m1", "index": 1},
            ],
        }
    )
    pool = FakeExpertPool()
    model = UnifiedRouterLLM(config=cfg, router=FakeRouter(), expert_pool=pool, judge=HeuristicJudge())
    out = model.generate("prompt")
    assert out == "answer from e1"
    assert pool.last_called == "e1"
