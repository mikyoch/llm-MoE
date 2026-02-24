from __future__ import annotations

from unified_llm.experts import ExpertResult
from unified_llm.judges import HeuristicJudge, LLMJudge


class FakeExpertPool:
    def __init__(self, judge_text: str, error: str | None = None) -> None:
        self.judge_text = judge_text
        self.error = error

    def generate_one(self, expert_name: str, prompt: str, generation_overrides=None):  # noqa: ANN001, ANN201
        return ExpertResult(
            expert_name=expert_name,
            text=self.judge_text,
            latency_ms=10.0,
            error=self.error,
        )


def test_heuristic_judge_prefers_valid_json_when_requested() -> None:
    judge = HeuristicJudge()
    prompt = "Return JSON only with keys: answer. required keywords: answer"
    c1 = ExpertResult(expert_name="e1", text='{"answer":"ok"}', latency_ms=1.0)
    c2 = ExpertResult(expert_name="e2", text="this is not json", latency_ms=1.0)
    result = judge.evaluate(prompt, [c1, c2])
    assert result.winner == "e1"
    assert result.scores["e1"] > result.scores["e2"]


def test_llm_judge_uses_winner_from_valid_json() -> None:
    fallback = HeuristicJudge()
    pool = FakeExpertPool(judge_text='{"winner":"e2","rationale":"better","scores":{"e1":6,"e2":9}}')
    judge = LLMJudge(judge_expert_name="judge", expert_pool=pool, rubric="accuracy", fallback_judge=fallback)

    c1 = ExpertResult(expert_name="e1", text="bad answer", latency_ms=1.0)
    c2 = ExpertResult(expert_name="e2", text="good answer", latency_ms=1.0)
    out = judge.evaluate("prompt", [c1, c2])
    assert out.winner == "e2"
    assert out.judge_type == "llm_judge"


def test_llm_judge_falls_back_on_invalid_output() -> None:
    pool = FakeExpertPool(judge_text="not-json")
    judge = LLMJudge(judge_expert_name="judge", expert_pool=pool, rubric="accuracy", fallback_judge=HeuristicJudge())
    c1 = ExpertResult(expert_name="e1", text='{"answer":"ok"}', latency_ms=1.0)
    c2 = ExpertResult(expert_name="e2", text="", latency_ms=1.0)
    out = judge.evaluate("return json", [c1, c2])
    assert out.winner == "e1"
    assert out.judge_type == "heuristic_judge"
