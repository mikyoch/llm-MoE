from __future__ import annotations

import json

from unified_llm.classifier import DummyClassifierModel


def test_dummy_classifier_keyword_match() -> None:
    model = DummyClassifierModel.default(num_experts_hint=4)
    out = model.predict("Please debug this stacktrace")
    assert out["label"] == "1"


def test_dummy_classifier_load_json(tmp_path) -> None:  # noqa: ANN001
    payload = {
        "rules": [{"index": 3, "keywords": ["finance"]}],
        "default_index": 1,
        "confidence": 0.95,
        "num_experts_hint": 5,
    }
    path = tmp_path / "dummy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model = DummyClassifierModel.from_json(path)
    out = model.predict("finance report")
    assert out["label"] == "3"
    assert out["confidence"] == 0.95


def test_dummy_classifier_fallback_is_index_string() -> None:
    model = DummyClassifierModel.default(num_experts_hint=3)
    out = model.predict("unseen content without matching keyword")
    assert out["label"] in {"0", "1", "2"}
