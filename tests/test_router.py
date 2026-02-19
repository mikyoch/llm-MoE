from __future__ import annotations

from unified_llm.config import RouterConfig
from unified_llm.router import CallableRouter, PythonModuleRouter, build_router


def test_callable_router_normalizes_dict() -> None:
    router = CallableRouter(lambda _: {"label": "A", "confidence": 0.99})
    decision = router.route("prompt")
    assert decision.label == "A"
    assert decision.confidence == 0.99


def test_callable_router_accepts_string_label() -> None:
    router = CallableRouter(lambda _: "B")
    decision = router.route("prompt")
    assert decision.label == "B"
    assert decision.confidence == 1.0


def test_python_module_router_loads_module_function() -> None:
    router = PythonModuleRouter("tests.mock_router_module:route")
    decision = router.route("alpha task")
    assert decision.label == "A"
    assert decision.confidence == 1.0


def test_build_router_callable_mode() -> None:
    cfg = RouterConfig(type="callable", confidence_threshold=0.0)
    router = build_router(cfg, callable_router=lambda _: {"label": "C", "confidence": 1.0})
    decision = router.route("anything")
    assert decision.label == "C"
