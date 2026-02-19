"""Expert model runtime management for heterogeneous LLMs."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unified_llm.config import ExpertConfig

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None


SAFE_GENERATION_DEFAULTS: Dict[str, Any] = {
    "max_new_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "do_sample": True,
}


@dataclass
class ExpertResult:
    expert_name: str
    text: str
    latency_ms: float
    error: Optional[str] = None


def _resolve_torch_dtype(dtype_name: str) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is required for expert generation but is not installed.")
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype={dtype_name!r}. Use one of {sorted(mapping)}")
    return mapping[dtype_name]


class ExpertRuntime:
    """
    Loads and executes one expert model with its own tokenizer.
    """

    def __init__(self, config: ExpertConfig) -> None:
        self.config = config
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._tokenizer is not None and self._model is not None

    def load(self) -> None:
        if self.is_loaded:
            return
        with self._load_lock:
            if self.is_loaded:
                return
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("transformers is required to load expert models.") from exc

            dtype = _resolve_torch_dtype(self.config.dtype)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.hf_model_id,
                trust_remote_code=self.config.trust_remote_code,
            )

            model_kwargs: Dict[str, Any] = {
                "torch_dtype": dtype,
                "trust_remote_code": self.config.trust_remote_code,
            }

            if self.config.device == "auto":
                model_kwargs["device_map"] = "auto"

            self._model = AutoModelForCausalLM.from_pretrained(self.config.hf_model_id, **model_kwargs)
            if self.config.device != "auto":
                self._model.to(self.config.device)
            self._model.eval()

    def unload_to_cpu(self) -> None:
        if self._model is None:
            return
        if torch is None:
            return
        self._model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _model_device(self) -> Any:
        if self._model is None:
            raise RuntimeError("Model is not loaded")
        if hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device

    def generate(self, prompt: str, generation_overrides: Optional[Dict[str, Any]] = None) -> ExpertResult:
        started = time.time()
        try:
            self.load()
            assert self._tokenizer is not None
            assert self._model is not None

            params = dict(SAFE_GENERATION_DEFAULTS)
            params.update(self.config.gen_defaults)
            if generation_overrides:
                params.update(generation_overrides)

            if float(params.get("temperature", 0.0)) <= 0:
                params["do_sample"] = False

            encoded = self._tokenizer(prompt, return_tensors="pt")
            device = self._model_device()
            encoded = {k: v.to(device) for k, v in encoded.items()}

            with self._run_lock:
                with torch.no_grad():  # type: ignore[union-attr]
                    output_ids = self._model.generate(**encoded, **params)

            text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return ExpertResult(
                expert_name=self.config.name,
                text=text,
                latency_ms=1000.0 * (time.time() - started),
            )
        except Exception as exc:  # noqa: BLE001
            return ExpertResult(
                expert_name=self.config.name,
                text="",
                latency_ms=1000.0 * (time.time() - started),
                error=str(exc),
            )
        finally:
            if self.config.offload_to_cpu:
                self.unload_to_cpu()


class ExpertPool:
    """
    Maintains all expert runtimes and supports controlled parallel generation.
    """

    def __init__(self, experts: List[ExpertConfig]) -> None:
        self._runtimes: Dict[str, ExpertRuntime] = {cfg.name: ExpertRuntime(cfg) for cfg in experts}
        self._configs: Dict[str, ExpertConfig] = {cfg.name: cfg for cfg in experts}

        for cfg in experts:
            if cfg.load_strategy == "eager":
                self._runtimes[cfg.name].load()

    def all_expert_names(self) -> List[str]:
        return list(self._runtimes.keys())

    def experts_for_task(self, task_label: str) -> List[str]:
        return [cfg.name for cfg in self._configs.values() if task_label in cfg.tasks]

    def generate_one(
        self,
        expert_name: str,
        prompt: str,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ) -> ExpertResult:
        if expert_name not in self._runtimes:
            raise ValueError(f"Unknown expert name={expert_name!r}")
        return self._runtimes[expert_name].generate(prompt, generation_overrides=generation_overrides)

    def generate_many(
        self,
        expert_names: List[str],
        prompt: str,
        max_parallel: int,
        generation_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[ExpertResult]:
        names = list(dict.fromkeys(expert_names))
        if not names:
            return []

        for name in names:
            if name not in self._runtimes:
                raise ValueError(f"Unknown expert name={name!r}")

        if max_parallel <= 1 or len(names) == 1:
            return [self.generate_one(name, prompt, generation_overrides=generation_overrides) for name in names]

        results_map: Dict[str, ExpertResult] = {}
        workers = min(max_parallel, len(names))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_name = {
                executor.submit(self.generate_one, name, prompt, generation_overrides): name
                for name in names
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                results_map[name] = future.result()

        return [results_map[name] for name in names]
