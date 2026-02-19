# UnifiedRouterLLM

Production-grade single-process LLM orchestration for combining many specialist
models into one deployable runtime package.

## Conceptual verdict

- "Run all experts in parallel then select one output using classifier" is valid.
  Correct term: **gated ensemble with late selection (late fusion)**.
- True **MoE** routes early so only selected experts execute.
- With different tokenizers and unrelated architectures, you cannot cleanly
  merge into one standard shared-tokenizer checkpoint without training/alignment.
- Under limited training data and quality-first constraints, a **single-process
  orchestrator** is the best practical primary solution.

## What this repository provides

- One Python package (`unified_llm`) with one public runtime class:
  `UnifiedRouterLLM`
- One process serving model with one generate API
- Dynamic expert set and task mapping through YAML config only
- Supports different tokenizers across experts
- OpenAI-compatible HTTP endpoints (optional FastAPI server)
- CLI
- Structured JSON logging for routing and judge decisions
- Built-in dummy classifier model backend with integer labels (`0`, `1`, `2`, ...)
- Optional distillation upgrade path in `distill/`

## Inference modes

1. `route_top1` (default)
   - Route by classifier label and execute one expert.
2. `run_all_select`
   - Run all experts and judge candidates (quality-first).
3. `route_topk_then_judge`
   - Route to shortlist, then judge candidates.
4. `cascade_refine`
   - Draft expert + refiner expert, then judge (or always take refined).

## Decision table

| Strategy | single artifact | single runtime | compute | expected quality | complexity | tokenizer constraints |
|---|---|---|---|---|---|---|
| run_all_select (late fusion) | Yes | Yes | High | Very high | Medium | None (text-level) |
| route_top1 (early routing) | Yes | Yes | Low/Medium | High when router is accurate | Low | None (text-level) |
| route_topk_then_judge | Yes | Yes | Medium/High | Very high | Medium/High | None (text-level) |
| cascade_refine | Yes | Yes | High | Maximum potential | High | None (text-level) |

## Installation

```bash
pip install -e .
# optional server deps
pip install -e ".[server]"
```

## CLI usage

```bash
unified-llm run --config examples/config.yaml --prompt "debug this stacktrace"
unified-llm run --config examples/config.yaml --prompt "return JSON only" --verbose
unified-llm serve --config examples/config.yaml --host 0.0.0.0 --port 8000
```

## Python usage

```python
from unified_llm import UnifiedRouterLLM

model = UnifiedRouterLLM.from_config("examples/config.yaml")
text = model.generate("Solve this task.")
result = model.generate("Return JSON", return_metadata=True, mode="run_all_select")
print(result["metadata"]["selected_expert"])
```

## API usage (OpenAI-compatible)

- `POST /v1/chat/completions`
- `POST /v1/completions`

The server returns standard choice fields plus optional `metadata` when
`verbose=true`.

## Config

See `examples/config.yaml`.

Highlights:
- dynamic experts list
- integer expert index routing support (`expert.index`)
- dynamic per-task routing and top-k shortlist overrides
- judge type (`heuristic_judge` or `llm_judge`)
- per-task refine chains
- load strategy (`eager` / `lazy`)
- per-expert generation defaults

### Dummy classifier model (replace later)

The repository includes a dummy classifier "model" JSON:

- `examples/dummy_classifier_model.json`

It outputs labels as integer strings (`"0"`, `"1"`, `"2"`, ...), where each
number maps to the best expert index.

Current default config already uses it:

```yaml
router:
  type: dummy_classifier
  dummy_model_path: "examples/dummy_classifier_model.json"
```

Expert mapping is configured by `expert.index` (fallback is list order).

## Fallback behavior

If a chosen expert fails in `route_top1`, the system automatically falls back
to judge-based selection over remaining experts.

## Distillation (optional)

Distillation is not required for main orchestration.
Optional scripts are in `distill/`:
- generate teacher pairs from routed ensemble
- run SFT distillation for a student model checkpoint

## Push to Hugging Face

- Export HF-style orchestrator artifact locally (Qwen-like repo layout):
  `python scripts/export_hf_router_artifact.py --router_config examples/config.yaml --output_dir outputs/hf_router`
- Push orchestrator artifact to HF (exports HF-style package by default):
  `python scripts/push_orchestrator_to_hf.py --repo_id org/unified-router-llm --router_config examples/config.yaml`
- Distilled student:
  `python scripts/push_student_to_hf.py --repo_id org/student --model_dir outputs/student`

The exported orchestrator folder contains a Hugging Face style structure:

- `config.json`
- `generation_config.json`
- `configuration_unified_router.py`
- `modeling_unified_router.py`
- `router_config.yaml`
- `README.md`
- optional `dummy_classifier_model.json`

## Notes

- Different tokenizers are handled by routing on raw prompt text and letting
  each expert tokenize independently.
- This is one runtime process and one deployable product, but memory includes
  all loaded expert weights.
