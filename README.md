# llm-MoE

Practical routing framework for building a **single runtime, single package**
from multiple specialist LLMs, including heterogeneous tokenizers.

## Direct answers to your latest questions

### 1) Do you need training data for distillation?
Yes. Distillation needs prompts at minimum (unlabeled is okay), and usually
teacher outputs/logits from your specialists.

- No prompts -> no distillation objective.
- Better coverage of production tasks -> better distilled quality.

### 2) Can you guarantee distillation performance?
No strict guarantee. You can reduce risk with evaluation and acceptance gates,
but distillation can regress on rare capabilities unless your distillation
data covers those cases.

### 3) Tokenizers differ across experts. Is that a blocker?
For a true token-level unified MoE model: usually yes (or very messy).
For a wrapper/orchestrator that routes full text requests to experts: no.

This repo uses the wrapper approach, so each expert can keep its own tokenizer.

## Best approach for your constraints

Given:
- minimal retraining preferred,
- output quality is more important than compute,
- dynamic number/type of experts and dynamic task ownership,
- one runtime process and one deployable package required,

the best practical approach is:

1. **Dynamic gated ensemble wrapper** (this repo):
   - top-1 route when classifier confidence is high,
   - parallel candidate execution (top-k) when confidence is lower,
   - deterministic or heuristic selector picks final output.
2. Keep your existing task classifier as the routing source.
3. Maintain task->expert rankings in config and update from offline evals.
4. Distillation is optional later for cost/latency optimization.

This avoids mandatory training while preserving specialist quality.

## Implemented modules

### A) Multi-expert dynamic wrapper (no retraining required)

File: `dynamic_router.py`

Main classes:
- `DynamicRoutedLLM`: single `generate()` entrypoint.
- `RoutingPolicy`: dynamic task->experts mapping and confidence policy.
- `ExpertRuntime`: one expert model + its own tokenizer.
- `ClassifierOutput`: label/confidence interface from your classifier.
- `FirstSuccessfulSelector`, `HeuristicQualitySelector`: candidate selection.

### B) Adapter routing on one base model

File: `adapter_router.py`

Main classes:
- `DynamicAdapterRoutedLLM`: single `generate()` entrypoint for base+LoRA adapters.
- `AdapterRoutingPolicy`: dynamic task->adapters mapping.
- `AdapterSpec`: adapter metadata and generation defaults.
- `ClassifierOutput`: label/confidence interface.

## Config formats

See:
- `router_config.example.json` (multi-expert wrapper)
- `adapter_router_config.example.json` (single base + adapters)

- Add/remove experts without code changes.
- Change task ownership/ranking per task in policy.
- Configure confidence threshold and low-confidence mode.

For adapter routing, you can also add/remove adapters by config only.

## Usage example

```python
from dynamic_router import (
    ClassifierOutput,
    DynamicRoutedLLM,
    HeuristicQualitySelector,
)

router = DynamicRoutedLLM.from_json_config(
    path="router_config.example.json",
    selector=HeuristicQualitySelector(),
)

clf = ClassifierOutput(label="B", confidence=0.86)
result = router.generate(
    prompt="Solve this task...",
    classifier_output=clf,
    request_id="req-001",
    generation_overrides={"max_new_tokens": 300},
    return_metadata=True,
)

print(result["selected_expert"])
print(result["text"])
```

## Deployment notes

- This is one runtime process with one application artifact.
- The package still includes all expert checkpoints (larger memory footprint).
- Tokenizers can differ safely because each expert handles its own tokenization.
- If using low-confidence parallel mode, monitor latency and GPU memory.
- Log routing decisions and confidence to track drift and misroutes.

## Suggested production rollout

1. Start with `confidence_threshold=0.98`, `low_confidence_mode=parallel`, top-2.
2. Evaluate on a fixed golden set and compare:
   - top-1 only,
   - parallel+selector,
   - current baseline.
3. Adjust per-task expert order and threshold from observed quality.
4. Add fallback experts for new tasks and unknown labels.
5. Optional later: distill selected behavior to a single student for efficiency.
