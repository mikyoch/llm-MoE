# Optional Distillation Upgrade Path

This folder is **optional**.  
The main orchestrator (`UnifiedRouterLLM`) works without training.

Use distillation only if you later want a single student checkpoint.

## What this pipeline does

1. Generate teacher responses using your existing routed ensemble.
2. Build `(prompt, response)` pairs in JSONL.
3. Run SFT training for a student model checkpoint.

## Why optional

- Distillation requires prompt data coverage.
- Quality is not guaranteed without representative data.
- For your current constraints (limited data, quality-first, training-light), orchestrator mode is the primary solution.

## Files

- `data.py`: prompt loading and teacher pair generation
- `train_utils.py`: dataset and collator helpers
- `sft_distill.py`: runnable SFT script

## Example

```bash
python -m distill.sft_distill \
  --router_config examples/config.yaml \
  --prompts_jsonl data/prompts.jsonl \
  --student_model mistralai/Mistral-7B-Instruct-v0.3 \
  --output_dir outputs/student-distilled
```
