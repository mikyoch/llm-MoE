"""Optional SFT distillation script."""

from __future__ import annotations

import argparse
from pathlib import Path

from distill.data import generate_teacher_pairs_from_config
from distill.train_utils import (
    DataCollatorForCausalLM,
    PromptCompletionDataset,
    load_pairs_jsonl,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optional SFT distillation pipeline")
    parser.add_argument("--router_config", required=True, help="Path to orchestrator config YAML")
    parser.add_argument("--prompts_jsonl", required=True, help="Input prompts JSONL")
    parser.add_argument("--teacher_pairs_jsonl", default=None, help="Optional pre-generated teacher pairs JSONL")
    parser.add_argument("--teacher_mode", default="run_all_select", help="Teacher generation mode")
    parser.add_argument("--student_model", required=True, help="Base student model id/path")
    parser.add_argument("--output_dir", required=True, help="Output directory for distilled checkpoint")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    pairs_path = args.teacher_pairs_jsonl
    if not pairs_path:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pairs_path = str(out / "teacher_pairs.jsonl")
        generated = generate_teacher_pairs_from_config(
            router_config=args.router_config,
            prompts_jsonl=args.prompts_jsonl,
            output_jsonl=pairs_path,
            mode=args.teacher_mode,
            max_samples=args.max_samples,
        )
        print(f"Generated {generated} teacher pairs at {pairs_path}")

    records = load_pairs_jsonl(pairs_path)
    if not records:
        raise RuntimeError("No training records found in teacher pairs JSONL")

    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("transformers is required for distillation training.") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.student_model)

    dataset = PromptCompletionDataset(records=records, tokenizer=tokenizer, max_length=args.max_length)
    collator = DataCollatorForCausalLM(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        fp16=False,
        bf16=False,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved distilled model to {args.output_dir}")


if __name__ == "__main__":
    main()
