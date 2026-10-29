"""Generate SQL predictions on the Gretel test/validation split."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm

from src.data.gretel import load_gretel_splits
from src.eval.eval_accuracy import normalize_sql
from src.utils.model_loader import load_model
from src.utils.prompting import build_prompt, extract_sql


def load_config():
    with open("configs/training_config.yaml") as f:
        return yaml.safe_load(f)


def generate_sql(model, tokenizer, question, schema, max_new_tokens):
    """Generate SQL with the same prompt layout used during training."""
    prompt = build_prompt(question, schema, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[-1]:]
    return extract_sql(tokenizer.decode(new_tokens, skip_special_tokens=True))


def save_results(output_path: Path, results: list[dict]):
    """Write results atomically so an interrupted run keeps finished samples."""
    temp_path = output_path.with_name(output_path.name + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    temp_path.replace(output_path)


def load_results(output_path: Path, valid_ids: set) -> list[dict]:
    """Load prior results that belong to the current split."""
    if not output_path.exists():
        return []
    try:
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if row.get("id") in valid_ids]


def main():
    parser = argparse.ArgumentParser(description="Generate SQL predictions")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "validation", "spider_dev"],
        help="Gretel split or held-out Spider development set",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum samples")
    parser.add_argument(
        "--data",
        default=None,
        help="Local JSON split to use instead of reloading the Hugging Face dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for the predictions JSON",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse predictions already present in the output file",
    )
    args = parser.parse_args()

    config = load_config()
    print("Loading dataset...")
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            test_data = json.load(f)
    elif args.split == "spider_dev":
        with open("data/processed/spider_dev.json", encoding="utf-8") as f:
            test_data = json.load(f)
    else:
        splits = load_gretel_splits(config)
        test_data = splits[args.split]
    if args.limit:
        if isinstance(test_data, list):
            test_data = test_data[: args.limit]
        else:
            test_data = test_data.select(range(min(args.limit, len(test_data))))

    default_output = (
        "data/processed/spider_dev_predictions.json"
        if args.split == "spider_dev"
        else "data/processed/predictions.json"
    )
    output_path = Path(args.output or default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid_ids = {sample["id"] for sample in test_data}
    results = load_results(output_path, valid_ids) if args.resume else []
    done_ids = {row["id"] for row in results}
    remaining = [sample for sample in test_data if sample["id"] not in done_ids]

    print(
        f"Generating predictions for {len(remaining)} {args.split} samples "
        f"({len(results)} already complete)..."
    )

    print("Loading base model...")
    model, tokenizer = load_model(config["model_name"])

    adapter_path = config["output_dir"]
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    max_new_tokens = config.get("generation_max_new_tokens", 256)
    exact_matches = sum(
        normalize_sql(row["predicted"]) == normalize_sql(row["output"])
        for row in results
    )

    for sample in tqdm(remaining, desc="Generating SQL"):
        predicted = generate_sql(
            model,
            tokenizer,
            sample["question"],
            sample["schema"],
            max_new_tokens,
        )
        if normalize_sql(predicted) == normalize_sql(sample["output"]):
            exact_matches += 1

        results.append(
            {
                "id": sample["id"],
                "domain": sample["domain"],
                "sql_complexity": sample["sql_complexity"],
                "sql_task_type": sample["sql_task_type"],
                "question": sample["question"],
                "schema": sample["schema"],
                "context": sample["context"],
                "output": sample["output"],
                "predicted": predicted,
            }
        )
        save_results(output_path, results)

    save_results(output_path, results)

    print(f"\nPredictions saved to {output_path}")
    print(f"Total samples: {len(results)}")
    print(
        f"Quick exact match: {exact_matches}/{len(results)} "
        f"({exact_matches / len(results) * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
