"""Compare base and fine-tuned models on the held-out Gretel finance test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from peft import PeftModel

from src.eval.metrics import evaluate_pair, summarize_latencies, validate_query_schema
from src.utils.model_loader import load_model
from src.utils.prompting import build_prompt, extract_sql


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def loss_for_example(model, tokenizer, prompt, target, max_length):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"]
    ids = prompt_ids + target_ids
    if len(ids) > max_length or not target_ids:
        return None, len(target_ids)
    device = next(model.parameters()).device
    input_ids = torch.tensor([ids], device=device)
    labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], device=device)
    with torch.inference_mode():
        loss = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels).loss
    return float(loss.detach().cpu()), len(target_ids)


def run_model(model, tokenizer, rows, name, args, max_length):
    model.eval()
    prompt0 = build_prompt(rows[0]["question"], rows[0]["schema"])
    warm = tokenizer(prompt0, return_tensors="pt").to(next(model.parameters()).device)
    with torch.inference_mode():
        model.generate(**warm, max_new_tokens=2, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    sync_device()

    predictions, losses, skipped = [], [], 0
    for index, row in enumerate(rows):
        prompt = build_prompt(row["question"], row["schema"])
        encoded = tokenizer(prompt, return_tensors="pt").to(next(model.parameters()).device)
        sync_device(); started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                                    do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        sync_device()
        latency = (time.perf_counter() - started) * 1000
        generated = output[0, encoded["input_ids"].shape[-1]:]
        sql = extract_sql(tokenizer.decode(generated, skip_special_tokens=True))
        metric = evaluate_pair(sql, row["output"], context=row.get("context"), timeout_s=args.sql_timeout)
        schema_metric = validate_query_schema(sql, row["schema"], timeout_s=args.sql_timeout)
        loss, target_tokens = loss_for_example(model, tokenizer, prompt, row["output"], max_length)
        if loss is None: skipped += 1
        else: losses.append((loss, target_tokens))
        predictions.append({
            "id": row.get("id"), "category": row.get("sql_complexity"), "task_type": row.get("sql_task_type"),
            "question": row["question"], "reference_sql": row["output"], "prediction": sql,
            "exact_match": metric["exact_match"], "execution_match": metric["execution_match"],
            "valid_sql": schema_metric["valid"], "error_type": metric["error_type"] or schema_metric["error_type"],
            "error": metric.get("error") or schema_metric.get("error"), "generation_latency_ms": latency,
            "generated_tokens": int(generated.numel()),
            "tokens_per_second": int(generated.numel()) / (latency / 1000) if latency else None,
            "gold_executable": metric["reference_executable"],
        })
        print(f"{name}: {index + 1}/{len(rows)}", flush=True)

    latency = summarize_latencies([r["generation_latency_ms"] for r in predictions])
    exact = sum(r["exact_match"] for r in predictions) / len(predictions)
    valid = sum(r["valid_sql"] for r in predictions) / len(predictions)
    executable = [r for r in predictions if r["gold_executable"]]
    execution = sum(r["execution_match"] for r in executable) / len(executable) if executable else None
    error_breakdown = {}
    for row in predictions:
        if not row["valid_sql"]:
            kind = row["error_type"] or "other_database_error"
            error_breakdown[kind] = error_breakdown.get(kind, 0) + 1
    token_weighted_loss = sum(loss * count for loss, count in losses) / sum(count for _, count in losses) if losses else None
    by_category = {}
    for category in sorted({str(r["category"]) for r in predictions}):
        group = [r for r in predictions if str(r["category"]) == category]
        executable_group = [r for r in group if r["gold_executable"]]
        by_category[category] = {"count": len(group), "exact_match": sum(r["exact_match"] for r in group) / len(group),
                                 "valid_sql": sum(r["valid_sql"] for r in group) / len(group),
                                 "execution_accuracy": sum(r["execution_match"] for r in executable_group) / len(executable_group) if executable_group else None}
    return {
        "model": name, "exact_match_accuracy": exact, "valid_sql_rate": valid,
        "execution_accuracy": execution, "execution_examples": len(executable),
        "valid_sql_error_breakdown": error_breakdown,
        "generation_latency": latency,
        "mean_tokens_per_second": statistics.mean([r["tokens_per_second"] for r in predictions if r["tokens_per_second"]]) if predictions else None,
        "test_loss": token_weighted_loss, "perplexity": min(float("inf"), __import__("math").exp(token_weighted_loss)) if token_weighted_loss is not None and token_weighted_loss < 700 else None,
        "test_loss_examples": len(losses), "test_loss_skipped_long": skipped,
        "by_complexity": by_category, "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/gretel_finance_test.json")
    parser.add_argument("--output", default="reports/evaluation")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--sql-timeout", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path("configs/training_config.yaml").read_text(encoding="utf-8"))
    args.max_new_tokens = args.max_new_tokens or config.get("generation_max_new_tokens", 256)
    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.limit: rows = rows[:args.limit]
    max_length = int(config.get("max_seq_length", 512))
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)

    base, tokenizer = load_model(config["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    results = []
    results.append(run_model(base, tokenizer, rows, "base", args, max_length))
    adapter_path = Path(config["output_dir"])
    if adapter_path.exists():
        adapted = PeftModel.from_pretrained(base, str(adapter_path)); adapted.eval()
        results.append(run_model(adapted, tokenizer, rows, "fine_tuned", args, max_length))
    else:
        print(f"Adapter missing at {adapter_path}; evaluated base only.")

    metadata = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "dataset": str(Path(args.data)), "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "examples": len(rows), "base_model": config["model_name"],
                "base_model_revision": getattr(base.config, "_commit_hash", None),
                "adapter": str(adapter_path), "adapter_present": adapter_path.exists(),
                "adapter_sha256": hashlib.sha256((adapter_path / "adapter_model.safetensors").read_bytes()).hexdigest() if (adapter_path / "adapter_model.safetensors").exists() else None,
                "max_new_tokens": args.max_new_tokens, "do_sample": False, "max_loss_sequence_length": max_length,
                "hardware": {"platform": platform.platform(), "python": platform.python_version(),
                             "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    summary = {"metadata": metadata, "models": [{k: v for k, v in result.items() if k != "predictions"} for result in results]}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            for record in result["predictions"]:
                f.write(json.dumps({"model": result["model"], **record}, ensure_ascii=False) + "\n")
    lines = ["# Text-to-SQL model evaluation", "", f"Dataset: `{metadata['dataset']}` ({len(rows)} examples)", "",
             "| Model | Exact match | Valid SQL | Execution accuracy* | Loss | P50 ms | P95 ms | P99 ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lat = r["generation_latency"]
        lines.append(f"| {r['model']} | {r['exact_match_accuracy']:.3f} | {r['valid_sql_rate']:.3f} | {r['execution_accuracy']} | {r['test_loss']} | {lat['p50_ms']} | {lat['p95_ms']} | {lat['p99_ms']} |")
    lines += ["", "* Execution accuracy uses executable reference examples only. Exact match ignores whitespace/comments and folds keyword case; it is not semantic SQL equivalence. Test loss scores target SQL tokens only; overlength examples are skipped and reported."]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__": main()
