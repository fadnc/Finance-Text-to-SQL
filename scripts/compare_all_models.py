"""Benchmark every trained adapter on the same held-out finance test set.

Each family loads its own base model once, scores the base, then attaches each
LoRA adapter in turn (unloading in between) so the numbers are directly
comparable.

Usage:
    python scripts/compare_all_models.py
    python scripts/compare_all_models.py --limit 10 --families tinyllama
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml
from peft import PeftModel

from src.utils.model_loader import load_model

# evaluate_models.py is a script, not an importable package module.
_spec = importlib.util.spec_from_file_location(
    "evaluate_models", ROOT / "scripts" / "evaluate_models.py"
)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

TINYLLAMA = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
QWEN = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

FAMILIES = [
    (
        "tinyllama",
        TINYLLAMA,
        [
            ("TinyLlama + Gretel finance", "models/tinyllama-gretel-finance-sql"),
            ("TinyLlama + multi-dataset", "models/tinyllama-multidataset-sql"),
            (
                "TinyLlama + multi-dataset (candidate)",
                "models/tinyllama-multidataset-sql-candidate",
            ),
            (
                "TinyLlama + multi-dataset + Spider (ckpt359)",
                "models/tinyllama-multidataset-sql-candidate-spider/checkpoint-359",
            ),
            ("phi3-sql (TinyLlama adapter)", "models/phi3-sql"),
        ],
    ),
    (
        "qwen",
        QWEN,
        [
            (
                "Qwen2.5-Coder + WikiSQL/Spider/Finance",
                "models/qwen2.5-coder-1.5b-finance-sql-lora",
            ),
        ],
    ),
]


def adapter_file(path: Path) -> Path | None:
    direct = path / "adapter_model.safetensors"
    if direct.exists():
        return direct
    checkpoints = sorted(
        path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
    )
    for ckpt in reversed(checkpoints):
        candidate = ckpt / "adapter_model.safetensors"
        if candidate.exists():
            return candidate
    return None


def resolve_adapter(path: Path) -> Path | None:
    if adapter_file(path) is not None:
        return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/gretel_finance_test.json")
    parser.add_argument("--output", default="reports/model_comparison")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--sql-timeout", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--families",
        default="tinyllama,qwen",
        help="Comma-separated subset of: tinyllama,qwen",
    )
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs/training_config.yaml").read_text("utf-8"))
    max_new_tokens = args.max_new_tokens or config.get("generation_max_new_tokens", 256)
    max_length = int(config.get("max_seq_length", 512))
    run_args = argparse.Namespace(
        max_new_tokens=max_new_tokens, sql_timeout=args.sql_timeout
    )

    rows = json.loads((ROOT / args.data).read_text("utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    selected = {name.strip() for name in args.families.split(",") if name.strip()}

    results: list[dict] = []
    adapter_meta: list[dict] = []
    for key, base_name, adapters in FAMILIES:
        if key not in selected:
            continue
        print(f"\n=== Loading base {base_name} ===", flush=True)
        base, tokenizer = load_model(base_name)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        results.append(
            _eval.run_model(
                base, tokenizer, rows, f"{key} base", run_args, max_length
            )
        )
        for label, relative in adapters:
            path = ROOT / relative
            resolved = resolve_adapter(path)
            if resolved is None:
                print(f"Skipping missing adapter: {relative}", flush=True)
                continue
            try:
                adapted = PeftModel.from_pretrained(base, str(resolved))
                adapted.eval()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                print(f"Failed to load {relative}: {exc}", flush=True)
                continue
            results.append(
                _eval.run_model(adapted, tokenizer, rows, label, run_args, max_length)
            )
            sha = hashlib.sha256(adapter_file(resolved).read_bytes()).hexdigest()
            adapter_meta.append({"label": label, "path": str(resolved), "sha256": sha})
            base = adapted.unload()
            del adapted
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del base, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.data,
        "dataset_sha256": hashlib.sha256((ROOT / args.data).read_bytes()).hexdigest(),
        "examples": len(rows),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "max_loss_sequence_length": max_length,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "adapters": adapter_meta,
    }
    summary = {
        "metadata": metadata,
        "models": [{k: v for k, v in r.items() if k != "predictions"} for r in results],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            for record in result["predictions"]:
                handle.write(
                    json.dumps(
                        {"model": result["model"], **record}, ensure_ascii=False
                    )
                    + "\n"
                )

    lines = [
        "# Model comparison - finance held-out test set",
        "",
        f"Dataset: `{args.data}` ({len(rows)} examples). "
        "Execution accuracy uses executable gold queries only.",
        "",
        "| Model | Exact match | Valid SQL | Execution | Loss | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lat = row["generation_latency"]
        exec_acc = row["execution_accuracy"]
        exec_text = f"{exec_acc:.1%}" if exec_acc is not None else "n/a"
        lines.append(
            f"| {row['model']} | {row['exact_match_accuracy']:.1%} | "
            f"{row['valid_sql_rate']:.1%} | {exec_text} | "
            f"{row['test_loss']:.3f} | {lat['p50_ms']:.0f} | {lat['p95_ms']:.0f} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nSaved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
