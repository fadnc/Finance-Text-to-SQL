"""
End-to-end pipeline for training, inference, and evaluation on the
gretelai/synthetic_text_to_sql dataset.
"""

import os
import subprocess
import sys


def run_command(cmd, description, env):
    """Run a command and report status."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {description}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        print(f" FAILED: {description}")
        return False
    print(f" SUCCESS: {description}")
    return True


def main():
    hf_cache = os.path.abspath(".hf_cache")
    env = os.environ.copy()
    env["HF_HOME"] = hf_cache
    env["HF_DATASETS_CACHE"] = os.path.join(hf_cache, "datasets")

    steps = [
        ([sys.executable, "data/prepare_dataset.py"], "Prepare Gretel finance dataset"),
        (
            [sys.executable, "data/prepare_multi_dataset.py"],
            "Prepare Spider data and dataset mixture",
        ),
        ([sys.executable, "-m", "src.training.train"], "Train model with QLoRA"),
        (
            [sys.executable, "-m", "scripts.generate_predictions"],
            "Generate predictions on the held-out test split",
        ),
        (
            [
                sys.executable,
                "-m",
                "src.eval.eval",
                "--dataset",
                "data/processed/predictions.json",
                "--output",
                "data/processed/eval_results.json",
                "--dashboard-output",
                "api/static/evaluation.json",
            ],
            "Evaluate model",
        ),
        (
            [sys.executable, "-m", "scripts.generate_predictions", "--split", "spider_dev"],
            "Generate predictions on held-out Spider dev databases",
        ),
        (
            [sys.executable, "-m", "src.eval.eval_spider"],
            "Evaluate exact SQL match on Spider dev",
        ),
    ]

    print("\n" + "=" * 60)
    print("FINANCE TEXT-TO-SQL (GRETEL) PIPELINE")
    print("=" * 60)

    for cmd, desc in steps:
        if not run_command(cmd, desc, env):
            print(f"\n Pipeline stopped at: {desc}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(" ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nCheck results in: data/processed/eval_results.json")


if __name__ == "__main__":
    main()
