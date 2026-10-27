"""Score Spider dev predictions without pretending empty schemas are databases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.eval_accuracy import normalize_sql


def evaluate(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        predictions = json.load(f)
    if not predictions:
        raise ValueError(f"No predictions found in {path}")

    correct = sum(
        normalize_sql(row["predicted"]) == normalize_sql(row["output"])
        for row in predictions
    )
    result = {
        "dataset": "Spider 1.0 development split",
        "metric": "normalized exact SQL match",
        "total_samples": len(predictions),
        "exact_matches": correct,
        "exact_match_accuracy": correct / len(predictions),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions", default="data/processed/spider_dev_predictions.json"
    )
    parser.add_argument("--output", default="data/processed/spider_dev_eval.json")
    args = parser.parse_args()

    result = evaluate(Path(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
