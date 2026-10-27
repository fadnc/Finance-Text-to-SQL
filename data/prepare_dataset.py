"""Prepare a local JSON snapshot of the configured Gretel dataset.

Training and evaluation load the dataset from the Hugging Face Hub directly,
but this snapshot is useful for inspection, offline experiments, and tools that
expect the older ``instruction`` / ``input`` / ``output`` JSON layout.

Usage:
    python data/prepare_dataset.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.gretel import load_gretel_splits


def main():
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    splits = load_gretel_splits(config)
    output_dir = Path(config["dataset"].get("prepared_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, dataset in splits.items():
        path = output_dir / f"gretel_finance_{split_name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_list(), f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(dataset):>5} rows to {path}")

    # Keep the historical filename available for consumers of the old layout.
    with open(output_dir / "finance_sql.json", "w", encoding="utf-8") as f:
        json.dump(splits["train"].to_list(), f, indent=2, ensure_ascii=False)

    metadata = {
        "source": config["dataset"]["name"],
        "domains": config["dataset"].get("domains") or [],
        "splits": {name: len(dataset) for name, dataset in splits.items()},
        "complexity": dict(Counter(splits["train"]["sql_complexity"])),
        "task_types": dict(Counter(splits["train"]["sql_task_type"])),
        "unique_schemas": len(set(splits["train"]["schema"])),
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset summary")
    print("=" * 60)
    print(f"Source:  {metadata['source']}")
    print(f"Domains: {', '.join(metadata['domains']) or 'all'}")
    for name, count in metadata["splits"].items():
        print(f"{name:>10}: {count}")
    print(f"Unique training schemas: {metadata['unique_schemas']}")
    print("\nTraining complexity:")
    for name, count in sorted(metadata["complexity"].items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
