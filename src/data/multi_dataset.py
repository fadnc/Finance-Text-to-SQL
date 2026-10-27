"""Combine prepared finance and cross-domain Spider data for QLoRA training."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets


def _read_json_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing prepared data file {path}. Run data/prepare_dataset.py "
            "and data/prepare_multi_dataset.py first."
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _split_by_database(rows: list[dict], validation_size: float, seed: int):
    """Hold out complete Spider databases for a less leaky validation split."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["db_id"]].append(index)

    database_ids = sorted(groups)
    random.Random(seed).shuffle(database_ids)
    target = max(1, math.ceil(len(rows) * validation_size))
    validation_ids: set[str] = set()
    validation_count = 0
    for db_id in database_ids:
        if validation_count >= target:
            break
        validation_ids.add(db_id)
        validation_count += len(groups[db_id])

    train_rows = [row for row in rows if row["db_id"] not in validation_ids]
    validation_rows = [row for row in rows if row["db_id"] in validation_ids]
    return train_rows, validation_rows


def _to_dataset(rows: list[dict]) -> Dataset:
    columns = ("id", "source", "domain", "sql_complexity", "sql_task_type",
               "question", "schema", "context", "output")
    normalized = []
    for row in rows:
        normalized.append(
            {
                key: (
                    str(row.get(key, ""))
                    if key == "id"
                    else row.get(key, "gretel_finance")
                    if key == "source"
                    else row.get(key, "")
                )
                for key in columns
            }
        )
    return Dataset.from_list(normalized)


def load_multi_dataset_splits(config: dict) -> DatasetDict:
    """Return mixed training/validation sets and the finance held-out test set.

    Gretel's prepared finance split stays unchanged. Spider's training
    validation is database-disjoint; its official dev split is saved for a
    separate exact-match evaluation and never enters training.
    """
    processed_dir = Path(config["dataset"].get("prepared_dir", "data/processed"))
    gretel_train = _read_json_rows(processed_dir / "gretel_finance_train.json")
    gretel_validation = _read_json_rows(
        processed_dir / "gretel_finance_validation.json"
    )
    gretel_test = _read_json_rows(processed_dir / "gretel_finance_test.json")
    spider_train = _read_json_rows(processed_dir / "spider_train.json")
    spider_train, spider_validation = _split_by_database(
        spider_train,
        float(config["training_datasets"]["spider"].get("validation_size", 0.1)),
        int(config.get("seed", 42)),
    )
    finance_repeat = max(
        1, int(config["training_datasets"]["spider"].get("finance_repeat", 1))
    )
    balanced_gretel_train = gretel_train * finance_repeat

    train = concatenate_datasets(
        [_to_dataset(balanced_gretel_train), _to_dataset(spider_train)]
    ).shuffle(seed=int(config.get("seed", 42)))
    validation = concatenate_datasets(
        [_to_dataset(gretel_validation), _to_dataset(spider_validation)]
    ).shuffle(seed=int(config.get("seed", 42)))

    return DatasetDict(
        {
            "train": train,
            "validation": validation,
            "test": _to_dataset(gretel_test),
        }
    )
