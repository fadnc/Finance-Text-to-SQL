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
            f"Missing prepared data file {path}. Run data/prepare_dataset.py, "
            "data/prepare_multi_dataset.py, and data/prepare_wikisql.py first."
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
    separate exact-match evaluation and never enters training. WikiSQL uses its
    official dev split for validation; that split is table-disjoint from train.
    """
    processed_dir = Path(config["dataset"].get("prepared_dir", "data/processed"))
    gretel_train = _read_json_rows(processed_dir / "gretel_finance_train.json")
    gretel_validation = _read_json_rows(
        processed_dir / "gretel_finance_validation.json"
    )
    gretel_test = _read_json_rows(processed_dir / "gretel_finance_test.json")
    finance_repeat = max(
        1, int(config["training_datasets"]["spider"].get("finance_repeat", 1))
    )
    balanced_gretel_train = gretel_train * finance_repeat

    train_parts = [_to_dataset(balanced_gretel_train)]
    validation_parts = [_to_dataset(gretel_validation)]
    if config["training_datasets"]["spider"].get("include", True):
        spider_train = _read_json_rows(processed_dir / "spider_train.json")
        spider_train, spider_validation = _split_by_database(
            spider_train,
            float(config["training_datasets"]["spider"].get("validation_size", 0.1)),
            int(config.get("seed", 42)),
        )
        max_spider_rows = config["training_datasets"]["spider"].get(
            "max_train_samples"
        )
        if max_spider_rows is not None and len(spider_train) > int(max_spider_rows):
            spider_train = _to_dataset(spider_train).shuffle(
                seed=int(config.get("seed", 42))
            ).select(range(int(max_spider_rows)))
            train_parts.append(spider_train)
        else:
            train_parts.append(_to_dataset(spider_train))
        validation_parts.append(_to_dataset(spider_validation))

    wikisql_config = config.get("training_datasets", {}).get("wikisql", {})
    if wikisql_config.get("include", False):
        wikisql_train = _read_json_rows(processed_dir / "wikisql_train.json")
        wikisql_validation = _read_json_rows(processed_dir / "wikisql_validation.json")
        max_wikisql_rows = wikisql_config.get("max_train_samples")
        if max_wikisql_rows is not None and len(wikisql_train) > int(max_wikisql_rows):
            wikisql_train = _to_dataset(wikisql_train).shuffle(
                seed=int(config.get("seed", 42))
            ).select(range(int(max_wikisql_rows)))
        else:
            wikisql_train = _to_dataset(wikisql_train)
        max_wikisql_validation = wikisql_config.get("max_validation_samples")
        if max_wikisql_validation is not None and len(wikisql_validation) > int(max_wikisql_validation):
            wikisql_validation = _to_dataset(wikisql_validation).select(
                range(int(max_wikisql_validation))
            )
        else:
            wikisql_validation = _to_dataset(wikisql_validation)
        train_parts.append(wikisql_train)
        validation_parts.append(wikisql_validation)

    train = concatenate_datasets(train_parts).shuffle(seed=int(config.get("seed", 42)))
    validation = concatenate_datasets(validation_parts).shuffle(seed=int(config.get("seed", 42)))

    return DatasetDict(
        {
            "train": train,
            "validation": validation,
            "test": _to_dataset(gretel_test),
        }
    )
