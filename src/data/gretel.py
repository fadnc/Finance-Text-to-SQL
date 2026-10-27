"""Load and normalize the gretelai/synthetic_text_to_sql dataset.

The source dataset contains many domains, each row carrying its own SQL schema
and (optionally) seed rows in ``sql_context``.  This module filters the dataset
to the configured domains, extracts DDL for prompting, and exposes train /
validation / test splits.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from src.training.data_split import grouped_split_indices
from src.utils.prompting import INSTRUCTION


def extract_schema(sql_context: str) -> str:
    """Return only the CREATE TABLE statements from a SQLite context script.

    ``sql_context`` often contains INSERT statements with sample rows.  Those
    rows must not be shown to the model, so the context is executed in an
    in-memory database and the canonical DDL is read from ``sqlite_master``.
    A regex fallback handles contexts that SQLite cannot parse.
    """
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(sql_context)
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY rootpage"
        ).fetchall()
        conn.close()
        statements = [row[0].strip() for row in rows if row[0]]
        if statements:
            return ";\n".join(statements)
    except sqlite3.Error:
        pass

    statements = re.findall(
        r"CREATE\s+TABLE[^;]+;", sql_context, flags=re.IGNORECASE | re.DOTALL
    )
    return ";\n".join(statement.strip().rstrip(";") for statement in statements)


def prepare_example(example: dict[str, Any]) -> dict[str, Any]:
    """Map a raw Gretel row to the project's instruction/input/output format."""
    schema = extract_schema(example["sql_context"])
    question = example["sql_prompt"].strip()
    sql = example["sql"].strip()

    return {
        "id": example["id"],
        "domain": example["domain"],
        "sql_complexity": example["sql_complexity"],
        "sql_task_type": example["sql_task_type"],
        "question": question,
        "schema": schema,
        "context": example["sql_context"],
        "output": sql,
        "instruction": INSTRUCTION,
        "input": f"Question: {question}\nSchema: {schema}",
    }


def _filter_and_prepare(dataset: Dataset, domains: set[str]) -> Dataset:
    if domains:
        dataset = dataset.filter(
            lambda row: row["domain"] in domains,
            desc="Filtering domains",
        )
    return dataset.map(
        prepare_example,
        remove_columns=dataset.column_names,
        desc="Preparing examples",
    )


def load_gretel_splits(config: dict[str, Any]) -> DatasetDict:
    """Load the configured HF dataset and return train/validation/test splits.

    The official test split is kept untouched for final evaluation.  A grouped
    validation split is carved out of the training split so identical prompts
    never straddle the boundary.
    """
    dataset_config = config["dataset"]
    domains = set(dataset_config.get("domains") or [])
    seed = config.get("seed", 42)

    raw = load_dataset(
        dataset_config["name"],
        cache_dir=dataset_config.get("cache_dir"),
    )

    train_split = dataset_config.get("train_split", "train")
    eval_split = dataset_config.get("eval_split", "test")

    train = _filter_and_prepare(raw[train_split], domains)
    test = _filter_and_prepare(raw[eval_split], domains)

    max_train = dataset_config.get("max_train_samples")
    if max_train:
        train = train.shuffle(seed=seed).select(range(min(max_train, len(train))))

    max_eval = dataset_config.get("max_eval_samples")
    if max_eval:
        test = test.select(range(min(max_eval, len(test))))

    validation_size = float(dataset_config.get("validation_size", 0.1))
    if validation_size > 0 and len(train) > 1:
        train_indices, validation_indices = grouped_split_indices(
            train["question"], validation_size, seed
        )
        train_split_dataset = train.select(train_indices)
        validation_dataset = train.select(validation_indices)
    else:
        train_split_dataset = train
        validation_dataset = train.select([])

    return DatasetDict(
        {
            "train": train_split_dataset,
            "validation": validation_dataset,
            "test": test,
        }
    )
