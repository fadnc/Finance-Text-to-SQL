"""Download and normalize Spider examples for multi-dataset training.

Spider is redistributed here by GRAST-SQL with schema metadata extracted from
the original SQLite databases. The upstream Spider license is CC BY-SA 4.0.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]


def quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def spider_schema_to_ddl(example: dict) -> str:
    """Render Spider's flattened schema metadata as CREATE TABLE statements."""
    tables: dict[str, list[str]] = {}
    for qualified_name in example.get("schema", []):
        if "." not in qualified_name:
            continue
        table, column = qualified_name.split(".", 1)
        tables.setdefault(table, [])
        if column not in tables[table]:
            tables[table].append(column)

    primary_keys = example.get("primary_keys", {})
    foreign_keys = example.get("foreign_keys", {})
    column_info = example.get("column_info", {})
    statements = []

    for table, columns in tables.items():
        definitions = []
        primary = primary_keys.get(table, [])
        for column in columns:
            info = column_info.get(f"{table}.{column}", {})
            data_type = str(info.get("type", "TEXT")).upper()
            if data_type not in {"TEXT", "INTEGER", "INT", "REAL", "NUMERIC", "BLOB"}:
                data_type = "TEXT"
            definition = f"{quote_identifier(column)} {data_type}"
            if column in primary and len(primary) == 1:
                definition += " PRIMARY KEY"
            definitions.append(definition)

        if len(primary) > 1:
            primary_columns = ", ".join(quote_identifier(c) for c in primary)
            definitions.append(f"PRIMARY KEY ({primary_columns})")

        for foreign_key in foreign_keys.get(table, []):
            source = foreign_key.get("from")
            target_table = foreign_key.get("ref_table")
            target_column = foreign_key.get("to")
            if source in columns and target_table in tables and target_column:
                definitions.append(
                    f"FOREIGN KEY ({quote_identifier(source)}) REFERENCES "
                    f"{quote_identifier(target_table)} ({quote_identifier(target_column)})"
                )

        body = ", ".join(definitions)
        statements.append(f"CREATE TABLE {quote_identifier(table)} ({body});")

    return "\n".join(statements)


def normalize_spider_row(row: dict, split: str) -> dict:
    question = row["question"].strip()
    sql = row["SQL"].strip()
    schema = spider_schema_to_ddl(row)
    if not question or not sql or not schema:
        raise ValueError(f"Incomplete Spider {split} example: {row.get('_id')}")
    return {
        "id": f"spider-{split}-{row.get('_id', 'unknown')}",
        "source": "spider",
        "domain": "spider",
        "sql_complexity": "cross-domain",
        "sql_task_type": "analytics and retrieval",
        "question": question,
        "schema": schema,
        "context": "",
        "output": sql,
        "db_id": row["db_id"],
    }


def atomic_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    temp_path.replace(path)


def main() -> None:
    with (ROOT / "configs" / "training_config.yaml").open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_dir = ROOT / config["dataset"].get("prepared_dir", "data/processed")
    required_gretel = [
        processed_dir / "gretel_finance_train.json",
        processed_dir / "gretel_finance_validation.json",
        processed_dir / "gretel_finance_test.json",
    ]
    missing = [path for path in required_gretel if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepare the Gretel finance snapshots first with "
            "python data/prepare_dataset.py; missing: "
            + ", ".join(str(path) for path in missing)
        )

    spider_config = config["training_datasets"]["spider"]
    raw_splits = {}
    for split, filename_key in (("train", "train_file"), ("dev", "dev_file")):
        source_path = hf_hub_download(
            repo_id=spider_config["repo_id"],
            filename=spider_config[filename_key],
            repo_type="dataset",
            revision=spider_config["revision"],
            cache_dir=str(ROOT / spider_config["cache_dir"]),
        )
        with open(source_path, encoding="utf-8") as f:
            raw_splits[split] = json.load(f)

    train_rows = [normalize_spider_row(row, "train") for row in raw_splits["train"]]
    dev_rows = [normalize_spider_row(row, "dev") for row in raw_splits["dev"]]
    atomic_json(processed_dir / "spider_train.json", train_rows)
    atomic_json(processed_dir / "spider_dev.json", dev_rows)

    gretel_train = json.loads(required_gretel[0].read_text(encoding="utf-8"))
    gretel_validation = json.loads(required_gretel[1].read_text(encoding="utf-8"))
    gretel_test = json.loads(required_gretel[2].read_text(encoding="utf-8"))
    info = {
        "datasets": {
            "gretelai/synthetic_text_to_sql": {
                "license": "Apache-2.0",
                "train": len(gretel_train),
                "validation": len(gretel_validation),
                "test": len(gretel_test),
            },
            spider_config["repo_id"]: {
                "license": spider_config["license"],
                "train": len(train_rows),
                "dev": len(dev_rows),
                "revision": spider_config["revision"],
            },
        },
        "spider_training_rows": len(train_rows),
        "spider_dev_rows": len(dev_rows),
    }
    wikisql_info_path = processed_dir / "wikisql_info.json"
    if wikisql_info_path.exists():
        info["wikisql"] = json.loads(wikisql_info_path.read_text(encoding="utf-8"))
    (processed_dir / "multi_dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(
        f"Prepared Spider: {len(train_rows)} train / {len(dev_rows)} dev; "
        f"Gretel finance: {len(gretel_train)} train / "
        f"{len(gretel_validation)} validation / {len(gretel_test)} test."
    )


if __name__ == "__main__":
    main()
