"""Download and normalize WikiSQL into the project's training row format.

WikiSQL ships as a 26 MB tarball on GitHub (train/dev/test .jsonl plus one
``*.tables.jsonl`` per split). The Hugging Face copy cannot be used with
``datasets>=4`` because it still requires a loading script.

The raw ``sql`` object stores conditions as ``[column_index, operator_index,
value]`` triples, and the table schema lives in a separate file keyed by
``table_id``. This module renders both into executable SQLite:

    question + CREATE TABLE schema -> single-table SELECT

Rows are validated by executing the rendered query against an in-memory copy
of the source table, so malformed headers or type mismatches are dropped rather
than poisoning training.

Usage:
    python data/prepare_wikisql.py
    python data/prepare_wikisql.py --no-validate --max-train-rows 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.progress import make_progress
DEFAULT_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
EXPECTED_SHA256 = "755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881"

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "!="]
INSERT_BATCH = 200


def quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def quote_text(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def is_numeric(value: str) -> bool:
    try:
        float(str(value).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def literal(value, column_type: str):
    """Render a SQL literal, using the column type to avoid text/number bugs."""
    if value is None:
        return "NULL"
    text = str(value)
    if column_type.upper() == "REAL" and is_numeric(text):
        number = float(text.replace(",", ""))
        return repr(int(number)) if number.is_integer() else repr(number)
    return quote_text(text)


def clean_headers(header: list) -> list[str]:
    """Trim, collapse whitespace, fill blanks and de-duplicate column names."""
    cleaned, seen = [], Counter()
    for index, name in enumerate(header):
        value = " ".join(str(name).split()) or f"column_{index}"
        seen[value] += 1
        if seen[value] > 1:
            value = f"{value}_{seen[value]}"
        cleaned.append(value)
    return cleaned


def column_types(table: dict, header: list[str]) -> list[str]:
    types = table.get("types") or []
    if len(types) != len(header):
        return ["TEXT"] * len(header)
    return ["REAL" if str(t).lower() == "real" else "TEXT" for t in types]


def table_name(table_id: str) -> str:
    return "table_" + str(table_id).replace("-", "_")


def build_schema(table: dict, headers: list[str], types: list[str]) -> str:
    columns = ", ".join(
        f"{quote_identifier(name)} {ctype}" for name, ctype in zip(headers, types)
    )
    return f"CREATE TABLE {quote_identifier(table_name(table['id']))} ({columns});"


def build_inserts(table: dict, headers: list[str], types: list[str]) -> list[str]:
    rows = table.get("rows") or []
    if not rows:
        return []
    columns = ", ".join(quote_identifier(name) for name in headers)
    prefix = f"INSERT INTO {quote_identifier(table_name(table['id']))} ({columns}) VALUES "
    statements = []
    for start in range(0, len(rows), INSERT_BATCH):
        batch = rows[start:start + INSERT_BATCH]
        values = []
        for row in batch:
            cells = list(row)[: len(headers)]
            cells += [None] * (len(headers) - len(cells))
            values.append(
                "(" + ", ".join(literal(cell, ctype) for cell, ctype in zip(cells, types)) + ")"
            )
        statements.append(prefix + ", ".join(values) + ";")
    return statements


def load_tables(path: Path) -> dict[str, dict]:
    tables = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            table = json.loads(line)
            tables[table["id"]] = table
    return tables


def load_examples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def download(url: str, destination: Path, expected_sha256: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"Downloading WikiSQL from {url}")
        temporary = destination.with_name(destination.name + ".part")
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length") or 0)
            bar = make_progress(
                total=total or None,
                desc="Downloading WikiSQL",
                unit="B",
                unit_scale=True,
            )
            with temporary.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    bar.update(len(chunk))
            bar.close()
        temporary.replace(destination)

    digest = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(
            f"WikiSQL archive checksum mismatch: {digest.hexdigest()} != "
            f"{expected_sha256}"
        )
    return destination


def extract(archive: Path, destination: Path) -> Path:
    data_dir = destination / "data"
    if data_dir.exists():
        return data_dir
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tar:
        try:
            tar.extractall(destination, filter="data")
        except TypeError:  # Python < 3.12 has no extraction filters.
            tar.extractall(destination)
    return data_dir


def normalize_split(
    split: str,
    examples: list[dict],
    tables: dict[str, dict],
    include_context: bool,
    validate: bool,
) -> tuple[list[dict], Counter]:
    """Render rows and optionally prove each gold query executes."""
    by_table: dict[str, list[dict]] = defaultdict(list)
    stats = Counter()
    for example in examples:
        by_table[example["table_id"]].append(example)

    rows: list[dict] = []
    bar = make_progress(
        total=len(by_table),
        desc=f"Validating {split}" if validate else f"Rendering {split}",
        unit="table",
    )
    for table_id, group in by_table.items():
        bar.update(1)
        table = tables.get(table_id)
        if table is None:
            stats["missing_table"] += len(group)
            continue

        headers = clean_headers(table["header"])
        types = column_types(table, headers)
        ddl = build_schema(table, headers, types)
        inserts = build_inserts(table, headers, types)
        context = "\n".join([ddl, *inserts]) if include_context else ""

        connection = None
        if validate:
            connection = sqlite3.connect(":memory:")
            try:
                connection.executescript(ddl)
                for statement in inserts:
                    connection.execute(statement)
            except sqlite3.Error:
                stats["bad_table"] += len(group)
                connection.close()
                continue

        for example in group:
            question = str(example.get("question", "")).strip()
            sql = example.get("sql") or {}
            if not question or not sql:
                stats["missing_fields"] += 1
                continue
            try:
                output = render_sql_with_table(sql, headers, types, table_id)
            except (KeyError, TypeError, ValueError):
                stats["bad_sql"] += 1
                continue

            if validate:
                try:
                    connection.execute(output).fetchall()
                except sqlite3.Error:
                    stats["unexecutable_sql"] += 1
                    continue

            rows.append(
                {
                    "id": f"wikisql-{split}-{example.get('id', len(rows))}",
                    "source": "wikisql",
                    "domain": "wikisql",
                    "sql_complexity": "basic SQL",
                    "sql_task_type": "data retrieval",
                    "question": question,
                    "schema": ddl,
                    "context": context,
                    "output": output,
                    "db_id": table_id,
                }
            )

        if connection is not None:
            connection.close()

    bar.close()
    return rows, stats


def render_sql_with_table(sql: dict, headers: list[str], types: list[str], table_id: str) -> str:
    """Render the gold query with the table name derived from the table id."""
    sel_index = int(sql["sel"])
    agg_index = int(sql["agg"])
    if not 0 <= sel_index < len(headers) or not 0 <= agg_index < len(AGG_OPS):
        raise ValueError("select/aggregation index out of range")

    selected = quote_identifier(headers[sel_index])
    select = f"{AGG_OPS[agg_index]}({selected})" if AGG_OPS[agg_index] else selected

    conditions = []
    for column_index, operator_index, value in sql.get("conds") or []:
        column_index, operator_index = int(column_index), int(operator_index)
        if not 0 <= column_index < len(headers) or not 0 <= operator_index < len(COND_OPS):
            raise ValueError("condition index out of range")
        conditions.append(
            f"{quote_identifier(headers[column_index])} "
            f"{COND_OPS[operator_index]} {literal(value, types[column_index])}"
        )

    query = f"SELECT {select} FROM {quote_identifier(table_name(table_id))}"
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    return query + ";"


def atomic_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare WikiSQL training data")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip execution checks (much faster, less safe)")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=2000,
                        help="Cap dev/test snapshots (they include row data)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs" / "training_config.yaml").read_text(encoding="utf-8"))
    wikisql_config = config["training_datasets"]["wikisql"]
    processed_dir = ROOT / config["dataset"].get("prepared_dir", "data/processed")
    cache_dir = ROOT / wikisql_config.get("cache_dir", ".hf_cache/wikisql")

    archive = download(
        wikisql_config.get("source_url", DEFAULT_URL),
        cache_dir / "data.tar.bz2",
        wikisql_config.get("sha256", EXPECTED_SHA256),
    )
    data_dir = extract(archive, cache_dir / "raw")

    info = {
        "dataset": "WikiSQL",
        "source_url": wikisql_config.get("source_url", DEFAULT_URL),
        "sha256": wikisql_config.get("sha256", EXPECTED_SHA256),
        "license": wikisql_config.get("license", ""),
        "splits": {},
    }

    for split, source in (("train", "train"), ("validation", "dev"), ("test", "test")):
        examples = load_examples(data_dir / f"{source}.jsonl")
        tables = load_tables(data_dir / f"{source}.tables.jsonl")
        print(f"{split}: {len(examples)} examples over {len(tables)} tables")

        rows, stats = normalize_split(
            split, examples, tables, include_context=split != "train",
            validate=not args.no_validate,
        )

        if split == "train" and args.max_train_rows:
            rows = rows[: args.max_train_rows]
        elif split != "train" and args.max_eval_rows:
            random.Random(args.seed).shuffle(rows)
            rows = rows[: args.max_eval_rows]

        atomic_json(processed_dir / f"wikisql_{split}.json", rows)
        print(f"  kept {len(rows)}; dropped {dict(stats) if stats else 'none'}")
        info["splits"][split] = {"rows": len(rows), "tables": len(tables),
                                 "dropped": dict(stats)}

    (processed_dir / "wikisql_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(f"Wrote WikiSQL snapshots to {processed_dir}")


if __name__ == "__main__":
    main()
