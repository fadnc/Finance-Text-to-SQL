"""Prepare completion-only SFT examples without truncating reference SQL."""

from __future__ import annotations

import re

import sqlparse
from datasets import Dataset
from sqlparse import tokens as T

from src.utils.prompting import build_prompt
from src.utils.progress import make_progress

MUTATING_KEYWORDS = {"INSERT", "UPDATE", "DELETE", "REPLACE", "CREATE", "ALTER", "DROP"}
TABLE_REFERENCE = re.compile(
    r"\b(?:FROM|JOIN|UPDATE|INTO)\s+(\[[^\]]+\]|`[^`]+`|\"[^\"]+\"|[\w.]+)",
    re.IGNORECASE,
)
CREATE_TABLE = re.compile(
    r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:\[([^\]]+)\]|`([^`]+)`|\"([^\"]+)\"|([\w.]+))",
    re.IGNORECASE,
)


def _is_read_only_select(sql: str) -> bool:
    statements = [statement for statement in sqlparse.parse(sql) if str(statement).strip()]
    if len(statements) != 1 or statements[0].get_type().upper() != "SELECT":
        return False
    return not any(
        token.ttype in T.Keyword.DML and token.normalized.upper() in MUTATING_KEYWORDS
        for token in statements[0].flatten()
    )


def _table_name(statement: str) -> str | None:
    match = CREATE_TABLE.search(statement)
    if not match:
        return None
    return next((part for part in match.groups() if part), "").split(".")[-1].casefold()


def _referenced_tables(sql: str) -> set[str]:
    names = set()
    for match in TABLE_REFERENCE.finditer(sql):
        name = match.group(1).strip("[]`\"").split(".")[-1]
        names.add(name.casefold())
    return names


def _fits(tokenizer, prompt: str, completion: str, max_length: int) -> bool:
    ids = tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
    # Reserve one token for the EOS marker added by SFTTrainer.
    return len(ids) + 1 <= max_length


def _fit_schema(example: dict, tokenizer, max_length: int) -> str | None:
    schema = str(example["schema"]).strip()
    question = str(example["question"])
    completion = str(example["output"]).strip()
    base_prompt = build_prompt(question, "", tokenizer)
    if not _fits(tokenizer, base_prompt, completion, max_length):
        return None
    if _fits(
        tokenizer,
        build_prompt(question, schema, tokenizer),
        completion,
        max_length,
    ):
        return schema

    statements = [part.strip() for part in sqlparse.split(schema) if part.strip()]
    if not statements:
        return "" if _fits(tokenizer, base_prompt, completion, max_length) else None

    table_statements = [(statement, _table_name(statement)) for statement in statements]
    references = _referenced_tables(completion)
    required = [statement for statement, name in table_statements if name and name in references]
    optional = [statement for statement, name in table_statements if statement not in required]

    selected = list(required)
    selected_schema = "\n".join(selected)
    if not _fits(tokenizer, build_prompt(question, selected_schema, tokenizer), completion, max_length):
        return None

    for statement in optional:
        candidate = "\n".join([*selected, statement])
        if _fits(tokenizer, build_prompt(question, candidate, tokenizer), completion, max_length):
            selected.append(statement)
    return "\n".join(selected)


def format_dataset(dataset, tokenizer, max_length: int):
    """Return TRL prompt/completion data, masking prompts from the training loss.

    Only read-only SELECT targets are kept because the application rejects
    writes. Referenced table DDL is prioritized when a schema must be shortened;
    rows are dropped when question + referenced schema + full target cannot fit.
    """
    formatted = []
    rejected_non_select = rejected_too_long = 0
    bar = make_progress(total=len(dataset), desc="Formatting SFT examples")
    for example in dataset:
        completion = str(example["output"]).strip()
        if not _is_read_only_select(completion):
            rejected_non_select += 1
            bar.update(1)
            continue
        schema = _fit_schema(example, tokenizer, max_length)
        if schema is None:
            rejected_too_long += 1
            bar.update(1)
            continue
        formatted.append(
            {
                "prompt": build_prompt(example["question"], schema, tokenizer),
                "completion": completion,
            }
        )
        bar.update(1)
    bar.close()
    print(
        f"Prepared {len(formatted)} read-only SFT examples; excluded "
        f"{rejected_non_select} non-SELECT rows and {rejected_too_long} rows "
        "that could not retain the full SQL target within max_seq_length."
    )
    return Dataset.from_list(formatted)
