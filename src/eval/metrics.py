"""Bounded, read-only SQL metrics used by model evaluation."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import sqlparse
from sqlparse import tokens as T


MAX_SQL_CHARS = 100_000
MAX_RESULT_ROWS = 100_000
MAX_VM_STEPS = 10_000_000
WRITE_ACTIONS = {
    getattr(sqlite3, name)
    for name in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TEMP_INDEX", "SQLITE_CREATE_TEMP_TABLE",
        "SQLITE_CREATE_TEMP_TRIGGER", "SQLITE_CREATE_TEMP_VIEW", "SQLITE_CREATE_TRIGGER",
        "SQLITE_CREATE_VIEW", "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE",
        "SQLITE_DROP_TEMP_INDEX", "SQLITE_DROP_TEMP_TABLE", "SQLITE_DROP_TEMP_TRIGGER",
        "SQLITE_DROP_TEMP_VIEW", "SQLITE_DROP_TRIGGER", "SQLITE_DROP_VIEW",
        "SQLITE_ALTER_TABLE", "SQLITE_ATTACH", "SQLITE_DETACH", "SQLITE_TRANSACTION",
        "SQLITE_SAVEPOINT", "SQLITE_PRAGMA", "SQLITE_REINDEX", "SQLITE_ANALYZE",
    )
    if hasattr(sqlite3, name)
}


def _parsed_statement(sql: str):
    statements = [statement for statement in sqlparse.parse(sql) if str(statement).strip()]
    if len(statements) != 1:
        raise ValueError("exactly one SQL statement is required")
    statement = statements[0]
    if statement.get_type().upper() != "SELECT":
        # sqlparse reports SELECT for SELECT and WITH ... SELECT statements.
        raise ValueError("only SELECT or WITH statements are permitted")
    return statement


def normalize_sql(sql: str) -> str:
    """Conservatively normalize tokens for string exact match.

    This ignores whitespace and comments, and folds SQL keyword case. It keeps
    literals, identifiers, operators, and token order unchanged. It is a string
    metric, not a proof that two different SQL expressions are equivalent.
    """
    result: list[tuple[str, str]] = []
    try:
        statement = _parsed_statement(sql)
        tokens = list(statement.flatten())
    except ValueError:
        # Preserve invalid SQL as tokenized text so invalid output never gains
        # equivalence through a broad regex rewrite.
        tokens = list(sqlparse.lexer.tokenize(sql))

    for item in tokens:
        if hasattr(item, "ttype"):
            token_type, value = item.ttype, item.value
        else:
            token_type, value = item
        if token_type in T.Whitespace or token_type in T.Comment:
            continue
        if token_type in T.Punctuation and value == ";":
            continue
        canonical = value.upper() if token_type in T.Keyword else value
        result.append((str(token_type), canonical))
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


def exact_match_accuracy(predicted_sql: str, ground_truth_sql: str) -> bool:
    return normalize_sql(predicted_sql) == normalize_sql(ground_truth_sql)


def _has_top_level_order_by(sql: str) -> bool:
    try:
        statement = _parsed_statement(sql)
    except ValueError:
        return False
    return any(
        token.ttype in T.Keyword.Order or token.normalized == "ORDER BY"
        for token in statement.tokens
    )


def _canonical_cell(value: Any) -> tuple[str, Any]:
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("number", str(int(value)))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return ("number", "NaN")
        if isinstance(value, float) and math.isinf(value):
            return ("number", "Infinity" if value > 0 else "-Infinity")
        try:
            number = Decimal(str(value)).normalize()
            return ("number", format(number, "f"))
        except InvalidOperation:
            pass
    if isinstance(value, bytes):
        return ("blob", value.hex())
    return ("text", str(value))


def equivalent_rows(
    predicted_rows: list[tuple],
    reference_rows: list[tuple],
    predicted_sql: str,
    reference_sql: str,
) -> bool:
    """Compare ordered rows when ORDER BY is present, otherwise multisets."""
    predicted = [tuple(_canonical_cell(value) for value in row) for row in predicted_rows]
    reference = [tuple(_canonical_cell(value) for value in row) for row in reference_rows]
    if _has_top_level_order_by(predicted_sql) or _has_top_level_order_by(reference_sql):
        return predicted == reference
    return Counter(predicted) == Counter(reference)


def _readonly_authorizer(action: int, arg1: str | None, arg2: str | None,
                         database: str | None, trigger: str | None) -> int:
    if action in WRITE_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == getattr(sqlite3, "SQLITE_FUNCTION", -1):
        function_name = (arg2 or arg1 or "").lower()
        if function_name in {"load_extension", "writefile", "readfile"}:
            return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _configure_limits(conn: sqlite3.Connection, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    vm_steps = 0
    timed_out = False
    step_limited = False

    def progress() -> int:
        nonlocal vm_steps, timed_out, step_limited
        vm_steps += 1000
        if time.monotonic() >= deadline:
            timed_out = True
            return 1
        if vm_steps > MAX_VM_STEPS:
            step_limited = True
            return 1
        return 0

    conn.set_progress_handler(progress, 1000)
    if hasattr(sqlite3, "SQLITE_LIMIT_SQL_LENGTH"):
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, MAX_SQL_CHARS)
    if hasattr(sqlite3, "SQLITE_LIMIT_COLUMN"):
        conn.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 2000)
    return {"timed_out": lambda: timed_out, "step_limited": lambda: step_limited}


def _open_database(db_path: str | None, context: str | None) -> sqlite3.Connection:
    if context:
        # Gretel contexts are prepared public test fixtures, not model output.
        conn = sqlite3.connect(":memory:")
        conn.executescript(context)
    elif db_path:
        path = Path(db_path).resolve()
        conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    else:
        raise ValueError("a disposable database path or test context is required")

    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_readonly_authorizer)
    return conn


def _classify_database_error(exc: Exception, status: dict[str, Any]) -> str:
    if status["timed_out"]() or "interrupted" in str(exc).lower():
        return "timeout"
    if status["step_limited"]():
        return "resource_limit"
    message = str(exc).lower()
    if "no such table" in message:
        return "missing_table"
    if "no such column" in message or "ambiguous column" in message:
        return "missing_column"
    if any(marker in message for marker in ("datatype mismatch", "type mismatch", "cannot store")):
        return "type_error"
    if "syntax error" in message or "incomplete input" in message or "near " in message:
        return "syntax_error"
    return "other_database_error"


def _fetch_bounded(conn: sqlite3.Connection, sql: str) -> list[tuple]:
    if len(sql) > MAX_SQL_CHARS:
        raise ValueError("SQL length limit exceeded")
    _parsed_statement(sql)
    cursor = conn.execute(sql)
    rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
    if len(rows) > MAX_RESULT_ROWS:
        raise OverflowError("result row limit exceeded")
    return rows


def execute_readonly(
    sql: str,
    *,
    db_path: str | None = None,
    context: str | None = None,
    timeout_s: float = 1.0,
) -> tuple[list[tuple], dict[str, Any]]:
    """Execute one read-only query against a local, disposable test fixture."""
    conn = None
    status = {"timed_out": lambda: False, "step_limited": lambda: False}
    started = time.monotonic()
    try:
        # Fixture setup can fail independently of the prediction (for example,
        # Gretel contexts containing MySQL-only views evaluated with SQLite).
        # Return that error as a reference-fixture failure instead of aborting
        # the full evaluation run.
        conn = _open_database(db_path, context)
        status = _configure_limits(conn, timeout_s)
        rows = _fetch_bounded(conn, sql)
        return rows, {"elapsed_s": time.monotonic() - started, "error_type": None}
    except OverflowError as exc:
        return [], {
            "elapsed_s": time.monotonic() - started,
            "error_type": "result_limit",
            "error": str(exc),
        }
    except Exception as exc:
        return [], {
            "elapsed_s": time.monotonic() - started,
            "error_type": _classify_database_error(exc, status),
            "error": str(exc),
        }
    finally:
        if conn is not None:
            conn.close()


def validate_query_schema(sql: str, schema: str, timeout_s: float = 1.0) -> dict[str, Any]:
    """Validate a generated query with EXPLAIN against a fresh schema-only DB."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema)
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_readonly_authorizer)
        status = _configure_limits(conn, timeout_s)
        _parsed_statement(sql)
        if len(sql) > MAX_SQL_CHARS:
            raise ValueError("SQL length limit exceeded")
        conn.execute("EXPLAIN QUERY PLAN " + sql).fetchmany(1000)
        return {"valid": True, "error_type": None, "error": None}
    except Exception as exc:
        return {
            "valid": False,
            "error_type": _classify_database_error(
                exc, locals().get("status", {"timed_out": lambda: False, "step_limited": lambda: False})
            ),
            "error": str(exc),
        }
    finally:
        conn.close()


def evaluate_pair(
    predicted_sql: str,
    reference_sql: str,
    *,
    context: str | None,
    timeout_s: float = 1.0,
) -> dict[str, Any]:
    """Compute exact match, validity, and execution equivalence for one case."""
    exact = exact_match_accuracy(predicted_sql, reference_sql)
    reference_rows, reference_status = execute_readonly(
        reference_sql, context=context, timeout_s=timeout_s
    )
    if reference_status["error_type"]:
        return {
            "exact_match": exact,
            "valid_sql": False,
            "execution_match": False,
            "reference_executable": False,
            "error_type": "reference_" + reference_status["error_type"],
            "error": reference_status.get("error"),
        }

    predicted_rows, predicted_status = execute_readonly(
        predicted_sql, context=context, timeout_s=timeout_s
    )
    if predicted_status["error_type"]:
        return {
            "exact_match": exact,
            "valid_sql": False,
            "execution_match": False,
            "reference_executable": True,
            "error_type": predicted_status["error_type"],
            "error": predicted_status.get("error"),
        }

    return {
        "exact_match": exact,
        "valid_sql": True,
        "execution_match": equivalent_rows(
            predicted_rows, reference_rows, predicted_sql, reference_sql
        ),
        "reference_executable": True,
        "error_type": None,
        "error": None,
    }


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float | int | None]:
    if not latencies_ms:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None}
    ordered = sorted(float(value) for value in latencies_ms)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered),
        "mean_ms": sum(ordered) / len(ordered),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }
