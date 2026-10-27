"""
SQL Query Evaluation Metrics.

Provides execution accuracy against either a shared SQLite database or the
per-example SQL context stored in the Gretel dataset.  The latter is necessary
because each dataset row can define its own schema and seed rows.
"""

import re
import sqlite3
from enum import Enum
from typing import Optional, Tuple


class ErrorType(Enum):
    """SQL error classification."""

    NONE = "none"
    SYNTAX = "syntax_error"
    SEMANTIC = "semantic_error"
    COLUMN_HALLUCINATION = "column_hallucination"
    TABLE_HALLUCINATION = "table_hallucination"
    LOGIC_ERROR = "logic_error"
    EXECUTION_ERROR = "execution_error"
    GOLD_EXECUTION_ERROR = "gold_execution_error"
    CONTEXT_ERROR = "context_error"


def normalize_sql(sql: str) -> str:
    """Normalize SQL for exact-string comparison."""
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";")
    return sql


def exact_match_accuracy(predicted_sql: str, ground_truth_sql: str) -> bool:
    """Check whether two SQL strings match after normalization."""
    return normalize_sql(predicted_sql) == normalize_sql(ground_truth_sql)


def _open_database(db_path: Optional[str], context: Optional[str]) -> sqlite3.Connection:
    """Open a read-only connection to a fixture DB or an in-memory context."""
    if context:
        conn = sqlite3.connect(":memory:")
        conn.executescript(context)
        conn.execute("PRAGMA query_only = ON")
        return conn

    if db_path:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA query_only = ON")
        return conn

    raise ValueError("Either db_path or context must be provided")


def _classify_sqlite_error(exc: Exception) -> Tuple[ErrorType, str]:
    error_msg = str(exc)
    lower = error_msg.lower()

    if "no such column" in lower:
        match = re.search(r"no such column: (\w+)", lower)
        column = match.group(1) if match else "unknown"
        return ErrorType.COLUMN_HALLUCINATION, f"Invalid column: {column}"

    if "no such table" in lower:
        match = re.search(r"no such table: (\w+)", lower)
        table = match.group(1) if match else "unknown"
        return ErrorType.TABLE_HALLUCINATION, f"Invalid table: {table}"

    if "syntax error" in lower or "near " in lower:
        return ErrorType.SYNTAX, error_msg

    return ErrorType.EXECUTION_ERROR, error_msg


def evaluate_example(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: Optional[str] = None,
    context: Optional[str] = None,
) -> dict:
    """Evaluate one prediction against a shared DB or an in-memory context."""
    exact_match = exact_match_accuracy(predicted_sql, ground_truth_sql)

    try:
        conn = _open_database(db_path, context)
    except Exception as exc:
        return {
            "predicted": predicted_sql,
            "expected": ground_truth_sql,
            "exec_match": False,
            "exact_match": exact_match,
            "gold_executable": False,
            "error_type": ErrorType.CONTEXT_ERROR.value,
            "error_msg": str(exc),
        }

    try:
        try:
            gold_rows = conn.execute(ground_truth_sql).fetchall()
        except Exception as exc:
            return {
                "predicted": predicted_sql,
                "expected": ground_truth_sql,
                "exec_match": False,
                "exact_match": exact_match,
                "gold_executable": False,
                "error_type": ErrorType.GOLD_EXECUTION_ERROR.value,
                "error_msg": str(exc),
            }

        try:
            predicted_rows = conn.execute(predicted_sql).fetchall()
        except Exception as exc:
            error_type, error_msg = _classify_sqlite_error(exc)
            return {
                "predicted": predicted_sql,
                "expected": ground_truth_sql,
                "exec_match": False,
                "exact_match": exact_match,
                "gold_executable": True,
                "error_type": error_type.value,
                "error_msg": error_msg,
            }

        exec_match = set(predicted_rows) == set(gold_rows)
        return {
            "predicted": predicted_sql,
            "expected": ground_truth_sql,
            "exec_match": exec_match,
            "exact_match": exact_match,
            "gold_executable": True,
            "error_type": ErrorType.NONE.value if exec_match else ErrorType.LOGIC_ERROR.value,
            "error_msg": "" if exec_match else "Results do not match",
        }
    finally:
        conn.close()


def execution_accuracy(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: Optional[str] = None,
    context: Optional[str] = None,
) -> bool:
    """Return True when predicted and ground-truth queries return the same rows."""
    result = evaluate_example(predicted_sql, ground_truth_sql, db_path, context)
    return result["exec_match"]


def classify_error(
    predicted_sql: str,
    ground_truth_sql: str,
    db_path: Optional[str] = None,
    schema_columns: Optional[list] = None,
    schema_tables: Optional[list] = None,
    context: Optional[str] = None,
) -> Tuple[bool, ErrorType, str]:
    """Classify a prediction as correct, logic error, or execution failure."""
    result = evaluate_example(predicted_sql, ground_truth_sql, db_path, context)
    error_type = ErrorType(result["error_type"])
    return result["exec_match"], error_type, result["error_msg"]


def evaluate_batch(
    predictions: list,
    ground_truths: list,
    db_path: Optional[str] = None,
    contexts: Optional[list] = None,
) -> dict:
    """Evaluate a batch of predictions.

    Execution accuracy is reported over samples whose ground-truth SQL executes
    successfully.  Some Gretel contexts use SQLite features (or write
    statements) that cannot be executed read-only; those are reported as
    ``gold_execution_errors`` instead of being counted as model failures.
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("predictions and ground_truths must have the same length")

    if contexts is None:
        contexts = [None] * len(predictions)
    elif len(contexts) != len(predictions):
        raise ValueError("contexts must have the same length as predictions")

    exec_correct = 0
    exact_correct = 0
    evaluated_samples = 0
    error_counts = {error_type.value: 0 for error_type in ErrorType}
    results = []

    for index, (prediction, ground_truth) in enumerate(zip(predictions, ground_truths)):
        result = evaluate_example(
            prediction,
            ground_truth,
            db_path=db_path,
            context=contexts[index],
        )
        if result["exec_match"]:
            exec_correct += 1
        if result["exact_match"]:
            exact_correct += 1
        if result["gold_executable"]:
            evaluated_samples += 1

        error_counts[result["error_type"]] += 1
        results.append({"index": index, **result})

    total = len(predictions)
    gold_errors = total - evaluated_samples

    return {
        "execution_accuracy": exec_correct / evaluated_samples if evaluated_samples else 0,
        "execution_accuracy_all_samples": exec_correct / total if total else 0,
        "exact_match_accuracy": exact_correct / total if total else 0,
        "total_samples": total,
        "evaluated_samples": evaluated_samples,
        "gold_execution_errors": gold_errors,
        "exec_correct": exec_correct,
        "exact_correct": exact_correct,
        "error_breakdown": error_counts,
        "detailed_results": results,
    }
