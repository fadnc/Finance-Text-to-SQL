"""
SQL Query Evaluation Metrics.
Provides execution accuracy, exact match, and error analysis.
"""

import sqlite3
import re
from enum import Enum
from typing import Tuple, Optional


class ErrorType(Enum):
    """SQL error classification."""
    NONE = "none"
    SYNTAX = "syntax_error"
    SEMANTIC = "semantic_error"
    COLUMN_HALLUCINATION = "column_hallucination"
    TABLE_HALLUCINATION = "table_hallucination"
    LOGIC_ERROR = "logic_error"
    EXECUTION_ERROR = "execution_error"


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL query for comparison.
    Removes extra whitespace, converts to lowercase, removes trailing semicolon.
    """
    sql = sql.strip().lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';')
    return sql


def exact_match_accuracy(predicted_sql: str, ground_truth_sql: str) -> bool:
    """
    Check if predicted SQL exactly matches ground truth after normalization.
    
    Args:
        predicted_sql: Generated SQL query
        ground_truth_sql: Expected SQL query
        
    Returns:
        True if queries match exactly after normalization
    """
    return normalize_sql(predicted_sql) == normalize_sql(ground_truth_sql)


def execution_accuracy(
    predicted_sql: str, 
    ground_truth_sql: str, 
    db_path: str
) -> bool:
    """
    Check if predicted SQL produces the same result as ground truth.
    
    Args:
        predicted_sql: Generated SQL query
        ground_truth_sql: Expected SQL query
        db_path: Path to SQLite database
        
    Returns:
        True if result sets match
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        cursor.execute(predicted_sql)
        pred_result = set(cursor.fetchall())

        cursor.execute(ground_truth_sql)
        gt_result = set(cursor.fetchall())

        return pred_result == gt_result

    except Exception:
        return False
    finally:
        conn.close()


def classify_error(
    predicted_sql: str, 
    ground_truth_sql: str, 
    db_path: str,
    schema_columns: Optional[list] = None,
    schema_tables: Optional[list] = None
) -> Tuple[bool, ErrorType, str]:
    """
    Classify the type of error in a predicted SQL query.
    
    Args:
        predicted_sql: Generated SQL query
        ground_truth_sql: Expected SQL query
        db_path: Path to SQLite database
        schema_columns: List of valid column names
        schema_tables: List of valid table names
        
    Returns:
        Tuple of (is_correct, error_type, error_message)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get schema info if not provided
    if schema_tables is None:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        schema_tables = [row[0].lower() for row in cursor.fetchall()]
    
    if schema_columns is None:
        schema_columns = []
        for table in schema_tables:
            cursor.execute(f"PRAGMA table_info({table})")
            schema_columns.extend([row[1].lower() for row in cursor.fetchall()])
    
    try:
        # Try executing predicted SQL
        cursor.execute(predicted_sql)
        pred_result = set(cursor.fetchall())
        
        # Execute ground truth
        cursor.execute(ground_truth_sql)
        gt_result = set(cursor.fetchall())
        
        if pred_result == gt_result:
            return True, ErrorType.NONE, ""
        else:
            return False, ErrorType.LOGIC_ERROR, "Results do not match"
            
    except sqlite3.OperationalError as e:
        error_msg = str(e).lower()
        
        # Check for hallucinated columns
        if "no such column" in error_msg:
            # Extract the column name
            match = re.search(r'no such column: (\w+)', error_msg)
            col_name = match.group(1) if match else "unknown"
            return False, ErrorType.COLUMN_HALLUCINATION, f"Invalid column: {col_name}"
        
        # Check for hallucinated tables
        if "no such table" in error_msg:
            match = re.search(r'no such table: (\w+)', error_msg)
            table_name = match.group(1) if match else "unknown"
            return False, ErrorType.TABLE_HALLUCINATION, f"Invalid table: {table_name}"
        
        # Generic syntax error
        if "syntax error" in error_msg or "near" in error_msg:
            return False, ErrorType.SYNTAX, error_msg
        
        return False, ErrorType.EXECUTION_ERROR, error_msg
        
    except Exception as e:
        return False, ErrorType.EXECUTION_ERROR, str(e)
    
    finally:
        conn.close()


def evaluate_batch(
    predictions: list,
    ground_truths: list,
    db_path: str
) -> dict:
    """
    Evaluate a batch of SQL predictions.
    
    Args:
        predictions: List of predicted SQL queries
        ground_truths: List of ground truth SQL queries
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with metrics and error breakdown
    """
    exec_correct = 0
    exact_correct = 0
    error_counts = {e.value: 0 for e in ErrorType}
    
    results = []
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        exec_match = execution_accuracy(pred, gt, db_path)
        exact_match = exact_match_accuracy(pred, gt)
        is_correct, error_type, error_msg = classify_error(pred, gt, db_path)
        
        if exec_match:
            exec_correct += 1
        if exact_match:
            exact_correct += 1
        
        error_counts[error_type.value] += 1
        
        results.append({
            "index": i,
            "predicted": pred,
            "expected": gt,
            "exec_match": exec_match,
            "exact_match": exact_match,
            "error_type": error_type.value,
            "error_msg": error_msg
        })
    
    total = len(predictions)
    
    return {
        "execution_accuracy": exec_correct / total if total > 0 else 0,
        "exact_match_accuracy": exact_correct / total if total > 0 else 0,
        "total_samples": total,
        "exec_correct": exec_correct,
        "exact_correct": exact_correct,
        "error_breakdown": error_counts,
        "detailed_results": results
    }