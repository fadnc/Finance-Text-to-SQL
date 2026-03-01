"""
Evaluation script for Text-to-SQL model.
Runs comprehensive evaluation with execution accuracy, exact match, and error analysis.
"""

import json
import yaml
import argparse
from pathlib import Path

from src.eval.eval_accuracy import (
    execution_accuracy, 
    exact_match_accuracy, 
    classify_error,
    evaluate_batch,
    ErrorType
)
from src.utils.logger import EvalLogger


def load_config():
    """Load configuration from YAML file."""
    with open("configs/training_config.yaml") as f:
        return yaml.safe_load(f)


def evaluate(dataset_path: str, db_path: str, output_path: str = None):
    """
    Run full evaluation on a dataset of predictions.
    
    Args:
        dataset_path: Path to JSON file with predictions
        db_path: Path to SQLite test database
        output_path: Optional path to save detailed results
    """
    logger = EvalLogger("sql_eval")
    
    with open(dataset_path) as f:
        data = json.load(f)

    predictions = [sample["predicted"] for sample in data]
    ground_truths = [sample["output"] for sample in data]
    
    # Run batch evaluation
    results = evaluate_batch(predictions, ground_truths, db_path)
    
    # Log individual samples
    for result in results["detailed_results"]:
        logger.log_sample(
            idx=result["index"],
            predicted=result["predicted"],
            expected=result["expected"],
            exec_match=result["exec_match"],
            exact_match=result["exact_match"],
            error_type=result["error_type"] if not result["exec_match"] else None
        )
    
    # Log summary
    logger.log_summary(
        exec_accuracy=results["execution_accuracy"],
        exact_accuracy=results["exact_match_accuracy"],
        error_breakdown=results["error_breakdown"]
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Samples: {results['total_samples']}")
    print(f"\nExecution Accuracy: {results['execution_accuracy']:.2%} ({results['exec_correct']}/{results['total_samples']})")
    print(f"Exact Match Accuracy: {results['exact_match_accuracy']:.2%} ({results['exact_correct']}/{results['total_samples']})")
    
    print("\nError Breakdown:")
    for error_type, count in results["error_breakdown"].items():
        if count > 0:
            pct = count / results['total_samples'] * 100
            print(f"  {error_type}: {count} ({pct:.1f}%)")
    
    # Save detailed results
    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {output_path}")
    
    return results


def evaluate_single(predicted_sql: str, ground_truth_sql: str, db_path: str):
    """
    Evaluate a single SQL query.
    
    Args:
        predicted_sql: Generated SQL
        ground_truth_sql: Expected SQL
        db_path: Path to database
        
    Returns:
        Dictionary with evaluation results
    """
    exec_match = execution_accuracy(predicted_sql, ground_truth_sql, db_path)
    exact_match = exact_match_accuracy(predicted_sql, ground_truth_sql)
    is_correct, error_type, error_msg = classify_error(predicted_sql, ground_truth_sql, db_path)
    
    return {
        "execution_match": exec_match,
        "exact_match": exact_match,
        "error_type": error_type.value,
        "error_message": error_msg
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL predictions")
    parser.add_argument("--dataset", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument("--db", type=str, default="data/processed/finance.db", help="Path to test database")
    parser.add_argument("--output", type=str, default=None, help="Path to save detailed results")
    
    args = parser.parse_args()
    
    evaluate(args.dataset, args.db, args.output)