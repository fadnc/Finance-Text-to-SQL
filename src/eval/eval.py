"""
Evaluation script for the Text-to-SQL model.

Runs execution accuracy, exact match, and error analysis.  When predictions
contain a ``context`` (as generated from the Gretel dataset), each sample is
executed against its own in-memory SQLite database.  Otherwise the shared
``--db`` fixture is used for backwards compatibility.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import yaml

from src.eval.eval_accuracy import (
    ErrorType,
    classify_error,
    evaluate_batch,
    execution_accuracy,
    exact_match_accuracy,
)
from src.utils.logger import EvalLogger


def load_config():
    with open("configs/training_config.yaml") as f:
        return yaml.safe_load(f)


def _write_dashboard_report(results, data, config, output_path):
    failures = [
        {
            "question": item.get("question", ""),
            "predicted": item["predicted"],
            "expected": item["expected"],
        }
        for item in results["detailed_results"]
        if not item["exec_match"] and item["gold_executable"]
    ][:3]

    report = {
        "model": f"{config['model_name']} + LoRA adapter",
        "evaluated_at": date.today().isoformat(),
        "test_samples": results["evaluated_samples"],
        "total_test_samples": results["total_samples"],
        "execution_accuracy": results["execution_accuracy"],
        "execution_correct": results["exec_correct"],
        "exact_match_accuracy": results["exact_match_accuracy"],
        "exact_match_correct": results["exact_correct"],
        "gold_execution_errors": results["gold_execution_errors"],
        "method": (
            "Greedy generation on the official gretelai/synthetic_text_to_sql "
            "finance test split; per-example SQLite context execution comparison "
            "and normalized exact SQL match."
        ),
        "error_breakdown": {
            key: value
            for key, value in results["error_breakdown"].items()
            if value and key not in {ErrorType.NONE.value, ErrorType.GOLD_EXECUTION_ERROR.value}
        },
        "failure_examples": failures,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Dashboard report saved to: {output_path}")


def evaluate(dataset_path: str, db_path: str = None, output_path: str = None,
             dashboard_output: str = None):
    """Run full evaluation on a dataset of predictions."""
    config = load_config()
    logger = EvalLogger("sql_eval")

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    predictions = [sample["predicted"] for sample in data]
    ground_truths = [sample["output"] for sample in data]
    contexts = [sample.get("context") for sample in data]
    use_contexts = any(contexts)

    if not use_contexts and db_path is None:
        db_path = config.get("db_path")

    results = evaluate_batch(
        predictions,
        ground_truths,
        db_path=None if use_contexts else db_path,
        contexts=contexts if use_contexts else None,
    )

    # Attach source metadata to detailed results for inspection.
    for result, sample in zip(results["detailed_results"], data):
        result["question"] = sample.get("question", "")
        result["id"] = sample.get("id")
        result["domain"] = sample.get("domain", "")
        result["sql_complexity"] = sample.get("sql_complexity", "")

    for result in results["detailed_results"]:
        logger.log_sample(
            idx=result["index"],
            predicted=result["predicted"],
            expected=result["expected"],
            exec_match=result["exec_match"],
            exact_match=result["exact_match"],
            error_type=result["error_type"] if not result["exec_match"] else None,
        )

    logger.log_summary(
        exec_accuracy=results["execution_accuracy"],
        exact_accuracy=results["exact_match_accuracy"],
        error_breakdown=results["error_breakdown"],
        evaluated_samples=results["evaluated_samples"],
        gold_execution_errors=results["gold_execution_errors"],
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total samples: {results['total_samples']}")
    print(f"Gold queries executable: {results['evaluated_samples']}")
    print(f"Gold execution errors: {results['gold_execution_errors']}")
    print(
        f"\nExecution Accuracy: {results['execution_accuracy']:.2%} "
        f"({results['exec_correct']}/{results['evaluated_samples']})"
    )
    print(
        f"Execution Accuracy (all samples): "
        f"{results['execution_accuracy_all_samples']:.2%} "
        f"({results['exec_correct']}/{results['total_samples']})"
    )
    print(
        f"Exact Match Accuracy: {results['exact_match_accuracy']:.2%} "
        f"({results['exact_correct']}/{results['total_samples']})"
    )

    print("\nError Breakdown:")
    for error_type, count in results["error_breakdown"].items():
        if count > 0:
            pct = count / results["total_samples"] * 100
            print(f"  {error_type}: {count} ({pct:.1f}%)")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")

    if dashboard_output:
        _write_dashboard_report(results, data, config, dashboard_output)

    return results


def evaluate_single(predicted_sql: str, ground_truth_sql: str, db_path: str = None,
                    context: str = None):
    """Evaluate a single SQL query."""
    exec_match = execution_accuracy(predicted_sql, ground_truth_sql, db_path, context)
    exact_match = exact_match_accuracy(predicted_sql, ground_truth_sql)
    is_correct, error_type, error_msg = classify_error(
        predicted_sql, ground_truth_sql, db_path, context=context
    )

    return {
        "execution_match": exec_match,
        "exact_match": exact_match,
        "error_type": error_type.value,
        "error_message": error_msg,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Text-to-SQL predictions")
    parser.add_argument("--dataset", type=str, required=True, help="Path to predictions JSON")
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to a shared SQLite database (legacy predictions without contexts)",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save detailed results")
    parser.add_argument(
        "--dashboard-output",
        type=str,
        default=None,
        help="Optional path for the dashboard evaluation JSON",
    )

    args = parser.parse_args()
    evaluate(args.dataset, args.db, args.output, args.dashboard_output)
