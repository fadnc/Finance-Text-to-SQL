"""
Quick validation script to test components without full training.
Run this first to verify setup before running the full pipeline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required packages are available."""
    print("Testing imports...")

    required = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("datasets", "Datasets"),
        ("trl", "TRL"),
        ("bitsandbytes", "BitsAndBytes"),
        ("mlflow", "MLflow"),
        ("fastapi", "FastAPI"),
        ("yaml", "PyYAML"),
    ]

    failed = []
    for module, name in required:
        try:
            __import__(module)
            print(f"   {name}")
        except ImportError:
            print(f"   {name} - MISSING")
            failed.append(name)

    return len(failed) == 0


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    import torch

    if torch.cuda.is_available():
        print(" CUDA available")
        print(f" Device: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(" CUDA not available - will use CPU (slower)")
    return True


def test_config():
    """Test config file."""
    print("\nTesting config...")
    import yaml

    try:
        with open("configs/training_config.yaml") as f:
            config = yaml.safe_load(f)

        required = ["model_name", "dataset", "output_dir", "schema_simple"]
        missing = [key for key in required if key not in config]

        if missing:
            print(f"   Missing config keys: {missing}")
            return False

        dataset = config["dataset"]
        for key in ["name", "train_split", "eval_split"]:
            if key not in dataset:
                print(f"   Missing dataset config key: {key}")
                return False

        print("   Config valid")
        print(f"     Model: {config['model_name']}")
        print(f"     Dataset: {dataset['name']}")
        print(f"     Domains: {dataset.get('domains') or 'all'}")
        return True

    except Exception as exc:
        print(f"   Config error: {exc}")
        return False


def test_dataset():
    """Load the configured dataset and verify the normalized fields."""
    print("\nTesting dataset (may download/cache the source dataset)...")
    import yaml

    from src.data.gretel import load_gretel_splits

    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    try:
        splits = load_gretel_splits(config)
    except Exception as exc:
        print(f"   Dataset load failed: {exc}")
        return False

    print(
        f"   Rows: train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, test={len(splits['test'])}"
    )

    required = ["question", "schema", "context", "output"]
    sample = splits["train"][0]
    missing = [key for key in required if key not in sample]
    if missing:
        print(f"   Missing prepared fields: {missing}")
        return False

    if not sample["schema"]:
        print("   Schema extraction produced an empty schema")
        return False

    print("   Format correct")
    return True


def test_supplementary_datasets():
    """Verify prepared Spider and WikiSQL snapshots and their mixture."""
    print("\nTesting supplementary datasets...")
    import json
    from pathlib import Path

    import yaml

    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    expected = {
        "spider": ("spider_train.json", "data/prepare_multi_dataset.py"),
        "wikisql": ("wikisql_train.json", "data/prepare_wikisql.py"),
    }
    missing = []
    for name, settings in (config.get("training_datasets") or {}).items():
        if not settings.get("include", False):
            continue
        filename, script = expected[name]
        path = Path("data/processed") / filename
        if not path.exists():
            missing.append(f"{path} (run {script})")
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        print(f"   {name}: {len(rows)} prepared training rows")

    if missing:
        print(f"   Missing snapshots: {missing}")
        return False
    return True


def test_database():
    """Test the optional demo SQLite database."""
    print("\nTesting demo database...")
    import sqlite3
    from pathlib import Path

    db_path = Path("data/processed/finance.db")

    if not db_path.exists():
        print("   Demo database not found (optional). Run: python data/setup_test_db.py")
        return True

    conn = sqlite3.connect(db_path)
    tables = [
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    ]
    conn.close()

    expected = ["users", "accounts", "expenses", "income", "budgets", "transactions"]
    missing = [table for table in expected if table not in tables]

    if missing:
        print(f"   Missing tables: {missing}")
        return False

    print(f"   Database ready with tables: {', '.join(tables)}")
    return True


def test_eval_sample():
    """Test evaluation against an in-memory SQL context."""
    print("\nTesting evaluation pipeline...")

    from src.eval.eval_accuracy import classify_error, execution_accuracy

    context = "CREATE TABLE t (x INT); INSERT INTO t VALUES (1), (2);"
    sql = "SELECT COUNT(*) FROM t"

    if not execution_accuracy(sql, sql, context=context):
        print("   Execution accuracy failed")
        return False

    is_correct, error_type, _ = classify_error(
        "SELECT * FROM nonexistent_table", sql, context=context
    )
    if is_correct or error_type.value != "table_hallucination":
        print(f"   Error classification failed: {error_type.value}")
        return False

    print("   Execution accuracy works")
    print(f"   Error classification works: {error_type.value}")
    return True


def test_model_download():
    """Test that the base tokenizer can be accessed."""
    print("\nTesting model access (this may download ~2GB)...")
    import yaml
    from transformers import AutoTokenizer

    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    try:
        AutoTokenizer.from_pretrained(config["model_name"])
        print(f"   Model accessible: {config['model_name']}")
        return True
    except Exception as exc:
        print(f"   Cannot access model: {exc}")
        return False


def main():
    print("=" * 60)
    print("FINANCE TEXT-TO-SQL - QUICK VALIDATION")
    print("=" * 60)

    tests = [
        (test_imports, "Package imports"),
        (test_cuda, "CUDA availability"),
        (test_config, "Configuration"),
        (test_dataset, "Training dataset"),
        (test_supplementary_datasets, "Supplementary datasets"),
        (test_database, "Demo database"),
        (test_eval_sample, "Evaluation pipeline"),
    ]

    results = []
    for test_fn, name in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as exc:
            print(f"  Error: {exc}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n All checks passed! Ready for training.")
        print("\nNext steps:")
        print("  1. Run training:      python -m src.training.train")
        print("  2. Or full pipeline:  python scripts/run_pipeline.py")
    else:
        print("\n Some checks failed. Fix issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
