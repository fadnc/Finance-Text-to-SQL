"""
Quick validation script to test components without full training.
Run this first to verify setup before running full pipeline.
"""

import sys

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
        print(f" CUDA available")
        print(f" Device: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print(" CUDA not available - will use CPU (slower)")
        return True  # Not a failure, just a warning


def test_dataset():
    """Test dataset exists and has correct format."""
    print("\nTesting dataset...")
    import json
    from pathlib import Path
    
    dataset_path = Path("data/processed/finance_sql.json")
    
    if not dataset_path.exists():
        print("   Dataset not found. Run: python data/gen_dataset.py")
        return False
    
    with open(dataset_path) as f:
        data = json.load(f)
    
    print(f"   Dataset loaded: {len(data)} samples")
    
    # Check format
    required_keys = ["instruction", "input", "output"]
    sample = data[0]
    
    for key in required_keys:
        if key not in sample:
            print(f" Missing key: {key}")
            return False
    
    print(f" Format correct")
    return True


def test_database():
    """Test SQLite database exists."""
    print("\nTesting database...")
    import sqlite3
    from pathlib import Path
    
    db_path = Path("data/processed/finance.db")
    
    if not db_path.exists():
        print("   Database not found. Run: python data/setup_test_db.py")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    expected = ["users", "accounts", "expenses", "income", "budgets", "transactions"]
    missing = [t for t in expected if t not in tables]
    
    if missing:
        print(f"   Missing tables: {missing}")
        return False
    
    print(f" Database ready with tables: {', '.join(tables)}")
    return True


def test_config():
    """Test config file."""
    print("\nTesting config...")
    import yaml
    
    try:
        with open("configs/training_config.yaml") as f:
            config = yaml.safe_load(f)
        
        required = ["model_name", "dataset_path", "output_dir", "schema_simple"]
        missing = [k for k in required if k not in config]
        
        if missing:
            print(f"   Missing config keys: {missing}")
            return False
        
        print(f"   Config valid")
        print(f"     Model: {config['model_name']}")
        return True
        
    except Exception as e:
        print(f"   Config error: {e}")
        return False


def test_model_download():
    """Test that model can be downloaded/accessed."""
    print("\nTesting model access (this may download ~2GB)...")
    
    from transformers import AutoTokenizer
    import yaml
    
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        print(f" Model accessible: {config['model_name']}")
        return True
    except Exception as e:
        print(f" Cannot access model: {e}")
        return False


def test_eval_sample():
    """Test evaluation on a sample query."""
    print("\nTesting evaluation pipeline...")
    
    import sys
    from pathlib import Path
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.eval.eval_accuracy import execution_accuracy, classify_error
    
    db_path = "data/processed/finance.db"
    
    # Test with valid query
    sql = "SELECT COUNT(*) FROM users"
    is_match = execution_accuracy(sql, sql, db_path)
    
    if is_match:
        print(" Execution accuracy works")
    else:
        print(" Execution accuracy failed")
        return False
    
    # Test error classification
    bad_sql = "SELECT * FROM nonexistent_table"
    is_correct, error_type, msg = classify_error(bad_sql, sql, db_path)
    
    print(f"   Error classification works: {error_type.value}")
    
    return True


def main():
    print("="*60)
    print("FINANCE TEXT-TO-SQL - QUICK VALIDATION")
    print("="*60)
    
    tests = [
        (test_imports, "Package imports"),
        (test_cuda, "CUDA availability"),
        (test_config, "Configuration"),
        (test_dataset, "Training dataset"),
        (test_database, "Test database"),
        (test_eval_sample, "Evaluation pipeline"),
    ]
    
    results = []
    for test_fn, name in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n All checks passed! Ready for training.")
        print("\nNext steps:")
        print("  1. Run training:  python -m src.training.train")
        print("  2. Or full pipeline: python scripts/run_pipeline.py")
    else:
        print("\n Some checks failed. Fix issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
