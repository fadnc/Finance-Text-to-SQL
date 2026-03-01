"""
End-to-end pipeline for training, inference, and evaluation.
Run this to verify the project works correctly.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"❌ FAILED: {description}")
        return False
    print(f"✅ SUCCESS: {description}")
    return True


def main():
    steps = [
        ("python data/gen_dataset.py", "Generate training dataset"),
        ("python data/setup_test_db.py", "Create test database"),
        ("python -m src.training.train", "Train model with QLoRA"),
        ("python scripts/generate_predictions.py", "Generate predictions on test set"),
        ("python -m src.eval.eval --dataset data/processed/predictions.json --db data/processed/finance.db --output data/processed/eval_results.json", "Evaluate model"),
    ]
    
    print("\n" + "="*60)
    print("FINANCE TEXT-TO-SQL PIPELINE")
    print("="*60)
    
    for cmd, desc in steps:
        if not run_command(cmd, desc):
            print(f"\n⚠️ Pipeline stopped at: {desc}")
            sys.exit(1)
    
    print("\n" + "="*60)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nCheck results in: data/processed/eval_results.json")


if __name__ == "__main__":
    main()
