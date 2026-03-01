# Finance Text-to-SQL

Domain-adaptive fine-tuning of small language models for natural language to SQL translation on personal finance data using QLoRA.

## Overview

This project demonstrates efficient domain adaptation of instruction-tuned LLMs for Text-to-SQL tasks. Key features:

- **Parameter-Efficient Fine-Tuning**: QLoRA with 4-bit NF4 quantization and LoRA adapters
- **Memory Optimization**: Gradient checkpointing and paged optimizers for training on consumer GPUs
- **Execution-Based Evaluation**: Beyond token matching—evaluates actual query results
- **Error Analysis**: Classifies failures as syntax, semantic, or hallucination errors

## Project Structure

```
├── api/                    # FastAPI inference server
│   └── main.py
├── configs/
│   └── training_config.yaml    # Centralized configuration
├── data/
│   ├── gen_dataset.py          # Synthetic dataset generator
│   ├── setup_test_db.py        # Test database creation
│   └── processed/
│       ├── finance_sql.json    # Training data
│       └── finance.db          # SQLite test database
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/                 # Saved model adapters
├── src/
│   ├── eval/
│   │   ├── eval.py            # Evaluation runner
│   │   └── eval_accuracy.py   # Metrics (exec accuracy, exact match, error analysis)
│   ├── inference/
│   │   └── gen_sql.py         # SQL generation utility
│   ├── training/
│   │   ├── train.py           # Main training script
│   │   ├── lora_config.py     # LoRA configuration
│   │   └── formatting.py      # Dataset formatting
│   └── utils/
│       ├── logger.py          # Logging utilities
│       └── model_loader.py    # Model loading with quantization
└── mlruns/                 # MLflow experiment tracking
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (8GB+ VRAM recommended)
- PyTorch with CUDA support

### Installation

```bash
# Clone repository
git clone <repo-url>
cd Finance-Text-to-SQL

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Generate Data

```bash
# Create synthetic training dataset (225+ samples)
python data/gen_dataset.py

# Create test database
python data/setup_test_db.py
```

## Training

### Configure Training

Edit `configs/training_config.yaml`:

```yaml
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0  # or microsoft/Phi-3-mini-4k-instruct
batch_size: 2
epochs: 3
learning_rate: 2e-4
```

### Run Training

```bash
python -m src.training.train
```

Training features:
- ✅ 4-bit NF4 quantization (reduces VRAM ~4x)
- ✅ LoRA adapters (trains only ~0.1% of parameters)
- ✅ Gradient checkpointing (trades compute for memory)
- ✅ Paged AdamW 8-bit optimizer (further memory reduction)
- ✅ MLflow experiment tracking

## Evaluation

### Run Evaluation

```bash
python -m src.eval.eval --dataset data/processed/predictions.json --db data/processed/finance.db
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Execution Accuracy** | Predicted SQL produces same result set as ground truth |
| **Exact Match Accuracy** | Normalized SQL strings match exactly |
| **Error Types** | Syntax, column hallucination, table hallucination, logic errors |

## Inference

### Python API

```python
from src.inference.gen_sql import generate_sql
from src.utils.model_loader import load_model
from peft import PeftModel

model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(model, "models/phi3-sql")

sql = generate_sql(
    model, tokenizer,
    question="How much did Fadhil spend on Food?",
    schema="users(user_id, name), expenses(expense_id, user_id, category, amount, date)"
)
print(sql)
```

### REST API

```bash
# Start server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or with Docker
docker-compose -f docker/docker-compose.yml up
```

```bash
# Query endpoint
curl -X POST "http://localhost:8000/generate-sql" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total spending by category?"}'
```

## Schema

The model is trained on a personal finance schema:

```sql
users(user_id, name, email, created_at)
accounts(account_id, user_id, account_type, balance, currency)
expenses(expense_id, user_id, category, amount, date, description)
income(income_id, user_id, source, amount, date)
budgets(budget_id, user_id, category, monthly_limit, start_date)
transactions(txn_id, account_id, txn_type, amount, date, merchant)
```

## Dataset Coverage

The training dataset includes diverse SQL patterns:

- ✅ Simple aggregations (SUM, COUNT, AVG, MIN, MAX)
- ✅ GROUP BY with HAVING clauses
- ✅ Multi-table JOINs
- ✅ Subqueries (correlated and non-correlated)
- ✅ Date filtering (relative dates, quarters, months)
- ✅ ORDER BY / LIMIT
- ✅ CASE statements
- ✅ Budget vs actual comparisons

## Technical Details

### QLoRA Configuration

```python
LoraConfig(
    r=16,                           # Rank
    lora_alpha=32,                  # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
```

### Quantization

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # Normal Float 4
    bnb_4bit_compute_dtype=torch.float16

)
```

## Hardware Requirements

| Configuration | VRAM | Training Time (3 epochs) |
|---------------|------|--------------------------|
| TinyLlama 1.1B | ~6GB | ~10 min |
| Phi-3 Mini 3.8B | ~10GB | ~30 min |

## Known Issues

- `bitsandbytes` may have compatibility issues on Windows
- 4-bit quantization requires CUDA compute capability 7.0+

## License

MIT