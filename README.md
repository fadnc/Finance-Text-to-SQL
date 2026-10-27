# Finance Text-to-SQL (Gretel)

Domain-adaptive fine-tuning of small language models for natural language to SQL
on the **finance subset of [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)** using QLoRA.

## Overview

This project demonstrates efficient domain adaptation of instruction-tuned LLMs
for Text-to-SQL. Key features:

- **Public dataset, not hand-written templates**: loads `gretelai/synthetic_text_to_sql` with `datasets.load_dataset`
- **Per-example schemas**: each training row ships its own DDL; the model is prompted with the schema instead of memorising one fixed database
- **Parameter-efficient fine-tuning**: QLoRA with 4-bit NF4 quantization and LoRA adapters
- **Memory optimization**: gradient checkpointing and paged optimizers for consumer GPUs
- **Execution-based evaluation**: creates an in-memory SQLite database from each sample's `sql_context` and compares actual query results
- **Error analysis**: classifies failures as syntax, column/table hallucination, logic, or gold-execution errors

## Dataset

The source dataset is loaded directly from the Hugging Face Hub:

```python
from datasets import load_dataset

ds = load_dataset("gretelai/synthetic_text_to_sql")
# DatasetDict({
#   train: 100,000 rows
#   test:   5,851 rows
# })
```

The full dataset covers 100+ domains. This project filters to the `finance`
domain by default:

| Split | Rows |
|---|---:|
| Train (after 10% validation holdout) | 945 |
| Validation | 106 |
| Official test | 72 |

Each row contains:

| Field | Use |
|---|---|
| `sql_prompt` | natural-language question |
| `sql_context` | `CREATE TABLE` + optional `INSERT` statements |
| `sql` | reference query |
| `sql_complexity` | basic SQL, aggregation, joins, subqueries, window functions, CTEs, set operations |
| `sql_task_type` | analytics/reporting, data retrieval, data manipulation, data definition |

Only the DDL portion of `sql_context` is inserted into the model prompt. The
full context (including seed rows) is kept separately for evaluation.

To use every domain instead of finance only, set `dataset.domains: []` in
`configs/training_config.yaml`. Expect a much longer training run.

## Project Structure

```
├── api/                          # FastAPI inference server + dashboard
│   └── main.py
├── configs/
│   └── training_config.yaml      # Centralized configuration
├── data/
│   ├── prepare_dataset.py        # Load + filter + snapshot Gretel dataset
│   ├── gen_dataset.py            # Deprecated shim; routes to prepare_dataset.py
│   ├── setup_test_db.py          # Optional personal-finance demo database
│   └── processed/                # Prepared JSON, predictions, evaluation output
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/                       # Saved LoRA adapters (git-ignored)
├── scripts/
│   ├── generate_predictions.py
│   ├── run_pipeline.py
│   └── validate_setup.py
├── src/
│   ├── data/
│   │   └── gretel.py             # Dataset loader + DDL extraction
│   ├── eval/
│   │   ├── eval.py
│   │   └── eval_accuracy.py      # Per-context execution accuracy + error analysis
│   ├── inference/
│   │   ├── gen_sql.py
│   │   ├── demo_sql.py           # Schema-aware fallback for demo questions
│   │   └── sql_guard.py          # SQLite validation + dialect cleanup
│   ├── training/
│   │   ├── train.py
│   │   ├── data_split.py
│   │   ├── formatting.py
│   │   └── lora_config.py
│   └── utils/
│       ├── logger.py
│       ├── model_loader.py
│       └── prompting.py          # Shared training/inference prompt builder
├── tests/
│   ├── test_api_generation.py    # Endpoint fallback/validation tests
│   └── test_sql_guard.py         # Regression tests for SQL guardrails
└── mlruns/                       # MLflow experiment tracking
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (4 GB VRAM demonstrated with TinyLlama; 8 GB+ recommended)
- PyTorch with CUDA support

### Installation

```bash
git clone <repo-url>
cd Finance-Text-to-SQL

python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

The first dataset/model download uses the Hugging Face Hub. Set `HF_TOKEN` if
you hit anonymous rate limits:

```bash
set HF_TOKEN=hf_...        # Windows
export HF_TOKEN=hf_...     # Linux/macOS
```

### Prepare Data

```bash
# Download/filter the dataset and write local JSON snapshots
python data/prepare_dataset.py
```

This writes:

```
data/processed/gretel_finance_train.json
data/processed/gretel_finance_validation.json
data/processed/gretel_finance_test.json
data/processed/finance_sql.json          # legacy-compatible training snapshot
data/processed/dataset_info.json
```

Training and inference load the dataset from the Hub directly; the JSON files
are for inspection and offline tooling.

### Validate Setup

```bash
python scripts/validate_setup.py
```

## Training

Edit `configs/training_config.yaml`:

```yaml
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
output_dir: models/tinyllama-gretel-finance-sql

dataset:
  name: gretelai/synthetic_text_to_sql
  domains: [finance]
  validation_size: 0.1

batch_size: 1
gradient_accumulation_steps: 4
epochs: 3
learning_rate: 0.0002
max_seq_length: 512
```

Run training:

```bash
python -m src.training.train
```

Training features:

- 4-bit NF4 quantization (reduces VRAM ~4x)
- LoRA adapters (rank 16, `q_proj`/`v_proj`)
- Gradient checkpointing
- Paged AdamW 8-bit optimizer
- Grouped train/validation split so identical questions never straddle the boundary
- MLflow parameter/metric logging (optional; failures do not stop training)

The official Gretel test split is never used during training.

## Evaluation

Generate predictions on the held-out test split:

```bash
python -m scripts.generate_predictions
```

Evaluate them:

```bash
python -m src.eval.eval \
  --dataset data/processed/predictions.json \
  --output data/processed/eval_results.json \
  --dashboard-output api/static/evaluation.json
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Execution Accuracy** | Predicted SQL returns the same result set as the reference SQL on the sample's own SQLite context |
| **Exact Match Accuracy** | Normalized SQL strings match exactly |
| **Error Types** | Syntax, column hallucination, table hallucination, logic error, gold execution error |

Some reference queries in the source dataset use SQLite-incompatible features
or write statements. Those are reported as `gold_execution_errors` and excluded
from the execution-accuracy denominator; they are not counted as model failures.

### Current Local Results

TinyLlama 1.1B + rank-16 QLoRA, evaluated on the official finance test split:

| Metric | Result |
|--------|--------|
| **Execution Accuracy** | 56.25% (27 / 48 SQLite-executable references) |
| **Exact Match Accuracy** | 19.44% (14 / 72) |
| **Gold Execution Errors** | 24 / 72 (excluded from execution accuracy) |
| **Model Failures** | 14 logic, 6 column hallucination, 1 execution |

These values come from `data/processed/eval_results.json`. The dashboard reads
the summary from `api/static/evaluation.json`, which is regenerated by the
evaluation command above.

Run the full pipeline end-to-end:

```bash
python scripts/run_pipeline.py
```

## Inference

### Python API

```python
from src.inference.gen_sql import generate_sql
from src.utils.model_loader import load_model
from peft import PeftModel

model, tokenizer = load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = PeftModel.from_pretrained(model, "models/tinyllama-gretel-finance-sql")

sql = generate_sql(
    model,
    tokenizer,
    question="What was the total value of loans issued to women?",
    schema=(
        "CREATE TABLE socially_responsible_loans ("
        "id INT, value DECIMAL(10, 2), client_gender VARCHAR(10), date DATE)"
    ),
)
print(sql)
```

### REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker-compose -f docker/docker-compose.yml up
```

Open `http://localhost:8000` for the Finance SQL Studio dashboard.

The dashboard is a self-contained page served by FastAPI. It renders the complete
`CREATE TABLE` schema from `/schema` with searchable tables and columns, syntax-
highlights generated SQL, labels each result as model, normalized, or
schema-safe fallback, keeps the last eight questions in browser storage, and
exports the current query as a `.sql` file. It also shows model runtime details
from `/model`, the live evaluation report, and a per-request `Latency X.XXs`
metric.

The API accepts an optional `schema`. If omitted, it uses the personal-finance
demo schema from `configs/training_config.yaml`.

Generated SQL is validated against a temporary in-memory SQLite database built
from the schema. The API rewrites common MySQL/Postgres date expressions,
rejects write statements, and uses a schema-aware fallback for common finance
demo questions when the model output cannot be validated. The response includes
`valid`, `source`, `latency_ms`, and an optional `warning`; the dashboard shows
this status and the request latency under the generated query.

Run the SQL guardrail regression tests with:

```bash
python -m unittest discover -s tests -v
```

```bash
curl -X POST "http://localhost:8000/generate-sql" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the total spending by category?",
    "schema": "expenses(expense_id, user_id, category, amount, date)"
  }'
```

## Demo Schema

The optional demo database in `data/processed/finance.db` uses a personal-finance
schema. The API's default prompt uses the full `CREATE TABLE` DDL from
`configs/training_config.yaml`:

```sql
CREATE TABLE users (user_id INT PRIMARY KEY, name TEXT, email TEXT, created_at DATE);
CREATE TABLE accounts (account_id INT PRIMARY KEY, user_id INT, account_type TEXT, balance DECIMAL, currency TEXT);
CREATE TABLE expenses (expense_id INT PRIMARY KEY, user_id INT, category TEXT, amount DECIMAL, date DATE, description TEXT);
CREATE TABLE income (income_id INT PRIMARY KEY, user_id INT, source TEXT, amount DECIMAL, date DATE);
CREATE TABLE budgets (budget_id INT PRIMARY KEY, user_id INT, category TEXT, monthly_limit DECIMAL, start_date DATE);
CREATE TABLE transactions (txn_id INT PRIMARY KEY, account_id INT, txn_type TEXT, amount DECIMAL, date DATE, merchant TEXT);
```

The trained adapter is not limited to this schema: every training example
carries its own schema in the prompt.

## Dataset Coverage (Finance Train Split, 945 rows)

| Complexity | Rows |
|---|---:|
| Basic SQL | 490 |
| Aggregation | 158 |
| Single join | 142 |
| Subqueries | 75 |
| Window functions | 64 |
| Multiple joins | 9 |
| Set operations | 4 |
| CTEs | 3 |

## Technical Details

### QLoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Quantization

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # or float16
)
```

## Hardware

| Configuration | VRAM | Notes |
|---|---|---|
| TinyLlama 1.1B, RTX 3050 Laptop GPU | 4 GB | Training and inference verified locally |
| Phi-3 Mini 3.8B | ~10 GB | May work with the same code after changing `model_name` |

## Known Issues

- `bitsandbytes` may have compatibility issues on Windows
- 4-bit quantization requires CUDA compute capability 7.0+
- Some source-dataset reference queries are not executable in SQLite; the
  evaluator reports these separately as `gold_execution_errors`
- The dataset is synthetic; it does not establish real-world accuracy

## License

Project code: MIT. Dataset: see the
[`gretelai/synthetic_text_to_sql` dataset card](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql).
