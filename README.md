# Finance Text-to-SQL (Gretel + Spider + WikiSQL)

Fine-tuning small language models for natural language to SQL using QLoRA and
three public datasets: finance examples from
[`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql),
cross-domain examples from [Spider](https://yale-lily.github.io/spider), and
single-table schema grounding from
[WikiSQL](https://github.com/salesforce/WikiSQL).

## Overview

This project demonstrates efficient domain adaptation of instruction-tuned LLMs
for Text-to-SQL. Key features:

- **Public datasets, not hand-written templates**: loads Gretel finance rows, Spider, and WikiSQL from pinned sources
- **Per-example schemas**: each training row ships its own DDL; the model is prompted with the schema instead of memorising one fixed database
- **Tokenizer-driven chat prompts**: the shared prompt builder uses `apply_chat_template`, so the base model can be swapped without rewriting the prompt layout
- **Parameter-efficient fine-tuning**: QLoRA with 4-bit NF4 quantization and architecture-agnostic `all-linear` LoRA targets
- **Memory optimization**: gradient checkpointing and paged optimizers for consumer GPUs
- **Execution-based evaluation**: creates an in-memory SQLite database from each sample's `sql_context` and compares actual query results
- **Base/model comparison**: scores the base model and fine-tuned adapter on the same held-out examples
- **Redis inference cache**: caches validated SQL with model/schema-aware keys, configurable TTL, and graceful fallback
- **Safe evaluation**: uses read-only SQLite execution with timeouts and resource limits

## Dataset

The upstream Gretel dataset contains 100,000 training rows, but the project
filters it to finance. That leaves 1,051 examples, split into 945 training and
106 validation rows, plus 72 official test rows. Spider adds 8,659 human-written
cross-domain training examples and 1,034 held-out development examples.
WikiSQL adds 56,355 single-table training examples over 18,585 Wikipedia tables,
with an official 8,421-row dev split used for validation.

The current mixture uses these sources:

| Dataset | Use | License |
|---|---|---|
| Gretel synthetic text-to-SQL | Finance train, validation, and SQLite-context test | Apache-2.0 |
| Spider 1.0 (GRAST-SQL schema metadata) | Cross-domain train, validation, and held-out dev exact-match evaluation | CC BY-SA 4.0 |
| WikiSQL | Single-table schema grounding; 8,000 sampled train rows and 1,000 dev rows per epoch | Wikipedia-derived (upstream code BSD-3-Clause; HF card lists license as unknown) |

The default training mixture is about 11,435 presentations: 8,000 WikiSQL rows,
600 Spider rows, and 945 finance examples repeated three times. WikiSQL's
official dev split is used for validation; it is table-disjoint from training.
Spider's dev databases remain out of the training and validation splits.

Spider and WikiSQL rows do not include database contents in the prompt, so
Spider is evaluated by normalized SQL exact match; WikiSQL snapshots do include
`INSERT` statements for the 2,000-row dev/test slices, so its gold queries can
also be executed. Execution accuracy is measured on Gretel's per-example SQLite
contexts.

The Gretel source can be loaded directly from the Hugging Face Hub:

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
│   ├── prepare_multi_dataset.py  # Fetch and normalize Spider schema metadata
│   ├── prepare_wikisql.py        # Download, render, and validate WikiSQL
│   ├── gen_dataset.py            # Deprecated shim; routes to prepare_dataset.py
│   ├── setup_test_db.py          # Optional personal-finance demo database
│   └── processed/                # Prepared JSON, predictions, evaluation output
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/                       # Saved LoRA adapters (git-ignored)
├── scripts/
│   ├── generate_predictions.py
│   ├── evaluate_models.py       # Base vs fine-tuned held-out comparison
│   ├── benchmark_cache.py       # Redis cache workload benchmark
│   ├── run_pipeline.py
│   └── validate_setup.py
├── src/
│   ├── data/
│   │   ├── gretel.py             # Dataset loader + DDL extraction
│   │   └── multi_dataset.py      # Combined Gretel + Spider + WikiSQL train/validation
│   ├── eval/
│   │   ├── eval.py
│   │   ├── metrics.py           # Safe execution, validity, exact match, latency metrics
│   │   └── eval_accuracy.py
│   ├── cache/
│   │   └── sql_cache.py          # Optional Redis SQL response cache
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
│   ├── test_eval_metrics.py      # Evaluation metrics and SQLite safety tests
│   ├── test_sql_cache.py         # Redis cache behavior and failure tests
│   └── test_sql_guard.py         # SQL guardrail regression tests
└── mlruns/                       # MLflow experiment tracking
```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (4 GB VRAM demonstrated with Qwen2.5-Coder 1.5B; 8 GB+ recommended)
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

The Gretel JSON snapshots are used by training. Spider data is downloaded from
the pinned public dataset revision and normalized to `CREATE TABLE` schemas:

```bash
python data/prepare_multi_dataset.py
```

WikiSQL is downloaded from the pinned 26 MB GitHub release, rendered into
`CREATE TABLE` + `INSERT` contexts, and checked by executing every gold query
against an in-memory SQLite copy of its table:

```bash
python data/prepare_wikisql.py
```

The script writes `wikisql_train.json`, `wikisql_validation.json`,
`wikisql_test.json`, and `wikisql_info.json` to `data/processed/`. Training rows
omit `INSERT` statements (only the schema is prompted); dev/test snapshots keep
them so gold queries remain executable. Use `--no-validate` for a faster run and
`--max-eval-rows` to change the dev/test slice size.

### Validate Setup

```bash
python scripts/validate_setup.py
```

## Training

Edit `configs/training_config.yaml`:

```yaml
model_name: Qwen/Qwen2.5-Coder-1.5B-Instruct
output_dir: models/qwen2.5-coder-1.5b-wikisql-sql

dataset:
  name: gretelai/synthetic_text_to_sql
  domains: [finance]
  validation_size: 0.1

training_datasets:
  spider:
    include: true
    max_train_samples: 600
    finance_repeat: 3
  wikisql:
    include: true
    max_train_samples: 8000
    max_validation_samples: 1000

batch_size: 1
gradient_accumulation_steps: 4
epochs: 1
learning_rate: 0.0002
max_seq_length: 512
```

Prepare all sources and train the combined model:

```bash
python -m src.training.train
```

Training features:

- 4-bit NF4 quantization (reduces VRAM ~4x)
- LoRA adapters (rank 16, `all-linear` targets so Qwen, Phi, and Llama-family projection names all resolve)
- Chat prompts rendered with the tokenizer's chat template (ChatML for Qwen)
- Gradient checkpointing
- Paged AdamW 8-bit optimizer
- Grouped train/validation split so identical questions never straddle the boundary
- MLflow parameter/metric logging (optional; failures do not stop training)

Gretel's official test split and Spider's official dev split are never used for
training. The new adapter is saved separately from the previous TinyLlama
adapters so prior models remain available for comparison.

### Progress bars

Training reports progress in three places, all using `tqdm` (with a no-op
fallback if it is missing):

- `Formatting SFT examples` — one row per training example while prompts are
  tokenized and pruned to `max_seq_length`
- `Training` — one tick per optimizer step, with `loss`, `lr`, and `epoch` in
  the postfix; validation prints `eval_loss` at the end of each epoch
- `Evaluating <model>` — one sample per generated/evaluated test row

Hugging Face's own trainer bar is disabled (`disable_tqdm=True`) so only one
bar is active. Use `--max-steps N` for a short VRAM smoke run:

```bash
python -m src.training.train --max-steps 20 --limit-train 200 --limit-validation 50 \
  --output-dir models/qwen-smoke
```

Without `--limit-train`, a smoke test still tokenizes the full 11,435-row
training set before the first optimizer step, so the GPU appears idle for
several minutes even though the model is already loaded. Limit the rows for a
true quick check.

### Live progress dashboard

`scripts\train_qwen_wikisql.cmd` starts the Qwen2.5-Coder + WikiSQL run and
writes a combined log to `logs\train_qwen_wikisql.log`. In a second terminal,
open the dashboard:

```powershell
start "" scripts\watch_training.cmd
```

It tails the log and redraws one screen with the current phase, step
count/percentage, elapsed time and ETA, loss, learning rate, epoch, and the
latest validation losses. `python scripts/watch_training.py --once` prints a
single frame.

MLflow tracking uses the local file store in `mlruns/`; view it with:

```powershell
.venv\Scripts\mlflow.exe ui --port 5000 --backend-store-uri file:///D:/PRJKT/py/Finance-Text-to-SQL/mlruns
```


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
| **Exact Match Accuracy** | Ignores whitespace/comments and folds keyword case while preserving identifiers and literals; not proof of SQL equivalence |
| **Valid SQL Rate** | Percentage of outputs accepted by SQLite against the sample schema, with errors categorized |
| **Generation Latency** | Per-example latency, mean, P50, P95, P99, and output tokens/second |
| **Test Loss** | Held-out target-SQL token loss with prompt tokens masked; includes perplexity |

Some reference queries in the source dataset use SQLite-incompatible features
or write statements. Those are reported as `gold_execution_errors` and excluded
from the execution-accuracy denominator; they are not counted as model failures.

### Latest Model Comparison

All rows below were re-benchmarked on the same 72 held-out Gretel finance
examples (48 with executable gold SQL) with greedy decoding and the current
ChatML prompt, so they are directly comparable. Full artifacts:
[`reports/model_comparison/`](reports/model_comparison/).

| Model | Exact match | Valid SQL | Execution | Test loss | P50 latency |
|---|---:|---:|---:|---:|---:|
| TinyLlama 1.1B base | 1.4% | 79.2% | 43.8% | 0.724 | 21,847 ms |
| TinyLlama + Gretel finance | 18.1% | 73.6% | 56.2% | 0.482 | 9,165 ms |
| TinyLlama + multi-dataset | 18.1% | 65.3% | 54.2% | 0.487 | 3,192 ms |
| TinyLlama + multi-dataset (candidate) | 15.3% | 66.7% | **58.3%** | 0.449 | 3,377 ms |
| TinyLlama + multi-dataset + Spider (ckpt359) | 16.7% | 63.9% | 54.2% | 0.460 | 3,357 ms |
| phi3-sql (TinyLlama adapter) | 4.2% | 31.9% | 22.9% | 0.935 | 2,743 ms |
| Qwen2.5-Coder 1.5B base | 12.5% | 62.5% | 52.1% | 0.476 | 3,032 ms |
| **Qwen2.5-Coder + WikiSQL/Spider/Finance** | **22.2%** | **68.1%** | 56.2% | **0.235** | 5,433 ms |

The Qwen adapter posts the best exact match (22.2%) and held-out loss (0.235),
and it adds +4.2 execution points over its base. Its 68.1% valid-SQL rate is
below the TinyLlama finance adapter (73.6%) and the TinyLlama base (79.2%), and
its P50 latency is the highest of the fine-tuned models (5,433 ms). Execution
accuracy is one example behind the best TinyLlama run on this 48-example
denominator, so that difference is not significant. The test set is small and
synthetic; these numbers do not establish real-world accuracy.

The dashboard's `/evaluation.json` is a separate single-model summary generated
by `src.eval.eval`; see [`reports/evaluation-qwen/`](reports/evaluation-qwen/)
for the base-versus-Qwen-adapter report and per-example predictions. It reports
56.25% execution and 25.0% exact match (18/72) for the Qwen adapter; the
3-point exact-match gap comes from its looser `normalize_sql` comparison, so use
the table above for cross-model comparisons.

Run the base-versus-adapter evaluation with:

```bash
python scripts/evaluate_models.py
```

Use `--limit 5` for a smoke run or `--output reports/my-run` to select an output
directory. The command writes `summary.json`, `report.md`, and `predictions.jsonl`.

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

model, tokenizer = load_model("Qwen/Qwen2.5-Coder-1.5B-Instruct")
model = PeftModel.from_pretrained(model, "models/qwen2.5-coder-1.5b-wikisql-sql")

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

### Vercel dashboard

`api/static/` is the linked Vercel project (a static deployment of the same
dashboard). It bundles no inference engine, so it connects to a FastAPI server
that loads the newest `Qwen2.5-Coder-1.5B` adapter. Point the hosted page at
that server in either of two ways:

- open it with `?api=https://your-api.example.com` (stored in `localStorage`
  for later visits), or
- set `<meta name="finance-sql-api" content="https://your-api.example.com">`
  in `api/static/index.html` before deploying.

With neither value set, a hosted page falls back to
`http://127.0.0.1:8000`, so it works when the Qwen API runs on the visitor's
machine. The header model pill and runtime card read `/health` and `/model`,
so they report the adapter actually being served instead of a hard-coded name.
Keep the API reachable over HTTPS when the dashboard is served over HTTPS;
`CORSMiddleware` already allows cross-origin requests.

The API accepts an optional `schema`. If omitted, it uses the personal-finance
demo schema from `configs/training_config.yaml`.

Generated SQL is validated against a temporary in-memory SQLite database built
from the schema. The API rewrites common MySQL/Postgres date expressions,
rejects write statements, and uses a schema-aware fallback for common finance
demo questions when the model output cannot be validated. The response includes
`valid`, `source`, and `latency_ms`, plus `cache_hit`, `model_version`,
`generation_latency_ms`, `response_latency_ms`, and `cache_error`. Generation
latency is null on a cache hit because inference did not run. Send
`X-Cache-Bypass: true` to bypass cache reads and writes.

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

WikiSQL contributes only single-table `SELECT ... WHERE` shapes (plain,
aggregation, and comparisons), so it improves column/table grounding but does
not add joins, subqueries, or CTEs. Spider and Gretel still carry those.

## Technical Details

### QLoRA Configuration

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",  # resolves per architecture
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
| Qwen2.5-Coder 1.5B, RTX 3050 Laptop GPU | 4 GB | Current configuration; 4-bit QLoRA + SDPA |
| TinyLlama 1.1B, RTX 3050 Laptop GPU | 4 GB | Previous baseline, kept for comparison |
| Qwen2.5-Coder 3B / Phi-4-mini 3.8B | ~6–8 GB | Training at 512 tokens is tight on 4 GB; inference only |
| Qwen2.5-Coder 7B / Qwen3 8B | ~12 GB | Recommended next step for accuracy |

## Known Issues

- `bitsandbytes` may have compatibility issues on Windows
- 4-bit quantization requires CUDA compute capability 7.0+
- Some source-dataset reference queries are not executable in SQLite; the
  evaluator reports these separately as `gold_execution_errors`
- The Gretel dataset is synthetic; it does not establish real-world accuracy
- WikiSQL is single-table and has no joins, ordering, or nesting; treat it as
  schema grounding rather than a replacement for Spider
- WikiSQL's Hugging Face copy requires a dataset loading script, which
  `datasets>=4` does not run; `data/prepare_wikisql.py` uses the pinned GitHub
  release instead

## Redis Cache and Cache Benchmark

Redis caching is optional. The API caches validated SQL responses for 300
seconds by default. Keys include digests for the question, schema, model
artifacts, prompt version, and generation settings. Redis failures fall back to
inference. Leave `REDIS_URL` unset or set `REDIS_CACHE_ENABLED=false` to bypass
the cache.

Start Redis and the API with Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up --build
```

For a local Python API, install dependencies and start Redis:

```powershell
.venv\Scripts\pip.exe install -r requirements.txt
docker compose -f docker/docker-compose.yml up -d redis
$env:REDIS_URL = "redis://localhost:6379/0"
$env:REDIS_CACHE_ENABLED = "true"
$env:REDIS_CACHE_TTL_SECONDS = "300"
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Run the repeated/unique-question cache benchmark:

```powershell
.venv\Scripts\python.exe scripts/benchmark_cache.py --requests 30 --unique-ratio 0.3 --concurrency 4
```

It saves measured hit/miss rates, hit/miss latency percentiles, generation
latency, throughput, and errors to `reports/cache-benchmark.json`. No live
Redis benchmark has been run in this environment; run this command with a live
Redis service and API to collect those measurements.

Set `CACHE_ADMIN_TOKEN` to enable `POST /cache/invalidate` and send the token in
`X-Cache-Admin-Token`. See
[docs/evaluation-and-cache.md](docs/evaluation-and-cache.md) for configuration,
evaluation details, and test instructions.

## License

Project code: MIT. Datasets: see the
[`gretelai/synthetic_text_to_sql` dataset card](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql),
the [Spider](https://yale-lily.github.io/spider) license, and the
[WikiSQL](https://github.com/salesforce/WikiSQL) repository.
