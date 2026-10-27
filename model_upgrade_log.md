# Model Upgrade Log — Finance Text-to-SQL

**Last updated:** 2026-10-27  
**Status:** Complete; API stopped after verification  
**Project:** `D:\PRJKT\py\Finance-Text-to-SQL`

| Item | Current value |
|---|---|
| Dataset | `gretelai/synthetic_text_to_sql`, finance domain |
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Adapter | `models/tinyllama-gretel-finance-sql` |
| Tuning method | 4-bit QLoRA |
| Training steps | 711 / 711 |
| Final validation loss | 0.4237 |
| Execution accuracy | **56.25%** (27 / 48 executable references) |
| Exact-match accuracy | **19.44%** (14 / 72) |
| Gold execution errors | 24 / 72 (excluded from execution accuracy) |
| Local API | `http://127.0.0.1:8000` |
| Regression tests | 16 passing |
| Per-request latency | Returned as `latency_ms`; shown in dashboard |

### Latest update

- Added a server-side `latency_ms` field to `/generate-sql`.
- Added a `Latency X.XXs` metric to the dashboard result header, with a
  server/round-trip tooltip.
- Extended the demo fallback for purchase-history questions such as
  “what did I buy in the last 3 weeks”.
- Added relative period parsing (`3 weeks` → `-21 days`), top-N item queries,
  and top-category selection.
- Expanded the regression suite to 16 tests.
- Upgraded the dashboard: full DDL parsing with collapsible tables and schema
  search, SQL syntax highlighting, source and validation badges, `.sql` export,
  browser-local query history, model runtime card, and responsive layout polish.

---

## 1. Executive summary

The project was refocused around the public **`gretelai/synthetic_text_to_sql`** dataset instead of a fixed/template-generated dataset. The finance subset was extracted, split into train/validation/test sets, and used to fine-tune a small instruction model with QLoRA.

The trained adapter was then connected to the FastAPI inference service and the Finance SQL Studio dashboard. The completed pipeline now supports:

- Finance-domain dataset loading through `datasets.load_dataset`
- Per-example DDL schemas in prompts
- QLoRA training on a 4 GB laptop GPU
- Held-out evaluation with SQLite execution comparison
- Live SQL generation through the REST API
- A dashboard that reads live model/evaluation metadata
- Schema-aware SQL validation and a safe demo fallback

The raw model remains a 1.1B model and is not highly accurate: it reaches **56.25% execution accuracy** on SQLite-executable reference queries. During dashboard testing it also produced invalid SQL for the personal-finance demo schema (a duplicate `JOIN` and a hallucinated category value). That serving issue was addressed with validation and fallback guardrails without changing the model's held-out metrics.

---

## 2. Goals and scope

The work covered four goals:

1. Load and prepare the finance subset of `gretelai/synthetic_text_to_sql`.
2. Fine-tune a small model on that dataset while keeping the project finance-focused.
3. Evaluate the model and record execution/exact-match metrics.
4. Connect the trained adapter to the API and webpage so the model can be used interactively.

The project remains finance-focused. The full 100,000-row multi-domain dataset is not used.

---

## 3. High-level changes

| Area | Change | Result |
|---|---|---|
| Dataset | Replaced the earlier data path with `gretelai/synthetic_text_to_sql` | Public, reproducible finance dataset |
| Data preparation | Added finance filtering, grouping, DDL extraction, JSON snapshots | 945 train / 106 validation / 72 test |
| Training | QLoRA fine-tuning of TinyLlama 1.1B | 711 steps, 0.4237 final validation loss |
| Evaluation | Per-example SQLite execution and error classification | 56.25% execution / 19.44% exact match |
| Serving | FastAPI adapter loading and shared greedy inference | `/generate-sql` endpoint |
| Dashboard | Live evaluation, schema, and generation integration | Finance SQL Studio UI |
| Reliability | SQL validation, dialect normalization, demo fallback | Invalid model SQL is no longer returned silently for the demo |
| Testing | Guardrail and endpoint regression tests | 16 tests passing |
| Observability | Server and end-to-end latency metric | `latency_ms` + dashboard display |

---

## 4. Dataset migration and preparation

### 4.1 Source

The source dataset is loaded from the Hugging Face Hub:

```python
load_dataset("gretelai/synthetic_text_to_sql")
```

Configuration:

- Domain filter: `finance`
- Source train split: 1,051 finance rows
- Official source test split: 72 rows
- Validation: grouped 10% split from the finance training rows
- Final split sizes:

| Split | Rows | Purpose |
|---|---:|---|
| Train | 945 | QLoRA training |
| Validation | 106 | Validation loss / model selection |
| Test | 72 | Held-out evaluation only |

The official test split is never used during training. The train/validation split is group-aware so identical questions do not cross the split boundary.

### 4.2 Prompt construction

Training and inference share the same prompt layout in `src/utils/prompting.py`:

```text
<|user|>
Write a SQL query for the following question

Question: <question>
Schema: <schema>
<|assistant|>
<SQL>
```

Only `CREATE TABLE` DDL is extracted from each row's `sql_context` for the prompt. The full `sql_context` is retained for evaluation so reference SQL can be executed against the same in-memory SQLite database.

SQL extraction handles:

- Assistant-tag output
- Markdown code fences
- `SQL:` prefixes
- Natural-language preambles

### 4.3 Dataset coverage

Finance training split coverage:

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
| **Total** | **945** |

Additional dataset facts:

- 768 unique finance schemas
- Task types: 836 analytics/reporting, 100 data manipulation, 6 data retrieval, 3 data definition
- `data/processed/dataset_info.json` stores the summary

### 4.4 Prepared artifacts

| File | Purpose |
|---|---|
| `data/processed/gretel_finance_train.json` | 945 training rows |
| `data/processed/gretel_finance_validation.json` | 106 validation rows |
| `data/processed/gretel_finance_test.json` | 72 test rows |
| `data/processed/finance_sql.json` | Legacy-compatible training snapshot |
| `data/processed/dataset_info.json` | Split and coverage summary |
| `data/processed/finance.db` | Optional personal-finance demo database |

`data/gen_dataset.py` is retained as a deprecated compatibility shim that routes to `data/prepare_dataset.py`.

---

## 5. Training

### 5.1 Base model and QLoRA configuration

| Setting | Value |
|---|---|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Quantization | 4-bit NF4 |
| Compute dtype | `bfloat16` / `float16` depending on hardware |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA target modules | `q_proj`, `v_proj` |
| LoRA dropout | 0.05 |
| Bias | `none` |
| Task type | `CAUSAL_LM` |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Max sequence length | 512 |
| Seed | 42 |
| Gradient checkpointing | Enabled |
| Optimizer | Paged AdamW 8-bit |
| Hardware | NVIDIA GeForce RTX 3050 Laptop GPU, 4.3 GB VRAM |

### 5.2 Training run

| Metric | Value |
|---|---|
| Steps completed | 711 / 711 |
| Runtime | ~52 minutes |
| Training loss | 0.4537 |
| Final validation loss | 0.4237 |
| Final validation token accuracy | 0.8810 |
| MLflow run name | `capable-auk-759` |
| MLflow run ID | `2f9f53592bee44699e2e5d29b69d55f3` |
| MLflow status | `FINISHED` |

Validation loss by epoch:

| Epoch | Validation loss | Token accuracy |
|---|---:|---:|
| 1 | 0.4544 | 0.8711 |
| 2 | 0.4272 | 0.8791 |
| 3 | 0.4237 | 0.8810 |

### 5.3 Adapter artifacts

Saved to `models/tinyllama-gretel-finance-sql`:

- `adapter_config.json`
- `adapter_model.safetensors` (~9 MB)
- `chat_template.jinja`
- `tokenizer.json`
- `tokenizer_config.json`
- `training_args.bin`
- `README.md`

The adapter was loaded back through PEFT and used for inference successfully.

---

## 6. Evaluation

### 6.1 Method

1. Load the fine-tuned adapter.
2. Generate predictions for the 72 official finance test rows using greedy decoding.
3. For each sample, create an in-memory SQLite database from the sample's full `sql_context`.
4. Execute the reference SQL and the predicted SQL in that context.
5. Compare result sets for execution accuracy.
6. Compare normalized SQL strings for exact-match accuracy.
7. Classify failures by type.

Reference queries that cannot execute in SQLite are reported separately as `gold_execution_errors` and excluded from the execution-accuracy denominator. They remain in the exact-match denominator because exact matching does not require execution.

### 6.2 Results

| Metric | Result |
|---|---:|
| Test samples | 72 |
| SQLite-executable references | 48 |
| Execution correct | 27 |
| **Execution accuracy** | **56.25%** |
| Execution accuracy over all 72 samples | 37.50% |
| Exact-match correct | 14 |
| **Exact-match accuracy** | **19.44%** |
| Gold execution errors | 24 / 72 |

### 6.3 Error breakdown

| Error type | Count |
|---|---:|
| No error / execution match | 27 |
| Logic error | 14 |
| Column hallucination | 6 |
| Execution error | 1 |
| Syntax error | 0 |
| Semantic error | 0 |
| Table hallucination | 0 |
| Context error | 0 |
| Gold execution error | 24 |

### 6.4 Interpretation

- The model learned the finance-domain prompt format and produces many executable queries.
- Its main weakness is query logic, not SQLite syntax: 14 logic errors and 6 column hallucinations.
- Nearly one third of the official test references are not executable under SQLite, which substantially affects the reported denominators.
- Exact-match accuracy is low because many predictions are semantically close but not textually identical.
- The guardrails added to the API do not change these held-out metrics; they improve serving reliability.

---

## 7. Serving and API integration

### 7.1 API implementation

The FastAPI service is implemented in `api/main.py`.

On startup it:

1. Loads the TinyLlama base model.
2. Loads the LoRA adapter with `PeftModel.from_pretrained`.
3. Sets the tokenizer pad token.
4. Switches the model to evaluation mode.

Generation uses the shared greedy path in `src/inference/gen_sql.py`, so the API and evaluation use the same prompt/SQL-extraction logic.

### 7.2 Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Finance SQL Studio dashboard |
| `/health` | GET | Health, model-loaded state, base model, adapter, dataset |
| `/model` | GET | Model, adapter, domains, split, max new tokens |
| `/schema` | GET | Default demo schema |
| `/dataset` | GET | Dataset source/domain/split |
| `/evaluation.json` | GET | Latest evaluation summary |
| `/generate-sql` | POST | Generate SQL for a question and optional schema |

Example request:

```json
{
  "question": "What is the total spending by category?",
  "schema": "expenses(expense_id, user_id, category, amount, date)"
}
```

Example response:

```json
{
  "sql": "SELECT category, COALESCE(SUM(amount), 0) AS total_spending FROM expenses GROUP BY category ORDER BY total_spending DESC;",
  "schema_used": "expenses(expense_id, user_id, category, amount, date)",
  "valid": true,
  "source": "safe-fallback",
  "warning": "The model query was not executable against this schema. A schema-safe query was generated instead.",
  "validation_error": "no such column: spending",
  "latency_ms": 11842.5
}
```

### 7.3 Dashboard

`api/static/index.html` is the Finance SQL Studio page. It:

- Displays model/evaluation metrics from `/evaluation.json`
- Falls back to the bundled static report if the API is unavailable
- Parses the full `CREATE TABLE` DDL from `/schema` and supports table/column search
- Sends questions to `/generate-sql`
- Syntax-highlights generated SQL
- Labels results as model, normalized, or schema-safe fallback and shows validation status
- Shows per-request latency in the result header
- Exports the current query as a `.sql` file
- Keeps the last eight questions and their SQL in browser-local storage
- Shows base model, adapter, dataset, token limit, and average latency from `/model`
- Allows the user to connect a different API base URL

The dashboard uses the live API when available instead of hard-coded metrics.
Query history never leaves the browser; only questions explicitly submitted to
`/generate-sql` reach the inference service.

---

## 8. SQL reliability guardrails

### 8.1 Problem found during browser testing

The model generated this query for the demo question **“what is my this week spending”**:

```sql
SELECT SUM(expenses.amount)
FROM expenses
JOIN accounts ON expenses.user_id = accounts.user_id
JOIN accounts ON accounts.user_id = accounts.account_type
WHERE accounts.user_id = (SELECT user_id FROM users WHERE name = 'Alice')
  AND accounts.account_type = 'Bank Account'
  AND accounts.currency = 'USD'
  AND expenses.category = 'Spending';
```

Problems:

- `accounts` was joined twice, causing `ambiguous column name: accounts.user_id`
- `'Spending'` is not a real category in the demo database
- `accounts.user_id = accounts.account_type` is a meaningless join condition
- The query would either fail or return no useful result

The endpoint was returning this SQL without validation.

### 8.2 Guardrail implementation

Added `src/inference/sql_guard.py`:

- Builds a temporary in-memory SQLite database from the supplied schema
- Supports full `CREATE TABLE` DDL and compact schemas such as `expenses(id, amount, date)`
- Runs `EXPLAIN QUERY PLAN` to catch:
  - Syntax errors
  - Missing tables/columns
  - Ambiguous column references
  - Duplicate joins
- Rejects write statements (`INSERT`, `UPDATE`, `DELETE`, `DROP`, etc.)
- Rejects multiple statements
- Normalizes common MySQL/Postgres date expressions to SQLite:
  - `DATE_SUB(CURDATE(), INTERVAL 7 DAY)` → `date('now', '-7 days')`
  - `DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)` → `date('now', '-1 month')`
  - `DATE_ADD(...)` → `date(..., '+N units')`
  - `CURDATE()` / `CURRENT_DATE` / `NOW()`
- Provides `deduplicate_joins` as a last-resort repair for custom schemas

Added `src/inference/demo_sql.py`:

- Handles common finance demo intents when the model SQL fails validation
- Covers:
  - This week / last week / this month / last month / this year / last year
  - Relative periods such as “last 3 weeks” → `date('now', '-21 days')`
  - Food, groceries, travel, utilities, entertainment, transportation, rent, healthcare, shopping
  - Spending totals
  - Spending by category
  - Purchase history / “what did I buy”
  - Top N items, including top 3 retrieved items
  - Top spending category in a period
  - Largest transactions
  - Transactions by merchant
  - Income totals and income by source
  - Budgets
  - Account balances
  - Users
- Only returns a fallback if the required table/columns exist
- The fallback is then validated by the same SQL guard

### 8.3 API response contract

`/generate-sql` now returns:

| Field | Meaning |
|---|---|
| `sql` | Returned SQL |
| `schema_used` | Schema used for validation/generation |
| `valid` | Whether the returned SQL passed schema validation |
| `source` | `model`, `model-normalized`, `safe-fallback`, or `invalid` |
| `warning` | User-facing note when a fallback/repair was needed |
| `validation_error` | Underlying SQLite error from the model query, when applicable |
| `latency_ms` | Server-side generation + validation time in milliseconds |

The webpage displays:

- Green “Validated against the schema” for valid model SQL
- Green “Schema-safe query generated for this demo” for fallback SQL
- Orange/red warning if the SQL still does not validate
- `Latency X.XXs` in the result header, with a tooltip showing server and round-trip time

### 8.4 Live result for the failing question

```sql
SELECT COALESCE(SUM(amount), 0) AS total_spending
FROM expenses
WHERE date >= date('now', '-7 days');
```

Live response:

```json
{
  "valid": true,
  "source": "safe-fallback",
  "validation_error": "ambiguous column name: accounts.user_id"
}
```

The query executes successfully. Because the demo fixture has no expense rows in the current week, it returns `0`.

### 8.4.1 Multi-week purchase-history issue

A second demo question exposed another model failure:

> “what add did u buy this 3 weeks and which category tops it with top 3 retrieved items”

The model returned a query that joined `accounts` to itself and referenced non-existent columns such as:

- `a.category`
- `a.created_at`
- `b.top_3_items`
- `b.top_3_items_amount`
- `b.top_3_items_category`

The fallback now recognizes this as a purchase-history question and returns:

```sql
WITH recent AS (
  SELECT date, description AS item, category, amount
  FROM expenses
  WHERE date >= date('now', '-21 days')
),
top_category AS (
  SELECT category, SUM(amount) AS total_spending
  FROM recent
  GROUP BY category
  ORDER BY total_spending DESC
  LIMIT 1
)
SELECT recent.date, recent.item, recent.category, recent.amount,
       (SELECT category FROM top_category) AS top_category,
       (SELECT total_spending FROM top_category) AS top_category_spending
FROM recent
ORDER BY recent.amount DESC
LIMIT 3;
```

This returns the top 3 recent purchases plus the top spending category and its total. It was verified live with `source: safe-fallback`, `valid: true`, and `validation_error: no such column: a.category`.

### 8.5 Note on behavior

This is a serving guardrail, not a retrain. The raw 1.1B model can still produce invalid or incorrect SQL. The guardrail:

- Prevents known-invalid SQL from being returned silently for the default demo schema
- Normalizes common dialect issues
- Keeps the model as the primary generator for questions outside the fallback's scope

---

## 9. Tests and validation

### 9.1 Regression tests

Added `tests/`:

| File | Coverage |
|---|---|
| `tests/test_sql_guard.py` | Validation, duplicate joins, date normalization, write rejection, compact schemas, fallback intents, multi-week purchase history |
| `tests/test_api_generation.py` | Endpoint fallback behavior, valid model output, invalid SQL without fallback, latency field |

Run:

```bash
python -m unittest discover -s tests -v
```

Result: **16 tests passed**.

### 9.2 Setup validation

```bash
python scripts/validate_setup.py
```

Verified:

- PyTorch, Transformers, PEFT, Datasets, TRL, BitsAndBytes, MLflow, FastAPI, PyYAML
- CUDA available on the RTX 3050
- Config valid
- Dataset split counts: 945 / 106 / 72
- Demo database tables: users, accounts, expenses, income, budgets, transactions
- Evaluation and error classification working

### 9.3 Live smoke tests

Verified:

- `/` returns `200`
- `/health` returns `200` with `model_loaded: true`
- `/model` returns `200`
- `/evaluation.json` returns `200`
- `/generate-sql` returns `200`
- The failing question now returns schema-valid SQL
- The three dashboard example questions return valid SQL:
  - Food spending last month
  - Spending by category
  - 5 largest transactions
- The multi-week purchase-history question returns valid CTE SQL with top 3 items and the top category
- `/generate-sql` returns `latency_ms`; the live test measured 11,351.8 ms
- A simple model-generated query (`Show all users`) returned `SELECT * FROM users;` with `source: model`
- The upgraded dashboard returned `200` with all new dashboard controls present
  (schema search, history, source badge, `.sql` export, runtime card)
- The dashboard JavaScript passed `node --check` and a DOM-stub harness covering
  DDL parsing, schema search, highlighting, history, export metadata, latency
  formatting, and error states
- Headless Chrome visual checks at 1440 px and 500 px confirmed the desktop and
  narrow layouts, schema accordions, syntax-highlighted result state, history
  replay, and evaluation panel
- The API was stopped after verification; ports `8000` and `8001` are free

---

## 10. Issues encountered and resolutions

| Issue | Resolution |
|---|---|
| Earlier dataset path was template/static and not reproducible | Migrated to `gretelai/synthetic_text_to_sql` with Hub loading |
| A prediction run stopped near the end and lost unsaved progress | Added `--resume` and atomic incremental saving to `scripts/generate_predictions.py` |
| Model produced duplicate joins / invalid SQL in the demo | Added schema validation, dialect normalization, and fallback |
| Pydantic warning because `schema` shadows `BaseModel.schema` | Renamed the field to `schema_text` with a `schema` alias |
| Dashboard could show stale metrics | Dashboard now fetches `/evaluation.json`; static report is a fallback |
| 24 test references cannot execute in SQLite | Reported separately as `gold_execution_errors`; excluded from execution denominator |
| Open-ended demo question produced incorrect SQL | Added guardrails and documented raw-model limits; not retrained |
| Purchase-history question (“buy / 3 weeks / top 3 items”) still returned invalid SQL | Added buy/purchase intent, relative multi-week filters, top-N items, and top-category fallback |
| Users could not see generation speed | Added server `latency_ms` and a dashboard `Latency X.XXs` metric |
| Inference is slow on 4 GB GPU | Greedy generation only; roughly 10–12 seconds per sample on the local RTX 3050 |

---

## 11. Current state

### Working

- Finance dataset preparation
- QLoRA training pipeline
- Adapter loading
- Prediction generation with resume support
- Execution/exact-match evaluation
- Error classification
- FastAPI inference service
- Dashboard with live evaluation and generation
- SQL validation and demo fallback
- Per-request latency metric (`latency_ms` + dashboard display)
- Regression tests
- Local server at `http://127.0.0.1:8000`

### Known limitations

- TinyLlama 1.1B is the quality bottleneck: 56.25% execution accuracy, 19.44% exact match.
- 24 / 72 official reference queries are not SQLite-executable and are excluded from execution accuracy.
- The demo fallback covers common finance intents only. Novel questions still rely on the raw model.
- The demo database has no expenses in the current week, so “this week” returns `0`.
- There is no SQL execution endpoint; generated SQL is returned for review, not run by the API.
- Generation is slow on the 4 GB GPU.
- Changes are currently uncommitted.

### Raw model vs. serving behavior

The guardrail improves what the webpage returns, but it does not improve the model's benchmark scores. Any future quality gain must come from better training data, schema linking, a larger model, or additional fine-tuning.

---

## 12. Files added or changed

| File | Change |
|---|---|
| `configs/training_config.yaml` | Gretel dataset, finance filter, split, model, adapter, schema, inference settings |
| `data/prepare_dataset.py` | Dataset preparation and JSON snapshots |
| `data/gen_dataset.py` | Deprecated compatibility shim |
| `src/data/gretel.py` | Dataset loading, finance filtering, DDL extraction |
| `src/utils/prompting.py` | Shared prompt construction and SQL extraction |
| `src/training/train.py` | QLoRA training and MLflow logging |
| `src/training/data_split.py` | Grouped train/validation split |
| `src/training/formatting.py` | Training example formatting |
| `src/training/lora_config.py` | Rank-16 LoRA configuration |
| `src/inference/gen_sql.py` | Shared greedy SQL generation |
| `src/inference/sql_guard.py` | Schema validation, dialect normalization, join repair |
| `src/inference/demo_sql.py` | Schema-aware fallback for demo intents, including multi-week purchase history |
| `scripts/generate_predictions.py` | Prediction generation with `--resume` and atomic saves |
| `scripts/validate_setup.py` | Environment/pipeline validation |
| `scripts/monitor_training.py` | Training progress monitor |
| `src/eval/eval.py` | Evaluation CLI and dashboard report writer |
| `src/eval/eval_accuracy.py` | SQLite execution accuracy and error classification |
| `data/processed/predictions.json` | 72 test predictions |
| `data/processed/eval_results.json` | Detailed evaluation output |
| `api/static/evaluation.json` | Dashboard evaluation summary |
| `api/main.py` | FastAPI endpoints, adapter startup, validation/fallback, latency timing |
| `api/static/index.html` | Finance SQL Studio dashboard with schema search, highlighting, history, `.sql` export, validation and latency display |
| `README.md` | Setup, pipeline, API, metrics, guardrails |
| `tests/test_sql_guard.py` | SQL guard/fallback tests |
| `tests/test_api_generation.py` | API endpoint tests |
| `models/tinyllama-gretel-finance-sql/` | Trained LoRA adapter |
| `mlflow.db` | MLflow tracking database |

---

## 13. Reproduction commands

### Prepare data

```bash
python data/prepare_dataset.py
```

### Validate environment

```bash
python scripts/validate_setup.py
```

### Train

```bash
python -m src.training.train
```

### Generate predictions

```bash
python -m scripts.generate_predictions
```

Resume an interrupted prediction run:

```bash
python -m scripts.generate_predictions --resume
```

### Evaluate

```bash
python -m src.eval.eval \
  --dataset data/processed/predictions.json \
  --output data/processed/eval_results.json \
  --dashboard-output api/static/evaluation.json
```

### Run tests

```bash
python -m unittest discover -s tests -v
```

### Start the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

---

## 14. Next steps / recommendations

1. **Commit the current work** once the dashboard is confirmed in the browser.
2. **Add CI** for `python -m unittest discover -s tests -v` and a lightweight API smoke test.
3. **Improve raw model quality**:
   - Try a larger base model (e.g. Phi-3 Mini / a 3B-class instruction model) if VRAM allows.
   - Add demo-schema examples or schema linking to reduce hallucinated columns.
   - Consider a second fine-tuning pass with more finance rows or curated hard examples.
4. **Expand the fallback** only where it is clearly schema-safe; keep it secondary to model generation.
5. **Deploy the API publicly** if the dashboard is hosted remotely, and update the frontend API URL.
6. **Add per-user context** if “my spending” should mean a specific authenticated user rather than the whole demo database.
7. **Document benchmark caveats** in any external report: 24/72 gold references are not SQLite-executable.

---

## Appendix A — Run metadata

| Field | Value |
|---|---|
| Evaluation date | 2026-10-27 |
| MLflow run | `capable-auk-759` |
| MLflow run ID | `2f9f53592bee44699e2e5d29b69d55f3` |
| Training status | `FINISHED` |
| Train rows | 945 |
| Validation rows | 106 |
| Test rows | 72 |
| Unique schemas | 768 |
| Max sequence length | 512 |
| Max new tokens at inference | 256 |
| Adapter size | ~9 MB |

## Appendix B — Current metric snapshot

```json
{
  "execution_accuracy": 0.5625,
  "execution_correct": 27,
  "evaluated_samples": 48,
  "total_samples": 72,
  "gold_execution_errors": 24,
  "exact_match_accuracy": 0.19444444444444445,
  "exact_match_correct": 14,
  "error_breakdown": {
    "none": 27,
    "logic_error": 14,
    "column_hallucination": 6,
    "execution_error": 1,
    "gold_execution_error": 24
  }
}
```
