# Finance Text-to-SQL: Deep Project Guide

> **State of the project:** training, evaluation, packaging, and local-plus-Vercel
> serving are complete as of 29 October 2026. The served model is
> `Qwen/Qwen2.5-Coder-1.5B-Instruct` with a rank-16 QLoRA adapter trained on
> finance, Spider, and WikiSQL data.
>
> This guide explains **how every part works and why it is built that way**. For
> the short version see [`README.md`](README.md); for the runbook see
> [`docs/local-api.md`](docs/local-api.md); for the evaluation audit see
> [`docs/PROJECT_TECHNICAL_GUIDE.md`](docs/PROJECT_TECHNICAL_GUIDE.md).

---

## 1. What the project actually does

The system translates a natural-language question plus a database schema into a
single read-only SQL query.

```text
question ─┐
          ├─► prompt ─► base model + LoRA adapter ─► raw text ─► guardrails ─► SQL
schema ───┘
```

Three distinct concerns are deliberately separated:

| Concern | What it means | Where it lives |
|---|---|---|
| Learning | Adapt a general code model to schema-grounded SQL | `src/training/` |
| Measurement | Compare predictions against reference SQL and real execution | `src/eval/`, `scripts/evaluate_models.py` |
| Serving | Validate, normalize, cache, and expose SQL over HTTP | `api/`, `src/inference/` |

The API **does not execute the generated SQL against a live database**. It only
validates that the statement is a single read-only query accepted by SQLite
against the supplied schema. Execution and authorization remain the caller's
responsibility.

---

## 2. The central design idea: schema in the prompt

A text-to-SQL model cannot know a private schema. Instead of fine-tuning one
schema into the weights, every training row and every request carries its own
DDL:

```text
Question: What is the total spending by category?
Schema: CREATE TABLE expenses (...); CREATE TABLE ...
```

This gives one adapter the ability to answer across thousands of unrelated
schemas. It is also why the API accepts an optional `schema` field: adding a new
database requires no retraining, only a correct schema.

The trade-off is context pressure. At `max_seq_length: 512`, a large schema can
crowd out the question and target SQL. The formatter therefore treats schema
selection as a first-class problem (section 5.3).

---

## 3. Repository map

| Path | Responsibility |
|---|---|
| `configs/training_config.yaml` | Single source of truth: model, paths, datasets, hyperparameters, demo schema |
| `src/data/` | Gretel loading, schema extraction, multi-dataset mixing |
| `data/prepare_*.py` | Download, normalize, validate, and snapshot each data source |
| `src/training/` | Prompt formatting, LoRA attachment, SFT loop, progress callbacks |
| `src/inference/` | Greedy generation, dialect normalization, schema validation, demo fallback |
| `src/eval/` | Exact match, execution accuracy, error taxonomy, latency/loss metrics |
| `scripts/` | Prediction generation, model comparison, cache benchmark, pipeline, launchers |
| `api/` | FastAPI service and the self-contained SQL Studio dashboard |
| `tests/` | 33 unit tests for guardrails, caching, formatting, API, and metrics |
| `reports/` | Tracked evaluation artifacts and per-example predictions |
| `docker/` | Compose stack for API + Redis (CPU image; see limitations) |

Artifacts that are **not** tracked: adapter weights (`models/`), prepared data
(`data/processed/*.json`), Hugging Face cache, logs, MLflow runs, and personal
files such as `RESUME.md`.

---

## 4. Data engineering in depth

### 4.1 Gretel finance: the domain anchor

Source: `gretelai/synthetic_text_to_sql` (Apache-2.0), filtered to
`domain == "finance"`.

| Split | Rows | Role |
|---|---:|---|
| Train (after grouped validation holdout) | 945 | Supervised fine-tuning |
| Validation | 106 | Epoch-level model selection/diagnostics |
| Test | 72 | Held-out evaluation only |

The raw dataset has 100,000 train and 5,851 test rows across many domains. The
finance filter produces 1,051 usable training rows, of which 10% (106) is carved
out as validation.

**Schema extraction.** Each Gretel row carries `sql_context`, a SQLite script
that may include `INSERT` seed rows. Showing those rows to the model would leak
answers and waste context. `extract_schema` (`src/data/gretel.py`) executes the
script in an in-memory SQLite database and reads canonical DDL back from
`sqlite_master`, with a regex fallback for contexts SQLite cannot parse. The full
context is retained only for evaluation; the model sees only DDL.

**Grouped validation split.** `grouped_split_indices`
(`src/training/data_split.py`) groups rows by exact question text before holding
out validation rows. Identical prompts therefore never straddle the train/val
boundary. It does not deduplicate paraphrases or template families; that is a
known limitation.

Finance training complexity (945 rows):

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

The distribution is heavily weighted toward single-table queries. That is
visible in the results: basic SQL improves most, while CTEs and subqueries
remain weak.

### 4.2 Spider: structural supervision

Source: `griffith-bigdata/GRAST-SQL-Spider`, pinned revision
`d6702d32cd917b57d32d21bb41d027ee49fa0c3e` (CC-BY-SA-4.0).

- 8,659 training rows, 1,034 official development rows.
- `spider_schema_to_ddl` reconstructs `CREATE TABLE` statements from flattened
  schema metadata, preserving column types, primary keys, and foreign keys.
- Validation is **database-disjoint**: complete `db_id` groups are held out, so
  no database appears in both training and validation.
- Only 600 training rows are sampled (`max_train_samples: 600`), shuffled with
  seed 42. The point is structural coverage (joins, nesting, cross-domain
  vocabulary), not volume.
- The official dev split never enters training. It is available for a separate
  normalized-exact-match check via `src/eval/eval_spider.py`.

### 4.3 WikiSQL: schema grounding at scale

Source: the pinned Salesforce GitHub tarball, SHA-256
`755c728ab188e364575705c8641f3fafd86fb089cb8b08e8c03f01832aae0881`.

The Hugging Face copy is unusable with `datasets>=4` because it still requires a
loading script, so `data/prepare_wikisql.py` downloads and verifies the official
archive. It then:

1. Joins each example to its table definition by `table_id`.
2. Cleans headers (trim, collapse whitespace, fill blanks, de-duplicate names).
3. Converts WikiSQL's index-based `sql` object into quoted SQL:
   `[column_index, operator_index, value]` becomes `WHERE "col" = 'value'`.
4. Executes every rendered gold query against an in-memory copy of its table.
   Rows whose SQL does not run are dropped instead of poisoning training.
5. Writes snapshots: 56,355 train, 2,000 dev, 2,000 test. Training uses 8,000
   sampled rows; validation caps at 1,000. The official dev split is
   table-disjoint from train.

WikiSQL contains only single-table `SELECT ... WHERE` shapes. It improves table
and column grounding but teaches no joins, subqueries, CTEs, or ordering. It is a
supplement, not a replacement for Spider or Gretel.

### 4.4 The mixture and the exact step count

The loader (`src/data/multi_dataset.py`) builds:

| Component | Rows |
|---|---:|
| Finance train, repeated 3× | 2,835 |
| Spider sampled | 600 |
| WikiSQL sampled | 8,000 |
| **Total training rows** | **11,435** |
| Validation (106 Gretel + 885 Spider + 1,000 WikiSQL) | 1,991 |
| Held-out Gretel test | 72 |

Why repeat finance 3×? With 8,000 WikiSQL rows, a single copy of 945 finance
rows would be under 8% of the mixture. Repetition keeps the target domain
represented without discarding structural data.

The formatter then removes rows that violate the serving contract:

- 309 non-`SELECT` targets (writes and DDL) are dropped.
- 4 rows cannot fit question + required schema + full SQL in 512 tokens.
- Result: **11,122 training examples** and **1,976 validation examples**.

With per-device batch 1 and gradient accumulation 4, the effective batch is 4:

```text
ceil(11,122 / 4) = 2,781 optimizer steps
```

That is exactly the observed run length.

### 4.5 Leakage and reproducibility controls

- Gretel's official test split is never trained on.
- Finance validation is grouped by exact question.
- Spider validation is database-disjoint.
- WikiSQL uses its official, table-disjoint dev split.
- Data snapshots, model revisions, and the WikiSQL archive are pinned by revision
  or checksum.
- Training uses `set_seed(42)` and a seeded shuffle.
- Prepared snapshots and adapters are git-ignored, so a fresh clone must
  regenerate data or download released artifacts; the tracked `reports/` prove
  what was measured.

---

## 5. Training in depth

### 5.1 Why QLoRA

Full fine-tuning a 1.5B model needs optimizer state and gradients for every
weight, which does not fit a 4 GB laptop GPU. QLoRA combines two memory
reductions:

1. **4-bit base weights.** `BitsAndBytesConfig(load_in_4bit=True,
   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=...)` stores the frozen base
   in NF4 (a 4-bit format optimized for normally distributed weights) and
   computes in bf16 when the GPU supports it, otherwise fp16.
2. **Low-rank adapters.** Only small matrices are trained. With `r=16`,
   `lora_alpha=32`, `lora_dropout=0.05`, and `target_modules="all-linear"`, the
   run trains **18,464,768 of 1,562,179,072 parameters (1.18%)**.

`all-linear` lets PEFT discover projection names for Qwen, TinyLlama, and Phi
instead of hardcoding `q_proj`/`v_proj`. The adapter is therefore portable
across model families as long as the base architecture matches.

Before attaching LoRA, `train.py` calls
`prepare_model_for_kbit_training` when the model is loaded in 4-bit. This casts
layer norms to fp32 and enables input gradients, which stabilizes quantized
training. It then enables non-reentrant gradient checkpointing and disables the
KV cache during training.

Memory intuition, not a benchmark: a 1.5B model at 16-bit is roughly 3 GB of
weights; at 4-bit it is roughly 0.75 GB. The remaining budget goes to
activations, LoRA parameters, and optimizer state. Gradient checkpointing trades
extra compute for lower activation memory; paged AdamW 8-bit keeps optimizer
state small and offloadable.

### 5.2 Completion-only SFT

Training uses TRL `SFTTrainer` with:

```python
completion_only_loss=True
packing=False
dataset_text_field="text"
```

`format_dataset` produces `{prompt, completion}` pairs. TRL masks the prompt
tokens from the loss, so the model is optimized only on SQL tokens. This matters:
the instruction and schema are long, and training on them would waste capacity
on reconstructing text the user already supplied.

`packing=False` keeps each example separate, so no completion is split across
sequence boundaries. The EOS marker is reserved during length checks
(`len(ids) + 1 <= max_length`).

### 5.3 Prompt policy and schema pruning

Training and serving must use the same prompt. `build_prompt`
(`src/utils/prompting.py`) renders:

```text
system:    You are a precise text-to-SQL assistant.
user:      Write one read-only SQL SELECT query ... 
           Question: <question>
           Schema: <DDL>
assistant:  <generated SQL>
```

For tokenizers with a chat template (Qwen uses ChatML), the prompt is rendered
with `apply_chat_template(..., add_generation_prompt=True)`. For older
tokenizers, a legacy `<|user|> ... <|assistant|>` layout is used. The Qwen path
also passes `enable_thinking=False` when supported, because this task wants a
direct SQL answer rather than a reasoning trace.

Schema pruning happens before tokenization (`src/training/formatting.py`):

1. If the full schema fits, keep it.
2. Otherwise parse the target SQL for table references.
3. Put DDL for referenced tables first.
4. Add optional tables only while the sequence still fits.
5. If even the required tables cannot fit, drop the example.

This prioritizes correctness of the target over breadth of context. It also
means an example never teaches the model to truncate a query mid-statement.

### 5.4 Hyperparameters

| Setting | Value | Reason |
|---|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Code/SQL pretraining fits 4 GB VRAM |
| LoRA rank / alpha | 16 / 32 | Capacity-to-memory balance; alpha = 2r |
| LoRA dropout | 0.05 | Light regularization on a small dataset |
| Target modules | `all-linear` | Architecture-agnostic attention + MLP adaptation |
| Quantization | 4-bit NF4 | Fits the 4 GB GPU |
| Compute dtype | bf16 if supported, else fp16 | Numerical stability and speed |
| Attention | SDPA | Predictable memory on small GPUs |
| Batch / accumulation | 1 × 4 | Effective batch 4 |
| Epochs | 1 | 2,781 steps is already enough for a 1.5B adapter |
| Learning rate | 2e-4 | Standard LoRA range |
| Schedule | Cosine, 3% warmup | Warmup avoids early divergence; cosine decays into the final steps |
| Optimizer | `paged_adamw_8bit` | Low optimizer-state memory |
| Max sequence | 512 | Covers question + pruned schema + SQL |
| Seed | 42 | Reproducible shuffles and splits |
| Evaluation | End of epoch | One validation pass per run |

### 5.5 Observed training run

From `logs/train_qwen_wikisql.log` (29 October 2026, RTX 3050 Laptop, 4.3 GB):

| Measurement | Value |
|---|---:|
| Optimizer steps | 2,781 |
| Wall clock | 5 h 20 min |
| Training loss (run average) | 0.0637 |
| Validation loss | 0.1321 |
| Validation token accuracy | 95.88% |
| Validation rows | 1,976 |
| Trainable parameters | 18.46 M (1.18%) |

Training loss is the average over noisy mini-batches; validation loss is the
more meaningful signal. The gap between 0.064 and 0.132 is expected when the
validation set contains Spider/WikiSQL schemas and questions not seen in
training, even though the finance test split is the real held-out target.

### 5.6 Monitoring

- `TrainingProgress` (`src/training/train.py`) owns a single tqdm bar over
  optimizer steps, with `loss`, `lr`, and `epoch` in the postfix.
- Validation prints `eval_loss` at epoch end.
- MLflow logs parameters and metrics to the local `mlruns/` file store. MLflow
  is optional: a logging failure does not stop training.
- `scripts/watch_training.py` renders a live dashboard from the training log.

---

## 6. Evaluation in depth

### 6.1 What is measured and on what denominator

All headline comparisons use the same 72 held-out Gretel finance examples. Of
those, only **48 reference queries execute successfully** in their SQLite
fixtures; the other 24 are reported as `gold_execution_errors` and excluded from
the execution-accuracy denominator. This is the single most important caveat
when reading the numbers: 48 examples means one example is about 2.1 percentage
points.

### 6.2 The four metrics

**Valid SQL rate.** Each prediction is parsed (exactly one statement, `SELECT`
or `WITH`) and validated with `EXPLAIN QUERY PLAN` against a fresh schema-only
in-memory SQLite database. The connection is read-only, has a 1-second timeout,
a 10-million VM-step ceiling, a SQL-length ceiling, and a result-row ceiling.
"Valid" means the schema accepts the query, not that it returns the right rows.

**Exact match.** `src/eval/metrics.py` tokenizes SQL with `sqlparse`, drops
whitespace and comments, folds keyword case, and compares the token sequence
including literals, identifiers, and punctuation. This is a lexical metric. It
does not understand aliases, equivalent expressions, or semantic equivalence.

**Execution accuracy.** Both reference and prediction run against the sample's
own `sql_context`. If either query has a top-level `ORDER BY`, rows must match in
order; otherwise rows are compared as multisets, preserving duplicates. Numbers
are normalized through `Decimal`, NULLs compare to NULLs, and blobs remain
distinct from text. Execution accuracy answers "does this produce the same
answer on this fixture?", not "is this query correct in general?"

**Test loss / perplexity.** The target SQL is tokenized with EOS appended, the
prompt is masked with `-100`, and the model's cross-entropy is weighted by target
token count across examples. Overlength examples are skipped and counted
(`test_loss_skipped_long`). Perplexity is `exp(loss)` when the loss is finite and
below 700. This measures how well the model ranks the reference SQL, not whether
generation succeeds.

### 6.3 Why the dashboard says 25.0% but the comparison says 22.2%

Two exact-match implementations exist:

- `src/eval/eval.py` uses `eval_accuracy.normalize_sql`: lowercase, collapse
  whitespace, strip the trailing semicolon. It reports **18/72 = 25.0%** and
  writes `api/static/evaluation.json`.
- `scripts/evaluate_models.py` uses `src/eval/metrics.normalize_sql`: token-level
  normalization. It reports **16/72 = 22.2%** and drives the cross-model table.

The token metric is stricter, so it is the one used for model comparisons. The
dashboard numbers are internally consistent but not directly comparable to the
comparison table.

### 6.4 Results

All rows: same 72 examples, greedy decoding, 256 new tokens, current ChatML
prompt.

| Model | Exact match | Valid SQL | Execution (n=48) | Test loss | P50 latency |
|---|---:|---:|---:|---:|---:|
| TinyLlama 1.1B base | 1.4% | 79.2% | 43.8% | 0.724 | 21,847 ms |
| TinyLlama + Gretel finance | 18.1% | 73.6% | 56.2% | 0.482 | 9,165 ms |
| TinyLlama + multi-dataset | 18.1% | 65.3% | 54.2% | 0.487 | 3,192 ms |
| TinyLlama + multi-dataset (candidate) | 15.3% | 66.7% | **58.3%** | 0.449 | 3,377 ms |
| TinyLlama + multi-dataset + Spider (ckpt359) | 16.7% | 63.9% | 54.2% | 0.460 | 3,357 ms |
| phi3-sql (mislabeled TinyLlama adapter) | 4.2% | 31.9% | 22.9% | 0.935 | 2,743 ms |
| Qwen2.5-Coder 1.5B base | 12.5% | 62.5% | 52.1% | 0.476 | 3,032 ms |
| **Qwen2.5-Coder + WikiSQL/Spider/Finance** | **22.2%** | 68.1% | 56.2% | **0.235** | 5,433 ms |

Qwen base → adapter:

| Metric | Base | Adapter | Delta |
|---|---:|---:|---:|
| Exact match | 12.5% | 22.2% | +9.7 pts |
| Valid SQL | 62.5% | 68.1% | +5.6 pts |
| Execution | 52.1% | 56.2% | +4.2 pts |
| Test loss | 0.476 | 0.235 | -50.6% |

Interpretation:

- The adapter is the best model on exact match and held-out loss, and it
  improves every headline metric over its base.
- It is **not** the best on valid-SQL rate: TinyLlama base and the TinyLlama
  finance adapter score higher. A model can produce schema-valid SQL that is
  semantically wrong.
- Execution accuracy is one example behind the best TinyLlama adapter on a
  48-example denominator, so that difference is not meaningful.
- Qwen latency (P50 5,433 ms) is higher than its own base (3,032 ms) and the
  multi-dataset TinyLlama adapters (about 3.2-3.4 s), but lower than the
  TinyLlama finance adapter (9,165 ms). Latency depends on generated length and
  GPU scheduling, not only parameter count.

By complexity for the Qwen adapter:

| Complexity | n | Exact | Valid | Execution |
|---|---:|---:|---:|---:|
| Basic SQL | 43 | 30.2% | 69.8% | 57.1% |
| Aggregation | 10 | 10.0% | 90.0% | 62.5% |
| Single join | 9 | 11.1% | 66.7% | 57.1% |
| Window functions | 5 | 20.0% | 40.0% | 33.3% |
| Subqueries | 2 | 0% | 50.0% | 0% |
| CTEs | 2 | 0% | 0% | n/a |

The pattern is consistent with the data mix: basic SQL and aggregation are
well represented; joins are harder; window functions, subqueries, and CTEs are
rare in both training and evaluation.

### 6.5 How to reproduce the comparison

```powershell
.venv\Scripts\python.exe scripts/evaluate_models.py
.venv\Scripts\python.exe scripts/compare_all_models.py
```

`evaluate_models.py` scores base versus configured adapter and writes
`reports/evaluation-qwen/`. `compare_all_models.py` loads each base family once,
scores the base, then attaches each adapter in turn (unloading in between) and
writes `reports/model_comparison/`. Both write `summary.json`, `report.md`, and
per-example `predictions.jsonl`.

---

## 7. Serving in depth

### 7.1 Request lifecycle

`POST /generate-sql` with `{ "question": "...", "schema": "..." }`:

1. If no schema is supplied, use the configured personal-finance DDL.
2. Build a cache key from normalized question, schema hash, model version,
   prompt version, generation settings, and scope.
3. Return a cache hit immediately if present and not bypassed.
4. Otherwise take a per-key async lock (prevents same-question stampedes inside
   one worker), re-check the cache, then generate.
5. Generation runs in a thread pool under a model lock; CUDA is synchronized
   before and after timing.
6. Extract SQL from the decoded text, normalize dialect differences, validate,
   and optionally fall back.
7. Cache the response only if it is valid.
8. Return SQL plus latency and provenance fields.

### 7.2 Output cleanup and guardrails

`extract_sql` (`src/utils/prompting.py`) strips markdown fences, an optional
`SQL:` prefix, any natural-language preamble, and trailing explanation after the
first semicolon.

`prepare_sql` (`src/inference/sql_guard.py`) then:

1. Rewrites MySQL/Postgres date expressions:
   `DATE_SUB(CURDATE(), INTERVAL 3 MONTH)` becomes
   `date('now', '-3 months')`; `CURDATE()`/`CURRENT_DATE` become `date('now')`;
   `NOW()` becomes `datetime('now')`.
2. Rejects writes. A word-boundary check blocks `INSERT`, `UPDATE`, `DELETE`,
   `DROP`, `ALTER`, `CREATE`, `REPLACE`, `ATTACH`, `DETACH`, `PRAGMA`, `VACUUM`,
   and `REINDEX` anywhere in the statement.
3. Requires the statement to start with `SELECT` or `WITH`.
4. Rejects multiple statements.
5. Executes `EXPLAIN QUERY PLAN` in a fresh in-memory database built from the
   schema.

If validation fails, the API tries a deterministic fallback
(`src/inference/demo_sql.py`) for common demo intents. The fallback maps
category synonyms (`groceries` → `Food`), relative time phrases (`last month`,
`last 3 weeks`), aggregation intents (`by category`, `by merchant`), and limits
(`top 5`). It is intentionally conservative: if the schema does not contain the
required table/column, it returns `None`.

Only if no fallback applies does the API try removing duplicate unaliased joins
and re-validating. This is a last resort because removing a join can change
meaning. Responses carry `source`:

| `source` | Meaning |
|---|---|
| `model` | Model output was valid unchanged |
| `model-normalized` | Date/dialect rewrite or duplicate-join repair made it valid |
| `safe-fallback` | Deterministic demo query replaced invalid model output |
| `invalid` | Neither model SQL nor fallback validated; response is flagged |

### 7.3 Response contract

| Field | Meaning |
|---|---|
| `sql` | Final statement |
| `schema_used` | Schema actually used |
| `valid` | Accepted by the schema-only SQLite check |
| `source` | Provenance, see table above |
| `warning` | Human-readable caveat when fallback or repair occurred |
| `validation_error` | Why the original model SQL failed |
| `latency_ms` | Total request latency |
| `response_latency_ms` | Same total, kept for comparison |
| `generation_latency_ms` | Model inference only; `null` on a cache hit |
| `cache_hit` | Whether the response came from Redis |
| `model_version` | Digest of base model, revision, adapter path, and adapter bytes |
| `cache_error` | Whether the cache failed during this request |

`model_version` is recomputed at startup from the base model name, resolved
revision, adapter path, and hashes of `adapter_config.json` and
`adapter_model.safetensors`. Replacing the adapter therefore invalidates cache
keys automatically.

### 7.4 Redis cache

The cache is optional and fail-open. If `REDIS_URL` is unset, the package is
missing, or Redis errors, the API generates normally. Defaults: 300-second TTL,
1-second connect/read timeout, 20 connections.

Keys look like:

```text
nl2sql:v1:<sha256 of canonical JSON>
```

The canonical JSON includes the case-folded, whitespace-normalized question, a
schema SHA-256, model version, prompt version (`finance-text-to-sql-v3-chatml`),
generation settings (`max_new_tokens`, `do_sample=false`), and scope (`public`).
Raw questions are not stored in keys.

Only validated responses are cached. `X-Cache-Bypass: true` skips both reads and
writes, which is what the benchmark uses. `POST /cache/invalidate` exists only
when `CACHE_ADMIN_TOKEN` is set and requires `X-Cache-Admin-Token`.

Limitations: locks are in-process, so multiple Uvicorn workers do not coordinate;
the cache is public-scoped because the app has no auth or tenant model.

### 7.5 API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | SQL Studio dashboard |
| GET | `/evaluation.json` | Bundled held-out summary for the dashboard |
| GET | `/health` | Status, model, adapter, dataset |
| GET | `/model` | Base model, adapter, dataset, domain, max tokens |
| GET | `/schema` | Default demo DDL |
| GET | `/dataset` | Source datasets and supplements |
| POST | `/generate-sql` | Generate, validate, fallback, cache |
| POST | `/cache/invalidate` | Hidden; token-protected |
| GET | `/docs` | OpenAPI UI (FastAPI default) |

CORS allows any origin for GET/POST and sets
`Access-Control-Allow-Private-Network: true` for Chrome's Local Network Access
preflight. That header is what permits an HTTPS page on Vercel to call
`http://127.0.0.1:8000`.

### 7.6 Dashboard and Vercel

`api/static/index.html` is a self-contained page with no build step. It renders
the searchable schema, highlights SQL, labels each result as model,
normalized, or fallback, keeps the last eight questions in `localStorage`,
exports `.sql`, shows runtime metadata, and renders the evaluation report.

API base resolution order:

1. `?api=<url>` query parameter (then stored in `localStorage`)
2. `localStorage.financeSqlApi`
3. `<meta name="finance-sql-api" content="...">`
4. `window.location.origin` when served by the API itself
5. `http://127.0.0.1:8000` when hosted elsewhere

The Vercel project is **static hosting only**. It cannot run PyTorch or hold GPU
weights, so the browser calls the local FastAPI directly. On a public HTTPS page,
Chrome asks for local network access; denying it produces the runtime note
"Allow local network access for this page".

Current production alias: `https://static-ten-pi.vercel.app`.

### 7.7 Docker

`docker/docker-compose.yml` starts Redis and an API image built from the
repository root. The image is `python:3.10` (CPU-only) and installs unpinned
dependencies. It is fine for exploring the API, but it will not provide GPU
inference; for the trained adapter, run Uvicorn in the project virtual
environment instead.

---

## 8. Tests and what they prove

`python -m unittest discover -s tests -v` runs **33 tests**:

| File | Tests | Guarantees |
|---|---:|---|
| `test_sql_guard.py` | 13 | Writes rejected, duplicate joins flagged, date rewrite, compact schema support, demo fallback intents |
| `test_eval_metrics.py` | 7 | Exact-match normalization, multiset/order comparison, numeric/NULL equivalence, read-only errors, timeouts, percentiles |
| `test_api_generation.py` | 5 | Valid SQL passthrough, fallback replacement, invalid flagging, cache-hit skips generation, PNA preflight |
| `test_sql_cache.py` | 4 | Key separation by schema/model, hit/miss/TTL, invalidation, fail-open |
| `test_training_formatting.py` | 4 | Read-only contract, schema pruning order, mutating targets dropped, overlength rows dropped |

What they do **not** prove: model accuracy, behavior against a live database,
browser permission flows, multi-worker cache coordination, or GPU/CPU parity.
Those require the evaluation scripts and a running service.

---

## 9. Reproducing from a fresh clone

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python data/prepare_dataset.py          # Gretel finance snapshots
python data/prepare_multi_dataset.py    # Spider snapshots
python data/prepare_wikisql.py          # pinned archive + validation
python scripts/validate_setup.py

python -m src.training.train            # ~5 h on a 4 GB RTX 3050
python scripts/evaluate_models.py       # base vs adapter
python scripts/compare_all_models.py    # all adapters on the same holdout
python -m unittest discover -s tests -v

scripts\start_api.cmd                   # http://127.0.0.1:8000
```

For a short GPU smoke test:

```powershell
python -m src.training.train --max-steps 20 --limit-train 200 --limit-validation 50 --output-dir models/qwen-smoke
```

Important: `--limit-train` matters. Without it, the formatter tokenizes the full
11,122-row set before the first optimizer step, so the GPU appears idle for
minutes.

Optional Redis:

```powershell
docker compose -f docker/docker-compose.yml up -d redis
$env:REDIS_URL = "redis://localhost:6379/0"
scripts\start_api.cmd
```

### Publishing the adapter to Hugging Face

The adapter directory contains everything PEFT needs:

| File | Upload? | Why |
|---|---|---|
| `adapter_config.json` | Yes | Declares base model and LoRA hyperparameters |
| `adapter_model.safetensors` | Yes | The 36 MB trained weights |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` | Yes | Makes the repo self-contained |
| `README.md` | Replace | Use it as the model card |
| `training_args.bin` | Optional | Contains local paths; usually omit |
| `checkpoint-2781/` | No | Duplicate weights and optimizer state |

Sketch:

```powershell
.venv\Scripts\pip.exe install huggingface_hub
.venv\Scripts\hf.exe auth login
.venv\Scripts\hf.exe repo create <your-user>/qwen2.5-coder-1.5b-finance-sql-lora --repo-type model --private

.venv\Scripts\hf.exe upload <your-user>/qwen2.5-coder-1.5b-finance-sql-lora `
  models\qwen2.5-coder-1.5b-finance-sql-lora . `
  --repo-type model `
  --exclude "checkpoint-2781/*" --exclude "training_args.bin"
```

Consumers then load it with:

```python
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "<your-user>/qwen2.5-coder-1.5b-finance-sql-lora")
```

Published copy (public):
https://huggingface.co/ncfad/qwen2.5-coder-1.5b-finance-sql-lora — adapter
weights, tokenizer, chat template, and card; `checkpoint-2781/` and
`training_args.bin` are excluded.

Model-card essentials: base model and revision, adapter type/rank, datasets
(Gretel finance, Spider, WikiSQL), training hyperparameters, held-out metrics
with the 48-example denominator, intended use, limitations, and licenses. The
card's YAML `datasets` list can only carry Hub dataset IDs, so it names Gretel
and Spider; WikiSQL was fetched from the pinned Salesforce GitHub tarball and is
documented in the card body and tags instead. The
base model is Apache-2.0; Gretel is Apache-2.0; Spider is CC-BY-SA-4.0; WikiSQL
is Wikipedia-derived. Because the adapter was trained on CC-BY-SA data, review
license compatibility before publishing; state the licenses explicitly in the
model card.

---

## 10. Design decisions and trade-offs

| Decision | Gain | Cost |
|---|---|---|
| Adapter instead of full fine-tune | Fits 4 GB, 36 MB artifact | Slightly less capacity; base required |
| 4-bit NF4 base | Roughly 4× smaller weights | Quantization noise; bitsandbytes/CUDA constraints |
| `all-linear` LoRA | Portable across model families | More trainable params than attention-only |
| Completion-only loss | Learns SQL, not prompt reconstruction | Cannot learn schema-reading formatting from loss |
| Schema pruning | Full SQL targets always fit | Some schemas are shortened; some rows dropped |
| Mixed datasets | Better structure and grounding | Domain dilution; finance needed 3× repetition |
| Greedy decoding | Reproducible evaluation | Less diverse; no sampling-based repair |
| SQLite fixture evaluation | Real execution signal per sample | 24/72 gold queries are not executable |
| Schema-only API validation | Catches hallucinations and writes | Does not prove semantic correctness |
| Demo fallback | Demo never returns guaranteed-broken SQL | Can mask model failure; only covers demo intents |
| Static dashboard + local API | No GPU hosting cost | Requires a running local server and browser permission |

---

## 11. Limitations and failure modes

1. **Small, synthetic holdout.** 72 examples, 48 executable. Differences of one
   or two examples are noise.
2. **Gold query defects.** 24 reference queries cannot run in SQLite; some source
   pairs are semantically questionable. The model is partly scored against noisy
   labels.
3. **Exact match is lexical.** It rewards phrasing, not equivalence, and two
   implementations disagree (22.2% vs 25.0%).
4. **Validity is not correctness.** 68.1% valid SQL coexists with 56.2%
   execution accuracy.
5. **Fallback can hide errors.** A `safe-fallback` response is valid but may not
   reflect what the model would have produced.
6. **The API does not execute SQL.** Downstream systems still need read-only
   connections, authorization, timeouts, and tenant scoping.
7. **No authentication or rate limiting.** The API is a local/demo service, not a
   public multi-tenant backend.
8. **Single-worker cache locks.** Multiple Uvicorn workers can duplicate work.
9. **CPU Docker image.** It cannot serve the GPU-loaded adapter at usable speed.
10. **Adapter and data are git-ignored.** A clone cannot reproduce results
    without retraining or downloading released artifacts.
11. **CTEs, subqueries, and window functions remain weak.** The data mix and
    model size limit them.
12. **Chrome local-network permission is a hard dependency** for the deployed
    dashboard calling `127.0.0.1`.

---

## 12. Prioritized next steps

1. **Expand the evaluation set.** Add real or independently generated finance
   questions, and report confidence intervals rather than point estimates.
2. **Add execution-guided checks.** Validate semantics on more fixtures and
   audit every gold pair.
3. **Improve structural coverage.** Oversample Spider joins/CTEs and add finance
   examples with windows and subqueries.
4. **Compare rank/target policies.** Run r=8/16/32 and attention-only vs
   all-linear ablations on the same holdout.
5. **Try a 3B-class coder model** if 6-8 GB VRAM is available; inference-only is
   already feasible on the current GPU.
6. **Add a real SQL execution sandbox** with read-only roles, row limits, and
   per-tenant authorization before any production use.
7. **Pin dependencies and build a GPU-aware container** if deployment moves
   beyond the local machine.
8. **Publish the adapter** with a complete model card so results are reusable.

---

## 13. Glossary

| Term | Meaning |
|---|---|
| SFT | Supervised fine-tuning: next-token training on input/target pairs |
| LoRA | Low-rank adaptation: small trainable matrices attached to frozen weights |
| QLoRA | LoRA on a quantized (usually 4-bit) base model |
| NF4 | 4-bit NormalFloat quantization optimized for normal weight distributions |
| Effective batch | Per-device batch × gradient accumulation × devices |
| Completion-only loss | Loss computed only on target tokens, prompt masked |
| Grouped split | Holdout that keeps identical prompts on one side |
| Database-disjoint | Holdout that keeps entire databases on one side |
| Greedy decoding | Always pick the highest-probability next token |
| Exact match | Token/string equality after normalization, not semantic equality |
| Execution accuracy | Predicted and reference queries return equivalent rows on a fixture |
| Gold execution error | Reference SQL that cannot run, excluded from execution denominator |
| P50/P95/P99 | Latency percentiles; P50 is the median |
| Fail-open cache | Cache errors fall back to normal inference |
| Local Network Access | Browser permission for a public page to call a loopback service |

---

## 14. Where to read what

| Question | File |
|---|---|
| How do I run the API? | `docs/local-api.md` |
| What were the measured results? | `reports/model_comparison/`, `reports/evaluation-qwen/` |
| How is evaluation implemented? | `src/eval/metrics.py`, `scripts/evaluate_models.py` |
| How is SQL validated? | `src/inference/sql_guard.py` |
| How does the fallback work? | `src/inference/demo_sql.py` |
| How is training configured? | `configs/training_config.yaml`, `src/training/train.py` |
| How is data prepared? | `data/prepare_dataset.py`, `data/prepare_multi_dataset.py`, `data/prepare_wikisql.py` |
| How does the dashboard connect? | `api/static/index.html`, `api/main.py` |
| What changed over time? | `model_upgrade_log.md` |

---

*Guide written 29 October 2026 against the working tree at commit `e62d4a3`.
Metrics are artifacts of the recorded run, not guarantees of future performance.*
