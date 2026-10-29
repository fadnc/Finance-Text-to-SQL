# Finance Text-to-SQL: Technical Guide

> **Migration update (2026-10-27):** The project no longer uses the hand-written
> template dataset. Training now loads
> [`gretelai/synthetic_text_to_sql`](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
> and filters it to the `finance` domain. Evaluation now executes each prediction
> against the sample's own `sql_context` in an in-memory SQLite database. Sections
> 3, 4, and 8 below still describe the pre-migration template pipeline; use the
> README and the migration notes in those sections for the current behavior.
>
> **Migration update (2026-10-28):** The configured base model changed from
> TinyLlama-1.1B to `Qwen/Qwen2.5-Coder-1.5B-Instruct`, prompts are rendered with
> the tokenizer's chat template instead of hardcoded `<|user|>` tags, LoRA targets
> changed to `all-linear`, and WikiSQL was added as a third training source
> (8,000 sampled rows per epoch). The Qwen adapter finished training on
> 2026-10-29 (2,781 steps, validation loss 0.1321) and scores 22.2% exact,
> 68.1% valid SQL, and 56.2% execution on the 72-row finance holdout; the full
> cross-model table is in `reports/model_comparison/`. See
> `model_upgrade_log.md` Appendix C.

**Repository investigation, interview reference, and operational notes**  
Repository state inspected: 20 October 2026; dataset migration applied 27 October 2026; model and WikiSQL migration applied 28 October 2026; Qwen training, evaluation, and serving completed 29 October 2026.  
Evidence labels: **Implemented** means directly present in the checked-in code or current repository artifacts; **Documented** means stated in repository documentation; **Inference** means a conclusion from implementation details; **Proposed** means future work.

## 1. Executive summary

**Implemented:** This repository is a prototype pipeline that fine-tunes a causal language model to translate questions about a supplied SQL schema into SQL. Training is supervised fine-tuning (SFT) over the finance subset of `gretelai/synthetic_text_to_sql`, plus Spider and WikiSQL cross-domain examples, with each example carrying its own DDL. [`src/utils/model_loader.py`](../src/utils/model_loader.py) loads the configured `Qwen/Qwen2.5-Coder-1.5B-Instruct` base in 4-bit NF4 when CUDA and bitsandbytes appear available. [`src/training/lora_config.py`](../src/training/lora_config.py) attaches rank-16 `all-linear` LoRA adapters. A FastAPI service generates SQL; a separate evaluator creates an in-memory SQLite database from each test sample's context and compares reference-query results.

**Intended users (inference):** Developers experimenting with domain adaptation, and consumers asking finance questions in natural language. No user study or product deployment evidence is present.

**System boundary:** The service returns generated SQL and a schema string. The API does not connect to a live finance database or execute the returned SQL. The local evaluator executes SQL against synthetic SQLite fixtures. No authentication, authorization, CI workflow, or hosted backend deployment is tracked; the linked Vercel project is a static dashboard that connects to a separately hosted FastAPI server. The API validates generated SQL against a schema-only SQLite database and rejects write statements, but that validation is not a write-capable execution sandbox. A 32-test `unittest` suite is tracked under `tests/`.

**Status:** This is best described as an educational/prototyping project. Production readiness, measured model quality, and operational performance are **not verified from the repository**.

## 2. Repository map and evidence

| Area | Main files | Responsibility |
|---|---|---|
| Configuration | [`configs/training_config.yaml`](../configs/training_config.yaml) | Base model, input/output paths, hyperparameters, schema text |
| Data preparation | [`data/prepare_dataset.py`](../data/prepare_dataset.py), [`data/prepare_multi_dataset.py`](../data/prepare_multi_dataset.py), [`data/prepare_wikisql.py`](../data/prepare_wikisql.py), [`src/data/gretel.py`](../src/data/gretel.py), [`src/data/multi_dataset.py`](../src/data/multi_dataset.py) | Load Gretel, Spider, and WikiSQL; filter domains, render DDL, build train/validation/test splits, write JSON snapshots |
| SQLite fixture | [`data/setup_test_db.py`](../data/setup_test_db.py) | Optional six-table personal-finance demo database for the API dashboard |
| Training | [`src/training/train.py`](../src/training/train.py), [`src/training/formatting.py`](../src/training/formatting.py), [`src/training/data_split.py`](../src/training/data_split.py), [`src/training/lora_config.py`](../src/training/lora_config.py) | Split, format, load base model, attach adapter, run TRL SFT |
| Model loading | [`src/utils/model_loader.py`](../src/utils/model_loader.py) | Tokenizer and causal LM loading with quantized/CUDA/CPU branches |
| Inference helper | [`src/inference/gen_sql.py`](../src/inference/gen_sql.py) | Python generation helper |
| HTTP API | [`api/main.py`](../api/main.py) | FastAPI schema, health, and SQL generation endpoints |
| Batch inference | [`scripts/generate_predictions.py`](../scripts/generate_predictions.py) | Generate predictions for the deterministic held-out split |
| Evaluation | [`src/eval/eval.py`](../src/eval/eval.py), [`src/eval/eval_accuracy.py`](../src/eval/eval_accuracy.py) | Exact string match, SQLite result comparison, error labels |
| Orchestration | [`scripts/run_pipeline.py`](../scripts/run_pipeline.py), [`scripts/validate_setup.py`](../scripts/validate_setup.py) | Sequential pipeline and setup checks |
| Packaging | [`requirements.txt`](../requirements.txt), [`docker/Dockerfile`](../docker/Dockerfile), [`docker/docker-compose.yml`](../docker/docker-compose.yml) | Python dependencies and a proposed containerized API |

Tracked files include no CI workflow, migration system, or separate production database configuration. The tracked `tests/` suite covers SQL guardrails, API generation, evaluation metrics, and Redis cache behavior (32 tests passing). `models/` has local adapter artifacts in this checkout, but `.gitignore` excludes model outputs; the tracked repository does not include the trained adapter weights. Generated data artifacts are also ignored. Local presence does not establish reproducible availability to a fresh clone; the held-out evaluation artifacts under `reports/` are tracked.

## 3. Problem and data contract

The model input is a natural-language question accompanied by a schema description. After the 2026-10-27 migration, the training record is derived directly from a Gretel row: `question` is `sql_prompt`, `schema` is the DDL extracted from `sql_context`, and `output` is `sql`. After the 2026-10-28 migration, the formatter renders that record through the tokenizer's chat template (ChatML for Qwen) using `Question: ...` and `Schema: ...`; the legacy `<|user|> ... <|assistant|>` layout remains only as a fallback for tokenizers without a template. The full `sql_context` (including seed rows) is retained for evaluation but is not shown to the model. The target remains SQL text; there is no structured SQL AST, tool call, or execution result in the model output contract.

The configured base is `Qwen/Qwen2.5-Coder-1.5B-Instruct` and the adapter output directory is `models/qwen2.5-coder-1.5b-finance-sql-lora`. The historical `models/phi3-sql` directory is a mislabeled TinyLlama adapter and is not used by the current configuration. The configured demo schema covers `users`, `accounts`, `expenses`, `income`, `budgets`, and `transactions`, with user/account foreign-key relationships described in the test DB setup.

**Migration update:** [`src/data/gretel.py`](../src/data/gretel.py) loads `gretelai/synthetic_text_to_sql` with `datasets.load_dataset`, filters to the configured domains (`finance` by default), and maps each row into the project format. The source dataset has 100,000 train and 5,851 test rows across many domains; the finance filter yields 1,051 train and 72 test rows before the validation holdout. `extract_schema` executes each context in an in-memory SQLite database and reads canonical `CREATE TABLE` DDL from `sqlite_master`, with a regex fallback. The official test split is held out; a grouped 10% validation split is carved from the training rows.

**WikiSQL update:** [`data/prepare_wikisql.py`](../data/prepare_wikisql.py) downloads the pinned 26 MB GitHub release rather than the HF copy, because `datasets 4.6.1` no longer runs dataset loading scripts. It joins `train/dev/test.jsonl` to the matching `*.tables.jsonl`, renders `sql.conds` triples into quoted single-table `SELECT` statements, and executes every gold query against an in-memory SQLite copy of its table before writing `wikisql_{train,validation,test}.json`. Training snapshots omit `INSERT` context; the 2,000-row dev/test slices retain it. [`src/data/multi_dataset.py`](../src/data/multi_dataset.py) mixes 8,000 WikiSQL rows, 600 Spider rows, and the 945 finance rows repeated three times.

**Pre-migration behavior (superseded):** `data/gen_dataset.py` created examples from hand-written templates and random choices of names, categories, thresholds, and limits. It covered aggregates, grouping, filters, joins, subqueries, ordering, budget comparisons, and CASE expressions. That generator is now a deprecated shim that routes to `data/prepare_dataset.py`.

The test database also uses random values and the current date for its date fields. It does not seed its random generator. A future regeneration can therefore change the fixture and recent-date query results. Query behavior for an empty sum or an absent category follows SQLite semantics; callers should not assume a non-null numeric answer.

## 4. Architecture and execution flow

```text
training_config.yaml ──► gretelai/synthetic_text_to_sql ──► domain filter + DDL extraction
                                                              │
                                                              ▼
                                              grouped train/validation split
                                                              │
                                                              ▼
format prompt ─► load base tokenizer/model ─► QLoRA + SFTTrainer
                                                              │
                                                              ▼
                                                        adapter output

question + per-sample schema ─► inference prompt ─► model + adapter ─► SQL string
                                                                        │
                                                                        ▼
                                                          FastAPI response (no DB execution)

held-out samples + sql_context ─► batch generation ─► predictions.json
                                                                        │
                                                                        ▼
                                          in-memory SQLite context per sample
                                                                        │
                                                                        ▼
                                          result-set / exact-match evaluation
```

### Training path

1. `src/training/train.py` loads YAML from a working-directory-relative path, seeds Transformers/PyTorch through `set_seed`, and calls `load_gretel_splits`.
2. [`src/data/gretel.py`](../src/data/gretel.py) loads the Hugging Face dataset, filters to the configured domains, extracts DDL, and returns train/validation/test splits. The official test split is untouched; `grouped_split_indices` carves a validation set from the training rows and keeps identical questions in the same side of the boundary. The split groups by exact prompt, not by semantic paraphrase.
3. The formatter adds the instruction, per-sample schema, and expected SQL into the sequence consumed by `SFTTrainer`.
4. `load_model` loads the tokenizer and base model. With CUDA and the bitsandbytes import/attribute check, the model uses 4-bit NF4 quantization and automatic device mapping. Otherwise it chooses CUDA FP16 or CPU FP32.
5. Training enables gradient checkpointing, disables the model KV cache during training, attaches rank-16 `all-linear` LoRA modules (architecture-agnostic; includes Qwen's attention and MLP projections), and runs TRL `SFTTrainer` with configured epochs, batch size, gradient accumulation, learning rate, sequence length, an 8-bit paged AdamW optimizer, cosine schedule, and epoch evaluation/checkpointing.
6. `trainer.save_model` writes the final adapter to the configured output directory. The validation split is used as the trainer evaluation set; the official test split is used only by the separate prediction/evaluation run.

### Inference paths

There are three implementations, not one shared serving path: `src/inference/gen_sql.py`, `scripts/generate_predictions.py`, and `api/main.py` each build their own prompt and call `generate`. The Python helper and API decode the entire generated sequence and split on the assistant marker; batch inference slices off the prompt token IDs and decodes only new tokens. Generation parameters differ. The batch path uses greedy decoding (`do_sample=False`); the API explicitly samples (`do_sample=True`) with temperature 0.2. The API does not set an explicit random seed per request, so repeated responses can vary.

The API loads configuration at module import and loads the base model plus adapter at application startup. `/health` returns whether the model object exists; `/schema` returns configured schema text; `/generate-sql` accepts `{ "question": "..." }` and returns `sql` and `schema_used`. The API uses `.to("cuda")` for tokenized input even though its startup can select devices automatically; CPU-only serving is therefore not supported by that path as written. `torch.float16` is passed when loading its model regardless of device. Startup/model download failures are not translated into a custom health state.

### Evaluation path

`scripts/generate_predictions.py` loads the official test split, generates SQL from each sample's question and schema, and writes `data/processed/predictions.json` including the full `sql_context`. `src/eval/eval.py` reads each record's `predicted`, `output`, and `context`; `evaluate_batch` creates an in-memory SQLite database by executing the context, sets `PRAGMA query_only = ON`, and compares predicted and reference result sets. Exact match lowercases, collapses whitespace, and removes trailing semicolons. Execution accuracy is reported over samples whose reference query executes successfully; reference queries that use unsupported SQLite features or write statements are counted as `gold_execution_errors`. The evaluator executes generated SQL directly against the in-memory connection without a read-only authorizer beyond `query_only`, statement allowlist, or timeout; the process is local and the contexts are dataset-provided.

Error classification attempts prediction and reference execution and distinguishes missing columns/tables, syntax-like operational errors, other execution errors, and unequal result sets. The declared `SEMANTIC` label is not returned in the current classifier, and schema-column/table lists are collected but not used for further validation. A prediction that fails execution is marked not correct. The evaluator executes generated SQL directly against its SQLite connection without a read-only authorizer, statement allowlist, or timeout.

## 5. Design decisions and trade-offs

| Choice | Evidence and rationale | Trade-off / interview answer |
|---|---|---|
| SFT on a causal LM | `SFTTrainer` learns next-token prediction over prompt plus SQL target. | Straightforward for text generation; model can emit malformed, unsafe, or incorrect SQL. |
| LoRA adapters | Rank 16, alpha 32, dropout 0.05; `all-linear` targets so projection names resolve per architecture. | Reduces trainable parameter count and adapter storage versus full fine-tuning. Rank and target policy were chosen in code; no repository ablation establishes they are optimal. |
| 4-bit NF4 quantization | `BitsAndBytesConfig` branch, with bfloat16 if supported else float16 compute. | Reduces base-weight memory, at cost of compatibility constraints and possible numerical effects. Fallback to full precision may exceed available hardware. |
| Gradient checkpointing | Enabled in training and SFT config. | Saves activation memory by recomputing activations during backward pass, increasing computation. |
| Paged 8-bit AdamW | Configured as `paged_adamw_8bit`. | Optimizer state memory is reduced; requires compatible bitsandbytes setup. |
| Per-example SQLite execution | Each Gretel `sql_context` is executed in an in-memory database, then predicted and reference queries run read-only. | Tests many schemas and seed datasets rather than one fixture; context scripts that SQLite cannot execute are reported as gold errors. Set comparison still ignores order/duplicates. |
| FastAPI | Typed Pydantic request/response, schema validation through `src/inference/sql_guard.py`, a schema-aware demo fallback, and an optional Redis response cache. | Generated SQL is checked against a temporary schema-only SQLite database, and write statements are rejected. No authentication, throttling, or concurrency/load design is present. |

Alternatives such as full fine-tuning, larger/richer datasets, constrained decoding, schema linking, SQL parser validation, database-aware prompting, or execution-guided repair are **proposed directions**, not implemented features.

## 6. Configuration, setup, and operations

Dependencies in `requirements.txt` are unpinned names, so a fresh install is not version-locked. The model and dataset require external model hub access on first use; actual access/cache behavior depends on environment. No `.env.example` or required environment variables were found in the tracked file list. `.env` is ignored; secret contents were not inspected or included.

Documented setup begins with Python 3.10+, a virtual environment, and `pip install -r requirements.txt`; dataset preparation is `python data/prepare_dataset.py`, followed by `python data/prepare_multi_dataset.py` and `python data/prepare_wikisql.py`, and training is launched with `python -m src.training.train`. The optional demo database is created separately with `python data/setup_test_db.py`. The repository also has `scripts/validate_setup.py` and `scripts/run_pipeline.py`; the latter runs data preparation, training, prediction, and evaluation sequentially and exits on the first failure.

Configuration paths are relative to the current working directory, so invoke commands from repository root. The Dockerfile uses `WORKDIR /app`, copies requirements and then the repository, and starts Uvicorn on port 8000. Compose declares `build: { context: .., dockerfile: docker/Dockerfile }`, so the root `requirements.txt` is available during the build. The image still installs unpinned dependencies on a CPU-only `python:3.10` base and copies the local repository subject to `.dockerignore` filtering. These are repository-level configuration observations; the image has not been rebuilt in this checkout.

The README's VRAM and training-time figures come from the local run log and cannot be reproduced from a fresh clone without the same hardware. The training script sets `report_to="none"` for the Hugging Face Trainer but opens an explicit MLflow run, logs parameters and metrics, and ends the run on completion. The held-out comparison is reproducible from `scripts/evaluate_models.py` when the adapters are present locally; adapter weights are git-ignored, so a fresh clone must retrain or obtain them.

## 7. Security, reliability, and correctness concerns

1. **Generated SQL is not execution-safe by default.** The API currently returns SQL only, but downstream execution must apply authorization, read-only connections, statement restrictions, query limits/timeouts, and tenant scoping. The evaluation code itself runs arbitrary predicted SQL on the fixture connection.
2. **No authentication or authorization.** The API exposes generation and schema endpoints with no access control in the implementation.
3. **Device mismatch.** API inputs are always moved to CUDA; CPU fallback in the shared loader does not make API inference CPU-compatible.
4. **Serving and training generation differ.** Stochastic API decoding, greedy batch decoding, and different output slicing make results and output cleanup non-equivalent.
5. **Metric limitations.** Set equality discards order and duplicates. Exact match is a lexical proxy, not semantic validation. One small changing SQLite fixture is insufficient as a generalization benchmark.
6. **Potential SQL/data semantic mismatch.** Some generated question wording and SQL templates deserve review, e.g. a “transactions in category” question queries `expenses`, while `transactions` has no category column. A quarter template subtracts two months from the start of the current month, which is not a calendar-quarter calculation. Such defects can make labels wrong even when a model reproduces them.
7. **Synthetic data leakage and diversity.** The split groups identical `input` strings, but many examples share fixed templates or are duplicated by random choice. This can make a row-level held-out score optimistic and does not show generalization to unseen phrasing/schema.
8. **Non-deterministic fixture/data generation.** Dataset generation and SQLite sample data use random values without seeding; SQLite rows use the current date. Training has a seed, but reproducibility begins only after generated inputs are fixed.
9. **Schema/config duplication.** `schema` and `schema_simple` are both maintained in YAML, and the test DB is maintained separately in Python. They can drift.
10. **Deployment configuration.** Compose points at the root build context and nested Dockerfile correctly, but the image uses a CPU-only base with unpinned dependencies and no GPU-specific selection. The Vercel project is static only and needs an externally hosted API for inference.

## 8. Reproduction and troubleshooting

From repository root, the current documented flow is:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python data/prepare_dataset.py
python scripts/validate_setup.py
python -m src.training.train
python -m scripts.generate_predictions
python -m src.eval.eval --dataset data/processed/predictions.json --output data/processed/eval_results.json --dashboard-output api/static/evaluation.json
```

`python scripts/run_pipeline.py` runs the last four data/training/evaluation steps sequentially. The optional demo database can be recreated with `python data/setup_test_db.py`; it is not required for the Gretel evaluation path.

Model hub/network access, sufficient disk, compatible CUDA/PyTorch/bitsandbytes builds, and enough RAM/VRAM are practical requirements. Windows bitsandbytes compatibility is explicitly listed by the README as a known issue. If quantized loading fails or its availability check falls back, inspect the branch and memory requirements before attributing the issue to the data pipeline. If generation works in batch but not API on a CPU host, inspect the API's device handling. If Compose cannot locate its Dockerfile, set the Dockerfile path in the Compose build definition or run a build that points to `docker/Dockerfile` (proposed operational fix; not applied here).

## 9. Interview-ready explanation

### 30-second overview

“This is a prototype for translating personal-finance questions into SQL. It fine-tunes a small instruction-tuned causal language model with a LoRA adapter on the finance subset of the public Gretel Text-to-SQL dataset, using an optional 4-bit NF4 base-model load to reduce training memory. Every example carries its own schema, and a FastAPI endpoint returns generated SQL for a supplied or default schema. Separately, a batch pipeline evaluates predictions by normalized exact match and by executing both predicted and reference SQL against each sample's own SQLite context. The implementation is a useful end-to-end experiment, but the test split is small, some reference queries are not SQLite-executable, and the API does not safely execute generated SQL or establish production readiness.”

### Likely follow-up questions

**Why LoRA/QLoRA?** The implementation configures 4-bit NF4 loading and rank-16 LoRA on query/value projections to reduce memory and update only adapter parameters. This is a plausible small-hardware adaptation strategy; no controlled comparison in the repository proves it beats full fine-tuning or another rank/target-module choice.

**How do you avoid duplicate prompt leakage?** The split groups exactly identical input strings before selecting the test partition. It does not deduplicate semantically equivalent questions or template families, and a prompt group can push test size above its target.

**How do you measure correctness?** The evaluator has normalized string exact match and SQLite result-set equality. They answer different questions: textual similarity versus equal outputs on one database. The latter ignores order/duplicate count and is not a proof of semantic equivalence.

**How is the system deployed?** A FastAPI application and Docker artifacts exist. The API loads the Qwen base model and adapter on startup and offers health/schema/generation routes, schema validation, a demo fallback, and an optional Redis cache. The Compose build context is configured correctly, but the image is CPU-only and the code lacks production controls such as authentication, a live SQL execution sandbox, monitoring, and resource limits. The Vercel project serves the static dashboard and must be pointed at a hosted API through `?api=<url>` or the `finance-sql-api` meta tag.

**What would you improve first?** Make data generation and fixtures deterministic; audit every question/SQL pair; strengthen evaluation with multiple fixtures, ordered/multiset-aware comparisons where appropriate, and parse/safety checks; unify prompt construction; add device-aware serving; constrain generated SQL to authorized read-only operations; pin dependencies; fix and verify container build; then report held-out results with environment and sample counts.

## 10. Evidence gaps and verification status

- Test suite/results: `python -m unittest discover -s tests -v` runs 32 tests covering SQL guardrails, API generation, evaluation metrics, Redis cache behavior, and training formatting; all pass in this checkout.
- Accuracy, latency, throughput, cost, memory consumption, scaling, and production uptime: **Not verified from the repository.**
- Dataset source: `gretelai/synthetic_text_to_sql` loaded from the Hugging Face Hub; the dataset is synthetic, not real customer data.
- Deployment environment and live database integration: **Not verified from the repository.**
- API behavior on CPU-only hosts and a fresh Docker Compose build: identified as code/configuration concerns; not executed here.
- Current local adapter files and generated datasets may exist in a developer checkout but are ignored by Git and are not portable evidence of a reproducible release.
- No application code or existing files were modified for this report. Only this guide and its PDF deliverable are added.

## Appendix A: source file index

See the repository map in section 2. Core implementation references are `configs/training_config.yaml`, `data/gen_dataset.py`, `data/setup_test_db.py`, `src/training/{train.py,formatting.py,data_split.py,lora_config.py}`, `src/utils/model_loader.py`, `src/inference/gen_sql.py`, `api/main.py`, `scripts/generate_predictions.py`, and `src/eval/{eval.py,eval_accuracy.py}`.

