# Model evaluation and Redis cache

## Evaluation

The held-out dataset is `data/processed/gretel_finance_test.json`. It is excluded from the Gretel training and validation splits. The file contains question/reference-SQL pairs and per-example SQLite fixture contexts. The evaluator labels this data as the public Gretel synthetic finance set; it is not a fresh evaluation set generated in this run.

Run a full base versus adapter comparison from the repository root:

```powershell
.venv\Scripts\python.exe scripts/evaluate_models.py
```

Use `--limit 5` for a quick smoke run, `--max-new-tokens 128` to cap generation, and `--output reports/my-run` to choose a report directory. The evaluator excludes one warm-up generation per model, uses greedy decoding for both, synchronizes CUDA around timed generation, masks prompt and padding tokens from test loss, and reports overlength test-loss examples separately. Execution accuracy is calculated only when the reference SQL can run in the fixture. SQL output is run only in fresh in-memory SQLite fixtures, with a read-only authorizer, one-second default timeout, VM-step ceiling, SQL-length ceiling, and result-row ceiling. Schema validity uses `EXPLAIN QUERY PLAN` in a schema-only in-memory database.

Outputs are `summary.json`, `report.md`, and `predictions.jsonl`. Exact match ignores whitespace/comments and folds keyword case while retaining literal, identifier, operator, and token order. It does not prove SQL semantic equivalence. Rows are compared in order when either query uses a top-level `ORDER BY`; otherwise row multisets are compared, preserving duplicates. Numeric representations are normalized, NULLs compare to NULLs, and blobs remain distinct from text.

### Latest held-out results

The newest adapter is `Qwen/Qwen2.5-Coder-1.5B-Instruct` with a rank-16 QLoRA
adapter trained on 8,000 WikiSQL rows, 600 Spider rows, and the 945 finance
rows repeated three times (11,435 examples, 1 epoch, 2,781 steps, validation
loss 0.1321). All rows below were scored on the same 72 held-out Gretel finance
examples (48 with executable reference SQL):

| Model | Exact match | Valid SQL | Execution | Test loss | P50 latency |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder 1.5B base | 12.5% | 62.5% | 52.1% | 0.476 | 3,032 ms |
| **Qwen2.5-Coder + WikiSQL/Spider/Finance** | **22.2%** | 68.1% | 56.2% | **0.235** | 5,433 ms |
| TinyLlama + Gretel finance | 18.1% | 73.6% | 56.2% | 0.482 | 9,165 ms |
| TinyLlama + multi-dataset (candidate) | 15.3% | 66.7% | 58.3% | 0.449 | 3,377 ms |

The Qwen adapter has the best exact match and loss but not the best valid-SQL
rate or execution accuracy. Execution is within one example of the best
TinyLlama adapter on this denominator. Full artifacts, including the remaining
TinyLlama runs, are in `reports/model_comparison/`; the base-versus-adapter
report for Qwen is in `reports/evaluation-qwen/`. The dashboard's bundled
`api/static/evaluation.json` is produced by the separate `src.eval.eval` path,
whose looser exact-match normalization reports 25.0% (18/72); use the table
above for cross-model comparisons.

## Redis

Install the project dependencies, including the optional cache client:

```powershell
.venv\Scripts\pip.exe install -r requirements.txt
```

Start Redis for local API development:

```powershell
docker compose -f docker/docker-compose.yml up -d redis
$env:REDIS_URL = "redis://localhost:6379/0"
$env:REDIS_CACHE_ENABLED = "true"
$env:REDIS_CACHE_TTL_SECONDS = "300"
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

`REDIS_URL` unset or `REDIS_CACHE_ENABLED=false` bypasses Redis. Redis connection/read timeouts are one second and Redis failures fall back to inference. The cache stores validated SQL responses only. Keys contain SHA-256 digests over normalized questions, schema content, model artifact version, prompt version, generation settings, and scope; raw questions and credentials are not logged. This app currently has no tenant/auth system, so scope is `public`; add an authorization identity to the scope before introducing user-specific schemas or permissions.

TTL defaults to 300 seconds. `REDIS_MAX_CONNECTIONS` defaults to 20. To enable the protected invalidation endpoint, set a strong `CACHE_ADMIN_TOKEN`; send it as `X-Cache-Admin-Token` to `POST /cache/invalidate`. The endpoint is hidden and returns 404 unless configured. Schema or model edits also naturally produce distinct keys. In-process locks suppress same-key concurrent inference within one API worker; they do not coordinate across multiple worker processes.

Responses retain existing fields, including `latency_ms`, and add `cache_hit`, `model_version`, `generation_latency_ms`, `response_latency_ms`, and `cache_error`. On cache hits, generation latency is `null`; response latency is reported separately. `X-Cache-Bypass: true` bypasses cache reads and writes for benchmarking.

## Cache benchmark

Start the API and Redis, then run:

```powershell
.venv\Scripts\python.exe scripts/benchmark_cache.py --requests 30 --unique-ratio 0.3 --concurrency 4
```

For a custom JSON array of question strings or objects with a `question` field, pass `--workload workload.json`. The script sends the same repeated/unique workload first with cache bypass, then with caching enabled, and saves actual observations to `reports/cache-benchmark.json`. It reports hit/miss rates, end-to-end and hit/miss latency percentiles, miss generation latency, avoided inference calls, throughput, cache errors, and HTTP errors. Benchmark measurements depend on the running model, host, Redis, and workload; do not treat sample or illustrative values as results.

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Cache unit tests use an in-memory fake Redis client, so they do not require a running Redis service. The live cache benchmark does.
