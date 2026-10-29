# Run the FastAPI server locally

This is the local inference host for the project. It loads the Qwen base model
and the LoRA adapter, serves the SQL Studio dashboard, and answers
`/generate-sql` requests from the local page or the deployed Vercel page.

- Model configuration: `configs/training_config.yaml` (`model_name`, `output_dir`)
- Server: `api/main.py`
- Launcher: `scripts/start_api.cmd`
- Adapter: `models/qwen2.5-coder-1.5b-finance-sql-lora`

## 1. Prerequisites

- Windows with Python 3.10+
- The repository virtual environment with dependencies installed
- The adapter directory present locally (it is git-ignored)
- An NVIDIA GPU with roughly 4 GB VRAM for the 4-bit load; CPU works but is slow
- Optional: Redis for the response cache

Set up the environment once:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Check that PyTorch, CUDA, the config, the dataset splits, and the demo database
are usable:

```powershell
.venv\Scripts\python.exe scripts\validate_setup.py
```

## 2. Start the server

From the repository root:

```powershell
scripts\start_api.cmd
```

The script activates `.venv`, binds `127.0.0.1:8000`, and prints the local and
Vercel URLs. To use another interface or port:

```powershell
scripts\start_api.cmd 0.0.0.0 8080
```

Equivalent manual command:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

On startup the server:

1. Reads `configs/training_config.yaml`.
2. Loads `Qwen/Qwen2.5-Coder-1.5B-Instruct` (downloaded to `.hf_cache/` on first use).
3. Applies the LoRA adapter from `models/qwen2.5-coder-1.5b-finance-sql-lora`.
4. Sets the pad token and puts the model in evaluation mode.

Loading can take 30-120 seconds the first time; wait for
`Application startup complete` in the console. The port is not reachable until
that finishes.

## 3. Open the app

| URL | Purpose |
|---|---|
| http://127.0.0.1:8000/ | SQL Studio dashboard |
| http://127.0.0.1:8000/docs | Interactive OpenAPI docs |
| http://127.0.0.1:8000/health | Health and `model_loaded` state |
| http://127.0.0.1:8000/model | Base model, adapter, dataset, token limit |
| http://127.0.0.1:8000/evaluation.json | Latest held-out evaluation summary |

Check it from PowerShell:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8000/model
```

Expected health shape:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "base_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
  "adapter": "models/qwen2.5-coder-1.5b-finance-sql-lora",
  "dataset": "gretelai/synthetic_text_to_sql"
}
```

Generate SQL directly:

```powershell
$body = @{ question = "What is the total spending by category?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/generate-sql `
  -ContentType "application/json" -Body $body
```

The first generation after startup is slower while CUDA warms up. Responses
include `valid`, `source`, `latency_ms`, `cache_hit`, and `model_version`.

## 4. Connect the Vercel dashboard

The Vercel project serves only the static dashboard, so the browser needs the
local API URL:

- open `https://static-ten-pi.vercel.app/?api=http://127.0.0.1:8000` once; the
  value is stored in `localStorage`, or
- set `<meta name="finance-sql-api" content="http://127.0.0.1:8000">` in
  `api/static/index.html` before deploying.

With neither set, the hosted page falls back to `http://127.0.0.1:8000`. When
Chrome asks for local network access, allow it. The API enables
`allow_private_network=True` so the HTTPS page can reach the loopback server.
For visitors other than you, host the API on a reachable HTTPS URL and use
`?api=https://...` instead.

## 5. Optional Redis cache

```powershell
docker compose -f docker/docker-compose.yml up -d redis
$env:REDIS_URL = "redis://localhost:6379/0"
$env:REDIS_CACHE_ENABLED = "true"
$env:REDIS_CACHE_TTL_SECONDS = "300"
scripts\start_api.cmd
```

Without `REDIS_URL` or with `REDIS_CACHE_ENABLED=false`, the API runs without a
cache. Send `X-Cache-Bypass: true` to skip cache reads and writes for one
request.

## 6. Stop the server

Press `Ctrl+C` in the terminal running `start_api.cmd`. If the port stays
bound, find and stop the process:

```powershell
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Only kill the PID you confirmed is the Uvicorn process.

## 7. Troubleshooting

| Symptom | Check |
|---|---|
| `Address already in use` | Another server owns port 8000; stop it or run `scripts\start_api.cmd 127.0.0.1 8001` |
| Browser shows `API unavailable` | Confirm `/health` returns 200 and the page's API URL has no typo or trailing slash |
| Vercel page cannot reach localhost | Allow the local-network prompt; confirm the API is running; use `?api=http://127.0.0.1:8000` |
| Startup fails on `PeftModel.from_pretrained` | The adapter directory is missing or incomplete; retrain or restore `models/qwen2.5-coder-1.5b-finance-sql-lora` |
| CUDA or bitsandbytes error | Install a CUDA-enabled PyTorch build; 4-bit loading needs CUDA compute capability 7.0+ |
| Very slow generation | A 1.5B model on a 4 GB laptop GPU is expected to take seconds per query; the first call also warms CUDA |
| `/generate-sql` returns a fallback | The model SQL failed schema validation; the response's `validation_error` explains why |

## Related files

- `api/main.py` - FastAPI app, middleware, endpoints
- `api/static/index.html` - dashboard and API URL resolution
- `configs/training_config.yaml` - model, dataset, and schema configuration
- `docs/evaluation-and-cache.md` - evaluation metrics and Redis behavior
