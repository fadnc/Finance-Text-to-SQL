"""
FastAPI server for Text-to-SQL inference.
"""

import os
import time
import hashlib
import asyncio
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))

import yaml
import torch
from fastapi import FastAPI, HTTPException, Header
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from peft import PeftModel

from src.inference.demo_sql import build_demo_sql
from src.inference.gen_sql import generate_sql
from src.inference.sql_guard import deduplicate_joins, prepare_sql, validate_sql
from src.utils.model_loader import load_model
from src.cache.sql_cache import SQLCache, canonical_key

app = FastAPI(
    title="Finance Text-to-SQL API",
    description="Natural language to SQL query generator fine-tuned on gretelai/synthetic_text_to_sql, Spider and WikiSQL",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the Finance SQL Studio dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/evaluation.json", include_in_schema=False)
def get_evaluation():
    """Return the latest checked-in model evaluation summary."""
    return FileResponse(STATIC_DIR / "evaluation.json", media_type="application/json")


with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

BASE_MODEL = config["model_name"]
ADAPTER_PATH = config["output_dir"]
SCHEMA = config["schema"]
MAX_NEW_TOKENS = config.get("generation_max_new_tokens", 256)

tokenizer = None
model = None
cache = SQLCache.from_env()
_model_lock = asyncio.Lock()
PROMPT_VERSION = "finance-text-to-sql-v3-chatml"
def _model_version(base_revision: str | None = None) -> str:
    digest = hashlib.sha256(f"{BASE_MODEL}:{base_revision or 'unresolved'}:{ADAPTER_PATH}".encode())
    for path in (Path(ADAPTER_PATH) / "adapter_config.json", Path(ADAPTER_PATH) / "adapter_model.safetensors"):
        if path.exists():
            with path.open("rb") as artifact:
                for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()[:16]


MODEL_VERSION = _model_version()


@app.on_event("startup")
async def load_model_on_startup():
    """Load the base model and adapter on startup."""
    global tokenizer, model, MODEL_VERSION

    model, tokenizer = load_model(BASE_MODEL)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    MODEL_VERSION = _model_version(getattr(model.get_base_model().config, "_commit_hash", None))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.eval()


@app.on_event("shutdown")
async def close_cache():
    await cache.close()


class Query(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "question": "What is the total spending by category?",
                "schema": "expenses(expense_id, user_id, category, amount, date)",
            }
        },
    )

    question: str
    schema_text: str | None = Field(default=None, alias="schema")


class SQLResponse(BaseModel):
    sql: str
    schema_used: str
    valid: bool = True
    source: str = "model"
    warning: str | None = None
    validation_error: str | None = None
    latency_ms: float | None = None
    cache_hit: bool = False
    model_version: str | None = None
    generation_latency_ms: float | None = None
    response_latency_ms: float | None = None
    cache_error: bool = False


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "dataset": config["dataset"]["name"],
    }


@app.get("/model")
def get_model_info():
    """Return the model and dataset backing this inference API."""
    return {
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "dataset": config["dataset"]["name"],
        "domains": config["dataset"].get("domains") or [],
        "split": config["dataset"].get("eval_split", "test"),
        "max_new_tokens": MAX_NEW_TOKENS,
    }


@app.get("/schema")
def get_schema():
    """Return the default database schema."""
    return {"schema": SCHEMA}


@app.get("/dataset")
def get_dataset_info():
    """Return the dataset sources used to train the adapter."""
    return {
        "name": config["dataset"]["name"],
        "domains": config["dataset"].get("domains") or [],
        "split": config["dataset"].get("eval_split", "test"),
        "supplementary": [
            name
            for name, settings in (config.get("training_datasets") or {}).items()
            if settings.get("include")
        ],
    }


@app.post("/generate-sql", response_model=SQLResponse)
async def generate(query: Query, x_cache_bypass: str | None = Header(default=None)):
    """Generate SQL from a natural language question."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    started_at = time.perf_counter()
    schema = query.schema_text or SCHEMA
    generation = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
    key = canonical_key(question=query.question, schema=schema,
                        model_version=MODEL_VERSION, prompt_version=PROMPT_VERSION,
                        generation=generation)
    bypass = (x_cache_bypass or "").lower() in {"1", "true", "yes"}
    cache_errors_before = cache.errors
    if not bypass:
        cached = await cache.get(key)
        if cached:
            elapsed = round((time.perf_counter() - started_at) * 1000, 1)
            return SQLResponse(**cached, cache_hit=True, model_version=MODEL_VERSION,
                               generation_latency_ms=None, response_latency_ms=elapsed,
                               latency_ms=elapsed, cache_error=cache.errors > cache_errors_before)

    async with cache.lock_for(key):
        if not bypass:
            cached = await cache.get(key)
            if cached:
                elapsed = round((time.perf_counter() - started_at) * 1000, 1)
                return SQLResponse(**cached, cache_hit=True, model_version=MODEL_VERSION,
                                   generation_latency_ms=None, response_latency_ms=elapsed,
                                   latency_ms=elapsed, cache_error=cache.errors > cache_errors_before)
        generation_started = time.perf_counter()
        try:
            async with _model_lock:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                raw_sql = await run_in_threadpool(
                    generate_sql, model, tokenizer, query.question, schema,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"SQL generation failed: {exc}") from exc
        generation_latency = round((time.perf_counter() - generation_started) * 1000, 1)

    sql, valid, validation_error, source = prepare_sql(raw_sql, schema)
    warning = None
    model_error = None if valid else validation_error

    if not valid:
        # Prefer a deterministic, schema-aware fallback for common demo
        # questions over returning SQL that is known to fail at execution.
        fallback = build_demo_sql(query.question, schema)
        if fallback:
            fallback_valid, _ = validate_sql(fallback, schema)
            if fallback_valid:
                sql = fallback
                valid = True
                source = "safe-fallback"
                warning = (
                    "The model query was not executable against this schema. "
                    "A schema-safe query was generated instead."
                )
            else:
                warning = (
                    "The generated SQL did not validate against the schema. "
                    "Review it before running."
                )
        else:
            repaired = deduplicate_joins(sql)
            repaired_valid, _ = validate_sql(repaired, schema)
            if repaired_valid and repaired.strip() != sql.strip():
                sql = repaired
                valid = True
                source = "model-normalized"
                warning = "Repeated joins were removed so the SQL executes."
            else:
                warning = (
                    "The generated SQL did not validate against the schema. "
                    "Review it before running."
                )

    response_latency = round((time.perf_counter() - started_at) * 1000, 1)
    response = SQLResponse(
        sql=sql,
        schema_used=schema,
        valid=valid,
        source=source,
        warning=warning,
        validation_error=model_error,
        latency_ms=response_latency,
        cache_hit=False,
        model_version=MODEL_VERSION,
        generation_latency_ms=generation_latency,
        response_latency_ms=response_latency,
        cache_error=cache.errors > cache_errors_before,
    )
    if response.valid and not bypass:
        cached_payload = response.model_dump(exclude={"cache_hit", "model_version", "generation_latency_ms", "response_latency_ms", "latency_ms", "cache_error"})
        await cache.set(key, cached_payload)
    response_latency = round((time.perf_counter() - started_at) * 1000, 1)
    response.response_latency_ms = response_latency
    response.latency_ms = response_latency
    response.cache_error = cache.errors > cache_errors_before
    return response


@app.post("/cache/invalidate", include_in_schema=False)
async def invalidate_cache(x_cache_admin_token: str | None = Header(default=None)):
    """Explicit cache invalidation; disabled unless an admin token is configured."""
    expected = os.getenv("CACHE_ADMIN_TOKEN")
    if not expected or x_cache_admin_token != expected:
        raise HTTPException(status_code=404, detail="Not found")
    deleted = await cache.invalidate()
    return {"invalidated": deleted}
