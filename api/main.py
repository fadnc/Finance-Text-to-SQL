"""
FastAPI server for Text-to-SQL inference.
"""

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from peft import PeftModel

from src.inference.demo_sql import build_demo_sql
from src.inference.gen_sql import generate_sql
from src.inference.sql_guard import deduplicate_joins, prepare_sql, validate_sql
from src.utils.model_loader import load_model

app = FastAPI(
    title="Finance Text-to-SQL API",
    description="Natural language to SQL query generator fine-tuned on gretelai/synthetic_text_to_sql",
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


@app.on_event("startup")
async def load_model_on_startup():
    """Load the base model and adapter on startup."""
    global tokenizer, model

    model, tokenizer = load_model(BASE_MODEL)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model.eval()


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
    """Return the dataset source used to train the adapter."""
    return {
        "name": config["dataset"]["name"],
        "domains": config["dataset"].get("domains") or [],
        "split": config["dataset"].get("eval_split", "test"),
    }


@app.post("/generate-sql", response_model=SQLResponse)
def generate(query: Query):
    """Generate SQL from a natural language question."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    started_at = time.perf_counter()
    schema = query.schema_text or SCHEMA
    try:
        raw_sql = generate_sql(
            model,
            tokenizer,
            query.question,
            schema,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {exc}") from exc

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

    latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
    return SQLResponse(
        sql=sql,
        schema_used=schema,
        valid=valid,
        source=source,
        warning=warning,
        validation_error=model_error,
        latency_ms=latency_ms,
    )
