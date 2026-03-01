"""
FastAPI server for Text-to-SQL inference.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = FastAPI(
    title="Finance Text-to-SQL API",
    description="Natural language to SQL query generator for personal finance domain",
    version="1.0.0"
)

# Load configuration
with open("configs/training_config.yaml") as f:
    config = yaml.safe_load(f)

BASE_MODEL = config["model_name"]
ADAPTER_PATH = config["output_dir"]
SCHEMA = config["schema_simple"]

tokenizer = None
model = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global tokenizer, model
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()


class Query(BaseModel):
    question: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "How much did Fadhil spend on Food last month?"
            }
        }


class SQLResponse(BaseModel):
    sql: str
    schema_used: str


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/schema")
def get_schema():
    """Return the database schema."""
    return {"schema": SCHEMA}


@app.post("/generate-sql", response_model=SQLResponse)
def generate(query: Query):
    """Generate SQL from natural language question."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = f"""<|user|>
Write a SQL query for the following question

Question: {query.question}
Schema: {SCHEMA}
<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

    result = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract SQL from response
    if "<|assistant|>" in result:
        result = result.split("<|assistant|>")[-1].strip()

    return SQLResponse(sql=result, schema_used=SCHEMA)