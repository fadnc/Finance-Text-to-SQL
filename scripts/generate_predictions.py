"""
Generate SQL predictions on test samples for evaluation.
"""

import json
import yaml
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config():
    with open("configs/training_config.yaml") as f:
        return yaml.safe_load(f)


def generate_sql(model, tokenizer, question, schema, device):
    """Generate SQL from natural language question."""
    prompt = f"""<|user|>
Write a SQL query for the following question

Question: {question}
Schema: {schema}
<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract SQL after assistant tag
    if "<|assistant|>" in result:
        result = result.split("<|assistant|>")[-1].strip()
    
    return result


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load test data
    with open(config["dataset_path"]) as f:
        data = json.load(f)
    
    # Use 20% for testing
    test_size = int(len(data) * 0.2)
    test_data = data[:test_size]
    
    print(f"Generating predictions for {len(test_data)} samples...")
    
    # Load model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load fine-tuned adapter
    adapter_path = config["output_dir"]
    print(f"Loading adapter from {adapter_path}...")
    
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        print(" Fine-tuned adapter loaded")
    except Exception as e:
        print(f" Could not load adapter: {e}")
        print("Using base model for inference...")
    
    model.eval()
    
    # Generate predictions
    results = []
    schema = config["schema_simple"]
    
    for sample in tqdm(test_data, desc="Generating SQL"):
        # Extract question from input
        input_text = sample["input"]
        question = input_text.split("Question: ")[1].split("\nSchema:")[0].strip()
        
        predicted = generate_sql(model, tokenizer, question, schema, device)
        
        results.append({
            "instruction": sample["instruction"],
            "input": sample["input"],
            "output": sample["output"],  # ground truth
            "predicted": predicted
        })
    
    # Save predictions
    output_path = "data/processed/predictions.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Predictions saved to {output_path}")
    print(f"   Total samples: {len(results)}")
    
    # Quick accuracy check
    exact_matches = sum(1 for r in results if r["predicted"].strip() == r["output"].strip())
    print(f"   Quick exact match: {exact_matches}/{len(results)} ({exact_matches/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
