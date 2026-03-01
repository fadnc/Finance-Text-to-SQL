import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sql(model, tokenizer, question, schema):
    prompt = f"""<|user|>
Write a SQL query for the following question

Question: {question}
Schema: {schema}
<|assistant|>"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Extract only the generated SQL after the prompt
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    if "<|assistant|>" in generated:
        generated = generated.split("<|assistant|>")[-1].strip()
    
    return generated