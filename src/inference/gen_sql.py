import torch

from src.utils.prompting import build_prompt, extract_sql


def generate_sql(model, tokenizer, question, schema, max_new_tokens=256):
    """Generate SQL from a question and schema using the trained adapter."""
    device = next(model.parameters()).device
    prompt = build_prompt(question, schema)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[-1]:]
    return extract_sql(tokenizer.decode(new_tokens, skip_special_tokens=True))
