from peft import LoraConfig, get_peft_model

def apply_lora(model, r: int = 16, alpha: int = 32, target_modules="all-linear"):
    """Attach QLoRA adapters in an architecture-agnostic way.

    ``target_modules="all-linear"`` lets PEFT discover the projection names for
    Qwen (q/k/v/o_proj + MLP), Phi (qkv_proj, gate_up_proj) and TinyLlama
    (q/v_proj) instead of hardcoding one model family. The output head is
    excluded by PEFT.
    """
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    return model
