import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_name: str, use_4bit: bool = True):
    """
    Load model with optional 4-bit quantization.
    Falls back to FP16/FP32 if quantization is unavailable.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if CUDA and bitsandbytes are available for 4-bit
    cuda_available = torch.cuda.is_available()
    bnb_available = False
    
    if use_4bit and cuda_available:
        try:
            import bitsandbytes
            # Check if it has GPU support
            bnb_available = hasattr(bitsandbytes, 'functional') 
        except ImportError:
            pass
    
    if use_4bit and cuda_available and bnb_available:
        print("Loading model with 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
    elif cuda_available:
        print("Loading model with FP16 (CUDA available, no 4-bit)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        print("Loading model on CPU (no CUDA)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

    return model, tokenizer