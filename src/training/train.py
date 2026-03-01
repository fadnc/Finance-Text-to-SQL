import yaml
import mlflow
import torch
from datasets import load_dataset
from transformers import TrainingArguments, set_seed
from trl import SFTTrainer

from src.utils.model_loader import load_model
from src.training.lora_config import apply_lora
from src.training.formatting import format_dataset


def main():

    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    seed = config.get("seed", 42)
    set_seed(seed)

    mlflow.set_experiment("finance-text-to-sql")

    dataset = load_dataset("json", data_files=config["dataset_path"])
    test_size = config.get("test_split", 0.1)
    dataset = dataset["train"].train_test_split(test_size=test_size, seed=seed)
    dataset = format_dataset(dataset)

    model, tokenizer = load_model(config["model_name"])
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    model = apply_lora(model)

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        logging_steps=10,
        fp16=True,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        # Memory-efficient training options
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
    )

    trainer.train()
    trainer.save_model(config["output_dir"])


if __name__ == "__main__":
    main()