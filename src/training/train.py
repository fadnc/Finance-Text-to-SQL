import os
from pathlib import Path

# Keep model and dataset downloads inside the repository cache.
os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))

import mlflow
import torch
import yaml
from transformers import set_seed
from trl import SFTConfig, SFTTrainer

from src.data.multi_dataset import load_multi_dataset_splits
from src.training.formatting import format_dataset
from src.training.lora_config import apply_lora
from src.utils.model_loader import load_model


def main():
    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    print("Loading the prepared Gretel finance + Spider training mixture...")
    splits = load_multi_dataset_splits(config)
    print(
        f"Rows: train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, test={len(splits['test'])}"
    )

    train_dataset = format_dataset(splits["train"])
    validation_dataset = format_dataset(splits["validation"])
    eval_dataset = validation_dataset if len(validation_dataset) > 0 else None

    model, tokenizer = load_model(config["model_name"])
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    model = apply_lora(model)
    model.print_trainable_parameters()

    training_output_dir = config.get("training_output_dir", config["output_dir"])
    training_args = SFTConfig(
        output_dir=training_output_dir,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        logging_steps=config.get("logging_steps", 10),
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=1,
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=seed,
        dataset_text_field="text",
        max_length=config.get("max_seq_length", 512),
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    mlflow_active = False
    try:
        mlflow.set_experiment("finance-text-to-sql")
        mlflow.start_run()
        mlflow.log_params(
            {
                "dataset": config["dataset"]["name"],
                "domains": ",".join(config["dataset"].get("domains") or []),
                "model_name": config["model_name"],
                "train_rows": len(splits["train"]),
                "validation_rows": len(splits["validation"]),
                "test_rows": len(splits["test"]),
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "gradient_accumulation_steps": config.get(
                    "gradient_accumulation_steps", 1
                ),
                "learning_rate": config["learning_rate"],
                "max_seq_length": config.get("max_seq_length", 512),
            }
        )
        mlflow_active = True
    except Exception as exc:  # MLflow is optional for the training run.
        print(f"MLflow logging disabled: {exc}")

    try:
        trainer.train()
        metrics = trainer.evaluate() if eval_dataset is not None else {}
        if mlflow_active:
            mlflow.log_metrics(
                {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
            )
    finally:
        if mlflow_active:
            mlflow.end_run()

    trainer.save_model(training_output_dir)
    tokenizer.save_pretrained(training_output_dir)

    print(f"Adapter saved to {training_output_dir}")
    if metrics:
        print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
