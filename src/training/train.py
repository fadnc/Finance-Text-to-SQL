import os
import argparse
from pathlib import Path

# Keep model and dataset downloads inside the repository cache.
os.environ.setdefault("HF_HOME", str(Path(".hf_cache").resolve()))

import mlflow
import torch
import yaml
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import TrainerCallback, set_seed
from trl import SFTConfig, SFTTrainer

from src.data.multi_dataset import load_multi_dataset_splits
from src.training.formatting import format_dataset
from src.training.lora_config import apply_lora
from src.utils.model_loader import load_model
from src.utils.progress import make_progress


class TrainingProgress(TrainerCallback):
    """Single tqdm bar over optimizer steps with live loss/LR postfix.

    Hugging Face's built-in bar is disabled so this is the only progress bar;
    it also keeps working when ``logging_steps`` is large.
    """

    def __init__(self):
        self.bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.bar = make_progress(total=state.max_steps, desc="Training", unit="step")
        if state.global_step:
            self.bar.update(state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.update(1)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.bar is None or not logs:
            return
        parts = []
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.2f}")
        if parts:
            self.bar.set_postfix_str(" ".join(parts))
        # Mirror optimizer-step stats into the active MLflow run.
        if "loss" in logs and mlflow.active_run() is not None:
            mlflow.log_metrics(
                {
                    "loss": float(logs["loss"]),
                    "learning_rate": float(logs.get("learning_rate") or 0.0),
                },
                step=int(state.global_step),
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            print(
                f"\nValidation @ step {state.global_step}: "
                f"loss={metrics['eval_loss']:.4f}"
            )
        if mlflow.active_run() is not None and metrics:
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
                step=int(state.global_step),
            )

    def on_train_end(self, args, state, control, **kwargs):
        if self.bar is not None:
            self.bar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Resume training state from an existing Trainer checkpoint.",
    )
    parser.add_argument(
        "--init-adapter",
        help="Continue fine-tuning from an adapter without restoring optimizer state.",
    )
    parser.add_argument("--output-dir", help="Override the configured candidate output path.")
    parser.add_argument("--epochs", type=float, help="Override the configured epoch count.")
    parser.add_argument("--learning-rate", type=float, help="Override the configured learning rate.")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop after N optimizer steps (use for a VRAM smoke test).",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        help="Format/train on only the first N training rows (fast smoke tests).",
    )
    parser.add_argument(
        "--limit-validation",
        type=int,
        help="Format/evaluate on only the first N validation rows.",
    )
    args = parser.parse_args()

    with open("configs/training_config.yaml") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    set_seed(seed)

    print("Loading the configured finance and optional Spider training data...")
    splits = load_multi_dataset_splits(config)
    print(
        f"Rows: train={len(splits['train'])}, "
        f"validation={len(splits['validation'])}, test={len(splits['test'])}"
    )

    if args.limit_train:
        splits["train"] = splits["train"].select(
            range(min(args.limit_train, len(splits["train"])))
        )
    if args.limit_validation:
        splits["validation"] = splits["validation"].select(
            range(min(args.limit_validation, len(splits["validation"])))
        )
    print(
        f"Formatting {len(splits['train'])} train / {len(splits['validation'])} "
        "validation examples on CPU; the GPU stays idle until this finishes."
    )

    model, tokenizer = load_model(config["model_name"])
    train_dataset = format_dataset(
        splits["train"], tokenizer, int(config.get("max_seq_length", 512))
    )
    validation_dataset = format_dataset(
        splits["validation"], tokenizer, int(config.get("max_seq_length", 512))
    )
    eval_dataset = validation_dataset if len(validation_dataset) > 0 else None
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # Cast quantized layer norms to fp32 and enable input gradients before LoRA
    # attachment; required for stable 4-bit QLoRA on Qwen/Phi checkpoints.
    if getattr(model, "is_loaded_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=False
        )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False

    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = apply_lora(model)
    model.print_trainable_parameters()

    training_output_dir = args.output_dir or config.get(
        "training_output_dir", config["output_dir"]
    )
    training_args = SFTConfig(
        output_dir=training_output_dir,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        num_train_epochs=args.epochs or config["epochs"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=args.learning_rate or config["learning_rate"],
        logging_steps=config.get("logging_steps", 10),
        fp16=torch.cuda.is_available() and not use_bf16,
        bf16=use_bf16,
        save_strategy="steps",
        save_steps=200,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        greater_is_better=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=seed,
        dataset_text_field="text",
        max_length=config.get("max_seq_length", 512),
        completion_only_loss=True,
        packing=False,
        disable_tqdm=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[TrainingProgress()],
    )

    # Track training stats in the local MLflow file store.
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", f"file://{Path('mlruns').resolve()}")
    )
    mlflow_active = False
    try:
        mlflow.set_experiment("finance-text-to-sql")
        mlflow.start_run(run_name=f"{Path(config['model_name']).name}-wikisql")
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
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
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
