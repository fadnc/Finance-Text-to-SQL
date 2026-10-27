"""Mirror the live training log into the active MLflow run.

MLflow has no tqdm-style progress bar.  This monitor reads the training log,
extracts the current step/percentage and loss values, and logs them as MLflow
metrics on the run that is currently training.

Usage:
    python scripts/monitor_training.py
"""

import argparse
import re
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


PROGRESS_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\[")
LOSS_RE = re.compile(r"'loss': '([\d.]+)'")
EVAL_LOSS_RE = re.compile(r"'eval_loss': '([\d.]+)'")
EPOCH_RE = re.compile(r"'epoch': '([\d.]+)'")


def is_training_progress(line: str) -> bool:
    """Return True only for the Trainer's optimizer-step progress bar.

    The log also contains dataset tokenization and model-loading bars.  Those
    use units such as ``examples/s`` or are labelled ``Loading weights`` and
    must not be treated as training progress.
    """
    if "Loading weights" in line:
        return False
    if any(unit in line for unit in ("examples/s", "ba/s", "B/s", "tokens/s")):
        return False
    return "s/it" in line or "it/s" in line


def find_active_run(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'RUNNING'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No RUNNING MLflow run found")
    return runs[0].info.run_id


def log_metric(client: MlflowClient, run_id: str, key: str, value: float, step: int):
    try:
        client.log_metric(run_id, key, value, step=step)
    except Exception:
        # The training process may have ended the run between checks.
        raise SystemExit("MLflow run is no longer active")


def step_from_epoch(line: str, state: dict) -> int | None:
    """Map a logged epoch fraction (e.g. 0.04233) to an optimizer step."""
    epoch_match = EPOCH_RE.search(line)
    if not epoch_match or not state["total_steps"] or not state["epochs"]:
        return None
    epoch = float(epoch_match.group(1))
    step = round(epoch * state["total_steps"] / state["epochs"])
    return max(0, min(state["total_steps"], step))


def process_line(line: str, client: MlflowClient, run_id: str, state: dict):
    progress = PROGRESS_RE.search(line)
    if progress and is_training_progress(line):
        percentage, step, total = (int(value) for value in progress.groups())
        # Ignore tiny progress bars from model loading or one-off downloads.
        if total < 100:
            return
        # Once the real training bar is found, ignore bars with other totals.
        if state["total_steps"] and total != state["total_steps"]:
            return
        state["total_steps"] = total
        if step != state["last_step"]:
            state["last_step"] = step
            log_metric(client, run_id, "progress_percent", percentage, step)
            if step == 0 or step % 25 == 0 or step == total:
                print(f"step {step}/{total} ({percentage}%)", flush=True)

    loss = LOSS_RE.search(line)
    loss_step = step_from_epoch(line, state)
    if loss and loss_step is not None:
        log_metric(client, run_id, "train_loss", float(loss.group(1)), loss_step)

    eval_loss = EVAL_LOSS_RE.search(line)
    eval_step = step_from_epoch(line, state)
    if eval_loss and eval_step is not None:
        log_metric(client, run_id, "eval_loss", float(eval_loss.group(1)), eval_step)


def main():
    parser = argparse.ArgumentParser(description="Mirror training log to MLflow")
    parser.add_argument("--log", default="logs/train_gretel.log")
    parser.add_argument("--experiment", default="finance-text-to-sql")
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()

    log_path = Path(args.log)
    client = MlflowClient()
    run_id = find_active_run(client, args.experiment)
    run = client.get_run(run_id)
    epochs = float(run.data.params.get("epochs", 1))
    print(f"Monitoring {log_path} -> MLflow run {run_id} ({epochs:g} epochs)", flush=True)

    # PowerShell's Tee-Object writes UTF-16 on Windows; GitHub Actions/Linux
    # shells write UTF-8.  Detect the encoding from the BOM.
    with open(log_path, "rb") as handle:
        bom = handle.read(2)
    encoding = "utf-16" if bom in (b"\xff\xfe", b"\xfe\xff") else "utf-8"

    state = {"last_step": -1, "total_steps": 0, "epochs": epochs}
    buffer = ""
    position = 0
    finished = False

    while not finished:
        if not log_path.exists():
            time.sleep(args.interval)
            continue

        with open(log_path, "r", encoding=encoding, errors="ignore") as handle:
            handle.seek(position)
            chunk = handle.read()
            position = handle.tell()

        if chunk:
            buffer += chunk
            lines = re.split(r"[\r\n]+", buffer)
            buffer = lines.pop()
            for line in lines:
                if "Adapter saved to" in line:
                    finished = True
                process_line(line, client, run_id, state)

        if not finished:
            time.sleep(args.interval)

    print("Training finished; monitor stopped.", flush=True)


if __name__ == "__main__":
    main()
