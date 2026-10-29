"""Live training dashboard for the Qwen2.5-Coder + WikiSQL fine-tune.

Run this in its own terminal while ``src.training.train`` writes to a log file.
It tails the log, parses the tqdm progress bars / validation lines, and redraws
a single-screen progress dashboard every second.

Usage (from the repository root, in a *new* terminal):
    .venv\\Scripts\\python.exe scripts/watch_training.py

Defaults to ``logs/train_qwen_wikisql.log`` (the file written by
``scripts/train_qwen_wikisql.cmd``).  Press Ctrl+C to stop watching; training
itself keeps running.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime
from pathlib import Path

BAR_RE = re.compile(
    r"^(?P<desc>[A-Za-z][^:|]*?):\s*\d+%\|[^|]*\|\s*(?P<step>\d+)/(?P<total>\d+)"
    r"\s*\[(?P<elapsed>[^<\]]*)(?:<(?P<remain>[^,\]]*))?[^\]]*\]"
    r"(?P<post>.*)$"
)
LOSS_RE = re.compile(r"\bloss=([\d.]+)")
LR_RE = re.compile(r"\blr=([\d.eE+-]+)")
EPOCH_RE = re.compile(r"\bepoch=([\d.]+)")
ROWS_RE = re.compile(r"Rows:\s*train=(\d+),\s*validation=(\d+),\s*test=(\d+)")
PREPARED_RE = re.compile(r"Prepared (\d+) read-only SFT examples")
TRAINABLE_RE = re.compile(r"trainable params: ([\d,]+)")
EVAL_RE = re.compile(r"Validation @ step (\d+): loss=([\d.]+)")
SAVED_RE = re.compile(r"Adapter saved to (.+)")
EXIT_RE = re.compile(r"EXITCODE=(-?\d+)")


def _decode(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


class LogState:
    """Incrementally parse a growing log file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.buffer = ""
        self.phase = "starting"
        self.detail = "waiting for training log"
        self.rows: tuple[int, int, int] | None = None
        self.prepared: int | None = None
        self.trainable: str | None = None
        self.step = 0
        self.total = 0
        self.post = ""
        self.elapsed = ""
        self.remain = ""
        self.evals: list[tuple[int, float]] = []
        self.saved: str | None = None
        self.exit_code: int | None = None
        self.started = datetime.now()

    def _handle(self, line: str) -> None:
        line = line.replace("\x1b[2K", "").strip()
        if not line:
            return

        rows = ROWS_RE.search(line)
        if rows:
            self.rows = tuple(int(v) for v in rows.groups())  # type: ignore[assignment]

        if "Loading model with 4-bit quantization" in line:
            self.phase = "loading"
            self.detail = "loading 4-bit base model"

        prepared = PREPARED_RE.search(line)
        if prepared:
            self.prepared = int(prepared.group(1))
            self.detail = f"{self.prepared:,} SFT examples prepared"

        trainable = TRAINABLE_RE.search(line)
        if trainable:
            self.trainable = trainable.group(1)

        bar = BAR_RE.search(line)
        if bar:
            desc = bar.group("desc").strip().lower()
            if desc.startswith("training"):
                self.phase = "training"
                self.step = int(bar.group("step"))
                self.total = int(bar.group("total"))
                self.elapsed = bar.group("elapsed").strip()
                self.remain = (bar.group("remain") or "").strip()
                self.post = line
                loss = LOSS_RE.search(self.post)
                if loss:
                    self.detail = f"loss={loss.group(1)}"
            elif desc.startswith("formatting"):
                self.phase = "formatting"
                self.step = int(bar.group("step"))
                self.total = int(bar.group("total"))
                self.elapsed = bar.group("elapsed").strip()
                self.remain = (bar.group("remain") or "").strip()
                self.post = bar.group("post").strip(" ,")
                self.detail = "tokenizing + pruning SFT examples"

        evaluation = EVAL_RE.search(line)
        if evaluation:
            self.evals.append((int(evaluation.group(1)), float(evaluation.group(2))))
            self.phase = "evaluating"
            self.detail = f"validation loss {evaluation.group(2)}"

        saved = SAVED_RE.search(line)
        if saved:
            self.saved = saved.group(1).strip()
            self.phase = "done"
            self.detail = "adapter saved"

        code = EXIT_RE.search(line)
        if code:
            self.exit_code = int(code.group(1))

    def read(self) -> None:
        try:
            size = self.path.stat().st_size
        except OSError:
            return
        if size < self.offset:  # log was truncated/rotated
            self.offset = 0
            self.buffer = ""
        if size == self.offset:
            return
        with open(self.path, "rb") as handle:
            handle.seek(self.offset)
            raw = handle.read()
            self.offset = handle.tell()
        self.buffer += _decode(raw)
        parts = re.split(r"[\r\n]+", self.buffer)
        self.buffer = parts.pop()
        for part in parts:
            self._handle(part)

    def metrics(self) -> dict[str, str]:
        values: dict[str, str] = {}
        loss = LOSS_RE.search(self.post)
        if loss:
            values["loss"] = loss.group(1)
        lr = LR_RE.search(self.post)
        if lr:
            values["lr"] = lr.group(1)
        epoch = EPOCH_RE.search(self.post)
        if epoch:
            values["epoch"] = epoch.group(1)
        return values


def render(state: LogState, width: int = 22) -> str:
    pct = (state.step / state.total * 100.0) if state.total else 0.0
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    bar = "#" * filled + "." * (width - filled)

    labels = {
        "starting": "STARTING",
        "formatting": "DATA + FORMAT",
        "loading": "MODEL LOAD",
        "training": "TRAINING",
        "evaluating": "VALIDATING",
        "done": "DONE",
    }
    label = labels.get(state.phase, state.phase.upper())

    rows = ""
    if state.rows:
        rows = (
            f"  Data       : {state.rows[0]:,} train / {state.rows[1]:,} validation"
            f" / {state.rows[2]:,} test rows\n"
        )

    step_line = (
        f"{state.step:,} / {state.total:,} steps" if state.total else "awaiting first step"
    )
    timing = ""
    if state.elapsed:
        timing = f"   elapsed {state.elapsed}"
        if state.remain:
            timing += f"   eta {state.remain}"

    metrics = state.metrics()
    metric_line = "   ".join(f"{key}={value}" for key, value in metrics.items())
    if state.trainable:
        metric_line = f"trainable={state.trainable}   {metric_line}".strip()

    evals = ""
    if state.evals:
        evals = "  " + "  ".join(f"eval@{step}={loss:.4f}" for step, loss in state.evals[-3:]) + "\n"

    footer = ""
    if state.phase == "done":
        footer = f"\n  Saved to {state.saved}\n"
    if state.exit_code is not None:
        footer += f"  Process exit code: {state.exit_code}\n"

    rule = "=" * 68
    lines = [
        rule,
        "  Qwen2.5-Coder-1.5B + WikiSQL  |  QLoRA training",
        rule,
        f"  Phase      : {label}",
        f"  Status     : {state.detail}",
    ]
    if rows:
        lines.append(rows.rstrip("\n"))
    lines += [
        "",
        f"  [{bar}] {pct:5.1f}%",
        f"  {step_line}{timing}",
        f"  {metric_line}",
    ]
    if evals:
        lines.append(evals.rstrip("\n"))
    lines += [rule, f"  log: {state.path}"]
    if footer:
        lines.append(footer.rstrip("\n"))
    lines.append("  Ctrl+C stops watching (training keeps running).")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Qwen+WikiSQL training dashboard")
    parser.add_argument("--log", default="logs/train_qwen_wikisql.log", help="training log file")
    parser.add_argument("--interval", type=float, default=1.0, help="refresh seconds")
    parser.add_argument("--once", action="store_true", help="render one frame and exit")
    args = parser.parse_args()

    state = LogState(Path(args.log))
    sys.stdout.reconfigure(encoding="utf-8")

    if args.once:
        state.read()
        print(render(state))
        return

    sys.stdout.write("\x1b[2J\x1b[H")
    try:
        while True:
            state.read()
            frame = render(state)
            sys.stdout.write("\x1b[H\x1b[2J" + frame)
            sys.stdout.flush()
            if state.phase == "done" or state.exit_code is not None:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
