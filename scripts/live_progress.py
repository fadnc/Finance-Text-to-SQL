"""Live ASCII progress bar for the Qwen2.5-Coder fine-tune training run.

Watches the active training log and re-renders a live progress bar to the
terminal. Phases: data/format -> model load -> LoRA training -> save/eval.

Usage (from the repository root, in any terminal):
    .venv\\Scripts\\python.exe scripts/live_progress.py --log logs/train_qwen-full.err.log

Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

TRAINING_RE = re.compile(r"Training:\s+\|\s*(\d+)%\s*\|\s*(\d+)/(\d+)")
PREPARED_RE = re.compile(r"Prepared (\d+) read-only SFT examples")
TRAINABLE_RE = re.compile(r"trainable params: ([\d,]+)")
SAVED_RE = re.compile(r"Adapter saved to")


def read_lines(path: Path, max_lines: int = 500) -> list[str]:
    try:
        with open(path, "rb") as handle:
            bom = handle.read(2)
        encoding = "utf-16" if bom in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
        with open(path, "r", encoding=encoding, errors="ignore") as handle:
            return handle.read().splitlines()[-max_lines:]
    except OSError:
        return []


def detect(lines: list[str]) -> dict:
    text = "\n".join(lines)
    state = {"phase": "starting", "step": 0, "total": 0, "pct": 0.0, "detail": ""}

    if SAVED_RE.search(text):
        state["phase"] = "done"
        state["detail"] = "Adapter saved to output dir"
        return state

    if TRAINABLE_RE.search(text):
        state["phase"] = "training"

    prepared = PREPARED_RE.search(text)
    if prepared:
        state["phase"] = "training"
        state["detail"] = f"Prepared {prepared.group(1)} SFT examples"

    if "Loading model with 4-bit quantization" in text:
        state["phase"] = "loading"
        state["detail"] = "Loading 4-bit model (warm cache)"

    if not (prepared or "Loading model with" in text):
        state["phase"] = "data"
        state["detail"] = "Loading data + formatting"

    match = TRAINING_RE.search(text)
    if match:
        state["phase"] = "training"
        state["step"] = int(match.group(2))
        state["total"] = int(match.group(3))
        state["pct"] = float(match.group(1))

    return state


def render(state: dict) -> str:
    pct = min(100.0, max(0.0, state["pct"]))
    width = 40
    filled = int(round(width * pct / 100))
    bar = "#" * filled + "." * (width - filled)

    phase_symbols = {
        "data": "📦",
        "loading": "⏳",
        "training": "🏋️",
        "done": "✅",
        "starting": "⏳",
    }
    phase_label = {
        "data": "DATA + FORMAT",
        "loading": "MODEL LOAD",
        "training": "TRAINING",
        "done": "DONE",
        "starting": "STARTING",
    }[state["phase"]]

    step_line = (
        f"step {state['step']}/{state['total']} ({pct:.0f}%)"
        if state["total"]
        else "awaiting first step"
    )
    return (
        f"\n{'=' * 70}\n"
        f"  Qwen2.5-Coder LoRA Training — Live Progress\n"
        f"{'=' * 70}\n"
        f"\n"
        f"  Phase: {phase_symbols[state['phase']]} {phase_label:<12} {state['detail']}\n"
        f"\n"
        f"  [{bar}]  {pct:6.2f}%\n"
        f"\n"
        f"  {step_line}\n"
        f"{'=' * 70}\n"
    )


def _key(state: dict) -> tuple:
    return (state["phase"], state["step"], state["total"], round(state["pct"], 2), state["detail"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Live training progress bar")
    parser.add_argument("--log", default="logs/train-qwen-full.out.log")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    path = Path(args.log)
    state = {"phase": "starting", "step": 0, "total": 0, "pct": 0.0, "detail": "waiting for log"}
    last_render = 0.0

    sys.stdout.reconfigure(encoding="utf-8")
    print("Live training progress — press Ctrl+C to stop.\n")

    while True:
        lines = read_lines(path)
        new_state = detect(lines)
        now = time.monotonic()
        if _key(new_state) != _key(state) or now - last_render > args.interval:
            state = new_state
            last_render = now
            # Clear last render (ANSI cursor up + clear line) and print new one.
            sys.stdout.write("\x1b[2F\x1b[2K\r" + render(state).rstrip("\n") + "\n")
            sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
