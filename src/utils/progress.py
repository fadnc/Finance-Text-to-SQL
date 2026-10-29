"""Small progress-bar helpers used across the pipeline.

Falls back to a no-op bar when ``tqdm`` is unavailable so scripts keep working
in minimal environments. Import ``tqdm.auto`` so notebooks and terminals both
render correctly.
"""

from __future__ import annotations


class _NoopBar:
    """Stand-in for tqdm with the subset of methods this project uses."""

    def __init__(self, total: int | None = None, **_: object) -> None:
        self.total = total
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix_str(self, _: str) -> None:
        return None

    def set_description_str(self, _: str) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def make_progress(total: int | None = None, desc: str = "", unit: str = "it",
                  unit_scale: bool = False):
    """Return a tqdm bar, or a no-op bar if tqdm is not installed."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _NoopBar(total)

    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        unit_scale=unit_scale,
        dynamic_ncols=True,
        leave=True,
    )


def progress_iter(iterable, total: int | None = None, desc: str = "", unit: str = "it"):
    """Wrap an iterable in a progress bar (no-op if tqdm is unavailable)."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable

    return tqdm(
        iterable,
        total=total if total is not None else len(iterable),
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )
