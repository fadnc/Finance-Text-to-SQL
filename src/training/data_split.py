"""Deterministic train/test splitting that keeps duplicate prompts together."""

import math
import random
from collections import defaultdict


def grouped_split_indices(inputs: list, test_size: float, seed: int) -> tuple[list, list]:
    """Return train and test row indices without splitting identical prompts."""
    groups = defaultdict(list)
    for index, prompt in enumerate(inputs):
        groups[prompt].append(index)

    prompt_groups = sorted(groups)
    random.Random(seed).shuffle(prompt_groups)
    target_test_rows = max(1, math.ceil(len(inputs) * test_size))

    test_prompts = set()
    test_rows = 0
    for prompt in prompt_groups:
        if test_rows >= target_test_rows:
            break
        test_prompts.add(prompt)
        test_rows += len(groups[prompt])

    train_indices = []
    test_indices = []
    for prompt, indices in groups.items():
        (test_indices if prompt in test_prompts else train_indices).extend(indices)

    return sorted(train_indices), sorted(test_indices)
