"""Shared prompt construction and SQL extraction.

Training and inference must use exactly the same prompt layout; otherwise the
adapter sees a different distribution at serving time. The layout is rendered
through the tokenizer's chat template when one is available, so swapping base
models does not require editing this module. The legacy TinyLlama/Zephyr tags
remain as a fallback for tokenizers without a chat template.
"""

from __future__ import annotations

import re

INSTRUCTION = (
    "Write one read-only SQL SELECT query for the following question using only "
    "the provided schema. Return SQL only."
)

SYSTEM = "You are a precise text-to-SQL assistant."

# Layout used before chat templates were supported (TinyLlama/Zephyr style).
LEGACY_PROMPT = (
    "<|user|>\n{instruction}\n\n"
    "Question: {question}\n"
    "Schema: {schema}\n"
    "<|assistant|>"
)


def _message_body(question: str, schema: str) -> str:
    return (
        f"{INSTRUCTION}\n\n"
        f"Question: {question.strip()}\n"
        f"Schema: {schema.strip()}"
    )


def build_prompt(question: str, schema: str, tokenizer=None) -> str:
    """Build the prompt used for both training and inference.

    When a tokenizer with a chat template is supplied, the prompt is rendered
    with ``apply_chat_template`` (ChatML for Qwen) and ends exactly at the
    assistant generation prompt. Callers must not append extra whitespace, or
    training and serving prompts will diverge.
    """
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_template is not None:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": _message_body(question, schema)},
        ]
        try:
            return apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older transformers releases do not accept enable_thinking.
            return apply_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return LEGACY_PROMPT.format(
        instruction=INSTRUCTION,
        question=question.strip(),
        schema=schema.strip(),
    )


def extract_sql(text: str) -> str:
    """Extract the first SQL statement from a raw model response."""
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]

    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"^(?:SQL|sql)\s*:\s*", "", text)

    # Drop a natural-language preamble such as "Here is the query:".
    statement = re.search(
        r"\b(SELECT|WITH|INSERT|UPDATE|DELETE|REPLACE|CREATE)\b",
        text,
        flags=re.IGNORECASE,
    )
    if statement and statement.start() > 0:
        text = text[statement.start():]

    # Models sometimes append an explanation after the query.  Keep the first
    # complete statement so the evaluator receives a single executable query.
    semicolon = text.find(";")
    if semicolon != -1:
        text = text[: semicolon + 1]

    return text.strip()
