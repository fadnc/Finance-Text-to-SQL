"""Shared prompt construction and SQL extraction.

Training and inference must use exactly the same prompt layout; otherwise the
adapter sees a different distribution at serving time.
"""

from __future__ import annotations

import re

INSTRUCTION = "Write a SQL query for the following question"


def build_prompt(question: str, schema: str) -> str:
    """Build the user/assistant prompt used for both training and inference."""
    return (
        f"<|user|>\n{INSTRUCTION}\n\n"
        f"Question: {question.strip()}\n"
        f"Schema: {schema.strip()}\n"
        f"<|assistant|>"
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
