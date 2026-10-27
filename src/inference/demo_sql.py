"""Small schema-aware fallback for common finance demo questions.

The fallback is deliberately conservative: it only handles questions whose
intent maps cleanly to the bundled personal-finance schema.  It is used by the
API when the model's SQL cannot be validated, so a demo request never returns a
query that is guaranteed to fail at execution time.
"""

from __future__ import annotations

import re
from typing import Optional

from src.inference.sql_guard import column_names, table_names

CATEGORY_SYNONYMS = {
    "groceries": "Food",
    "grocery": "Food",
    "restaurant": "Food",
    "dining": "Food",
    "food": "Food",
    "flights": "Travel",
    "flight": "Travel",
    "hotel": "Travel",
    "travel": "Travel",
    "electricity": "Utilities",
    "electric": "Utilities",
    "internet": "Utilities",
    "utilities": "Utilities",
    "utility": "Utilities",
    "movies": "Entertainment",
    "movie": "Entertainment",
    "streaming": "Entertainment",
    "entertainment": "Entertainment",
    "gas": "Transportation",
    "fuel": "Transportation",
    "transport": "Transportation",
    "transportation": "Transportation",
    "uber": "Transportation",
    "housing": "Rent",
    "rent": "Rent",
    "doctor": "Healthcare",
    "medical": "Healthcare",
    "health": "Healthcare",
    "healthcare": "Healthcare",
    "clothes": "Shopping",
    "shopping": "Shopping",
}

TIME_FILTERS = (
    ("today", "date = date('now')"),
    ("yesterday", "date = date('now', '-1 day')"),
    ("this week", "date >= date('now', '-7 days')"),
    ("last week", "date >= date('now', '-14 days') AND date < date('now', '-7 days')"),
    ("this month", "date >= date('now', 'start of month')"),
    (
        "last month",
        "date >= date('now', 'start of month', '-1 month') "
        "AND date < date('now', 'start of month')",
    ),
    ("this year", "date >= date('now', 'start of year')"),
    (
        "last year",
        "date >= date('now', 'start of year', '-1 year') "
        "AND date < date('now', 'start of year')",
    ),
)


def _has_columns(schema: str, table: str, required: set[str]) -> bool:
    return required.issubset(column_names(schema, table))


def _time_filter(question: str, column: str = "date") -> Optional[str]:
    # Handle relative periods such as "last 3 weeks" / "this 3 weeks" before
    # the fixed phrases below. SQLite has no "weeks" modifier, so convert weeks
    # to days.
    relative = re.search(
        r"\b(?:last|past|previous|this)?\s*(\d{1,3})\s+(day|week|month|year)s?\b",
        question,
    )
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        if unit == "week":
            amount *= 7
            unit = "day"
        suffix = "" if amount == 1 else "s"
        return f"{column} >= date('now', '-{amount} {unit}{suffix}')"

    for phrase, condition in TIME_FILTERS:
        if re.search(rf"\b{re.escape(phrase)}\b", question):
            return condition.replace("date", column, 1)
    return None


def _category_filter(question: str) -> Optional[str]:
    for word in sorted(CATEGORY_SYNONYMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", question):
            return CATEGORY_SYNONYMS[word]
    return None


def _limit(question: str, default: int = 5) -> int:
    patterns = (
        r"\btop\s+(\d{1,3})\b",
        r"\b(\d{1,3})\s+(?:largest|biggest|items|results|transactions|expenses)\b",
        r"\b(\d{1,3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return max(1, min(int(match.group(1)), 100))
    return default


def _where(conditions: list[str]) -> str:
    return f" WHERE {' AND '.join(conditions)}" if conditions else ""


def build_demo_sql(question: str, schema: str) -> Optional[str]:
    """Build a validated SQLite query for a common demo intent, or return None."""
    q = " ".join(question.lower().split())
    tables = table_names(schema)
    if not tables:
        return None

    # "What did I buy ... by category" belongs to expenses because the
    # transactions table has no category column in the demo schema.
    wants_expense_history = (
        "expenses" in tables
        and "category" in q
        and any(
            word in q
            for word in ("buy", "bought", "purchase", "purchased", "shop", "shopping")
        )
    )

    if not wants_expense_history and any(
        word in q
        for word in ("transaction", "purchase", "purchased", "buy", "bought", "merchant")
    ):
        if "transactions" not in tables or not _has_columns(
            schema, "transactions", {"amount"}
        ):
            return None
        conditions = []
        time_filter = _time_filter(q)
        if time_filter and "date" in column_names(schema, "transactions"):
            conditions.append(time_filter)
        if "by merchant" in q or "per merchant" in q:
            return (
                "SELECT merchant, COALESCE(SUM(amount), 0) AS total_spending "
                f"FROM transactions{_where(conditions)} "
                "GROUP BY merchant ORDER BY total_spending DESC;"
            )
        if any(word in q for word in ("largest", "biggest", "top", "highest")):
            return (
                f"SELECT * FROM transactions{_where(conditions)} "
                f"ORDER BY amount DESC LIMIT {_limit(q)};"
            )
        if any(word in q for word in ("recent", "latest")):
            return (
                f"SELECT * FROM transactions{_where(conditions)} "
                "ORDER BY date DESC LIMIT 10;"
            )
        return f"SELECT * FROM transactions{_where(conditions)} ORDER BY date DESC;"

    if any(word in q for word in ("income", "salary", "earn")):
        if "income" not in tables or not _has_columns(schema, "income", {"amount"}):
            return None
        conditions = []
        time_filter = _time_filter(q)
        if time_filter and "date" in column_names(schema, "income"):
            conditions.append(time_filter)
        if "by source" in q or "per source" in q:
            return (
                "SELECT source, COALESCE(SUM(amount), 0) AS total_income "
                f"FROM income{_where(conditions)} "
                "GROUP BY source ORDER BY total_income DESC;"
            )
        return (
            "SELECT COALESCE(SUM(amount), 0) AS total_income "
            f"FROM income{_where(conditions)};"
        )

    if "budget" in q:
        if "budgets" not in tables:
            return None
        if "total" in q and _has_columns(schema, "budgets", {"monthly_limit"}):
            return (
                "SELECT COALESCE(SUM(monthly_limit), 0) AS total_budget "
                "FROM budgets;"
            )
        return "SELECT category, monthly_limit FROM budgets ORDER BY category;"

    if any(
        word in q
        for word in (
            "spend",
            "spent",
            "expense",
            "cost",
            "paid",
            "pay",
            "buy",
            "bought",
            "purchase",
            "purchased",
            "shop",
            "shopping",
        )
    ):
        if "expenses" not in tables or not _has_columns(
            schema, "expenses", {"amount"}
        ):
            return None
        columns = column_names(schema, "expenses")
        conditions = []
        category = _category_filter(q)
        if category and "category" in columns:
            conditions.append(f"category = '{category}'")
        time_filter = _time_filter(q)
        if time_filter and "date" in columns:
            conditions.append(time_filter)

        if "by category" in q or "per category" in q or "category breakdown" in q:
            if "category" not in columns:
                return None
            return (
                "SELECT category, COALESCE(SUM(amount), 0) AS total_spending "
                f"FROM expenses{_where(conditions)} "
                "GROUP BY category ORDER BY total_spending DESC;"
            )

        wants_items = any(
            word in q
            for word in ("top", "largest", "biggest", "items", "retrieved")
        )
        wants_top_category = any(
            phrase in q
            for phrase in (
                "category tops",
                "top category",
                "which category",
                "category has the most",
                "highest category",
            )
        )
        item_columns = {"date", "description", "category", "amount"}
        if wants_items and item_columns.issubset(columns):
            top_n = _limit(q, default=3)
            recent = (
                "SELECT date, description AS item, category, amount "
                f"FROM expenses{_where(conditions)}"
            )
            if wants_top_category:
                return (
                    "WITH recent AS ("
                    f"{recent}"
                    "), top_category AS ("
                    "SELECT category, SUM(amount) AS total_spending FROM recent "
                    "GROUP BY category ORDER BY total_spending DESC LIMIT 1"
                    ") "
                    "SELECT recent.date, recent.item, recent.category, recent.amount, "
                    "(SELECT category FROM top_category) AS top_category, "
                    "(SELECT total_spending FROM top_category) AS top_category_spending "
                    "FROM recent ORDER BY recent.amount DESC "
                    f"LIMIT {top_n};"
                )
            return f"{recent} ORDER BY amount DESC LIMIT {top_n};"

        if "how many" in q:
            return f"SELECT COUNT(*) AS expense_count FROM expenses{_where(conditions)};"
        if "average" in q:
            return f"SELECT AVG(amount) AS average_spending FROM expenses{_where(conditions)};"
        return (
            "SELECT COALESCE(SUM(amount), 0) AS total_spending "
            f"FROM expenses{_where(conditions)};"
        )

    if "balance" in q or "account" in q:
        if "accounts" not in tables:
            return None
        columns = column_names(schema, "accounts")
        if "total" in q and "balance" in columns:
            return "SELECT COALESCE(SUM(balance), 0) AS total_balance FROM accounts;"
        if "how many" in q:
            return "SELECT COUNT(*) AS account_count FROM accounts;"
        if {"account_type", "balance"}.issubset(columns):
            return (
                "SELECT account_type, balance, currency FROM accounts "
                "ORDER BY balance DESC;"
            )
        return "SELECT * FROM accounts;"

    if "user" in q or "customer" in q:
        if "users" not in tables:
            return None
        if "how many" in q:
            return "SELECT COUNT(*) AS user_count FROM users;"
        return "SELECT * FROM users;"

    return None
