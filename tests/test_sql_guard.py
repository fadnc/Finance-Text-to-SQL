"""Regression tests for API SQL validation and the demo fallback."""

from __future__ import annotations

import unittest

import yaml

from src.inference.demo_sql import build_demo_sql
from src.inference.sql_guard import (
    prepare_sql,
    validate_sql,
)

with open("configs/training_config.yaml", encoding="utf-8") as handle:
    SCHEMA = yaml.safe_load(handle)["schema"]

DUPLICATE_JOIN_SQL = (
    "SELECT SUM(expenses.amount) FROM expenses "
    "JOIN accounts ON expenses.user_id = accounts.user_id "
    "JOIN accounts ON accounts.user_id = accounts.account_type "
    "WHERE accounts.user_id = (SELECT user_id FROM users WHERE name = 'Alice') "
    "AND accounts.account_type = 'Bank Account' "
    "AND accounts.currency = 'USD' "
    "AND expenses.category = 'Spending';"
)


class SqlGuardTests(unittest.TestCase):
    def test_valid_query(self) -> None:
        self.assertEqual(validate_sql("SELECT * FROM users;", SCHEMA), (True, None))

    def test_duplicate_join_is_rejected(self) -> None:
        valid, error = validate_sql(DUPLICATE_JOIN_SQL, SCHEMA)
        self.assertFalse(valid)
        self.assertIn("ambiguous", (error or "").lower())

    def test_prepare_does_not_hide_duplicate_join(self) -> None:
        _, valid, error, source = prepare_sql(DUPLICATE_JOIN_SQL, SCHEMA)
        self.assertFalse(valid)
        self.assertIn("ambiguous", (error or "").lower())
        self.assertEqual(source, "invalid")

    def test_mysql_date_functions_are_normalized(self) -> None:
        sql = (
            "SELECT SUM(amount) FROM expenses "
            "WHERE date >= DATE_SUB(CURDATE(), INTERVAL 7 DAY);"
        )
        prepared, valid, error, source = prepare_sql(sql, SCHEMA)
        self.assertTrue(valid, error)
        self.assertIn("date('now', '-7 days')", prepared)
        self.assertEqual(source, "model-normalized")

    def test_write_statements_are_rejected(self) -> None:
        valid, error = validate_sql("DELETE FROM users;", SCHEMA)
        self.assertFalse(valid)
        self.assertIn("read-only", (error or "").lower())

    def test_compact_schema_is_supported(self) -> None:
        compact = "expenses(expense_id, user_id, category, amount, date)"
        valid, error = validate_sql(
            "SELECT category, SUM(amount) FROM expenses GROUP BY category;",
            compact,
        )
        self.assertTrue(valid, error)


class DemoFallbackTests(unittest.TestCase):
    def assert_fallback(self, question: str, *fragments: str) -> str:
        sql = build_demo_sql(question, SCHEMA)
        self.assertIsNotNone(sql, question)
        self.assertTrue(validate_sql(sql, SCHEMA)[0], sql)
        for fragment in fragments:
            self.assertIn(fragment, sql)
        return sql

    def test_this_week_spending(self) -> None:
        sql = self.assert_fallback(
            "what is my this week spending",
            "FROM expenses",
            "date('now', '-7 days')",
            "COALESCE(SUM(amount), 0)",
        )
        self.assertNotIn("JOIN", sql)

    def test_food_last_month(self) -> None:
        self.assert_fallback(
            "How much did I spend on Food last month?",
            "category = 'Food'",
            "start of month",
        )

    def test_spending_by_category(self) -> None:
        self.assert_fallback(
            "What is the total spending by category?",
            "GROUP BY category",
        )

    def test_largest_transactions(self) -> None:
        self.assert_fallback(
            "Show my 5 largest transactions",
            "ORDER BY amount DESC LIMIT 5",
        )

    def test_three_week_purchase_history_with_top_category(self) -> None:
        self.assert_fallback(
            "what add did u buy this 3 weeks and which category tops it "
            "with top 3 retrieved items",
            "WITH recent AS",
            "date('now', '-21 days')",
            "top_category",
            "LIMIT 3",
        )

    def test_top_items_use_relative_week_filter(self) -> None:
        self.assert_fallback(
            "Show my top 3 expenses in the last 3 weeks",
            "date('now', '-21 days')",
            "ORDER BY amount DESC LIMIT 3",
        )

    def test_unknown_question_returns_none(self) -> None:
        self.assertIsNone(build_demo_sql("Explain the weather forecast", SCHEMA))


if __name__ == "__main__":
    unittest.main()
