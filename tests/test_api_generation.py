"""Endpoint-level tests that do not load the language model."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from api import main as api_main

DUPLICATE_JOIN_SQL = (
    "SELECT SUM(expenses.amount) FROM expenses "
    "JOIN accounts ON expenses.user_id = accounts.user_id "
    "JOIN accounts ON accounts.user_id = accounts.account_type "
    "WHERE expenses.category = 'Spending';"
)


class ApiGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._model = api_main.model
        self._tokenizer = api_main.tokenizer
        api_main.model = object()
        api_main.tokenizer = object()

    def tearDown(self) -> None:
        api_main.model = self._model
        api_main.tokenizer = self._tokenizer

    def test_invalid_model_sql_returns_schema_safe_fallback(self) -> None:
        with patch.object(api_main, "generate_sql", return_value=DUPLICATE_JOIN_SQL):
            response = api_main.generate(
                api_main.Query(question="what is my this week spending")
            )

        self.assertTrue(response.valid)
        self.assertEqual(response.source, "safe-fallback")
        self.assertIn("FROM expenses", response.sql)
        self.assertIn("date('now', '-7 days')", response.sql)
        self.assertNotIn("JOIN", response.sql)
        self.assertIn("ambiguous", (response.validation_error or "").lower())
        self.assertIsNotNone(response.warning)

    def test_valid_model_sql_is_returned_unchanged(self) -> None:
        with patch.object(api_main, "generate_sql", return_value="SELECT * FROM users;"):
            response = api_main.generate(api_main.Query(question="Show all users"))

        self.assertTrue(response.valid)
        self.assertEqual(response.source, "model")
        self.assertEqual(response.sql, "SELECT * FROM users;")
        self.assertIsNone(response.warning)
        self.assertIsNotNone(response.latency_ms)
        self.assertGreaterEqual(response.latency_ms, 0)

    def test_invalid_sql_without_fallback_is_flagged(self) -> None:
        schema = "CREATE TABLE foo (id INT);"
        with patch.object(api_main, "generate_sql", return_value="SELECT missing FROM foo;"):
            response = api_main.generate(
                api_main.Query(question="Explain the weather", schema=schema)
            )

        self.assertFalse(response.valid)
        self.assertIn("did not validate", (response.warning or ""))
        self.assertIn("no such column", (response.validation_error or "").lower())


if __name__ == "__main__":
    unittest.main()
