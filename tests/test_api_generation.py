"""Endpoint-level tests that do not load the language model."""

from __future__ import annotations

import unittest
import asyncio
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
            response = asyncio.run(api_main.generate(
                api_main.Query(question="what is my this week spending"), None
            ))

        self.assertTrue(response.valid)
        self.assertEqual(response.source, "safe-fallback")
        self.assertIn("FROM expenses", response.sql)
        self.assertIn("date('now', '-7 days')", response.sql)
        self.assertNotIn("JOIN", response.sql)
        self.assertIn("ambiguous", (response.validation_error or "").lower())
        self.assertIsNotNone(response.warning)

    def test_valid_model_sql_is_returned_unchanged(self) -> None:
        with patch.object(api_main, "generate_sql", return_value="SELECT * FROM users;"):
            response = asyncio.run(api_main.generate(api_main.Query(question="Show all users"), None))

        self.assertTrue(response.valid)
        self.assertEqual(response.source, "model")
        self.assertEqual(response.sql, "SELECT * FROM users;")
        self.assertIsNone(response.warning)
        self.assertIsNotNone(response.latency_ms)
        self.assertGreaterEqual(response.latency_ms, 0)

    def test_invalid_sql_without_fallback_is_flagged(self) -> None:
        schema = "CREATE TABLE foo (id INT);"
        with patch.object(api_main, "generate_sql", return_value="SELECT missing FROM foo;"):
            response = asyncio.run(api_main.generate(
                api_main.Query(question="Explain the weather", schema=schema), None
            ))

        self.assertFalse(response.valid)
        self.assertIn("did not validate", (response.warning or ""))
        self.assertIn("no such column", (response.validation_error or "").lower())

    def test_successful_response_is_cached_and_cache_hit_skips_generation(self) -> None:
        class MemoryCache:
            def __init__(self): self.values = {}; self.errors = 0; self.enabled = True
            async def get(self, key): return self.values.get(key)
            async def set(self, key, value): self.values[key] = value; return True
            def lock_for(self, key):
                import asyncio
                return locks.setdefault(key, asyncio.Lock())

        locks = {}
        cache = MemoryCache()
        original_cache = api_main.cache
        api_main.cache = cache
        try:
            with patch.object(api_main, "generate_sql", return_value="SELECT * FROM users;") as generate:
                async def concurrent_calls():
                    return await asyncio.gather(*[
                        api_main.generate(api_main.Query(question="Show all users"), None)
                        for _ in range(2)
                    ])
                first, second = asyncio.run(concurrent_calls())
            self.assertEqual(sum(r.cache_hit for r in (first, second)), 1)
            cached = first if first.cache_hit else second
            self.assertIsNone(cached.generation_latency_ms)
            self.assertEqual(generate.call_count, 1)
        finally:
            api_main.cache = original_cache


if __name__ == "__main__":
    unittest.main()
