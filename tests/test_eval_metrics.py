import unittest

from src.eval.metrics import (
    equivalent_rows, evaluate_pair, exact_match_accuracy, execute_readonly,
    summarize_latencies, validate_query_schema,
)


class MetricsTests(unittest.TestCase):
    def test_exact_match_ignores_case_whitespace_comments(self):
        self.assertTrue(exact_match_accuracy("select /*x*/ a from t;", "SELECT a\nFROM t"))

    def test_result_multiset_preserves_duplicates_and_null_numeric_equivalence(self):
        self.assertTrue(equivalent_rows([(1, None), (1, None)], [(1.0, None), (1, None)],
                                        "select a", "select a"))
        self.assertFalse(equivalent_rows([(1,), (1,)], [(1,)], "select a", "select a"))

    def test_ordered_rows_respect_order_by(self):
        self.assertFalse(equivalent_rows([(1,), (2,)], [(2,), (1,)],
                                         "select x order by x", "select x order by x"))

    def test_readonly_execution_and_missing_table_classification(self):
        context = "CREATE TABLE t(x); INSERT INTO t VALUES(1);"
        rows, status = execute_readonly("SELECT x FROM t", context=context)
        self.assertEqual(rows, [(1,)])
        self.assertIsNone(status["error_type"])
        _, failure = execute_readonly("DELETE FROM t", context=context)
        self.assertEqual(failure["error_type"], "other_database_error")
        _, missing = execute_readonly("SELECT * FROM absent", context=context)
        self.assertEqual(missing["error_type"], "missing_table")

    def test_validation_and_pair(self):
        schema = "CREATE TABLE t(x INTEGER);"
        self.assertTrue(validate_query_schema("SELECT x FROM t", schema)["valid"])
        self.assertEqual(validate_query_schema("SELECT z FROM t", schema)["error_type"], "missing_column")
        result = evaluate_pair("SELECT x FROM t", "select x from t", context=schema + "INSERT INTO t VALUES (7);")
        self.assertTrue(result["execution_match"])

    def test_syntax_failure_and_timeout(self):
        context = "CREATE TABLE t(x INTEGER);"
        _, invalid = execute_readonly("SELECT FROM", context=context)
        self.assertEqual(invalid["error_type"], "syntax_error")
        recursive = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT SUM(x) FROM c"
        _, timed = execute_readonly(recursive, context=context, timeout_s=0.001)
        self.assertEqual(timed["error_type"], "timeout")

    def test_percentiles(self):
        summary = summarize_latencies([1, 2, 3, 4, 5])
        self.assertEqual(summary["mean_ms"], 3)
        self.assertEqual(summary["p50_ms"], 3)


if __name__ == "__main__":
    unittest.main()
