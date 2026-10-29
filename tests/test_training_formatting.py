import unittest

from datasets import Dataset

from src.training.formatting import format_dataset
from src.utils.prompting import INSTRUCTION


class SpaceTokenizer:
    def __call__(self, text, add_special_tokens=True):
        # Deterministic budget proxy suitable for schema-pruning unit tests.
        return {"input_ids": text.split()}


class TrainingFormattingTests(unittest.TestCase):
    def test_instruction_matches_read_only_api_contract(self):
        self.assertIn("read-only SQL SELECT", INSTRUCTION)

    def test_preserves_full_sql_and_referenced_schema_before_optional_tables(self):
        schema = (
            "CREATE TABLE unrelated (" + " ".join(f"col{i} TEXT," for i in range(80)) + "id INT);"
            " CREATE TABLE expenses (id INT, amount DECIMAL);"
        )
        dataset = Dataset.from_list([{
            "question": "What is the expense amount?",
            "schema": schema,
            "output": "SELECT amount FROM expenses WHERE id = 7;",
        }])
        result = format_dataset(dataset, SpaceTokenizer(), max_length=100)
        self.assertEqual(len(result), 1)
        self.assertIn("CREATE TABLE expenses", result[0]["prompt"])
        self.assertNotIn("CREATE TABLE unrelated", result[0]["prompt"])
        self.assertEqual(result[0]["completion"], "SELECT amount FROM expenses WHERE id = 7;")

    def test_filters_mutating_targets(self):
        dataset = Dataset.from_list([
            {"question": "List rows", "schema": "CREATE TABLE t (id INT);", "output": "SELECT id FROM t;"},
            {"question": "Update rows", "schema": "CREATE TABLE t (id INT);", "output": "UPDATE t SET id=1;"},
        ])
        result = format_dataset(dataset, SpaceTokenizer(), max_length=100)
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["completion"].startswith("SELECT"))

    def test_drops_example_when_question_and_target_cannot_fit(self):
        dataset = Dataset.from_list([{
            "question": "word " * 30,
            "schema": "CREATE TABLE t (id INT);",
            "output": "SELECT id FROM t;",
        }])
        result = format_dataset(dataset, SpaceTokenizer(), max_length=10)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
