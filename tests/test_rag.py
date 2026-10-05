import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))

from rag import LocalRAG


class RagTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.rag = LocalRAG(self.root / "knowledge" / "sql_rag.jsonl")

    def test_retrieves_join_guidance_for_join_query(self):
        results = self.rag.retrieve_for_case(
            schema="users(id), orders(user_id, total)",
            sql="SELECT u.id, o.total FROM users u JOIN orders o ON o.user_id = u.id",
            categories=["JOIN"],
            top_k=3,
        )[0]
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.id, "joins")

    def test_retrieval_is_deterministic(self):
        query = "SELECT department, COUNT(*) FROM employees GROUP BY department HAVING COUNT(*) > 2"
        a = self.rag.retrieve(query, top_k=3)
        b = self.rag.retrieve(query, top_k=3)
        self.assertEqual(
            [(x.chunk.id, x.score) for x in a],
            [(x.chunk.id, x.score) for x in b],
        )

    def test_context_is_grounded_and_not_query_answer(self):
        results, context = self.rag.retrieve_for_case(
            schema="users(id, name)",
            sql="SELECT name FROM users WHERE id = 1",
            categories=["WHERE"],
            top_k=2,
        )
        self.assertIn("generic supporting knowledge", context)
        self.assertNotIn("name is Alice", context)
        self.assertTrue(all("score" in {"score": r.score} for r in results))

    def test_empty_retrieval_has_safe_fallback(self):
        context = self.rag.format_context([])
        self.assertIn("SQL and schema only", context)


if __name__ == "__main__":
    unittest.main()
