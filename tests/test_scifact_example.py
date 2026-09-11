"""Small offline fixtures exercise the adapter, not the real benchmark scores."""
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from examples import scifact_retrieval as example


def members(qrels="query-id\tcorpus-id\tscore\nq1\td1\t1\n"):
    return {"scifact/corpus.jsonl": b'{"_id":"d1","title":"alpha","text":"beta"}\n',
            "scifact/queries.jsonl": b'{"_id":"q1","text":"beta"}\n{"_id":"train-query","text":"not test"}\n',
            "scifact/qrels/test.tsv": qrels.encode()}


class SciFactAdapterTests(unittest.TestCase):
    def test_only_test_judged_queries_are_selected(self):
        data = example.parse_benchmark(members())
        self.assertEqual(data.queries, {"q1": "beta"})
        self.assertEqual(data.relevant, {"q1": {"d1"}})

    def test_rejects_duplicate_ids_and_qrels(self):
        with self.assertRaises(ValueError):
            example.jsonl_records(b'{"_id":"q","text":"a"}\n{"_id":"q","text":"b"}')
        with self.assertRaises(ValueError):
            example.parse_benchmark(members("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td1\t1\n"))

    def test_rejects_graded_or_unresolved_judgments(self):
        for qid, did, score in (("q1", "d1", "2"), ("absent", "d1", "1"), ("q1", "absent", "1")):
            with self.subTest(qid=qid, did=did, score=score), self.assertRaises(ValueError):
                example.parse_benchmark(members(f"query-id\tcorpus-id\tscore\n{qid}\t{did}\t{score}\n"))

    def test_source_selection_explicit(self):
        for options in ({}, {"archive": "unused", "download": True}):
            with self.assertRaises(ValueError):
                example.load_scifact(**options)

    def test_changed_archive_rejected(self):
        with patch("pathlib.Path.open", return_value=BytesIO(b"changed")), self.assertRaisesRegex(ValueError, "checksum"):
            example.load_scifact(archive="unused")

    def test_bounded_read(self):
        with self.assertRaises(ValueError):
            example.bounded_read(BytesIO(b"1234"), 3)

    def test_cohorts_have_fixed_boundary(self):
        groups = example.query_groups({"a": "word " * 12, "b": "word " * 13})
        self.assertEqual(groups, {"a": "short_queries", "b": "long_queries"})

    def test_policy_requires_all_cohorts_and_full_coverage(self):
        policy = example.release_policy()
        self.assertEqual(policy.require_reports, example.GROUPS)
        self.assertEqual(policy.default.min_coverage["retrieval_ranking"], 1)
        self.assertEqual(policy.default.max_drop["retrieval_ranking.recall@10"], 0.05)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn example dependency not installed")
class RankingTests(unittest.TestCase):
    def test_actual_fields_change_ranking_without_using_judgments(self):
        corpus = {"a": {"title": "other", "text": "needle needle"}, "b": {"title": "needle", "text": "other " * 20}}
        query = {"q": "needle"}
        self.assertEqual(example.rank_tfidf(corpus, query, k=1)["q"], ["a"])
        self.assertEqual(example.rank_tfidf(corpus, query, fields=("title",), k=1)["q"], ["b"])

    def test_ties_stable_and_oov_empty(self):
        corpus = {"b": {"title": "same", "text": "same"}, "a": {"title": "same", "text": "same"}}
        rankings = example.rank_tfidf(corpus, {"match": "same", "oov": "unseen"}, k=10)
        self.assertEqual(rankings, {"match": ["a", "b"], "oov": []})
        self.assertEqual(rankings, example.rank_tfidf(dict(reversed(list(corpus.items()))), {"oov": "unseen", "match": "same"}, k=10))

    def test_invalid_ranking_options(self):
        for kwargs in ({"k": 0}, {"k": True}, {"fields": ()}, {"fields": ("unknown",)}):
            with self.assertRaises(ValueError):
                example.rank_tfidf({"a": {"title": "same", "text": "same"}}, {"q": "same"}, **kwargs)

    def test_integration_keeps_real_population_and_marks_tiny_slices_insufficient(self):
        data = example.parse_benchmark(members())
        ranks = example.rank_tfidf(data.corpus, data.queries)
        baseline, candidate, decision = example.compare_rankings(data, ranks, ranks)
        self.assertEqual(candidate.metrics["overall"]["retrieval_ranking"]["recall@10"], 1)
        self.assertEqual(decision.status, "insufficient")
        self.assertIn("long_queries", {i.evidence["report"] for i in decision.issues})


class NotebookStructureTests(unittest.TestCase):
    def test_notebooks_have_executable_python_and_teaching_sections(self):
        root = Path(__file__).resolve().parents[1]
        paths = list((root / "examples/notebooks").glob("*.ipynb"))
        self.assertTrue({"scifact_retrieval.ipynb", "bank_marketing.ipynb"} <= {p.name for p in paths})
        for path in paths:
            notebook = json.loads(path.read_text())
            self.assertEqual(notebook["nbformat"], 4)
            self.assertGreaterEqual(sum(c["cell_type"] == "markdown" for c in notebook["cells"]), 5)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    self.assertIsNotNone(cell["execution_count"], "Commit notebooks only after executing them")
                    source = cell["source"]
                    compile("".join(source) if isinstance(source, list) else source, str(path), "exec")
                    self.assertFalse(any(o.get("output_type") == "error" for o in cell["outputs"]))
