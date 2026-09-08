"""Hand-calculated ranking cases, malformed input, provenance, and CI policy."""
import importlib.util
import json
import math
import random
import unittest

from proofml import AuditFailed, CheckResult, RetrievalConfig, audit_retrieval
from proofml.reporting import render_html
from proofml.retrieval import default_retrieval_checks


class RetrievalTests(unittest.TestCase):
    def codes(self, report):
        self.assertFalse([c for c in report.checks if c.status == "error"])
        return {f.code for f in report.findings}

    def metrics(self, report):
        return report.metrics["retrieval_ranking"]

    def test_perfect_retrieval(self):
        report = audit_retrieval([["a", "b"]], [{"a", "b"}], k=2, corpus_ids={"a", "b"}, min_recall=1, min_ndcg=1)
        self.assertEqual(self.codes(report), set())
        for metric in ("precision", "recall", "hit_rate", "mrr", "ndcg"):
            self.assertEqual(self.metrics(report)[f"{metric}@2"], 1)
        report.raise_for_issues(require_checks=("retrieval_ranking", "retrieval_corpus"))

    def test_hand_calculated_macro_metrics(self):
        report = audit_retrieval([["x", "a", "b"], ["y", "c"]], [{"a", "b"}, {"c"}], k=3)
        values = self.metrics(report)
        self.assertAlmostEqual(values["precision@3"], 0.5)
        self.assertEqual(values["recall@3"], 1)
        self.assertEqual(values["hit_rate@3"], 1)
        self.assertEqual(values["mrr@3"], 0.5)
        expected = ((1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3)) + 1 / math.log2(3)) / 2
        self.assertAlmostEqual(values["ndcg@3"], expected)

    def test_cutoff_excludes_later_hits(self):
        report = audit_retrieval([["x", "a"]], [{"a"}], k=1)
        for metric in ("precision", "recall", "hit_rate", "mrr", "ndcg"):
            self.assertEqual(self.metrics(report)[f"{metric}@1"], 0)

    def test_empty_results_count_as_zero_for_judged_queries(self):
        report = audit_retrieval([[], ["a"]], [{"a"}, {"a"}], k=1)
        self.assertIn("empty_retrieval_results", self.codes(report))
        self.assertEqual(self.metrics(report)["recall@1"], 0.5)

    def test_short_rankings_do_not_inflate_precision(self):
        report = audit_retrieval([["a"]], [{"a"}], k=10)
        self.assertEqual(self.metrics(report)["precision@10"], 0.1)
        self.assertEqual(self.metrics(report)["recall@10"], 1)

    def test_duplicate_ids_never_receive_multiple_credit(self):
        report = audit_retrieval([["a", "a", "b"]], [{"a", "b"}], k=3)
        self.assertIn("duplicate_retrieval_results", self.codes(report))
        self.assertEqual(self.metrics(report)["recall@3"], 1)
        self.assertEqual(self.metrics(report)["precision@3"], 2 / 3)
        self.assertAlmostEqual(self.metrics(report)["ndcg@3"], 1.5 / (1 + 1 / math.log2(3)))

    def test_duplicate_judgments_have_set_semantics(self):
        report = audit_retrieval([["a"]], [["a", "a"]], k=1)
        self.assertEqual(self.metrics(report)["recall@1"], 1)

    def test_unjudged_queries_excluded_not_perfect(self):
        report = audit_retrieval([["a"], ["a"]], [[], ["b"]], k=1)
        self.assertEqual(self.metrics(report)["evaluated_queries"], 1)
        self.assertEqual(self.metrics(report)["excluded_queries"], 1)
        self.assertEqual(self.metrics(report)["recall@1"], 0)
        self.assertIn("missing_relevance_judgments", self.codes(report))

    def test_all_unjudged_skips_metrics_and_required_gate_fails(self):
        report = audit_retrieval([["a"]], [[]], min_recall=1)
        self.assertNotIn("retrieval_ranking", report.metrics)
        self.assertEqual(next(c.status for c in report.checks if c.check_id == "retrieval_ranking"), "skipped")
        with self.assertRaises(AuditFailed):
            report.raise_for_issues(require_checks=("retrieval_ranking",))

    def test_opt_in_thresholds_and_equality(self):
        report = audit_retrieval([["a"]], [{"a", "b"}], k=2, min_recall=0.5)
        self.assertNotIn("retrieval_recall_below_threshold", self.codes(report))
        report = audit_retrieval([["x"]], [{"a"}], k=1, min_recall=0.1, min_ndcg=0.1)
        self.assertIn("retrieval_recall_below_threshold", self.codes(report))
        self.assertIn("retrieval_ndcg_below_threshold", self.codes(report))
        with self.assertRaises(AuditFailed):
            report.raise_for_issues(require_checks=("retrieval_ranking",))
        self.assertNotIn("retrieval_recall_below_threshold", self.codes(audit_retrieval([["x"]], [{"a"}])))

    def test_missing_corpus_and_unknown_ids(self):
        report = audit_retrieval([["a"]], [{"b"}], corpus_ids={"c"})
        self.assertIn("unknown_retrieved_ids", self.codes(report))
        self.assertIn("unreachable_relevant_ids", self.codes(report))
        report = audit_retrieval([["a"]], [{"a"}], corpus_ids=[])
        self.assertIn("unknown_retrieved_ids", self.codes(report))
        report = audit_retrieval([["a"]], [{"a"}])
        self.assertEqual(next(c.status for c in report.checks if c.check_id == "retrieval_corpus"), "skipped")

    def test_mapping_alignment_uses_query_ids_not_order(self):
        report = audit_retrieval({"q1": ["a"], "q2": ["b"]}, {"q2": {"b"}, "q1": {"a"}}, k=1)
        self.assertEqual(self.metrics(report)["recall@1"], 1)

    def test_mapping_key_mismatch_and_mixed_styles_rejected(self):
        for retrieved, relevant in (({"q1": ["a"]}, {"q2": {"a"}}), ({"q1": ["a"]}, [["a"]]),
                                    ([["a"]], {"q1": ["a"]}), ({1: ["a"]}, {1: ["a"]})):
            with self.subTest(retrieved=retrieved), self.assertRaises((TypeError, ValueError)):
                audit_retrieval(retrieved, relevant)

    def test_generators_and_no_mutation(self):
        rankings, relevant = [["b", "a"]], [["a"]]
        report = audit_retrieval(iter(rankings), iter(relevant), k=2)
        self.assertEqual(rankings, [["b", "a"]])
        self.assertEqual(relevant, [["a"]])
        self.assertEqual(self.metrics(report)["mrr@2"], 0.5)

    def test_invalid_and_unordered_inputs(self):
        for retrieved, relevant in (([], []), ([["a"]], []), ([], [["a"]]), ("a", "a"),
                                    ([{"a"}], [["a"]]), ([[1]], [["a"]]), ([[""]], [["a"]]),
                                    ([["a"]], ["a"]), ([["a"]], [[None]]), ([["a"]], [{"a": 1}])):
            with self.subTest(retrieved=retrieved, relevant=relevant), self.assertRaises((TypeError, ValueError)):
                audit_retrieval(retrieved, relevant)

    def test_limits_refuse_without_sampling(self):
        for options in ({"max_queries": 1}, {"max_ids_per_query": 1}, {"max_bytes": 1}, {"max_corpus_ids": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                audit_retrieval([["a", "b"], ["a", "b"]], [["a"], ["a"]], corpus_ids=["a", "b"], **options)

    def test_configuration_validation(self):
        for options in ({"k": 0}, {"k": True}, {"k": 1.5}, {"min_recall": -1}, {"min_ndcg": float("nan")},
                        {"min_ndcg": float("inf")}, {"min_recall": True}, {"max_queries": 0}, {"disabled_checks": "retrieval_inputs"}):
            with self.subTest(options=options), self.assertRaises((ValueError, TypeError)):
                RetrievalConfig(**options)
        with self.assertRaises(TypeError):
            audit_retrieval([["a"]], [["a"]], min_recal=1)
        report = audit_retrieval([["a"]], [["a"]], config=RetrievalConfig(k=10), k=1)
        self.assertIn("recall@1", self.metrics(report))

    def test_id_matching_is_exact(self):
        report = audit_retrieval([["A"]], [["a"]], k=1)
        self.assertEqual(self.metrics(report)["recall@1"], 0)

    def test_reports_omit_queries_and_document_ids(self):
        private_query, private_id = "PRIVATE QUERY", "PRIVATE ID"
        report = audit_retrieval({private_query: [private_id, private_id]}, {private_query: {private_id}}, k=2)
        for output in (json.dumps(report.to_dict()), render_html(report)):
            self.assertNotIn(private_query, output)
            self.assertNotIn(private_id, output)
        self.assertIn("Measurements", render_html(report))
        self.assertIn("recall@2", render_html(report))

    def test_fingerprint_tracks_ranks_judgments_and_corpus(self):
        def fingerprint(rank, truth, **kwargs):
            return audit_retrieval([rank], [truth], **kwargs).datasets["evaluation"]["sha256"]
        base = fingerprint(["a", "b"], {"a", "b"})
        self.assertEqual(base, fingerprint(["a", "b"], ["b", "a"]))
        self.assertNotEqual(base, fingerprint(["b", "a"], {"a", "b"}))
        self.assertNotEqual(base, fingerprint(["a", "b"], {"b"}))
        self.assertNotEqual(base, fingerprint(["a", "b"], {"a", "b"}, corpus_ids={"a", "b"}))

    def test_metrics_bounded_for_random_duplicate_rankings(self):
        rng = random.Random(31)
        ranks = [rng.choices("abcdef", k=rng.randrange(20)) for _ in range(100)]
        truth = [set(rng.choices("abcde", k=rng.randrange(1, 10))) for _ in range(100)]
        metrics = self.metrics(audit_retrieval(ranks, truth, k=8))
        for key, value in metrics.items():
            if "@" in key:
                self.assertGreaterEqual(value, 0)
                self.assertLessEqual(value, 1)

    def test_plugins_and_disabled_metric_checks(self):
        class Custom:
            id = "custom"
            def run(self, ctx):
                return CheckResult.complete(self.id, [], metrics={"queries": len(ctx.data.retrieved)})
        report = audit_retrieval([["a"]], [["a"]], checks=[*default_retrieval_checks(), Custom()], disabled_checks=["retrieval_ranking"])
        self.assertEqual(report.metrics["custom"]["queries"], 1)
        self.assertNotIn("retrieval_ranking", report.metrics)
        with self.assertRaises(ValueError):
            audit_retrieval([["a"]], [["a"]], disabled_checks=["unknown"])

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "optional reference library not installed")
    def test_ndcg_matches_sklearn_binary_reference(self):
        from sklearn.metrics import ndcg_score
        report = audit_retrieval([["a", "x", "b", "y"]], [{"a", "b"}], k=3)
        expected = ndcg_score([[1, 0, 1, 0]], [[4, 3, 2, 1]], k=3)
        self.assertAlmostEqual(self.metrics(report)["ndcg@3"], expected)


class MetricContractTests(unittest.TestCase):
    def test_invalid_metrics_rejected(self):
        for metrics in ({"x": float("nan")}, {"x": float("inf")}, {"x": True}, {"x": "secret"}, {"": 1}):
            with self.subTest(metrics=metrics), self.assertRaises(ValueError):
                CheckResult.complete("test", [], metrics=metrics)
        with self.assertRaises(ValueError):
            CheckResult("test", "skipped", reason="No data", metrics={"value": 1})

    def test_invalid_plugin_result_and_metric_never_pass(self):
        class Invalid:
            id = "invalid"
            def run(self, ctx):
                return CheckResult.complete("wrong", [])
        class Nonfinite:
            id = "nonfinite"
            def run(self, ctx):
                result = CheckResult.complete(self.id, [], metrics={"value": 1})
                result.metrics["value"] = float("nan")
                return result
        report = audit_retrieval([["a"]], [["a"]], checks=[Invalid(), Nonfinite()])
        self.assertTrue(all(c.status == "error" for c in report.checks))
        self.assertEqual(report.metrics, {})
        with self.assertRaises(AuditFailed):
            report.raise_for_issues()

    def test_mutated_nonnumeric_plugin_metric_is_rejected(self):
        class Mutated:
            id = "mutated"
            def run(self, ctx):
                result = CheckResult.complete(self.id, [], metrics={"value": 1})
                result.metrics["value"] = "PRIVATE VALUE"
                return result
        report = audit_retrieval([["a"]], [["a"]], checks=[Mutated()])
        self.assertEqual(report.checks[0].status, "error")
        self.assertNotIn("PRIVATE VALUE", json.dumps(report.to_dict()))

    def test_metric_names_are_html_escaped(self):
        class Metric:
            id = "custom"
            def run(self, ctx):
                return CheckResult.complete(self.id, [], metrics={"<script>": 1})
        report = audit_retrieval([["a"]], [["a"]], checks=[Metric()])
        self.assertNotIn("<script>", render_html(report))
        self.assertIn("&lt;script&gt;", render_html(report))
