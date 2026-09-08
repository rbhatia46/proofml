"""Never compare different benchmarks as though they were the same experiment."""
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from proofml import AuditPolicy, AuditReport, CheckResult, audit_retrieval, audit_text
from proofml.retrieval import default_retrieval_checks


class ComparisonTests(unittest.TestCase):
    def report(self, ranked=None, relevant=None, **options):
        return audit_retrieval(ranked or {"q1": ["a"], "q2": ["b"]}, relevant or {"q1": {"a"}, "q2": {"b"}}, k=1, **options)

    def test_output_changes_do_not_change_benchmark_identity(self):
        baseline = self.report()
        candidate = self.report({"q1": ["x"], "q2": ["b"]})
        a, b = baseline.datasets["evaluation"], candidate.datasets["evaluation"]
        self.assertEqual(a["evaluation_sha256"], b["evaluation_sha256"])
        self.assertNotEqual(a["output_sha256"], b["output_sha256"])
        comparison = candidate.compare(baseline)
        self.assertTrue(comparison.compatible)
        change = next(c for c in comparison.changes if c.check_id == "retrieval_ranking" and c.metric == "recall@1")
        self.assertEqual((change.baseline, change.candidate, change.delta), (1, 0.5, -0.5))

    def test_mapping_order_and_judgment_set_order_do_not_matter(self):
        baseline = self.report(relevant={"q1": ["a", "c"], "q2": ["b"]})
        candidate = self.report({"q2": ["b"], "q1": ["a"]}, {"q2": ["b"], "q1": ["c", "a"]})
        self.assertTrue(candidate.compare(baseline).compatible)
        self.assertEqual(candidate.datasets["evaluation"]["output_sha256"], baseline.datasets["evaluation"]["output_sha256"])

    def test_changed_query_ids_or_judgments_incompatible(self):
        baseline = self.report()
        for candidate in (self.report(relevant={"q1": {"x"}, "q2": {"b"}}),
                          self.report({"new": ["a"], "q2": ["b"]}, {"new": {"a"}, "q2": {"b"}})):
            self.assertFalse(candidate.compare(baseline).compatible)
            self.assertFalse(candidate.compare(baseline).changes)

    def test_changed_corpus_or_missing_snapshot_incompatible(self):
        baseline = self.report(corpus_ids={"a", "b"})
        for candidate in (self.report(corpus_ids={"a", "b", "c"}), self.report()):
            self.assertFalse(candidate.compare(baseline).compatible)
        self.assertTrue(self.report(corpus_ids=["b", "a"]).compare(baseline).compatible)

    def test_positional_inputs_cannot_establish_query_identity(self):
        report = audit_retrieval([["a"]], [["a"]], k=1)
        self.assertFalse(report.compare(report).compatible)

    def test_tool_config_and_registry_changes_are_incompatible(self):
        baseline = self.report()
        for candidate in (replace(baseline, tool_version="other"), replace(baseline, schema_version="2.0"),
                          self.report(min_recall=0.8), replace(baseline, checks=baseline.checks[:-1])):
            with self.subTest(candidate=candidate.config):
                self.assertFalse(candidate.compare(baseline).compatible)

    def test_legacy_report_missing_identity_refuses_comparison(self):
        baseline = self.report()
        metadata = {key: value for key, value in baseline.datasets["evaluation"].items()
                    if key in {"rows", "columns", "sha256", "format", "size_bytes"}}
        legacy = replace(baseline, datasets={"evaluation": metadata})
        self.assertFalse(baseline.compare(legacy).compatible)

    def test_unsupported_domain_is_explicit(self):
        report = audit_text(["document"])
        self.assertFalse(report.compare(report).compatible)
        self.assertEqual(AuditPolicy().evaluate(report, baseline=report).status, "incompatible")

    def test_saved_report_has_same_comparison_behavior(self):
        baseline = self.report()
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = baseline.save(Path(tmp) / "baseline")
            loaded = AuditReport.load(path)
            self.assertEqual(loaded.to_dict(), baseline.to_dict())
            self.assertEqual(self.report().compare(loaded), self.report().compare(baseline))

    def test_missing_metrics_never_become_zero(self):
        baseline = self.report()
        checks = tuple(replace(c, metrics={}) if c.check_id == "retrieval_ranking" else c for c in baseline.checks)
        candidate = replace(baseline, checks=checks)
        comparison = candidate.compare(baseline)
        change = next(c for c in comparison.changes if c.metric == "recall@1")
        self.assertIsNone(change.candidate)
        self.assertIsNone(change.delta)
        result = AuditPolicy(max_drop={"retrieval_ranking.recall@1": 1}).evaluate(candidate, baseline=baseline)
        self.assertEqual(result.status, "insufficient")

    def test_unversioned_custom_checks_cannot_masquerade_as_builtin_metrics(self):
        class Custom:
            id = "retrieval_ranking"
            def run(self, ctx):
                return CheckResult.complete(self.id, [], metrics={"recall@1": 1})
        report = self.report(checks=[Custom()])
        self.assertFalse(report.compare(report).compatible)

    def test_regression_boundary_is_absolute_and_rounding_tolerant(self):
        baseline = self.report()
        def with_recall(value):
            checks = tuple(replace(c, metrics={**c.metrics, "recall@1": value}) if c.check_id == "retrieval_ranking" else c for c in baseline.checks)
            return replace(baseline, checks=checks)
        policy = AuditPolicy(max_drop={"retrieval_ranking.recall@1": 0.1})
        self.assertTrue(policy.evaluate(with_recall(0.7), baseline=with_recall(0.8)).passed)
        self.assertFalse(policy.evaluate(with_recall(0.699), baseline=with_recall(0.8)).passed)

    def test_zero_tolerance_does_not_hide_small_real_regressions(self):
        baseline = self.report()
        checks = tuple(replace(c, metrics={**c.metrics, "recall@1": 1 - 1e-13}) if c.check_id == "retrieval_ranking" else c for c in baseline.checks)
        candidate = replace(baseline, checks=checks)
        self.assertEqual(AuditPolicy(max_drop={"retrieval_ranking.recall@1": 0}).evaluate(candidate, baseline=baseline).status, "failed")
