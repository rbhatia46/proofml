"""Slice evidence and gates must catch failures hidden by an improving aggregate."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proofml import AuditFailed, AuditPolicy, AuditSuite, CheckResult, SuitePolicy, audit_retrieval, audit_retrieval_slices
from proofml.suite_reporting import render_suite


def make_example(candidate=False):
    # Six English queries and two Spanish queries. Overall goes 2/8 -> 6/8,
    # while the smaller Spanish cohort goes from perfect retrieval to no hits.
    truth = {f"q{i}": {f"d{i}"} for i in range(8)}
    groups = {f"q{i}": "english" if i < 6 else "spanish" for i in range(8)}
    ranks = {f"q{i}": [f"d{i}" if (i < 6 if candidate else i >= 6) else "miss"] for i in range(8)}
    return audit_retrieval_slices(ranks, truth, groups=groups, k=1)


class SliceTests(unittest.TestCase):
    def test_overall_improves_but_slice_regression_blocks(self):
        baseline, candidate = make_example(), make_example(True)
        self.assertEqual(baseline.metrics["overall"]["retrieval_ranking"]["recall@1"], 0.25)
        self.assertEqual(candidate.metrics["overall"]["retrieval_ranking"]["recall@1"], 0.75)
        policy = SuitePolicy(default=AuditPolicy(require_checks=("retrieval_ranking",), min_evaluated={"retrieval_ranking": 2},
                                                max_drop={"retrieval_ranking.recall@1": 0.01}),
                             require_reports=("overall", "english", "spanish"))
        result = policy.evaluate(candidate, baseline=baseline)
        self.assertEqual(result.results["overall"].status, "passed")
        self.assertEqual(result.results["english"].status, "passed")
        self.assertEqual(result.results["spanish"].status, "failed")
        self.assertEqual(result.status, "failed")
        issue = result.results["spanish"].issues[0]
        self.assertEqual(issue.evidence["baseline"], 1)
        self.assertEqual(issue.evidence["candidate"], 0)
        with self.assertRaises(AuditFailed) as caught:
            policy.enforce(candidate, baseline=baseline)
        self.assertIsInstance(caught.exception.report, AuditSuite)
        self.assertEqual(caught.exception.decision.status, "failed")

    def test_group_order_and_input_mapping_order_do_not_change_identity(self):
        one = audit_retrieval_slices({"q1": ["a"], "q2": ["b"]}, {"q1": {"a"}, "q2": {"b"}}, groups={"q1": "en", "q2": "es"}, k=1)
        two = audit_retrieval_slices({"q2": ["b"], "q1": ["a"]}, {"q2": {"b"}, "q1": {"a"}}, groups={"q2": "es", "q1": "en"}, k=1)
        self.assertTrue(SuitePolicy().evaluate(two, baseline=one).passed)
        self.assertEqual(one.metrics, two.metrics)

    def test_fractional_metrics_repeat_exactly_across_query_order(self):
        keys = [f"q{i}" for i in range(40)]
        ranks = {key: ["miss"] * (i % 7) + ["hit"] for i, key in enumerate(keys)}
        truth = {key: {"hit"} for key in keys}
        groups = {key: "all" for key in keys}
        one = audit_retrieval_slices(ranks, truth, groups=groups, k=10)
        two = audit_retrieval_slices(dict(reversed(list(ranks.items()))), truth, groups=groups, k=10)
        self.assertEqual(one.metrics, two.metrics)
        policy = SuitePolicy(default=AuditPolicy(severity="critical", max_drop={"retrieval_ranking.mrr@10": 0}))
        self.assertTrue(policy.evaluate(two, baseline=one).passed)

    def test_slice_metrics_match_independent_unsliced_reference(self):
        ranks = {"q1": ["x", "a"], "q2": ["b"], "q3": ["c"]}
        truth = {"q1": {"a"}, "q2": {"b"}, "q3": {"missing"}}
        suite = audit_retrieval_slices(ranks, truth, slices={"subset": {"q1", "q3"}}, k=2, corpus_ids={"a", "b", "c", "x", "missing"})
        reference = audit_retrieval({q: ranks[q] for q in ("q1", "q3")}, {q: truth[q] for q in ("q1", "q3")}, k=2,
                                    corpus_ids={"a", "b", "c", "x", "missing"})
        self.assertEqual(suite.reports["subset"].checks, reference.checks)
        self.assertTrue(suite.reports["subset"].compare(reference).compatible)

    def test_overlapping_slices_do_not_reweight_overall(self):
        suite = audit_retrieval_slices({"q1": ["a"], "q2": ["x"]}, {"q1": {"a"}, "q2": {"b"}},
                                      slices={"first": ["q1"], "both": ["q1", "q2"]}, k=1)
        self.assertEqual(suite.metrics["overall"]["retrieval_ranking"]["recall@1"], 0.5)
        self.assertEqual(suite.metrics["first"]["retrieval_ranking"]["recall@1"], 1)
        self.assertEqual(suite.metrics["both"]["retrieval_ranking"]["evaluated_queries"], 2)

    def test_empty_requested_slice_is_not_dropped_or_perfect(self):
        suite = audit_retrieval_slices({"q1": ["a"]}, {"q1": {"a"}}, slices={"absent_cohort": []}, k=1)
        self.assertIn("absent_cohort", suite.reports)
        self.assertEqual(suite.reports["absent_cohort"].datasets["evaluation"]["rows"], 0)
        self.assertNotIn("retrieval_ranking", suite.metrics["absent_cohort"])
        result = SuitePolicy(default=AuditPolicy(min_coverage={"retrieval_ranking": 1})).evaluate(suite)
        self.assertEqual(result.results["absent_cohort"].status, "insufficient")

    def test_unjudged_slice_and_small_slice_block(self):
        suite = audit_retrieval_slices({"q1": ["a"], "q2": ["b"]}, {"q1": {"a"}, "q2": set()}, groups={"q1": "judged", "q2": "unjudged"}, k=1)
        result = SuitePolicy(default=AuditPolicy(min_evaluated={"retrieval_ranking": 2})).evaluate(suite)
        self.assertEqual(result.results["judged"].status, "insufficient")
        self.assertEqual(result.results["unjudged"].status, "insufficient")

    def test_membership_changes_invalidate_same_named_slice(self):
        args = ({"q1": ["a"], "q2": ["b"]}, {"q1": {"a"}, "q2": {"b"}})
        baseline = audit_retrieval_slices(*args, slices={"cohort": ["q1"]}, k=1)
        candidate = audit_retrieval_slices(*args, slices={"cohort": ["q2"]}, k=1)
        result = SuitePolicy().evaluate(candidate, baseline=baseline)
        self.assertEqual(result.results["overall"].status, "passed")
        self.assertEqual(result.results["cohort"].status, "incompatible")

    def test_disappearing_group_is_missing_not_silently_ignored(self):
        baseline = make_example()
        candidate = AuditSuite({k: v for k, v in baseline.reports.items() if k != "spanish"})
        result = SuitePolicy().evaluate(candidate, baseline=baseline)
        self.assertEqual(result.status, "insufficient")
        self.assertTrue(any(i.code == "candidate_report_removed" for i in result.issues))

    def test_retrieval_and_corpus_generators_are_consumed_once(self):
        visits = []
        def corpus():
            for value in ("a", "b"):
                visits.append(value)
                yield value
        ranks = {"q1": iter(["a"]), "q2": iter(["b"])}
        truth = {"q1": iter(["a"]), "q2": iter(["b"])}
        suite = audit_retrieval_slices(ranks, truth, groups={"q1": "en", "q2": "es"}, corpus_ids=corpus(), k=1)
        self.assertEqual(visits, ["a", "b"])
        self.assertTrue(all(r.metrics["retrieval_ranking"]["recall@1"] == 1 for r in suite.reports.values()))

    def test_unknown_ids_duplicates_and_bad_grouping_fail(self):
        args = ({"q1": ["a"], "q2": ["b"]}, {"q1": {"a"}, "q2": {"b"}})
        options = ({}, {"groups": {}, "slices": {"x": []}}, {"groups": {"q1": "en"}},
                   {"groups": {"q1": "en", "unknown": "es"}}, {"groups": {"q1": "en", "q2": None}},
                   {"slices": {}}, {"slices": {"x": ["unknown"]}}, {"slices": {"x": ["q1", "q1"]}},
                   {"slices": {"overall": ["q1"]}}, {"slices": {"x": "q1"}}, {"slices": {"": []}})
        for value in options:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                audit_retrieval_slices(*args, **value)

    def test_grouping_requires_query_id_mappings(self):
        with self.assertRaises(TypeError):
            audit_retrieval_slices([["a"]], [["a"]], groups={"q1": "en"})

    def test_budgets_reject_before_any_check_runs(self):
        args = ({"q1": ["a"], "q2": ["b"]}, {"q1": {"a"}, "q2": {"b"}})
        options = ({"max_slices": 1}, {"max_memberships": 1}, {"max_slices": True}, {"max_memberships": 0})
        for limits in options:
            with self.subTest(limits=limits), patch("proofml.retrieval.slices.build_report") as build, self.assertRaises(ValueError):
                audit_retrieval_slices(*args, groups={"q1": "en", "q2": "es"}, **limits)
            build.assert_not_called()

    def test_input_lists_are_not_modified(self):
        ranks, truth, groups = {"q1": ["a"]}, {"q1": ["a"]}, {"q1": "en"}
        before = json.dumps([ranks, truth, groups], sort_keys=True)
        audit_retrieval_slices(ranks, truth, groups=groups)
        self.assertEqual(json.dumps([ranks, truth, groups], sort_keys=True), before)

    def test_privacy_and_escaping(self):
        suite = audit_retrieval_slices({"PRIVATE QUERY": ["PRIVATE DOC"]}, {"PRIVATE QUERY": {"PRIVATE DOC"}}, groups={"PRIVATE QUERY": "<script>"}, k=1)
        html = render_suite(suite)
        for value in (json.dumps(suite.to_dict()), html):
            self.assertNotIn("PRIVATE QUERY", value)
            self.assertNotIn("PRIVATE DOC", value)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn('sandbox=""', suite._repr_html_())

    def test_slice_names_do_not_become_filesystem_paths(self):
        suite = audit_retrieval_slices({"q1": ["a"]}, {"q1": {"a"}}, slices={"../../outside": ["q1"]}, k=1)
        with tempfile.TemporaryDirectory() as tmp:
            paths = suite.save(Path(tmp) / "report")
            self.assertEqual({p.name for p in paths}, {"suite.json", "suite.html"})
            self.assertEqual(AuditSuite.load(paths[0]).metrics, suite.metrics)

    @unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas extra not installed")
    def test_tidy_dataframe_has_slice_metrics_and_sample_counts(self):
        frame = make_example().to_frame()
        self.assertEqual(list(frame.columns), ["report", "check", "metric", "value", "evaluated", "total", "unit"])
        row = frame[(frame["report"] == "spanish") & (frame["metric"] == "recall@1")].iloc[0]
        self.assertEqual(row["evaluated"], 2)
        self.assertEqual(row["value"], 1)
