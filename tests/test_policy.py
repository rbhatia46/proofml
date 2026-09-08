"""Release decisions must fail closed on weak, missing, or incompatible evidence."""
from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from proofml import AuditFailed, AuditPolicy, AuditReport, CheckResult, Coverage, audit, audit_retrieval, audit_text
from proofml.retrieval import default_retrieval_checks


def retrieval(rank=None, truth=None, **options):
    return audit_retrieval(rank if rank is not None else {"q1": ["a"]},
                           truth if truth is not None else {"q1": {"a"}}, k=1, **options)


class PolicyTests(unittest.TestCase):
    def test_one_of_one_hundred_judged_is_insufficient(self):
        report = retrieval({str(i): ["a"] for i in range(100)}, {str(i): {"a"} if i == 0 else set() for i in range(100)})
        report.raise_for_issues(require_checks=("retrieval_ranking",))  # Old API remains compatible.
        policy = AuditPolicy(min_coverage={"retrieval_ranking": 0.95}, min_evaluated={"retrieval_ranking": 20})
        result = policy.evaluate(report)
        self.assertEqual(result.status, "insufficient")
        self.assertEqual({i.code for i in result.issues}, {"insufficient_coverage", "insufficient_evaluated"})
        self.assertEqual(report.metrics["retrieval_ranking"]["recall@1"], 1)
        with self.assertRaises(AuditFailed) as caught:
            policy.enforce(report)
        self.assertEqual(caught.exception.decision.status, "insufficient")

    def test_complete_coverage_passes_at_equality(self):
        policy = AuditPolicy(require_checks=["retrieval_ranking"], min_coverage={"retrieval_ranking": 1},
                             min_evaluated={"retrieval_ranking": 1}, min_rows={"evaluation": 1},
                             min_metrics={"retrieval_ranking.recall@1": 1}, max_metrics={"retrieval_inputs.empty_result_queries": 0})
        self.assertTrue(policy.enforce(retrieval()).passed)

    def test_no_judgments_and_skipped_checks_block(self):
        policy = AuditPolicy(require_checks=("retrieval_ranking",), min_coverage={"retrieval_ranking": 0.5})
        result = policy.evaluate(retrieval(truth={"q1": set()}))
        self.assertEqual(result.status, "insufficient")
        self.assertIn("required_check_unassessed", [i.code for i in result.issues])

    def test_missing_declared_coverage_is_not_inferred_from_metrics(self):
        report = retrieval()
        checks = tuple(replace(c, coverage=None) for c in report.checks)
        result = AuditPolicy(min_coverage={"retrieval_ranking": 0.1}).evaluate(replace(report, checks=checks))
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.issues[0].code, "coverage_unavailable")

    def test_missing_metric_and_typo_cannot_pass(self):
        for rules in ({"min_metrics": {"retrieval_ranking.recal@1": 0}}, {"max_metrics": {"no.check": 100}},
                      {"require_checks": ("typo",)}, {"min_rows": {"missing": 1}}):
            with self.subTest(rules=rules):
                self.assertEqual(AuditPolicy(**rules).evaluate(retrieval()).status, "insufficient")

    def test_check_errors_block_even_at_critical_severity(self):
        report = replace(retrieval(), checks=(CheckResult("broken", "error", reason="Error"),))
        result = AuditPolicy(severity="critical").evaluate(report)
        self.assertEqual(result.status, "insufficient")

    def test_severity_gate_is_not_disabled_by_metric_rules(self):
        report = retrieval(rank={"q1": ["a", "a"]})
        self.assertEqual(AuditPolicy().evaluate(report).status, "failed")
        self.assertTrue(AuditPolicy(severity="critical").evaluate(report).passed)

    def test_absolute_minimum_and_maximum(self):
        report = retrieval(rank={"q1": ["b"]})
        result = AuditPolicy(min_metrics={"retrieval_ranking.recall@1": 0.8},
                             max_metrics={"retrieval_inputs.queries": 0}).evaluate(report)
        self.assertEqual(result.status, "failed")
        self.assertEqual(len(result.issues), 2)

    def test_regression_requires_baseline(self):
        result = AuditPolicy(max_drop={"retrieval_ranking.recall@1": 0.1}).evaluate(retrieval())
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.issues[0].code, "baseline_required")

    def test_regression_and_improvement(self):
        good, bad = retrieval(), retrieval(rank={"q1": ["x"]})
        policy = AuditPolicy(max_drop={"retrieval_ranking.recall@1": 0.1})
        result = policy.evaluate(bad, baseline=good)
        self.assertEqual(result.status, "failed")
        issue = next(i for i in result.issues if i.code == "metric_regression")
        self.assertEqual(issue.evidence["deterioration"], 1)
        self.assertTrue(policy.evaluate(good, baseline=bad).passed)

    def test_regression_tolerance_equality_and_lower_is_better(self):
        good = retrieval()
        empty = retrieval(rank={"q1": []})
        self.assertTrue(AuditPolicy(max_drop={"retrieval_ranking.recall@1": 1}).evaluate(empty, baseline=good).passed)
        result = AuditPolicy(max_increase={"retrieval_inputs.empty_result_queries": 0}).evaluate(empty, baseline=good)
        self.assertEqual(result.status, "failed")

    def test_baseline_evidence_must_meet_same_coverage_contract(self):
        report = retrieval(truth={"q1": set()})
        result = AuditPolicy(min_coverage={"retrieval_ranking": 1}, max_drop={"retrieval_ranking.recall@1": 0}).evaluate(report, baseline=report)
        self.assertEqual(result.status, "insufficient")
        self.assertTrue(any(i.evidence.get("role") == "baseline" for i in result.issues))

    def test_baseline_quality_minimum_does_not_block_an_improvement(self):
        result = AuditPolicy(min_metrics={"retrieval_ranking.recall@1": 0.9}, max_drop={"retrieval_ranking.recall@1": 0}).evaluate(
            retrieval(), baseline=retrieval(rank={"q1": ["x"]}))
        self.assertTrue(result.passed)

    def test_incompatible_baseline_is_not_a_zero_regression(self):
        result = AuditPolicy(max_drop={"retrieval_ranking.recall@1": 0}).evaluate(retrieval(), baseline=retrieval(truth={"q1": {"b"}}))
        self.assertEqual(result.status, "incompatible")
        self.assertEqual(result.comparison.changes, ())

    def test_policies_apply_to_tabular_and_text_reports(self):
        text = audit_text(["one", "two"])
        self.assertTrue(AuditPolicy(require_checks=("text_quality",), min_rows={"train": 2}, min_coverage={"text_quality": 1}).evaluate(text).passed)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("x\n1\n2\n")
            report = audit(path)
            self.assertTrue(AuditPolicy(min_rows={"train": 2}, min_coverage={"data_quality": 1}).evaluate(report).passed)
            self.assertEqual(AuditPolicy(min_rows={"train": 3}).evaluate(report).status, "insufficient")

    def test_policy_mapping_is_copied_and_read_only(self):
        settings = {"train": 10}
        policy = AuditPolicy(min_rows=settings)
        settings["train"] = 1
        self.assertEqual(policy.min_rows["train"], 10)
        with self.assertRaises(TypeError):
            policy.min_rows["train"] = 0

    def test_decision_snapshots_inputs(self):
        report = retrieval()
        result = AuditPolicy().evaluate(report)
        report.datasets["evaluation"]["rows"] = 999
        report.checks[-1].metrics["recall@1"] = 0
        self.assertEqual(result.report.datasets["evaluation"]["rows"], 1)
        self.assertEqual(result.report.metrics["retrieval_ranking"]["recall@1"], 1)

    def test_policy_validation(self):
        invalid = ({"severity": "none"}, {"name": ""}, {"min_rows": {"train": True}}, {"min_evaluated": {"x": 0}},
                   {"min_coverage": {"x": 0}}, {"min_coverage": {"x": 1.1}}, {"min_metrics": {"x.m": float("nan")}},
                   {"max_drop": {"x.m": -1}}, {"max_increase": {"x.m": float("inf")}}, {"min_metrics": {"recall": 1}},
                   {"require_checks": "a"}, {"require_checks": ["a", "a"]}, {"min_rows": []},
                   {"min_metrics": {"a.m": 1}, "max_metrics": {"a.m": 0}}, {"max_drop": {"a.m": 0}, "max_increase": {"a.m": 0}})
        for options in invalid:
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                AuditPolicy(**options)

    def test_policy_roundtrip_and_overwrite_protection(self):
        policy = AuditPolicy(require_checks=("retrieval_ranking",), min_coverage={"retrieval_ranking": 0.9})
        with tempfile.TemporaryDirectory() as tmp:
            path = policy.save(Path(tmp) / "policy.json")
            self.assertEqual(AuditPolicy.from_file(path).to_dict(), policy.to_dict())
            with self.assertRaises(FileExistsError):
                policy.save(path)
            policy.save(path, overwrite=True)
            for payload in ('{"schema_version":"1.0","unknown":1}', '{"schema_version":"1.0","name":"a","name":"b"}',
                            '{"schema_version":"2.0"}', '{"schema_version":"1.0","max_metrics":{"x.y":NaN}}'):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    AuditPolicy.from_file(path)

    def test_junit_and_json_for_each_outcome(self):
        decisions = (AuditPolicy().evaluate(retrieval()),
                     AuditPolicy(min_metrics={"retrieval_ranking.recall@1": 2}).evaluate(retrieval()),
                     AuditPolicy(min_evaluated={"retrieval_ranking": 2}).evaluate(retrieval()),
                     AuditPolicy().evaluate(retrieval(), baseline=retrieval(truth={"q1": {"b"}})))
        self.assertEqual([r.status for r in decisions], ["passed", "failed", "insufficient", "incompatible"])
        with tempfile.TemporaryDirectory() as tmp:
            for i, result in enumerate(decisions):
                xml = ET.fromstring(result.to_junit())
                self.assertEqual(xml.attrib["failures"], str(int(result.status == "failed")))
                self.assertEqual(xml.attrib["errors"], str(int(result.status in {"insufficient", "incompatible"})))
                path = result.save(Path(tmp) / f"{i}.json")
                self.assertEqual(json.loads(path.read_text())["status"], result.status)
                result.save_junit(Path(tmp) / f"{i}.xml")

    def test_no_private_query_or_document_ids_in_decisions(self):
        report = retrieval({"PRIVATE QUERY": ["PRIVATE DOCUMENT"]}, {"PRIVATE QUERY": {"PRIVATE DOCUMENT"}})
        result = AuditPolicy(min_evaluated={"retrieval_ranking": 2}, name="<release>").evaluate(report)
        for value in (json.dumps(result.to_dict()), result.to_junit()):
            self.assertNotIn("PRIVATE QUERY", value)
            self.assertNotIn("PRIVATE DOCUMENT", value)
        self.assertIn("&lt;release&gt;", result.to_junit())

    def test_plugin_coverage_contract(self):
        class Custom:
            id = "custom"
            def run(self, ctx):
                return CheckResult.complete(self.id, [], coverage=Coverage(10, 5, "examples"), metrics={"score": 0.5})
        report = retrieval(checks=[*default_retrieval_checks(), Custom()])
        self.assertTrue(AuditPolicy(min_coverage={"custom": 0.5}, min_metrics={"custom.score": 0.5}).evaluate(report).passed)

    def test_coverage_validation(self):
        for total, evaluated in ((1, 2), (-1, 0), (True, 1), (2, 0.5)):
            with self.subTest(total=total, evaluated=evaluated), self.assertRaises(ValueError):
                Coverage(total, evaluated)
        with self.assertRaises(ValueError):
            CheckResult("x", "skipped", reason="not run", coverage=Coverage(1, 1))
        self.assertIsNone(Coverage(0, 0).fraction)
