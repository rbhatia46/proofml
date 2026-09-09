"""Reusable multi-report policies, storage, CLI behavior, and error isolation."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree as ET

from proofml import AuditFailed, AuditPolicy, AuditSuite, CheckResult, SuitePolicy, audit, audit_retrieval, audit_text
from proofml.cli import main


def report():
    return audit_retrieval({"q1": ["a"]}, {"q1": {"a"}}, k=1)


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suite = AuditSuite({"overall": report(), "cohort": report()})

    def test_required_report_and_override_typos_fail_closed(self):
        for policy in (SuitePolicy(require_reports=("missing",)), SuitePolicy(overrides={"typo": AuditPolicy()})):
            result = policy.evaluate(self.suite)
            self.assertEqual(result.status, "insufficient")
            self.assertTrue(any(i.code == "required_report_missing" for i in result.issues))
            with self.assertRaises(AuditFailed):
                result.raise_for_issues()

    def test_default_applies_to_every_report_and_override_replaces(self):
        policy = SuitePolicy(default=AuditPolicy(min_evaluated={"retrieval_ranking": 2}),
                             overrides={"cohort": AuditPolicy(min_evaluated={"retrieval_ranking": 1})})
        result = policy.evaluate(self.suite)
        self.assertEqual(result.results["overall"].status, "insufficient")
        self.assertEqual(result.results["cohort"].status, "passed")

    def test_new_candidate_report_requires_baseline_counterpart(self):
        baseline = AuditSuite({"overall": report()})
        result = SuitePolicy().evaluate(self.suite, baseline=baseline)
        self.assertEqual(result.status, "insufficient")
        self.assertTrue(any(i.code == "baseline_report_missing" for i in result.issues))

    def test_regression_policy_without_any_baseline_blocks(self):
        policy = SuitePolicy(default=AuditPolicy(max_drop={"retrieval_ranking.recall@1": 0.01}))
        result = policy.evaluate(self.suite)
        self.assertEqual(result.status, "insufficient")
        self.assertTrue(all(r.status == "insufficient" for r in result.results.values()))

    def test_suite_and_single_report_types_are_not_silently_mixed(self):
        with self.assertRaises(TypeError):
            SuitePolicy().evaluate(report())
        with self.assertRaises(TypeError):
            SuitePolicy().evaluate(self.suite, baseline=report())
        with self.assertRaises(TypeError):
            AuditPolicy().evaluate(self.suite)

    def test_generic_suite_supports_tabular_and_text_policies(self):
        path = self.root / "data.csv"
        path.write_text("x\n1\n2\n")
        suite = AuditSuite({"table": audit(path), "text": audit_text(["one", "two"])})
        policy = SuitePolicy(default=AuditPolicy(min_rows={"train": 2}),
                             overrides={"table": AuditPolicy(require_checks=("data_quality",), min_rows={"train": 2})},
                             require_reports=("table", "text"))
        self.assertTrue(policy.enforce(suite).passed)

    def test_errored_slice_cannot_be_ignored_by_severity(self):
        broken = replace(report(), checks=(CheckResult("broken", "error", reason="check failed"),))
        suite = AuditSuite({"overall": report(), "broken": broken})
        result = SuitePolicy(default=AuditPolicy(severity="critical")).evaluate(suite)
        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.results["overall"].status, "passed")

    def test_input_name_maps_are_copied_and_read_only(self):
        source = {"one": report()}
        suite = AuditSuite(source)
        source.clear()
        self.assertIn("one", suite.reports)
        with self.assertRaises(TypeError):
            suite.reports["two"] = report()
        overrides = {"one": AuditPolicy()}
        policy = SuitePolicy(overrides=overrides)
        overrides.clear()
        self.assertIn("one", policy.overrides)
        with self.assertRaises(TypeError):
            policy.overrides["two"] = AuditPolicy()

    def test_suite_validation_and_bounds(self):
        for values in ({}, {"": report()}, {"x": self.suite}, {"x": {}}, {"x" * 257: report()},
                       {str(i): report() for i in range(1001)}):
            with self.subTest(count=len(values)), self.assertRaises((ValueError, TypeError)):
                AuditSuite(values)

    def test_suite_policy_validation(self):
        invalid = ({"default": {}}, {"overrides": []}, {"overrides": {"x": self.suite}}, {"overrides": {"": AuditPolicy()}},
                   {"require_reports": "a"}, {"require_reports": ["a", "a"]}, {"require_reports": [""]})
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises((TypeError, ValueError)):
                SuitePolicy(**kwargs)

    def test_suite_roundtrip_and_comparison_after_load(self):
        paths = self.suite.save(self.root / "suite")
        loaded = AuditSuite.load(paths[0])
        self.assertEqual(loaded.metrics, self.suite.metrics)
        self.assertEqual(loaded.summary, self.suite.summary)
        self.assertTrue(SuitePolicy().evaluate(self.suite, baseline=loaded).passed)
        self.assertEqual(AuditSuite.from_dict(self.suite.to_dict()).metrics, self.suite.metrics)
        with self.assertRaises(FileExistsError):
            self.suite.save(self.root / "suite")
        self.suite.save(self.root / "suite", overwrite=True)

    def test_nested_and_malformed_suite_snapshots_are_rejected(self):
        path = self.root / "bad.json"
        for value in ({"schema_version": "1.0", "kind": "audit_suite", "reports": {}},
                      {"schema_version": "1.0", "kind": "audit_suite", "reports": {"nested": self.suite.to_dict()}},
                      {**self.suite.to_dict(), "extra": True}, {**self.suite.to_dict(), "schema_version": "2.0"}):
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                AuditSuite.load(path)
        path.write_text(json.dumps(self.suite.to_dict()))
        with self.assertRaises(ValueError):
            AuditSuite.load(path, max_bytes=10)

    def test_policy_roundtrip_and_invalid_nested_rule(self):
        policy = SuitePolicy(default=AuditPolicy(min_coverage={"retrieval_ranking": 1}),
                             overrides={"cohort": AuditPolicy(min_evaluated={"retrieval_ranking": 1})}, require_reports=("overall",))
        path = policy.save(self.root / "policy.json")
        loaded = SuitePolicy.from_file(path)
        self.assertEqual(loaded.to_dict(), policy.to_dict())
        self.assertTrue(loaded.enforce(self.suite).passed)
        for value in ({**policy.to_dict(), "typo": 1}, {**policy.to_dict(), "default": {"schema_version": "1.0", "min_covrage": {"x": 1}}},
                      {**policy.to_dict(), "overrides": {"x": policy.to_dict()}}):
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                SuitePolicy.from_file(path)

    def test_junit_has_one_case_per_report_with_numeric_evidence(self):
        policy = SuitePolicy(overrides={"cohort": AuditPolicy(min_metrics={"retrieval_ranking.recall@1": 2})})
        decision = policy.evaluate(self.suite)
        xml = ET.fromstring(decision.to_junit())
        self.assertEqual(xml.attrib, {"name": "proofml.suite", "tests": "2", "failures": "1", "errors": "0"})
        failure = next(c for c in xml if c.attrib["name"] == "cohort").find("failure")
        self.assertIn('"observed": 1.0', failure.text)
        self.assertIn('"limit": 2', failure.text)
        decision.save(self.root / "decision.json")
        decision.save_junit(self.root / "decision.xml")

    def test_junit_contract_error_is_separate(self):
        decision = SuitePolicy(require_reports=("missing",)).evaluate(self.suite)
        xml = ET.fromstring(decision.to_junit())
        self.assertEqual(xml.attrib["tests"], "3")
        self.assertEqual(xml.attrib["errors"], "1")

    def test_xml_controls_are_printed_safely(self):
        suite = AuditSuite({"cohort\x00name": report()})
        xml = SuitePolicy().evaluate(suite).to_junit()
        self.assertIn("\\u0000", xml)
        ET.fromstring(xml)
        ET.fromstring(AuditPolicy(name="bad\ud800").evaluate(report()).to_junit())

    def test_suite_cli_and_baseline(self):
        suite_path, _ = self.suite.save(self.root / "suite")
        policy_path = SuitePolicy(default=AuditPolicy(max_drop={"retrieval_ranking.recall@1": 0})).save(self.root / "policy.json")
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            code = main(["gate", str(suite_path), "--suite", "--baseline", str(suite_path), "--policy", str(policy_path),
                         "--output", str(self.root / "decision.json"), "--junit", str(self.root / "decision.xml")])
        self.assertEqual(code, 0)
        self.assertIn("[cohort] passed", stdout.getvalue())
        self.assertEqual(json.loads((self.root / "decision.json").read_text())["status"], "passed")

    def test_suite_cli_failed_insufficient_and_input_protection(self):
        suite_path, _ = self.suite.save(self.root / "suite")
        cases = ((SuitePolicy(default=AuditPolicy(min_metrics={"retrieval_ranking.recall@1": 2})), 1),
                 (SuitePolicy(require_reports=("missing",)), 2))
        for policy, expected in cases:
            policy_path = policy.save(self.root / "policy.json", overwrite=True)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["gate", str(suite_path), "--suite", "--policy", str(policy_path)]), expected)
        before = suite_path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["gate", str(suite_path), "--suite", "--policy", str(policy_path), "--output", str(suite_path), "--overwrite"]), 2)
        self.assertEqual(suite_path.read_bytes(), before)
