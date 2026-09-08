"""JSON snapshots reject malformed structure and cannot overwrite source inputs."""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

from proofml import AuditPolicy, AuditReport, audit_retrieval, audit_text
from proofml.cli import main
from proofml.snapshots import report_from_dict


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.report = audit_retrieval({"q1": ["a"]}, {"q1": {"a"}}, k=1)
        self.path, _ = self.report.save(self.root / "report")

    def test_text_and_legacy_reports_load(self):
        report = audit_text(["same", "same"])
        path, _ = report.save(self.root / "text")
        self.assertEqual(AuditReport.load(path).to_dict(), report.to_dict())
        value = self.report.to_dict()
        for check in value["checks"]:
            check.pop("coverage")
        self.path.write_text(json.dumps(value))
        loaded = AuditReport.load(self.path)
        self.assertTrue(all(c.coverage is None for c in loaded.checks))

    def test_json_limits_and_nonfinite_rejection(self):
        with self.assertRaises(ValueError):
            AuditReport.load(self.path, max_bytes=2)
        for payload in ('{"schema_version":"1.0","schema_version":"1.0"}', '{"value":NaN}',
                        '[1,2]', '{"schema_version":"2.0"}', '[' * 2000 + ']' * 2000):
            with self.subTest(payload=payload[:60]):
                self.path.write_text(payload)
                with self.assertRaises(ValueError):
                    AuditReport.load(self.path)

    def test_invalid_snapshot_fields(self):
        def payload():
            return json.loads(json.dumps(self.report.to_dict()))
        cases = []
        for key, value in (("unknown", 1), ("checks", {}), ("config", []), ("tool_version", None), ("summary", {}), ("metrics", {})):
            record = payload()
            record[key] = value
            cases.append(record)
        record = payload(); record["datasets"]["evaluation"]["rows"] = True; cases.append(record)
        record = payload(); record["datasets"]["evaluation"]["sha256"] = "bad"; cases.append(record)
        record = payload(); record["checks"].append(record["checks"][0]); cases.append(record)
        record = payload(); record["checks"][0]["metrics"]["queries"] = float("inf"); cases.append(record)
        record = payload(); record["checks"][0]["coverage"]["evaluated"] = 99; cases.append(record)
        record = payload(); record["checks"][0]["findings"] = {}; cases.append(record)
        record = payload(); record["checks"][0].pop("status"); cases.append(record)
        for value in cases:
            with self.subTest(value=str(value)[:80]):
                self.path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    AuditReport.load(self.path)

    def test_gate_cli_passes_and_writes_artifacts(self):
        policy_path = AuditPolicy(min_coverage={"retrieval_ranking": 1}).save(self.root / "policy.json")
        output, junit = self.root / "decision.json", self.root / "decision.xml"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["gate", str(self.path), "--policy", str(policy_path), "--output", str(output), "--junit", str(junit)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.read_text())["status"], "passed")
        self.assertIn('failures="0"', junit.read_text())

    def test_gate_cli_failure_insufficiency_and_incompatibility_codes(self):
        for policy, code in ((AuditPolicy(min_metrics={"retrieval_ranking.recall@1": 2}), 1),
                             (AuditPolicy(min_evaluated={"retrieval_ranking": 2}), 2),
                             (AuditPolicy(max_drop={"retrieval_ranking.recall@1": 0}), 2)):
            policy_path = policy.save(self.root / "policy.json", overwrite=True)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["gate", str(self.path), "--policy", str(policy_path)]), code)
        baseline = audit_retrieval({"q1": ["a"]}, {"q1": {"b"}}, k=1)
        baseline_path, _ = baseline.save(self.root / "baseline")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["gate", str(self.path), "--policy", str(policy_path), "--baseline", str(baseline_path)]), 2)

    def test_cli_protects_inputs_and_existing_outputs(self):
        policy_path = AuditPolicy().save(self.root / "policy.json")
        before = self.path.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            for args in (("--output", str(self.path), "--overwrite"), ("--output", str(policy_path), "--overwrite"),
                         ("--output", str(self.root / "same"), "--junit", str(self.root / "same")),
                         ("--output", str(self.path))):
                self.assertEqual(main(["gate", str(self.path), "--policy", str(policy_path), *args]), 2)
        self.assertEqual(self.path.read_bytes(), before)

    def test_cli_invalid_policy_is_an_error_not_a_failed_assertion(self):
        path = self.root / "invalid.json"
        path.write_text('{"schema_version":"1.0","unexpected":1}')
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["gate", str(self.path), "--policy", str(path)]), 2)
