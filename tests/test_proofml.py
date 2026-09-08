"""Behavior and failure-mode tests; fixtures are synthetic and contain no PII."""
import contextlib
import csv
import importlib.util
import io
import json
import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from proofml import AuditConfig, CheckResult, Finding, audit
from proofml.checks import default_checks
from proofml.checks.drift import ks_distance
from proofml.checks.leakage import correlation
from proofml.cli import main
from proofml.data import load_dataset
from proofml.reporting import render_html, write_reports


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def csv(self, name, columns, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def codes(self, report):
        self.assertFalse([c for c in report.checks if c.status == "error"])
        return {f.code for f in report.findings}

    def result(self, report, check_id):
        return next(c for c in report.checks if c.check_id == check_id)

    def test_clean_classification_has_no_findings(self):
        rng = random.Random(42)
        train = self.csv("train.csv", ["id", "x", "target"], [(i, rng.random(), i % 2) for i in range(300)])
        test = self.csv("test.csv", ["id", "x", "target"], [(i, rng.random(), i % 2) for i in range(300, 600)])
        report = audit(train, test=test, config=AuditConfig(target="target", entity_id="id", require_disjoint_entities=True))
        self.assertEqual(self.codes(report), set())
        self.assertEqual(self.result(report, "feature_availability").status, "skipped")

    def test_target_copy_is_suspicious_not_certified_leakage(self):
        path = self.csv("data.csv", ["copy", "target"], [(i % 2, i % 2) for i in range(40)])
        report = audit(path, config=AuditConfig(target="target"))
        finding = next(f for f in report.findings if f.code == "target_copy")
        self.assertEqual(finding.confidence, "suspicious")
        self.assertEqual(finding.evidence["agreement"], 1.0)

    def test_declared_unavailable_feature_is_critical(self):
        path = self.csv("data.csv", ["post_event", "y"], [(0, 0), (1, 1)])
        report = audit(path, config=AuditConfig(target="y", unavailable_features=("post_event",)))
        self.assertIn("unavailable_feature", self.codes(report))
        self.assertEqual(next(f for f in report.findings if f.code == "unavailable_feature").severity, "critical")

    def test_regression_linear_proxy(self):
        path = self.csv("data.csv", ["x", "y"], [(i * 3 + 7, i) for i in range(50)])
        report = audit(path, config=AuditConfig(target="y", task="regression"))
        self.assertIn("high_target_correlation", self.codes(report))

    def test_multiclass_codes_do_not_use_pearson(self):
        path = self.csv("data.csv", ["x", "y"], [(i % 3 + 10, i % 3) for i in range(60)])
        self.assertNotIn("high_target_correlation", self.codes(audit(path, config=AuditConfig(target="y"))))

    def test_invalid_regression_labels(self):
        path = self.csv("data.csv", ["x", "y"], [(1, "nan"), (2, "inf"), (3, "oops")])
        self.assertIn("invalid_regression_target", self.codes(audit(path, config=AuditConfig(target="y", task="regression"))))

    def test_extreme_values_do_not_crash_correlation(self):
        self.assertAlmostEqual(correlation([1e308, -1e308, 0.0], [1e308, -1e308, 0.0]), 1.0)
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3]))

    def test_small_sample_is_skipped(self):
        path = self.csv("data.csv", ["x", "y"], [(0, 0), (1, 1)])
        self.assertEqual(self.result(audit(path, config=AuditConfig(target="y")), "target_association").status, "skipped")

    def test_missing_and_imbalanced_labels(self):
        path = self.csv("data.csv", ["x", "y"], [(i, "yes" if i == 1 else "no") for i in range(40)] + [(41, "")])
        self.assertTrue({"missing_target", "class_imbalance"} <= self.codes(audit(path, config=AuditConfig(target="y"))))

    def test_missing_and_mixed_numeric(self):
        path = self.csv("data.csv", ["empty", "mixed"], [("", i if i < 9 else "unknown") for i in range(10)])
        self.assertTrue({"missing_values", "mixed_numeric"} <= self.codes(audit(path)))

    def test_overlap_aligns_reordered_columns_without_test_labels(self):
        train = self.csv("train.csv", ["a", "b", "y"], [(1, 2, 0), (3, 4, 1)])
        test = self.csv("test.csv", ["b", "a"], [(2, 1)])
        report = audit(train, test=test, config=AuditConfig(target="y"))
        self.assertIn("train_test_overlap", self.codes(report))
        self.assertEqual(self.result(report, "split_schema").status, "passed")

    def test_schema_mismatch_prevents_partial_overlap_claim(self):
        train = self.csv("train.csv", ["a", "b"], [(1, 2)])
        test = self.csv("test.csv", ["a", "c"], [(1, 2)])
        report = audit(train, test=test)
        self.assertIn("schema_mismatch", self.codes(report))
        self.assertEqual(self.result(report, "split_overlap").status, "skipped")

    def test_entity_overlap_requires_context_unless_declared(self):
        train = self.csv("train.csv", ["id", "x"], [(1, 2)])
        test = self.csv("test.csv", ["id", "x"], [(1, 3)])
        for strict, confidence in ((False, "needs_context"), (True, "confirmed")):
            report = audit(train, test=test, config=AuditConfig(entity_id="id", require_disjoint_entities=strict))
            self.assertEqual(next(f for f in report.findings if f.code == "shared_entities").confidence, confidence)

    def test_missing_entity_ids_do_not_count_as_shared_entities(self):
        train = self.csv("train.csv", ["id", "x"], [("", 2)])
        test = self.csv("test.csv", ["id", "x"], [("", 3)])
        self.assertNotIn("shared_entities", self.codes(audit(train, test=test, config=AuditConfig(entity_id="id"))))

    def test_temporal_order_normalizes_timezones(self):
        train = self.csv("train.csv", ["time", "x"], [("2026-01-01T10:00:00+05:30", 1)])
        test = self.csv("test.csv", ["time", "x"], [("2026-01-01T05:00:00Z", 2)])
        config = AuditConfig(time_column="time", expect_temporal_split=True)
        self.assertEqual(self.result(audit(train, test=test, config=config), "temporal_order").status, "passed")

    def test_temporal_equality_violates_future_only_holdout(self):
        train = self.csv("train.csv", ["time"], [("2026-01-01",)])
        report = audit(train, test=train, config=AuditConfig(time_column="time", expect_temporal_split=True))
        self.assertIn("temporal_overlap", self.codes(report))

    def test_invalid_time_is_a_finding(self):
        train = self.csv("train.csv", ["time"], [("not-a-date",)])
        self.assertIn("invalid_timestamp", self.codes(audit(train, test=train, config=AuditConfig(time_column="time", expect_temporal_split=True))))

    def test_numeric_drift_and_categorical_drift(self):
        train = self.csv("train.csv", ["numeric", "category"], [(i, "old") for i in range(40)])
        test = self.csv("test.csv", ["numeric", "category"], [(i + 100, "new") for i in range(40)])
        report = audit(train, test=test)
        shifted = [f for f in report.findings if f.code == "feature_distribution_shift"]
        self.assertEqual(len(shifted), 2)
        self.assertEqual({f.evidence["distance"] for f in shifted}, {1.0})

    def test_ks_handles_ties(self):
        self.assertEqual(ks_distance([1, 1, 2], [1, 1, 2]), 0)
        self.assertEqual(ks_distance([1, 1], [2, 2]), 1)

    def test_duplicate_and_empty_headers_rejected(self):
        for headers in (["x", "x"], ["x", ""], ["x", " x "]):
            with self.subTest(headers=headers):
                path = self.csv("bad.csv", headers, [(1, 2)])
                with self.assertRaises(ValueError):
                    audit(path)

    def test_bad_row_width_and_empty_data_rejected(self):
        for rows in ([(1,)], []):
            path = self.csv("bad.csv", ["a", "b"], rows)
            with self.assertRaises(ValueError):
                audit(path)

    def test_limits_never_silently_sample(self):
        path = self.csv("data.csv", ["x"], [(1,), (2,)])
        for config in (AuditConfig(max_rows=1), AuditConfig(max_bytes=1)):
            with self.assertRaises(ValueError):
                audit(path, config=config)

    def test_literal_na_is_preserved(self):
        path = self.csv("data.csv", ["region"], [("NA",), ("EU",)])
        self.assertEqual(load_dataset(path, AuditConfig()).column("region"), ("NA", "EU"))

    def test_bad_config_values_rejected(self):
        for args in ({"task": "forecast"}, {"missing_threshold": float("nan")}, {"min_samples": True},
                     {"disabled_checks": "target_health"}, {"require_disjoint_entities": True}, {"max_rows": 1.5}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                AuditConfig(**args)

    def test_duplicate_and_unknown_config_fields_rejected(self):
        path = self.root / "config.json"
        for text in ('{"targte":"y"}', '{"task":"classification","task":"regression"}', '[]'):
            path.write_text(text)
            with self.assertRaises(ValueError):
                AuditConfig.from_file(path)

    def test_missing_declared_column_fails_early(self):
        path = self.csv("data.csv", ["x"], [(1,)])
        with self.assertRaises(ValueError):
            audit(path, config=AuditConfig(target="typo"))

    def test_unknown_disabled_check_rejected(self):
        path = self.csv("data.csv", ["x"], [(1,)])
        with self.assertRaises(ValueError):
            audit(path, config=AuditConfig(disabled_checks=("typo",)))

    def test_plugin_failure_is_reported_without_exception_data(self):
        class Broken:
            id = "broken"
            def run(self, ctx):
                raise RuntimeError("private@example.com")
        path = self.csv("data.csv", ["x"], [(1,)])
        report = audit(path, checks=[*default_checks(), Broken()])
        self.assertEqual(self.result(report, "broken").status, "error")
        self.assertNotIn("private@example.com", json.dumps(report.to_dict()))
        self.assertEqual(len(report.checks), 10)

    def test_duplicate_plugin_ids_rejected(self):
        path = self.csv("data.csv", ["x"], [(1,)])
        with self.assertRaises(ValueError):
            audit(path, checks=[default_checks()[0], default_checks()[0]])

    def test_nonfinite_plugin_evidence_is_error(self):
        class Invalid:
            id = "invalid"
            def run(self, ctx):
                return CheckResult.complete(self.id, [Finding("bad", "low", "confirmed", "Bad", "", "", evidence={"x": float("nan")})])
        path = self.csv("data.csv", ["x"], [(1,)])
        self.assertEqual(audit(path, checks=[Invalid()]).checks[0].status, "error")

    def test_html_escapes_untrusted_column_names(self):
        path = self.csv("data.csv", ['<script>alert("x")</script>'], [("",)])
        html = render_html(audit(path))
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Content-Security-Policy", html)

    def test_raw_values_not_in_reports(self):
        path = self.csv("data.csv", ["email"], [("private@example.com",), ("private@example.com",)])
        report = audit(path)
        self.assertNotIn("private@example.com", json.dumps(report.to_dict()))
        self.assertNotIn("private@example.com", render_html(report))

    def test_report_no_clobber_and_explicit_overwrite(self):
        path = self.csv("data.csv", ["x"], [(1,)])
        report = audit(path)
        output = self.root / "reports"
        paths = write_reports(report, output)
        original = paths[0].read_bytes()
        with self.assertRaises(FileExistsError):
            write_reports(report, output)
        self.assertEqual(original, paths[0].read_bytes())
        write_reports(report, output, overwrite=True)
        self.assertEqual(json.loads(paths[0].read_text())["schema_version"], "1.0")

    def test_input_unchanged_and_findings_reproducible(self):
        path = self.csv("data.csv", ["x"], [(1,), (1,)])
        before = path.read_bytes()
        one, two = audit(path), audit(path)
        self.assertEqual(one.checks, two.checks)
        self.assertEqual(one.datasets, two.datasets)
        self.assertEqual(before, path.read_bytes())

    def test_cli_success_failure_and_input_exit_codes(self):
        path = self.csv("data.csv", ["x", "y"], [(1, 0), (2, 1)])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["audit", str(path), "--target", "y", "--output", str(self.root / "ok")]), 0)
            self.assertEqual(main(["demo", "--output", str(self.root / "demo")]), 1)
            self.assertEqual(main(["audit", str(path), "--target", "typo", "--output", str(self.root / "bad")]), 2)
            self.assertEqual(main(["demo", "--output", str(self.root / "demo"), "--overwrite", "--fail-on", "none"]), 0)

    def test_bundled_demos_expected_findings(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["demo", "--output", str(self.root / "faulty")]), 1)
            self.assertEqual(main(["demo", "--clean", "--output", str(self.root / "clean")]), 0)
        faulty = json.loads((self.root / "faulty/report.json").read_text())
        clean = json.loads((self.root / "clean/report.json").read_text())
        codes = {f["code"] for c in faulty["checks"] for f in c["findings"]}
        self.assertEqual(codes, {"unavailable_feature", "shared_entities", "target_copy", "temporal_overlap",
                                 "train_test_overlap", "duplicate_rows", "feature_distribution_shift", "missing_values"})
        self.assertEqual(clean["summary"]["findings"], 0)
        self.assertEqual(clean["summary"]["check_counts"]["skipped"], 1)

    def test_cli_check_errors_override_severity_gating(self):
        path = self.csv("data.csv", ["x"], [(1,)])
        failed = replace(audit(path), checks=(CheckResult("broken", "error", reason="Failed"),))
        with patch("proofml.cli.audit", return_value=failed), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["audit", str(path), "--fail-on", "none", "--output", str(self.root / "error")]), 2)

    def test_cli_refuses_config_output_collision(self):
        path = self.csv("data.csv", ["x"], [(1,)])
        config = self.root / "report.json"
        config.write_text("{}")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["audit", str(path), "--config", str(config), "--output", str(self.root), "--overwrite"]), 2)
        self.assertEqual(config.read_text(), "{}")

    def test_missing_all_candidate_pairs_skips_association(self):
        path = self.csv("data.csv", ["x", "y"], [("", i % 2) for i in range(40)])
        self.assertEqual(self.result(audit(path, config=AuditConfig(target="y")), "target_association").status, "skipped")

    def test_wrong_plugin_result_id_becomes_error(self):
        class Wrong:
            id = "expected"
            def run(self, ctx):
                return CheckResult.complete("different", [])
        path = self.csv("data.csv", ["x"], [(1,)])
        self.assertEqual(audit(path, checks=[Wrong()]).checks[0].status, "error")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow") and importlib.util.find_spec("pandas"), "optional Parquet dependencies not installed")
    def test_parquet_adapter_and_row_limit(self):
        import pandas as pd
        path = self.root / "data.parquet"
        pd.DataFrame({"x": [1.0, None, 3.0], "y": [0, 1, 0]}).to_parquet(path)
        self.assertEqual(load_dataset(path, AuditConfig()).column("x"), ("1.0", "", "3.0"))
        self.assertIn("missing_values", self.codes(audit(path, config=AuditConfig(target="y"))))
        with self.assertRaises(ValueError):
            audit(path, config=AuditConfig(max_rows=2))


if __name__ == "__main__":
    unittest.main()
