"""The short examples users copy must work against the installed public API."""
import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from proofml import AuditConfig, AuditFailed, CheckResult, audit, __version__
from proofml.cli import main


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data.csv"
        self.data.write_text("feature,label\n1,0\n2,1\n3,0\n")

    def test_short_file_api_and_save(self):
        report = audit(self.data, target="label")
        paths = report.save(self.root / "report")
        self.assertEqual(len(paths), 2)
        self.assertEqual(json.loads(paths[0].read_text())["tool_version"], __version__)
        self.assertIn("skipped", str(report))

    def test_keywords_override_config(self):
        report = audit(self.data, target="label", task="regression", config=AuditConfig(target="wrong"), min_samples=3)
        self.assertEqual(report.config["target"], "label")
        self.assertEqual(report.config["task"], "regression")
        self.assertEqual(report.config["min_samples"], 3)

    def test_unknown_keyword_fails_early(self):
        with self.assertRaises(TypeError):
            audit(self.data, targte="label")

    def test_positional_test_input(self):
        report = audit(self.data, self.data, target="label")
        self.assertIn("train_test_overlap", [f.code for f in report.findings])

    def test_pipeline_gate_thresholds_and_attached_report(self):
        report = audit(self.data, self.data, target="label")
        with self.assertRaises(AuditFailed) as caught:
            report.raise_for_issues()
        self.assertIs(caught.exception.report, report)
        report.raise_for_issues("critical")

    def test_pipeline_gate_requires_coverage(self):
        report = audit(self.data)
        report.raise_for_issues()
        report.raise_for_issues(require_checks=("data_quality",))
        for required in ("split_overlap", "typo"):
            with self.assertRaises(AuditFailed):
                report.raise_for_issues(require_checks=(required,))
        with self.assertRaises(TypeError):
            report.raise_for_issues(require_checks="data_quality")

    def test_check_errors_always_fail_gate(self):
        report = replace(audit(self.data), checks=(CheckResult("broken", "error", reason="Failure"),))
        with self.assertRaises(AuditFailed):
            report.raise_for_issues("critical")
        with self.assertRaises(ValueError):
            report.raise_for_issues("none")

    def test_notebook_view_is_isolated_and_escaped(self):
        self.data.write_text('<script>,label\n1,0\n1,1\n')
        rendered = audit(self.data)._repr_html_()
        self.assertTrue(rendered.startswith("<iframe"))
        self.assertNotIn("<script>", rendered)
        self.assertIn('sandbox=""', rendered)

    def test_version_agrees_with_cli_and_reports(self):
        with contextlib.redirect_stdout(io.StringIO()) as captured, self.assertRaises(SystemExit):
            main(["--version"])
        self.assertIn(__version__, captured.getvalue())
        self.assertEqual(audit(self.data).tool_version, __version__)


@unittest.skipUnless(importlib.util.find_spec("pandas"), "optional pandas not installed")
class DataFrameApiTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        self.pd = pd

    def test_dataframe_is_not_mutated_and_index_is_ignored(self):
        df = self.pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}, index=[5, 7, 8])
        before = df.copy(deep=True)
        report = audit(df, target="y")
        self.pd.testing.assert_frame_equal(df, before)
        self.assertEqual(report.datasets["train"]["columns"], ["x", "y"])
        self.assertEqual(report.datasets["train"]["format"], "dataframe")

    def test_file_and_frame_produce_same_findings_for_equivalent_values(self):
        df = self.pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            df.to_csv(path, index=False)
            self.assertEqual(audit(df, target="y").checks, audit(path, target="y").checks)

    def test_dataframe_holdout_and_mixed_input_types(self):
        df = self.pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        report = audit(df, df.drop(columns="y"), target="y")
        self.assertIn("train_test_overlap", [f.code for f in report.findings])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            df.to_csv(path, index=False)
            self.assertIn("train_test_overlap", [f.code for f in audit(df, path, target="y").findings])

    def test_nullable_types_and_categories(self):
        df = self.pd.DataFrame({"x": self.pd.Series([1, None, 3], dtype="Int64"),
                               "region": self.pd.Categorical(["NA", "EU", None]),
                               "y": [1, 2, 3]})
        report = audit(df, target="y", task="regression")
        self.assertEqual(sum(f.code == "missing_values" for f in report.findings), 2)
        self.assertFalse([c for c in report.checks if c.status == "error"])

    def test_forecasting_without_config_object(self):
        df = self.pd.DataFrame({"store": ["a", "b", "a", "b"],
                               "date": self.pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], utc=True),
                               "demand": [1, 2, 4, 3]})
        test = df.copy()
        test["date"] += self.pd.Timedelta(days=5)
        report = audit(df, test, target="demand", task="forecasting", time_column="date", series_id="store",
                       expected_interval_seconds=86400, label_horizon_seconds=86400)
        self.assertEqual(next(c.status for c in report.checks if c.check_id == "forecast_label_boundary"), "passed")

    def test_invalid_frames_rejected(self):
        cases = [self.pd.DataFrame(), self.pd.DataFrame({0: [1]}),
                 self.pd.DataFrame({"x": [[1, 2]]}), self.pd.DataFrame([[1, 2]], columns=["x", "x"])]
        for frame in cases:
            with self.subTest(shape=frame.shape), self.assertRaises(ValueError):
                audit(frame)
        with self.assertRaises(TypeError):
            audit([{"x": 1}])

    def test_memory_and_row_limits(self):
        df = self.pd.DataFrame({"x": [1, 2]})
        for options in ({"max_rows": 1}, {"max_bytes": 1}):
            with self.assertRaises(ValueError):
                audit(df, **options)

    def test_fingerprint_is_stable_and_value_sensitive(self):
        df = self.pd.DataFrame({"x": [1, 2]})
        one = audit(df).datasets["train"]["sha256"]
        self.assertEqual(one, audit(df.copy()).datasets["train"]["sha256"])
        df.loc[0, "x"] = 9
        self.assertNotEqual(one, audit(df).datasets["train"]["sha256"])

    def test_to_frame_has_same_schema_when_empty(self):
        clean = audit(self.pd.DataFrame({"x": [1, 2, 3]})).to_frame()
        faulty = audit(self.pd.DataFrame({"x": [1, 1, 1]})).to_frame()
        self.assertTrue(clean.empty)
        self.assertFalse(faulty.empty)
        self.assertEqual(list(clean.columns), list(faulty.columns))

    def test_column_names_are_not_domain_specific(self):
        df = self.pd.DataFrame({"任意": [0, 1] * 20, "outcome": [0, 1] * 20})
        report = audit(df, target="outcome")
        self.assertIn("target_copy", [f.code for f in report.findings])


if __name__ == "__main__":
    unittest.main()
