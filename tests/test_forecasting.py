"""Forecasting contracts and legitimate counterexamples, using only synthetic data."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from proofml import AuditConfig, audit
from proofml.cli import main
from proofml.context import AuditContext
from proofml.data import load_dataset


class ForecastTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = AuditConfig(task="forecasting", target="y", time_column="time", series_id="series",
                                  expected_interval_seconds=86400, label_horizon_seconds=86400)

    def data(self, name, rows, columns=("series", "time", "y")):
        path = self.root / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def run_audit(self, rows, test=None, config=None):
        train = self.data("train.csv", rows)
        holdout = self.data("test.csv", test) if test is not None else None
        report = audit(train, test=holdout, config=config or self.config)
        self.assertFalse([c for c in report.checks if c.status == "error"])
        return report

    def check(self, report, name):
        return next(c for c in report.checks if c.check_id == name)

    def test_clean_multiseries_and_repeated_series_across_splits(self):
        report = self.run_audit([("A", "2026-01-01", 1), ("B", "2026-01-01", 2),
                                 ("A", "2026-01-02", 3), ("B", "2026-01-02", 4)],
                                [("A", "2026-01-04", 5), ("B", "2026-01-04", 6)])
        for name in ("forecast_index", "forecast_cadence", "forecast_label_boundary", "temporal_order"):
            self.assertEqual(self.check(report, name).status, "passed")
        self.assertNotIn("shared_entities", [f.code for f in report.findings])

    def test_duplicate_origin_with_different_targets_and_timezone_spelling(self):
        report = self.run_audit([("A", "2026-01-01T00:00:00Z", 1), ("A", "2026-01-01T05:30:00+05:30", 2)])
        result = self.check(report, "forecast_index")
        self.assertEqual(result.findings[0].evidence["extra_rows_at_duplicate_times"], 1)
        self.assertEqual(self.check(report, "forecast_cadence").status, "skipped")

    def test_sorting_warning_is_not_a_leakage_claim(self):
        report = self.run_audit([("A", "2026-01-02", 2), ("A", "2026-01-01", 1)])
        self.assertEqual(self.check(report, "forecast_index").findings[0].confidence, "suspicious")
        self.assertEqual(self.check(report, "forecast_cadence").status, "passed")

    def test_missing_grid_slots_counted_per_series(self):
        report = self.run_audit([("A", "2026-01-01", 1), ("A", "2026-01-04", 2)])
        evidence = self.check(report, "forecast_cadence").findings[0].evidence
        self.assertEqual(evidence["missing_slots_in_exact_multiple_gaps"], 2)

    def test_off_grid_does_not_invent_missing_slots(self):
        report = self.run_audit([("A", "2026-01-01", 1), ("A", "2026-01-02T12:00:00", 2)])
        evidence = self.check(report, "forecast_cadence").findings[0].evidence
        self.assertEqual(evidence["irregular_intervals"], 1)
        self.assertEqual(evidence["missing_slots_in_exact_multiple_gaps"], 0)

    def test_invalid_index_does_not_silently_pass_dependent_checks(self):
        for key, time in (("", "2026-01-01"), ("A", "oops"), ("A", "")):
            with self.subTest(key=key, time=time):
                report = self.run_audit([(key, time, 1)], [("A", "2026-01-04", 2)])
                self.assertEqual(self.check(report, "forecast_index").findings[0].code, "invalid_forecast_index")
                for name in ("forecast_cadence", "forecast_label_boundary"):
                    self.assertEqual(self.check(report, name).status, "skipped")

    def test_horizon_equality_fails_but_strictly_earlier_passes(self):
        for date, expected in (("2026-01-02", "findings"), ("2026-01-03", "passed")):
            report = self.run_audit([("A", "2026-01-01", 1)], [("A", date, 2)])
            self.assertEqual(self.check(report, "forecast_label_boundary").status, expected)

    def test_embargo_adds_to_horizon(self):
        report = self.run_audit([("A", "2026-01-01", 1)], [("A", "2026-01-03", 2)],
                                replace(self.config, embargo_seconds=86400))
        self.assertEqual(self.check(report, "forecast_label_boundary").status, "findings")

    def test_global_cutoff_includes_other_series(self):
        report = self.run_audit([("A", "2026-01-01", 1), ("B", "2026-01-04", 2)],
                                [("A", "2026-01-03", 3), ("B", "2026-01-10", 4)])
        self.assertEqual(self.check(report, "forecast_label_boundary").findings[0].evidence["affected_training_rows"], 1)

    def test_unknown_horizon_and_cadence_are_skipped(self):
        report = self.run_audit([("A", "2026-01-01", 1)], config=replace(self.config, label_horizon_seconds=None, expected_interval_seconds=None))
        for name in ("forecast_cadence", "forecast_label_boundary"):
            self.assertEqual(self.check(report, name).status, "skipped")

    def test_forecast_numeric_targets_not_class_imbalance(self):
        report = self.run_audit([("A", "2026-01-01", "oops"), ("A", "2026-01-02", "2")])
        self.assertIn("invalid_regression_target", [f.code for f in report.findings])
        self.assertNotIn("class_imbalance", [f.code for f in report.findings])

    def test_series_metadata_is_not_a_predictor(self):
        path = self.data("train.csv", [("A", "2026-01-01", 1)])
        ctx = AuditContext(load_dataset(path, self.config), None, self.config)
        self.assertEqual(ctx.features, ())

    def test_configuration_validation(self):
        for values in ({"time_column": None}, {"target": None}, {"expected_interval_seconds": 0},
                       {"label_horizon_seconds": -1}, {"embargo_seconds": True},
                       {"series_id": "y"}, {"embargo_seconds": 1, "label_horizon_seconds": None},
                       {"label_horizon_seconds": float("inf")}, {"embargo_seconds": None}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(self.config, **values)
        with self.assertRaises(ValueError):
            AuditConfig(series_id="x")

    def test_single_series_without_key(self):
        path = self.data("train.csv", [("2026-01-01", 1), ("2026-01-02", 2)], columns=("time", "y"))
        report = audit(path, config=replace(self.config, series_id=None))
        self.assertEqual(self.check(report, "forecast_index").status, "passed")

    def test_unlabelled_holdout_missing_series_key_is_reported(self):
        train = self.data("train.csv", [("A", "2026-01-01", 1)])
        test = self.data("test.csv", [("2026-01-04",)], columns=("time",))
        report = audit(train, test=test, config=self.config)
        self.assertEqual(self.check(report, "forecast_index").status, "findings")
        self.assertEqual(self.check(report, "forecast_label_boundary").status, "skipped")

    def test_private_series_values_omitted_from_reports(self):
        report = self.run_audit([("secret-store", "2026-01-01", 1), ("secret-store", "2026-01-01", 2)])
        self.assertNotIn("secret-store", json.dumps(report.to_dict()))

    def test_forecast_demos_and_cli_registry(self):
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(["checks", "--task", "forecasting"]), 0)
            self.assertIn("forecast_label_boundary", captured.getvalue())
            self.assertEqual(main(["demo", "--problem", "forecasting", "--output", str(self.root / "faulty")]), 1)
            self.assertEqual(main(["demo", "--problem", "forecasting", "--clean", "--output", str(self.root / "clean")]), 0)
        bad = json.loads((self.root / "faulty/report.json").read_text())
        good = json.loads((self.root / "clean/report.json").read_text())
        self.assertEqual(good["summary"]["findings"], 0)
        codes = {f["code"] for c in bad["checks"] for f in c["findings"]}
        self.assertTrue({"duplicate_series_timestamp", "unsorted_forecast_series", "irregular_forecast_cadence", "forecast_label_boundary_overlap"} <= codes)


if __name__ == "__main__":
    unittest.main()
