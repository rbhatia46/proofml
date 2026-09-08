"""General-purpose interoperability and distribution-boundary regressions."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

from proofml import audit
from proofml.context import AuditContext
from proofml.data import load_dataset
from proofml.config import AuditConfig


class GeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def data(self, name, rows, columns=("x", "y")):
        path = self.root / name
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def codes(self, report):
        self.assertFalse([c for c in report.checks if c.status == "error"])
        return {f.code for f in report.findings}

    def test_unseen_class_in_test(self):
        train = self.data("train.csv", [(i, i % 2) for i in range(40)])
        test = self.data("test.csv", [(41, 2), (42, 1)])
        self.assertIn("unseen_test_labels", self.codes(audit(train, test, target="y")))
        self.assertNotIn("unseen_test_labels", self.codes(audit(train, test, target="y", task="regression")))

    def test_numeric_parse_change(self):
        train = self.data("train.csv", [(i, i % 2) for i in range(40)])
        test = self.data("test.csv", [("unknown", 1)])
        self.assertIn("test_numeric_parse_change", self.codes(audit(train, test, target="y")))

    def test_numeric_parse_small_training_sample_is_unassessed(self):
        train = self.data("train.csv", [(1, 0), (2, 1)])
        test = self.data("test.csv", [("unknown", 1)])
        self.assertNotIn("test_numeric_parse_change", self.codes(audit(train, test, target="y")))

    def test_unseen_categories(self):
        train = self.data("train.csv", [("a" if i % 2 else "b", i % 2) for i in range(40)])
        test = self.data("test.csv", [("private-category", 1)])
        report = audit(train, test, target="y")
        self.assertIn("unseen_feature_categories", self.codes(report))
        self.assertNotIn("private-category", str(report.to_dict()))

    def test_high_cardinality_text_not_mistaken_for_categories(self):
        train = self.data("train.csv", [(f"text-{i}", i % 2) for i in range(60)])
        test = self.data("test.csv", [("new-document", 1)])
        self.assertNotIn("unseen_feature_categories", self.codes(audit(train, test, target="y")))

    def test_missingness_increase(self):
        train = self.data("train.csv", [(i, i % 2) for i in range(40)])
        test = self.data("test.csv", [("", 0), ("", 1), (3, 0), (4, 1)])
        self.assertIn("test_missingness_increase", self.codes(audit(train, test, target="y")))

    def test_no_warning_for_improved_missingness(self):
        train = self.data("train.csv", [("", 0), ("", 1), (3, 0), (4, 1)])
        test = self.data("test.csv", [(1, 0), (2, 1)])
        self.assertNotIn("test_missingness_increase", self.codes(audit(train, test, target="y")))

    def forecast(self, available, horizon=None, embargo=0):
        train = self.data("train.csv", [("2026-01-01", available, 1), ("2026-01-02", "2026-01-03", 2)],
                          ("origin", "available", "y"))
        test = self.data("test.csv", [("2026-01-05", 3)], ("origin", "y"))
        return audit(train, test, task="forecasting", target="y", time_column="origin",
                     label_available_column="available", label_horizon_seconds=horizon, embargo_seconds=embargo)

    def test_variable_availability_reaches_cutoff(self):
        report = self.forecast("2026-01-05")
        self.assertIn("forecast_label_boundary_overlap", self.codes(report))
        self.assertNotIn("schema_mismatch", self.codes(report))

    def test_variable_availability_before_cutoff(self):
        self.assertNotIn("forecast_label_boundary_overlap", self.codes(self.forecast("2026-01-04")))

    def test_variable_availability_embargo(self):
        self.assertIn("forecast_label_boundary_overlap", self.codes(self.forecast("2026-01-04", embargo=86400)))

    def test_invalid_variable_availability(self):
        for value in ("", "bad timestamp"):
            with self.subTest(value=value):
                self.assertIn("invalid_label_availability", self.codes(self.forecast(value)))

    def test_variable_availability_before_origin(self):
        self.assertIn("label_available_before_origin", self.codes(self.forecast("2025-12-31")))

    def test_conflicting_horizons_rejected(self):
        with self.assertRaises(ValueError):
            self.forecast("2026-01-04", horizon=86400)

    def test_availability_is_not_model_feature(self):
        config = AuditConfig(task="forecasting", target="y", time_column="origin", label_available_column="available")
        path = self.data("train.csv", [("2026-01-01", "2026-01-02", 1)], ("origin", "available", "y"))
        ctx = AuditContext(load_dataset(path, config), None, config)
        self.assertEqual(ctx.features, ())


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas extra not installed")
class SeparateLabelsTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pandas as pd
        self.np, self.pd = np, pd
        self.x = np.array([[1, 2], [3, 4], [5, 6]])
        self.y = np.array([0, 1, 0])

    def test_numpy_X_y(self):
        report = audit(self.x, y=self.y)
        self.assertEqual(report.config["target"], "__proofml_target__")
        self.assertEqual(report.datasets["train"]["columns"], ["feature_0", "feature_1", "__proofml_target__"])

    def test_numpy_holdout(self):
        report = audit(self.x, self.x, y=self.y, y_test=self.y)
        self.assertIn("train_test_overlap", [f.code for f in report.findings])

    def test_pandas_preserves_values_and_indexes(self):
        x = self.pd.DataFrame(self.x, columns=["a", "b"], index=[7, 2, 9])
        y = self.pd.Series(self.y, index=x.index)
        before = x.copy(deep=True)
        report = audit(x, y=y, target="outcome")
        self.pd.testing.assert_frame_equal(before, x)
        self.assertEqual(report.config["target"], "outcome")

    def test_pandas_index_order_mismatch_fails(self):
        x = self.pd.DataFrame(self.x, columns=["a", "b"], index=[7, 2, 9])
        y = self.pd.Series(self.y, index=[2, 7, 9])
        with self.assertRaisesRegex(ValueError, "indexes"):
            audit(x, y=y)

    def test_invalid_label_shapes_and_lengths(self):
        for y in ([1, 2], [[1], [0], [1]]):
            with self.subTest(y=y), self.assertRaises(ValueError):
                audit(self.x, y=y)

    def test_no_test_for_test_labels(self):
        with self.assertRaises(ValueError):
            audit(self.x, y=self.y, y_test=self.y)

    def test_target_collision_fails(self):
        x = self.pd.DataFrame({"outcome": [1, 2, 3]})
        with self.assertRaisesRegex(ValueError, "collides"):
            audit(x, y=self.y, target="outcome")

    def test_one_dimensional_feature_array_fails(self):
        with self.assertRaises(ValueError):
            audit(self.y, y=self.y)

    def test_numpy_features_without_target(self):
        self.assertIsNone(audit(self.x).config["target"])

    def test_empty_feature_matrix_fails(self):
        with self.assertRaises(ValueError):
            audit(self.np.empty((3, 0)), y=self.y)

    def test_array_row_limit(self):
        with self.assertRaises(ValueError):
            audit(self.x, y=self.y, max_rows=1)


if __name__ == "__main__":
    unittest.main()
