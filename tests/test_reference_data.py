"""Cross-domain input compatibility, not a benchmark of leakage detection.

scikit-learn bundles these datasets; the tests do not fetch data at runtime.
It is a development extra only, not a ProofML runtime dependency.
"""
import importlib.util
import unittest

from proofml import audit


@unittest.skipUnless(importlib.util.find_spec("sklearn") and importlib.util.find_spec("pandas"),
                     "development reference-dataset dependencies not installed")
class ReferenceDatasetTests(unittest.TestCase):
    def verify_dataset(self, loader, task):
        from sklearn.model_selection import train_test_split
        frame = loader(as_frame=True).frame
        train, test = train_test_split(frame, test_size=0.25, random_state=42)
        report = audit(train, test, target="target", task=task)
        self.assertEqual(report.datasets["train"]["rows"], len(train))
        self.assertFalse([c for c in report.checks if c.status == "error"])
        # Do not assert zero findings: real datasets can legitimately trigger
        # quality heuristics. Verify stable serialization and broad compatibility.
        self.assertEqual(len(report.to_frame()), len(report.findings))
        return report

    def test_iris_multiclass(self):
        from sklearn.datasets import load_iris
        self.verify_dataset(load_iris, "classification")

    def test_wine_multiclass(self):
        from sklearn.datasets import load_wine
        self.verify_dataset(load_wine, "classification")

    def test_breast_cancer_binary(self):
        from sklearn.datasets import load_breast_cancer
        self.verify_dataset(load_breast_cancer, "classification")

    def test_diabetes_regression(self):
        from sklearn.datasets import load_diabetes
        self.verify_dataset(load_diabetes, "regression")


if __name__ == "__main__":
    unittest.main()
