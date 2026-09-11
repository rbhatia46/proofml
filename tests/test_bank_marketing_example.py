"""Offline harness tests, not substitute evidence for the separately run UCI case."""
from hashlib import sha256
import importlib.util
from io import BytesIO
from pathlib import Path
import tempfile
import ssl
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from examples import bank_marketing as example


class DownloadSafetyTests(unittest.TestCase):
    def test_download_keeps_tls_verification_enabled(self):
        context = example.download_context()
        self.assertTrue(context.check_hostname)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)

    def test_bounded_read_rejects_excess(self):
        self.assertEqual(example.bounded_read(BytesIO(b"abc"), 3), b"abc")
        with self.assertRaises(ValueError):
            example.bounded_read(BytesIO(b"abcd"), 3)

    def test_changed_archive_is_rejected_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "checksum changed"):
            example.verified_csv(b"not the official archive")

    def test_pinned_nested_member_and_csv_hash(self):
        csv = b"fixture only; not UCI data"
        nested = BytesIO()
        with ZipFile(nested, "w") as archive:
            archive.writestr(example.CSV_MEMBER, csv)
            archive.writestr("../../unwanted", "must never be extracted")
        outer = BytesIO()
        with ZipFile(outer, "w") as archive:
            archive.writestr("bank-additional.zip", nested.getvalue())
        payload = outer.getvalue()
        with patch.object(example, "ARCHIVE_SHA256", sha256(payload).hexdigest()):
            with self.assertRaisesRegex(ValueError, "CSV checksum changed"):
                example.verified_csv(payload)
            with patch.object(example, "CSV_SHA256", sha256(csv).hexdigest()):
                self.assertEqual(example.verified_csv(payload), csv)

    def test_zip_member_size_limit(self):
        payload = BytesIO()
        with ZipFile(payload, "w") as archive:
            archive.writestr("large", b"1234")
        with ZipFile(BytesIO(payload.getvalue())) as archive, self.assertRaises(ValueError):
            example.member_bytes(archive, "large", 3)

    def test_existing_output_refused_before_download(self):
        with tempfile.TemporaryDirectory() as output, patch.object(example, "read_dataset") as read:
            with self.assertRaises(FileExistsError):
                example.run(output, download=True)
            read.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("pandas"), "pandas example dependency not installed")
class CaseStudyTests(unittest.TestCase):
    def fixture(self):
        import pandas as pd
        count = 100
        values = {column: [i % 7 for i in range(count)] for column in example.NUMERIC}
        values.update({column: ["first" if i % 3 else "unknown" for i in range(count)] for column in example.CATEGORICAL})
        values["y"] = ["yes" if i % 4 == 0 else "no" for i in range(count)]
        return pd.DataFrame(values)

    def test_source_must_be_explicit_and_unambiguous(self):
        for kwargs in ({}, {"archive": Path("unused"), "download": True}):
            with self.assertRaises(ValueError):
                example.read_dataset(**kwargs)

    def test_holdout_preserves_order_without_overlap_or_mutation(self):
        frame = self.fixture()
        train, test = example.chronological_holdout(frame)
        self.assertEqual(list(train.index), list(range(80)))
        self.assertEqual(list(test.index), list(range(80, 100)))
        train.iloc[0, 0] = -999
        self.assertNotEqual(frame.iloc[0, 0], -999)

    def test_same_contract_blocks_removed_and_undeclared_scenarios_correctly(self):
        frame = self.fixture()
        train, test = example.chronological_holdout(frame)
        suite, policy, decisions = example.audit_scenarios(train, test)
        self.assertEqual({name: d.status for name, d in decisions.items()},
                         {"undeclared": "insufficient", "duration_included": "failed", "duration_removed": "passed"})
        self.assertEqual(suite.reports["duration_included"].config, suite.reports["duration_removed"].config)
        self.assertIn("duration", train.columns)
        self.assertEqual(policy.require_checks, ("feature_availability",))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn example dependency not installed")
    def test_model_feature_sets_and_finite_scores(self):
        train, test = example.chronological_holdout(self.fixture())
        for included in (True, False):
            metrics = example.evaluate_model(train, test, include_duration=included)
            self.assertEqual("duration" in metrics["features"], included)
            self.assertNotIn("y", metrics["features"])
            for name in ("roc_auc", "average_precision"):
                self.assertTrue(0 <= metrics[name] <= 1)
