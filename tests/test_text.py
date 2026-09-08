"""Text-domain contracts, counterexamples, privacy, and bounded execution."""
import json
import random
import tempfile
import unittest
from pathlib import Path

from proofml import AuditFailed, CheckResult, TextConfig, audit_text
from proofml.reporting import render_html
from proofml.text import default_text_checks


class TextTests(unittest.TestCase):
    def codes(self, report):
        self.assertFalse([c for c in report.checks if c.status == "error"])
        return {f.code for f in report.findings}

    def check(self, report, identifier):
        return next(c for c in report.checks if c.check_id == identifier)

    def test_clean_corpora(self):
        report = audit_text(["A red bird flies", "The tall trees sway"], ["Snow covers every mountain"], y=[0, 1], y_test=[1])
        self.assertEqual(self.codes(report), set())
        self.assertTrue(all(c.status == "passed" for c in report.checks))
        report.raise_for_issues(require_checks=("text_overlap", "text_near_duplicates"))

    def test_unicode_normalization_and_whitespace(self):
        report = audit_text(["ＣＡＦÉ\n  GOOD", "café good"], ["CAFÉ good"])
        self.assertIn("duplicate_documents", self.codes(report))
        self.assertIn("text_train_test_overlap", self.codes(report))

    def test_exact_normalization_preserves_case_and_spacing(self):
        report = audit_text(["HELLO world"], ["hello  world"], normalization="exact")
        self.assertNotIn("text_train_test_overlap", self.codes(report))

    def test_blank_inputs_are_quality_findings_not_overlap(self):
        report = audit_text(["", "   "], ["\t"])
        self.assertEqual(self.codes(report), {"empty_documents"})
        self.assertEqual(self.check(report, "text_near_duplicates").status, "skipped")

    def test_unlabelled_and_no_test_coverage(self):
        report = audit_text(["a document"])
        self.assertEqual(self.check(report, "text_labels").status, "skipped")
        self.assertEqual(self.check(report, "text_overlap").status, "skipped")
        with self.assertRaises(AuditFailed):
            report.raise_for_issues(require_checks=("text_overlap",))

    def test_conflicting_labels_within_and_across_splits(self):
        self.assertIn("conflicting_text_labels", self.codes(audit_text(["same", "same"], y=[0, 1])))
        self.assertIn("conflicting_text_labels", self.codes(audit_text(["same"], ["SAME"], y=[0], y_test=[1])))

    def test_integer_and_string_classes_remain_distinct(self):
        self.assertIn("conflicting_text_labels", self.codes(audit_text(["same", "same"], y=[1, "1"])))

    def test_missing_and_unseen_labels(self):
        report = audit_text(["a", "b", "c"], ["d"], y=[0, None, ""], y_test=["new"])
        self.assertIn("missing_text_labels", self.codes(report))
        self.assertIn("unseen_text_labels", self.codes(report))

    def test_near_duplicate_positive_and_negative(self):
        source = "one two three four five six seven eight nine ten"
        similar = source + " eleven"
        report = audit_text([source], [similar, "totally different tokens here"])
        finding = next(f for f in report.findings if f.code == "text_near_train_test_overlap")
        self.assertEqual(finding.evidence["affected_test_rows"], 1)
        self.assertNotIn("text_near_train_test_overlap", self.codes(audit_text([source], [similar], near_duplicate_threshold=1)))

    def test_exact_matches_do_not_double_count_as_near(self):
        report = audit_text(["one two three four"], ["one two three four"])
        self.assertIn("text_train_test_overlap", self.codes(report))
        self.assertNotIn("text_near_train_test_overlap", self.codes(report))

    def test_short_document_coverage(self):
        report = audit_text(["one"], ["two"])
        self.assertEqual(self.check(report, "text_near_duplicates").status, "skipped")
        report = audit_text(["one"], ["two"], shingle_size=1)
        self.assertEqual(self.check(report, "text_near_duplicates").status, "passed")

    def test_partial_budget_never_looks_like_pass(self):
        for options in ({"max_tokens_per_document": 2}, {"max_shingles": 1}, {"max_candidate_visits": 1}):
            with self.subTest(options=options):
                report = audit_text(["one two three four five"], ["one two three four five six"], **options)
                self.assertEqual(self.check(report, "text_near_duplicates").status, "skipped")
                self.assertNotIn("text_near_duplicates", report.metrics)
                with self.assertRaises(AuditFailed):
                    report.raise_for_issues(require_checks=("text_near_duplicates",))

    def test_duplicate_test_rows_counted(self):
        text = "one two three four five six seven eight nine ten"
        report = audit_text([text], [text + " eleven"] * 3)
        finding = next(f for f in report.findings if f.code == "text_near_train_test_overlap")
        self.assertEqual(finding.evidence["affected_test_rows"], 3)
        self.assertEqual(finding.evidence["distinct_test_documents"], 1)

    def test_near_duplicates_match_bruteforce_reference(self):
        rng = random.Random(17)
        train = [" ".join(rng.choices("abcdefghi", k=8)) for _ in range(30)]
        test = [" ".join(rng.choices("abcdefghi", k=8)) for _ in range(20)]
        def terms(text):
            tokens = text.split()
            return set(zip(tokens, tokens[1:]))
        expected = sum(any(len(terms(a) & terms(b)) / len(terms(a) | terms(b)) >= 0.2 for a in train)
                       for b in test if b not in train)
        report = audit_text(train, test, shingle_size=2, near_duplicate_threshold=0.2)
        observed = sum(f.evidence["affected_test_rows"] for f in report.findings if f.code == "text_near_train_test_overlap")
        self.assertEqual(observed, expected)

    def test_inputs_are_not_mutated_and_generators_work(self):
        docs, labels = ["HELLO", "World"], [0, 1]
        audit_text(docs, y=labels)
        self.assertEqual(docs, ["HELLO", "World"])
        self.assertEqual(labels, [0, 1])
        self.assertEqual(audit_text(iter(docs), y=iter(labels)).checks, audit_text(docs, y=labels).checks)

    def test_label_length_and_types_fail(self):
        for labels in ([1], [1, 2, 3], [True, 1], [[1], [2]], "ab"):
            with self.subTest(labels=labels), self.assertRaises((TypeError, ValueError)):
                audit_text(["a", "b"], y=labels)
        with self.assertRaises(ValueError):
            audit_text(["a"], ["b"], y_test=[1])

    def test_invalid_documents(self):
        for documents in ([], "document", {"a", "b"}, {"a": "b"}, [None], [b"bytes"]):
            with self.subTest(documents=documents), self.assertRaises((TypeError, ValueError)):
                audit_text(documents)

    def test_input_limits_refuse_not_truncate(self):
        for options in ({"max_documents": 1}, {"max_bytes": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                audit_text(["first", "second"], **options)

    def test_config_validation_and_overrides(self):
        for options in ({"normalization": "lower"}, {"shingle_size": True}, {"near_duplicate_threshold": float("nan")},
                        {"near_duplicate_threshold": 0}, {"max_candidate_visits": 0}, {"disabled_checks": "text_overlap"}):
            with self.subTest(options=options), self.assertRaises((TypeError, ValueError)):
                TextConfig(**options)
        with self.assertRaises(TypeError):
            audit_text(["a"], normlization="exact")
        report = audit_text(["a"], config=TextConfig(normalization="exact"), normalization="unicode")
        self.assertEqual(report.config["normalization"], "unicode")

    def test_plugins_and_disabled_checks(self):
        class Broken:
            id = "custom"
            def run(self, ctx):
                raise RuntimeError("PRIVATE TEXT")
        report = audit_text(["a"], checks=[*default_text_checks(), Broken()], disabled_checks=["text_overlap"])
        self.assertEqual(self.check(report, "custom").status, "error")
        self.assertNotIn("PRIVATE TEXT", json.dumps(report.to_dict()))
        with self.assertRaises(ValueError):
            audit_text(["a"], disabled_checks=["unknown"])
        with self.assertRaises(ValueError):
            audit_text(["a"], checks=[Broken(), Broken()])

    def test_reports_do_not_disclose_content_or_labels(self):
        secret, label = "PRIVATE-SENSITIVE-DOCUMENT", "PRIVATE-LABEL"
        report = audit_text([secret, secret], [secret], y=[label, "different"])
        for output in (json.dumps(report.to_dict()), render_html(report)):
            self.assertNotIn(secret, output)
            self.assertNotIn(label, output)
        with tempfile.TemporaryDirectory() as tmp:
            json_path, html_path = report.save(Path(tmp) / "text")
            self.assertEqual(json.loads(json_path.read_text())["config"]["modality"], "text")
            self.assertIn("text_overlap", html_path.read_text())

    def test_fingerprint_tracks_raw_content_and_labels(self):
        def fingerprint(docs, **kwargs):
            return audit_text(docs, **kwargs).datasets["train"]["sha256"]
        self.assertEqual(fingerprint(["hello"]), fingerprint(["hello"]))
        self.assertNotEqual(fingerprint(["hello"]), fingerprint(["HELLO"]))
        self.assertNotEqual(fingerprint(["hello"], y=[1]), fingerprint(["hello"], y=[2]))
