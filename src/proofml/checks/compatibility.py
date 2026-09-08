"""Train/test compatibility signals with counts, never raw category values."""
from ..context import AuditContext
from ..data import number
from ..models import CheckResult, Finding


class SplitCompatibilityCheck:
    id = "split_compatibility"

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.test is None:
            return CheckResult(self.id, "skipped", reason="A test dataset is required.")
        findings = []
        assessed = 0
        target = ctx.config.target
        if ctx.config.task == "classification" and target and target in ctx.test.columns:
            known = set(ctx.train.column(target)) - {""}
            unknown = [v for v in ctx.test.column(target) if v and v not in known]
            assessed += 1
            if unknown:
                findings.append(Finding(
                    "unseen_test_labels", "high", "confirmed", "Test labels are absent from training",
                    "Some observed test classes have no labelled training examples. This can invalidate closed-set classification evaluation.",
                    "Review label encoding and split coverage; open-set tasks need a separate evaluation contract.", (target,),
                    {"unseen_classes": len(set(unknown)), "affected_test_rows": len(unknown)},
                ))
        for column in ctx.features:
            if column not in ctx.test.columns:
                continue
            a, b = ctx.train.column(column), ctx.test.column(column)
            nonmissing_a, nonmissing_b = [v for v in a if v], [v for v in b if v]
            assessed += 1
            increase = b.count("") / len(b) - a.count("") / len(a)
            if increase >= ctx.config.missing_threshold:
                findings.append(Finding(
                    "test_missingness_increase", "medium", "confirmed", f"Missingness increased: {column}",
                    "The test missing-value share exceeds training by at least the configured missing threshold.",
                    "Check collection and preprocessing differences before interpreting holdout performance.", (column,),
                    {"train_missing_fraction": a.count("") / len(a), "test_missing_fraction": b.count("") / len(b),
                     "increase": increase, "threshold": ctx.config.missing_threshold},
                ))
            if len(nonmissing_a) < ctx.config.min_samples or not nonmissing_b:
                continue
            if all(number(v) is not None for v in nonmissing_a):
                invalid = sum(number(v) is None for v in nonmissing_b)
                if invalid:
                    findings.append(Finding(
                        "test_numeric_parse_change", "high", "suspicious", f"Numeric parsing changes: {column}",
                        "All observed training values parse as finite numbers, but some test values do not. Numeric-looking categories may be intentional.",
                        "Review units, sentinels, encoding, and preprocessing type expectations.", (column,),
                        {"non_numeric_test_rows": invalid, "observed_training_rows": len(nonmissing_a)},
                    ))
            elif len(set(nonmissing_a)) <= 50:
                known_values = set(nonmissing_a)
                unknown = [v for v in nonmissing_b if v not in known_values]
                if unknown:
                    findings.append(Finding(
                        "unseen_feature_categories", "medium", "needs_context", f"New test categories: {column}",
                        "A low-cardinality nonnumeric training feature has values not seen during training; an encoder may already handle them.",
                        "Verify unknown-category handling in the fitted preprocessing pipeline.", (column,),
                        {"unseen_categories": len(set(unknown)), "affected_test_rows": len(unknown)},
                    ))
        if not assessed:
            return CheckResult(self.id, "skipped", reason="No shared candidate features or classification labels to assess.")
        result = CheckResult.complete(self.id, findings)
        return CheckResult(self.id, result.status, result.findings,
                           reason="Type/category signals require min_samples nonmissing training values; categorical training cardinality is limited to 50. No encoder is inspected.")
