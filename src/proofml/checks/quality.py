"""Data integrity checks, independent of any training framework."""
from collections import Counter
from ..context import AuditContext
from ..data import number
from ..models import CheckResult, Finding


class QualityCheck:
    id = "data_quality"

    def run(self, ctx: AuditContext) -> CheckResult:
        findings = []
        for split, data in (("train", ctx.train), ("test", ctx.test)):
            if data is None:
                continue
            duplicate_count = len(data.rows) - len(set(data.rows))
            if duplicate_count:
                findings.append(Finding("duplicate_rows", "medium", "confirmed", f"Repeated rows in {split}",
                    "Repeated observations can bias estimates; their presence alone does not prove leakage.",
                    "Check the unit of observation before removing duplicates.",
                    evidence={"split": split, "extra_duplicate_rows": duplicate_count}))
            for column in data.columns:
                values = data.column(column)
                missing = values.count("")
                present = [v for v in values if v]
                if missing / len(values) >= ctx.config.missing_threshold:
                    findings.append(Finding("missing_values", "medium", "confirmed", f"Missing values in {column}",
                        "The missing fraction reaches the configured threshold.",
                        "Investigate collection failures and fit any imputation on training data only.", (column,),
                        {"split": split, "missing": missing, "rows": len(values), "fraction": missing / len(values)}))
                if present and len(set(present)) == 1:
                    findings.append(Finding("constant_column", "low", "confirmed", f"Constant column: {column}",
                        "All observed values are identical within this split.",
                        "Check whether this feature is meaningful or unexpectedly constant.", (column,), {"split": split}))
                numeric = sum(number(v) is not None for v in present)
                if present and 0.8 <= numeric / len(present) < 1:
                    findings.append(Finding("mixed_numeric", "medium", "suspicious", f"Mixed numeric values in {column}",
                        "Most nonempty values parse as finite numbers; some do not. Categories may be intentional.",
                        "Check units, sentinels, and parsing before converting this column.", (column,),
                        {"split": split, "numeric": numeric, "non_numeric": len(present) - numeric}))
        return CheckResult.complete(self.id, findings)


class TargetCheck:
    id = "target_health"

    def run(self, ctx: AuditContext) -> CheckResult:
        target = ctx.config.target
        if not target:
            return CheckResult(self.id, "skipped", reason="No target was declared.")
        findings = []
        for split, data in (("train", ctx.train), ("test", ctx.test)):
            if data is None or target not in data.columns:
                continue  # Unlabelled inference/test datasets are supported.
            values = data.column(target)
            missing = values.count("")
            if missing:
                findings.append(Finding("missing_target", "high", "confirmed", f"Missing labels in {split}",
                    "Supervised evaluation requires observed labels.", "Separate unlabelled rows from supervised fitting and evaluation.",
                    (target,), {"split": split, "missing": missing}))
            present = [v for v in values if v]
            counts = Counter(present)
            if len(counts) <= 1:
                findings.append(Finding("constant_target", "high", "confirmed", f"Insufficient target variation in {split}",
                    "Fewer than two distinct nonempty target values are present.", "Inspect labels and split construction.",
                    (target,), {"split": split, "distinct": len(counts)}))
            if ctx.config.task in {"regression", "forecasting"}:
                invalid = sum(number(v) is None for v in present)
                if invalid:
                    findings.append(Finding("invalid_regression_target", "high", "confirmed", "Regression labels are not finite numbers",
                        "Nonempty regression targets must parse as finite numbers.", "Correct parsing or declare classification if labels are categories.",
                        (target,), {"split": split, "invalid": invalid}))
            elif len(counts) > 1 and min(counts.values()) / len(present) < ctx.config.imbalance_threshold:
                findings.append(Finding("class_imbalance", "medium", "confirmed", f"Uneven class representation in {split}",
                    "At least one class is below the configured share. This is a dataset property, not automatically a defect.",
                    "Report per-class metrics and a majority-class baseline; choose metrics for the task.", (target,),
                    {"split": split, "classes": len(counts), "minority_fraction": min(counts.values()) / len(present),
                     "majority_baseline_accuracy": max(counts.values()) / len(present)}))
        return CheckResult.complete(self.id, findings)
