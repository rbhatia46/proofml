"""Leakage signals with deliberately limited claims; association is not causation."""
import math
from ..context import AuditContext
from ..data import number
from ..models import CheckResult, Finding


def correlation(xs: list[float], ys: list[float]) -> float | None:
    # Scale first so extreme finite inputs do not overflow variance calculations.
    sx, sy = max(map(abs, xs)), max(map(abs, ys))
    if sx == 0 or sy == 0:
        return None
    x = [v / sx for v in xs]
    y = [v / sy for v in ys]
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    vx, vy = math.fsum(v * v for v in dx), math.fsum(v * v for v in dy)
    if not vx or not vy:
        return None
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(dx, dy)) / math.sqrt(vx * vy)))


class AvailabilityCheck:
    id = "feature_availability"

    def run(self, ctx: AuditContext) -> CheckResult:
        if not ctx.config.unavailable_features:
            return CheckResult(self.id, "skipped", reason="Prediction-time availability cannot be inferred; declare unavailable_features.")
        findings = [Finding("unavailable_feature", "critical", "confirmed", f"Unavailable predictor present: {column}",
            "The user declared this column unavailable when predictions are made, but it is present among candidate predictors.",
            "Remove it from model inputs or rebuild it using only information available at prediction time.", (column,))
            for column in ctx.config.unavailable_features if column in ctx.features]
        return CheckResult.complete(self.id, findings)


class AssociationCheck:
    id = "target_association"

    def run(self, ctx: AuditContext) -> CheckResult:
        if not ctx.config.target:
            return CheckResult(self.id, "skipped", reason="No target was declared.")
        target = ctx.train.column(ctx.config.target)
        labelled = sum(bool(v) for v in target)
        if labelled < ctx.config.min_samples:
            return CheckResult(self.id, "skipped", reason=f"At least {ctx.config.min_samples} observed targets are required.")
        if len(set(target) - {""}) < 2:
            return CheckResult(self.id, "skipped", reason="Target has insufficient variation.")
        findings = []
        checked = 0
        for column in ctx.features:
            pairs = [(a, b) for a, b in zip(ctx.train.column(column), target) if a and b]
            if len(pairs) < ctx.config.min_samples:
                continue
            checked += 1
            identical = sum(a == b for a, b in pairs) / len(pairs)
            if identical >= ctx.config.association_threshold:
                findings.append(Finding("target_copy", "high", "suspicious", f"Predictor closely matches target: {column}",
                    "This feature has near-identical observed values to the target. A legitimate proxy is possible.",
                    "Verify lineage and prediction-time availability; rerun evaluation without this feature.", (column,),
                    {"agreement": identical, "paired_rows": len(pairs)}))
                continue
            # Pearson is meaningful here for regression and binary numeric labels,
            # not arbitrary numeric encodings of multiclass categories.
            if ctx.config.task == "classification" and len(set(target) - {""}) != 2:
                continue
            numeric = [(number(a), number(b)) for a, b in pairs]
            if any(a is None or b is None for a, b in numeric):
                continue
            result = correlation([a for a, _ in numeric], [b for _, b in numeric])
            if result is not None and abs(result) >= ctx.config.association_threshold:
                findings.append(Finding("high_target_correlation", "high", "suspicious", f"Strong target association: {column}",
                    "Absolute Pearson correlation reaches the threshold. This is a review signal, not measured predictive accuracy or proof of leakage.",
                    "Inspect feature provenance and validation design; test performance without this feature.", (column,),
                    {"pearson_r": result, "paired_rows": len(pairs), "threshold": ctx.config.association_threshold}))
        if not checked:
            return CheckResult(self.id, "skipped", reason="No candidate feature has enough observed target-feature pairs.")
        result = CheckResult.complete(self.id, findings)
        return CheckResult(result.check_id, result.status, result.findings,
                           reason=f"Target-copy comparisons: {checked} features. Pearson only on fully numeric regression/binary pairs; other associations are unassessed.")
