"""Descriptive split differences; thresholds are effect sizes, not p-values."""
from collections import Counter
from ..context import AuditContext
from ..data import number
from ..models import CheckResult, Finding


def ks_distance(a: list[float], b: list[float]) -> float:
    """Two-sample empirical CDF distance, including ties correctly."""
    a, b = sorted(a), sorted(b)
    i = j = 0
    distance = 0.0
    for value in sorted(set(a) | set(b)):
        while i < len(a) and a[i] <= value:
            i += 1
        while j < len(b) and b[j] <= value:
            j += 1
        distance = max(distance, abs(i / len(a) - j / len(b)))
    return distance


class DriftCheck:
    id = "feature_drift"

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.test is None:
            return CheckResult(self.id, "skipped", reason="A test dataset is required.")
        findings = []
        checked = 0
        for column in ctx.features:
            if column not in ctx.test.columns:
                continue
            a = [v for v in ctx.train.column(column) if v]
            b = [v for v in ctx.test.column(column) if v]
            if min(len(a), len(b)) < ctx.config.min_samples:
                continue
            na, nb = [number(v) for v in a], [number(v) for v in b]
            if all(v is not None for v in na + nb):
                distance, method = ks_distance(na, nb), "empirical_ks_distance"
            else:
                # High-cardinality strings (free text/IDs) produce trivial drift.
                if len(set(a) | set(b)) > 50:
                    continue
                ca, cb = Counter(a), Counter(b)
                distance = sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in ca.keys() | cb.keys()) / 2
                method = "total_variation_distance"
            checked += 1
            if distance >= ctx.config.drift_threshold:
                findings.append(Finding("feature_distribution_shift", "medium", "suspicious", f"Distribution differs: {column}",
                    "Observed split distributions differ by the configured effect-size threshold. Intended population or time shifts may explain this.",
                    "Review sampling and population changes; examine performance on the deployment population.", (column,),
                    {"method": method, "distance": distance, "train_nonmissing": len(a), "test_nonmissing": len(b)}))
        if not checked:
            return CheckResult(self.id, "skipped", reason="No shared feature meets sample-size and type/cardinality requirements.")
        result = CheckResult.complete(self.id, findings)
        return CheckResult(result.check_id, result.status, result.findings,
                           reason=f"Assessed {checked} of {len(ctx.features)} candidate features; sample-size, type, and cardinality limits apply.")
