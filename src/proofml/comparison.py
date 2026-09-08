"""Conservative comparison: incompatible evidence never becomes a zero regression."""
from dataclasses import asdict, dataclass
import json
import math
import re

from .models import AuditReport


@dataclass(frozen=True)
class MetricChange:
    check_id: str
    metric: str
    baseline: int | float | None
    candidate: int | float | None

    @property
    def delta(self):
        if self.candidate is None or self.baseline is None:
            return None
        difference = self.candidate - self.baseline
        return difference if math.isfinite(difference) else None


@dataclass(frozen=True)
class ReportComparison:
    reasons: tuple[str, ...]
    changes: tuple[MetricChange, ...] = ()

    @property
    def compatible(self):
        return not self.reasons

    def to_dict(self):
        return {"compatible": self.compatible, "reasons": list(self.reasons),
                "changes": [{**asdict(change), "delta": change.delta} for change in self.changes]}


def metric_value(report, path):
    """Resolve explicit check.metric names; reject ambiguous plugin namespaces."""
    matches = [value for check in report.checks if check.status in {"passed", "findings"}
               for name, value in check.metrics.items() if f"{check.check_id}.{name}" == path]
    if len(matches) != 1:
        return None
    value = matches[0]
    return value if type(value) in {int, float} and math.isfinite(value) else None


def compare_reports(candidate: AuditReport, baseline: AuditReport) -> ReportComparison:
    if not isinstance(candidate, AuditReport) or not isinstance(baseline, AuditReport):
        raise TypeError("Comparison requires AuditReport instances")
    reasons = []
    if candidate.schema_version != baseline.schema_version or candidate.tool_version != baseline.tool_version:
        reasons.append("Report schema and tool versions must match.")
    if json.dumps(candidate.config, sort_keys=True, allow_nan=False) != json.dumps(baseline.config, sort_keys=True, allow_nan=False):
        reasons.append("Audit configurations must match, including metric definitions and budgets.")
    if sorted(c.check_id for c in candidate.checks) != sorted(c.check_id for c in baseline.checks):
        reasons.append("Check registries must match.")
    if candidate.config.get("modality") != "retrieval" or baseline.config.get("modality") != "retrieval":
        reasons.append("Strict metric comparison currently supports retrieval reports only.")
    a, b = candidate.datasets.get("evaluation", {}), baseline.datasets.get("evaluation", {})
    for report, metadata in ((candidate, a), (baseline, b)):
        contracts = metadata.get("builtin_check_contracts")
        if not isinstance(contracts, dict) or set(contracts) != {c.check_id for c in report.checks} or any(v != "1" for v in contracts.values()):
            reasons.append("Strict comparison requires versioned built-in retrieval checks; custom/legacy check definitions are unverified.")
            break
    for metadata in (a, b):
        if metadata.get("comparison_contract") != "retrieval.binary.v1" or metadata.get("alignment") != "query_id":
            reasons.append("Both reports require the versioned retrieval identity contract and stable query-ID mappings.")
            break
    for name in ("evaluation_sha256", "corpus_sha256"):
        values = (a.get(name), b.get(name))
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in values):
            reasons.append(f"Missing or invalid {name}; legacy snapshots cannot establish compatibility.")
        elif values[0] != values[1]:
            reasons.append(f"Evaluation identity differs: {name}.")
    if a.get("rows") != b.get("rows"):
        reasons.append("Query counts must match.")
    if reasons:
        return ReportComparison(tuple(reasons))
    # Keep check/metric names separate internally, avoiding collisions from dots.
    keys = {(c.check_id, name) for report in (candidate, baseline) for c in report.checks for name in c.metrics}
    changes = []
    for check_id, name in sorted(keys):
        def get(report):
            values = [c.metrics.get(name) for c in report.checks if c.check_id == check_id and c.status in {"passed", "findings"}]
            value = values[0] if len(values) == 1 else None
            return value if type(value) in {int, float} and math.isfinite(value) else None
        changes.append(MetricChange(check_id, name, get(baseline), get(candidate)))
    if any(change.baseline is not None and change.candidate is not None and change.delta is None for change in changes):
        return ReportComparison(("A metric difference exceeds finite numeric range.",))
    return ReportComparison((), tuple(changes))
