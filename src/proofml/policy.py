"""Versionable, fail-closed release policies across audit domains."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from hashlib import sha256
import json
import math
from types import MappingProxyType
from xml.etree import ElementTree as ET

from .comparison import ReportComparison, compare_reports, metric_value
from .exceptions import AuditFailed
from .models import AuditReport, SEVERITY_ORDER
from .snapshots import read_json, report_from_dict, write_json, write_text


def _finite(value):
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def _above(value, limit):
    # Absolute units, not relative percent. Tolerate only floating-point rounding
    # at a boundary, not materially worse metrics.
    return value > limit and (limit == 0 or not math.isclose(value, limit, rel_tol=1e-12, abs_tol=0))


def _xml_text(value):
    """Represent XML-disallowed code points explicitly instead of emitting bad XML."""
    return "".join(char if char in "\t\n\r" or 0x20 <= ord(char) <= 0xD7FF or 0xE000 <= ord(char) <= 0xFFFD or 0x10000 <= ord(char) <= 0x10FFFF
                   else f"\\u{ord(char):04x}" for char in value)


@dataclass(frozen=True)
class GateIssue:
    code: str
    kind: str
    message: str
    evidence: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GateResult:
    policy: AuditPolicy
    report: AuditReport
    baseline: AuditReport | None
    issues: tuple[GateIssue, ...]
    comparison: ReportComparison | None = None

    @property
    def status(self):
        for kind in ("incompatible", "insufficient", "failed"):
            if any(issue.kind == kind for issue in self.issues):
                return kind
        return "passed"

    @property
    def passed(self):
        return self.status == "passed"

    def raise_for_issues(self):
        if not self.passed:
            error = AuditFailed(f"Release gate {self.status}: " + "; ".join(issue.message for issue in self.issues), self.report)
            error.decision = self
            raise error
        return self

    def to_dict(self):
        def fingerprint(report):
            return sha256(json.dumps(report.to_dict(), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()
        return {"schema_version": "1.0", "status": self.status, "passed": self.passed,
                "policy": self.policy.to_dict(), "candidate_report_sha256": fingerprint(self.report),
                "baseline_report_sha256": fingerprint(self.baseline) if self.baseline is not None else None,
                "issues": [asdict(issue) for issue in self.issues],
                "comparison": self.comparison.to_dict() if self.comparison is not None else None}

    def save(self, path, *, overwrite=False):
        """Save one decision JSON file; source audit reports remain separate."""
        return write_json(path, self.to_dict(), overwrite=overwrite)

    def to_junit(self):
        """One policy decision becomes one CI test; unassessable evidence is an error."""
        suite = ET.Element("testsuite", name="proofml.release", tests="1",
                           failures=str(int(self.status == "failed")), errors=str(int(self.status in {"insufficient", "incompatible"})))
        case = ET.SubElement(suite, "testcase", classname="proofml", name=_xml_text(self.policy.name))
        if not self.passed:
            node = ET.SubElement(case, "failure" if self.status == "failed" else "error", message=f"Release gate {self.status}")
            node.text = _xml_text("\n".join(f"{issue.code}: {issue.message} {json.dumps(issue.evidence, sort_keys=True)}" for issue in self.issues))
        return ET.tostring(suite, encoding="unicode", xml_declaration=True)

    def save_junit(self, path, *, overwrite=False):
        return write_text(path, self.to_junit() + "\n", overwrite=overwrite)


@dataclass(frozen=True)
class AuditPolicy:
    """Opt-in evidence and quality requirements, never inferred from business intent.

    Keys for metric rules are 'check_id.metric_name'. Coverage keys name checks;
    min_rows keys name datasets (e.g. train, test, evaluation). Missing evidence
    blocks the gate. max_drop/max_increase require a compatible baseline.
    """
    name: str = "release"
    severity: str = "high"
    require_checks: tuple[str, ...] = ()
    min_rows: Mapping[str, int] = field(default_factory=dict)
    min_coverage: Mapping[str, float] = field(default_factory=dict)
    min_evaluated: Mapping[str, int] = field(default_factory=dict)
    min_metrics: Mapping[str, float] = field(default_factory=dict)
    max_metrics: Mapping[str, float] = field(default_factory=dict)
    max_drop: Mapping[str, float] = field(default_factory=dict)
    max_increase: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Policy name must be nonempty")
        if self.severity not in SEVERITY_ORDER:
            raise ValueError("Policy severity must be low, medium, high, or critical")
        if not isinstance(self.require_checks, (tuple, list)) or any(not isinstance(v, str) or not v for v in self.require_checks):
            raise ValueError("require_checks must be a list/tuple of nonempty check IDs")
        if len(set(self.require_checks)) != len(self.require_checks):
            raise ValueError("require_checks cannot contain duplicate IDs")
        object.__setattr__(self, "require_checks", tuple(self.require_checks))
        for name in ("min_rows", "min_coverage", "min_evaluated", "min_metrics", "max_metrics", "max_drop", "max_increase"):
            values = getattr(self, name)
            if not isinstance(values, Mapping):
                raise TypeError(f"{name} must map names to numeric requirements")
            values = dict(values)
            for key, value in values.items():
                if not isinstance(key, str) or not key.strip() or not _finite(value):
                    raise ValueError(f"{name} requires nonempty names and finite numeric values")
                if name in {"min_rows", "min_evaluated"} and (type(value) is not int or value < 1):
                    raise ValueError(f"{name} requires positive integer counts")
                if name == "min_coverage" and not 0 < value <= 1:
                    raise ValueError("min_coverage must be in (0, 1]")
                if name in {"max_drop", "max_increase"} and value < 0:
                    raise ValueError("Regression tolerances must be nonnegative absolute differences")
                if name in {"min_metrics", "max_metrics", "max_drop", "max_increase"} and ("." not in key or key.startswith(".") or key.endswith(".")):
                    raise ValueError("Metric paths must use check_id.metric_name")
            object.__setattr__(self, name, MappingProxyType(values))
        for key in self.min_metrics.keys() & self.max_metrics.keys():
            if self.min_metrics[key] > self.max_metrics[key]:
                raise ValueError("Metric minimum cannot exceed its maximum")
        if self.max_drop.keys() & self.max_increase.keys():
            raise ValueError("Choose one regression direction per metric")

    def to_dict(self):
        result = {f.name: getattr(self, f.name) for f in fields(self)}
        return {"schema_version": "1.0", **{name: dict(value) if isinstance(value, Mapping) else list(value) if isinstance(value, tuple) else value
                                             for name, value in result.items()}}

    @classmethod
    def from_file(cls, path, *, max_bytes=1_000_000):
        return cls.from_dict(read_json(path, max_bytes=max_bytes))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.get("schema_version") != "1.0":
            raise ValueError("Policy requires schema_version '1.0'")
        if set(value) - {f.name for f in fields(cls)} - {"schema_version"}:
            raise ValueError("Unknown policy field")
        return cls(**{k: v for k, v in value.items() if k != "schema_version"})

    def save(self, path, *, overwrite=False):
        return write_json(path, self.to_dict(), overwrite=overwrite)

    def enforce(self, report, *, baseline=None):
        """Return a passing decision or raise AuditFailed with .report/.decision."""
        return self.evaluate(report, baseline=baseline).raise_for_issues()

    def evaluate(self, report: AuditReport, *, baseline: AuditReport | None = None) -> GateResult:
        if not isinstance(report, AuditReport) or (baseline is not None and not isinstance(baseline, AuditReport)):
            raise TypeError("Policy evaluation requires AuditReport instances")
        # Take validated snapshots: caller mutation after evaluation must not
        # silently change which evidence the saved decision claims to assess.
        report = report_from_dict(json.loads(json.dumps(report.to_dict(), allow_nan=False)))
        if baseline is not None:
            baseline = report_from_dict(json.loads(json.dumps(baseline.to_dict(), allow_nan=False)))
        issues = []

        def add(code, kind, message, **evidence):
            issues.append(GateIssue(code, kind, message, evidence))

        def evidence_requirements(current, role):
            checks = {check.check_id: check for check in current.checks}
            for check in current.checks:
                if check.status == "error":
                    add("check_error", "insufficient", f"{role}: check {check.check_id} errored.", role=role, check=check.check_id)
            for identifier in self.require_checks:
                check = checks.get(identifier)
                if check is None or check.status not in {"passed", "findings"}:
                    add("required_check_unassessed", "insufficient", f"{role}: required check {identifier} did not complete.", role=role, check=identifier)
            for dataset, minimum in self.min_rows.items():
                rows = current.datasets.get(dataset, {}).get("rows")
                if type(rows) is not int or rows < minimum:
                    add("insufficient_rows", "insufficient", f"{role}: dataset {dataset} does not meet the minimum row count.", role=role, dataset=dataset, observed=rows, minimum=minimum)
            for name in ("min_coverage", "min_evaluated"):
                for identifier, minimum in getattr(self, name).items():
                    check = checks.get(identifier)
                    if check is None or check.status not in {"passed", "findings"} or check.coverage is None:
                        add("coverage_unavailable", "insufficient", f"{role}: {identifier} has no completed, declared coverage.", role=role, check=identifier, rule=name)
                        continue
                    observed = check.coverage.fraction if name == "min_coverage" else check.coverage.evaluated
                    if observed is None or (minimum > observed if name == "min_evaluated" else _above(minimum, observed)):
                        add("insufficient_coverage" if name == "min_coverage" else "insufficient_evaluated", "insufficient",
                            f"{role}: {identifier} does not meet {name}.", role=role, check=identifier,
                            observed=observed, minimum=minimum, total=check.coverage.total, evaluated=check.coverage.evaluated, unit=check.coverage.unit)

        evidence_requirements(report, "candidate")
        for check in report.checks:
            for finding in check.findings:
                if SEVERITY_ORDER[finding.severity] >= SEVERITY_ORDER[self.severity]:
                    add("finding_threshold", "failed", f"candidate: {check.check_id}/{finding.code} meets the severity threshold.", check=check.check_id, finding=finding.code, severity=finding.severity)
        for rule in ("min_metrics", "max_metrics"):
            for path, limit in getattr(self, rule).items():
                value = metric_value(report, path)
                if value is None:
                    add("metric_unavailable", "insufficient", f"candidate: metric {path} is missing, ambiguous, or unassessed.", metric=path)
                elif _above(limit, value) if rule == "min_metrics" else _above(value, limit):
                    add("metric_threshold", "failed", f"candidate: {path} violates {rule}.", metric=path, observed=value, limit=limit, rule=rule)
        comparison = None
        if baseline is not None:
            evidence_requirements(baseline, "baseline")
            comparison = compare_reports(report, baseline)
            for reason in comparison.reasons:
                add("incompatible_baseline", "incompatible", reason)
        elif self.max_drop or self.max_increase:
            add("baseline_required", "insufficient", "Regression rules require a baseline report.")
        if comparison is not None and comparison.compatible:
            for rule in ("max_drop", "max_increase"):
                for path, tolerance in getattr(self, rule).items():
                    before, after = metric_value(baseline, path), metric_value(report, path)
                    if before is None or after is None:
                        add("regression_metric_unavailable", "insufficient", f"Both reports must assess metric {path}.", metric=path)
                        continue
                    deterioration = before - after if rule == "max_drop" else after - before
                    if not _finite(deterioration):
                        add("metric_difference_nonfinite", "insufficient", f"Metric difference for {path} exceeds finite numeric range.", metric=path)
                    elif _above(deterioration, tolerance):
                        add("metric_regression", "failed", f"{path} changed from {before:g} to {after:g}, exceeding {rule} tolerance {tolerance:g}.", metric=path,
                            baseline=before, candidate=after, deterioration=deterioration, tolerance=tolerance, rule=rule)
        return GateResult(self, report, baseline, tuple(issues), comparison)
