"""Serializable contracts shared by checks and report renderers."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal

Severity = Literal["critical", "high", "medium", "low"]
Confidence = Literal["confirmed", "suspicious", "needs_context"]
Status = Literal["passed", "findings", "skipped", "error"]
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass(frozen=True)
class Finding:
    """Evidence contains counts and column names, never raw values or labels."""
    code: str
    severity: Severity
    confidence: Confidence
    title: str
    explanation: str
    recommendation: str
    columns: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in SEVERITY_ORDER:
            raise ValueError(f"Invalid severity: {self.severity}")
        if self.confidence not in {"confirmed", "suspicious", "needs_context"}:
            raise ValueError(f"Invalid confidence: {self.confidence}")
        if not self.code or not self.title:
            raise ValueError("Findings need a stable code and a title")


@dataclass(frozen=True)
class Coverage:
    """Check-specific assessed population; counts do not establish statistical power."""
    total: int
    evaluated: int
    unit: str = "observations"

    def __post_init__(self):
        if type(self.total) is not int or type(self.evaluated) is not int or not 0 <= self.evaluated <= self.total:
            raise ValueError("Coverage requires integer counts with 0 <= evaluated <= total")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("Coverage requires a nonempty unit")

    @property
    def fraction(self) -> float | None:
        return self.evaluated / self.total if self.total else None


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: Status
    findings: tuple[Finding, ...] = ()
    reason: str = ""
    metrics: dict[str, int | float] = field(default_factory=dict)
    coverage: Coverage | None = None

    def __post_init__(self) -> None:
        if self.status not in {"passed", "findings", "skipped", "error"}:
            raise ValueError("Invalid check status")
        if bool(self.findings) != (self.status == "findings"):
            raise ValueError("Only a findings result can contain findings, and it cannot be empty")
        if self.status in {"skipped", "error"} and not self.reason:
            raise ValueError("Skipped and error results require an explanation")
        if self.metrics and self.status not in {"passed", "findings"}:
            raise ValueError("Only completed checks may expose metrics")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name or type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("Metrics need nonempty names and finite numeric values")
        if self.coverage is not None:
            if not isinstance(self.coverage, Coverage):
                raise TypeError("coverage must be Coverage or None")
            self.coverage.__post_init__()
            if self.status == "error" or (self.status == "skipped" and self.coverage.evaluated):
                raise ValueError("Uncompleted checks cannot claim evaluated coverage")

    @classmethod
    def complete(cls, check_id: str, findings: list[Finding], *, metrics: dict[str, int | float] | None = None,
                 coverage: Coverage | None = None) -> CheckResult:
        return cls(check_id, "findings" if findings else "passed", tuple(findings), metrics=metrics or {}, coverage=coverage)


@dataclass(frozen=True)
class AuditReport:
    schema_version: str
    tool_version: str
    created_at: str
    config: dict[str, Any]
    datasets: dict[str, dict[str, Any]]
    checks: tuple[CheckResult, ...]

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(sorted(
            (f for check in self.checks for f in check.findings),
            key=lambda f: (-SEVERITY_ORDER[f.severity], f.code, f.columns),
        ))

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "findings": len(self.findings),
            "severity_counts": {s: sum(f.severity == s for f in self.findings) for s in reversed(SEVERITY_ORDER)},
            "check_counts": {s: sum(c.status == s for c in self.checks) for s in ("passed", "findings", "skipped", "error")},
        }

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "summary": self.summary, "metrics": self.metrics}

    @property
    def metrics(self) -> dict[str, dict[str, int | float]]:
        """Measurements grouped by check ID; omitted checks have no measurements."""
        return {check.check_id: dict(check.metrics) for check in self.checks if check.metrics}

    def save(self, output: str, *, overwrite: bool = False):
        """Save report.html and report.json in a directory; return both paths."""
        from .reporting import write_reports
        return write_reports(self, output, overwrite=overwrite)

    @classmethod
    def load(cls, path, *, max_bytes: int = 100_000_000) -> AuditReport:
        """Load a validated report.json snapshot; never unpickle or execute code."""
        from .snapshots import load_report
        return load_report(path, max_bytes=max_bytes)

    def compare(self, baseline: AuditReport):
        """Compare retrieval metrics only when evaluation identities are compatible."""
        from .comparison import compare_reports
        return compare_reports(self, baseline)

    def to_frame(self):
        """Return one pandas row per finding, with a stable empty-table schema.

        Coverage lives in ``report.checks``; an empty findings table alone does
        not mean every check ran. pandas is an optional dependency.
        """
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("to_frame() requires pandas; install the pandas extra") from error
        from dataclasses import fields
        return pd.DataFrame([asdict(f) for f in self.findings], columns=[f.name for f in fields(Finding)])

    def raise_for_issues(self, severity: Severity = "high", *, require_checks: tuple[str, ...] = ()) -> None:
        """Raise AuditFailed for failed checks, missing coverage, or findings.

        ``require_checks`` lists IDs that must run successfully. Without it,
        skipped checks are permitted, but errored checks always fail the gate.
        """
        from .exceptions import AuditFailed
        if severity not in SEVERITY_ORDER:
            raise ValueError("severity must be low, medium, high, or critical")
        if isinstance(require_checks, str):
            raise TypeError("require_checks must be a collection of check IDs, not a string")
        completed = {c.check_id for c in self.checks if c.status in {"passed", "findings"}}
        missing = set(require_checks) - completed
        errors = sum(c.status == "error" for c in self.checks)
        issues = sum(SEVERITY_ORDER[f.severity] >= SEVERITY_ORDER[severity] for f in self.findings)
        if errors or missing or issues:
            raise AuditFailed(f"Audit gate failed: {issues} findings at/above {severity}, {errors} errored checks, "
                              f"{len(missing)} required checks unassessed.", self)

    def __str__(self) -> str:
        counts = self.summary["check_counts"]
        return (f"ProofML: {len(self.findings)} findings; {counts['passed']} checks passed, "
                f"{counts['findings']} with findings, {counts['skipped']} skipped, {counts['error']} errored")

    def _repr_html_(self) -> str:
        """Isolate report styling from the surrounding notebook document."""
        from html import escape
        from .reporting import render_html
        return ('<iframe title="ProofML audit report" sandbox="" '
                'style="width:100%;height:720px;border:0" srcdoc="'
                + escape(render_html(self), quote=True) + '"></iframe>')
