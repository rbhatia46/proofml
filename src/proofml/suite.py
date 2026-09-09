"""Named audit collections for slices, folds, or independent pipeline stages."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from xml.etree import ElementTree as ET

from .exceptions import AuditFailed
from .models import AuditReport
from .policy import AuditPolicy, GateIssue, GateResult, _xml_text
from .snapshots import read_json, report_from_dict, write_json, write_text

MAX_REPORTS = 1000


def _names(values):
    if any(not isinstance(name, str) or not name.strip() or len(name) > 256 for name in values):
        raise ValueError("Report names must be nonempty strings of at most 256 characters")


@dataclass(frozen=True)
class AuditSuite:
    """Keep per-report evidence separate. Overlapping slices are not pooled."""
    reports: Mapping[str, AuditReport]

    def __post_init__(self):
        if not isinstance(self.reports, Mapping) or not 1 <= len(self.reports) <= MAX_REPORTS:
            raise ValueError(f"An audit suite requires 1 to {MAX_REPORTS} named reports")
        _names(self.reports)
        if any(not isinstance(report, AuditReport) for report in self.reports.values()):
            raise TypeError("Suite values must be AuditReport instances, not nested suites")
        object.__setattr__(self, "reports", MappingProxyType(dict(self.reports)))

    @property
    def metrics(self):
        return {name: report.metrics for name, report in self.reports.items()}

    @property
    def summary(self):
        return {"reports": len(self.reports), "findings": sum(len(r.findings) for r in self.reports.values()),
                "errored_checks": sum(c.status == "error" for r in self.reports.values() for c in r.checks),
                "skipped_checks": sum(c.status == "skipped" for r in self.reports.values() for c in r.checks)}

    def to_dict(self):
        return {"schema_version": "1.0", "kind": "audit_suite", "reports": {name: r.to_dict() for name, r in self.reports.items()}}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"schema_version", "kind", "reports"} or value.get("schema_version") != "1.0" or value.get("kind") != "audit_suite":
            raise ValueError("Unsupported or malformed suite snapshot")
        reports = value["reports"]
        if not isinstance(reports, dict) or not 1 <= len(reports) <= MAX_REPORTS:
            raise ValueError("Invalid suite report count")
        _names(reports)
        return cls({name: report_from_dict(report) for name, report in reports.items()})

    @classmethod
    def load(cls, path, *, max_bytes=100_000_000):
        try:
            return cls.from_dict(read_json(path, max_bytes=max_bytes))
        except (KeyError, TypeError, AttributeError, OverflowError) as error:
            raise ValueError("Malformed suite snapshot") from error

    def save(self, output, *, overwrite=False):
        """Save suite.json and a portable summary in suite.html; no name-based paths."""
        from .suite_reporting import render_suite
        root = Path(output)
        paths = (root / "suite.json", root / "suite.html")
        if not overwrite and any(p.exists() or p.is_symlink() for p in paths):
            raise FileExistsError("Suite output exists; use a new directory or overwrite=True")
        # Render and validate before publishing either file. As with reports,
        # each file is atomic but the pair is not a filesystem transaction.
        payload = json.dumps(self.to_dict(), allow_nan=False, indent=2) + "\n"
        html = render_suite(self)
        write_text(paths[0], payload, overwrite=overwrite)
        write_text(paths[1], html, overwrite=overwrite)
        return paths

    def to_frame(self):
        """Optional tidy pandas table: one row per report/check/metric."""
        try:
            import pandas as pd
        except ImportError as error:
            raise ImportError("to_frame() requires the pandas extra") from error
        columns = ["report", "check", "metric", "value", "evaluated", "total", "unit"]
        rows = []
        for name, report in self.reports.items():
            for check in report.checks:
                for metric, value in check.metrics.items():
                    coverage = check.coverage
                    rows.append([name, check.check_id, metric, value, coverage.evaluated if coverage else None,
                                 coverage.total if coverage else None, coverage.unit if coverage else None])
        return pd.DataFrame(rows, columns=columns)

    def __str__(self):
        return f"ProofML suite: {self.summary['reports']} reports, {self.summary['findings']} findings, {self.summary['errored_checks']} errored checks"

    def _repr_html_(self):
        from html import escape
        from .suite_reporting import render_suite
        return '<iframe title="ProofML suite" sandbox="" style="width:100%;height:720px;border:0" srcdoc="' + escape(render_suite(self), quote=True) + '"></iframe>'


@dataclass(frozen=True)
class SuiteResult:
    policy: SuitePolicy
    results: Mapping[str, GateResult]
    issues: tuple[GateIssue, ...] = ()

    def __post_init__(self):
        if not isinstance(self.results, Mapping) or not self.results or any(not isinstance(value, GateResult) for value in self.results.values()):
            raise ValueError("SuiteResult requires named gate results")
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))

    @property
    def status(self):
        states = {issue.kind for issue in self.issues} | {result.status for result in self.results.values()}
        return next((state for state in ("incompatible", "insufficient", "failed") if state in states), "passed")

    @property
    def passed(self):
        return self.status == "passed"

    def to_dict(self):
        from dataclasses import asdict
        return {"schema_version": "1.0", "kind": "suite_decision", "status": self.status,
                "passed": self.passed, "policy": self.policy.to_dict(), "issues": [asdict(i) for i in self.issues],
                "results": {name: result.to_dict() for name, result in self.results.items()}}

    def raise_for_issues(self):
        if not self.passed:
            details = [i.message for i in self.issues]
            details.extend(f"{name}: {result.status}" for name, result in self.results.items() if not result.passed)
            error = AuditFailed(f"Suite gate {self.status}: " + "; ".join(details), AuditSuite({name: result.report for name, result in self.results.items()}))
            error.decision = self
            raise error
        return self

    def save(self, path, *, overwrite=False):
        return write_json(path, self.to_dict(), overwrite=overwrite)

    def to_junit(self):
        suite = ET.Element("testsuite", name="proofml.suite")
        for name, result in self.results.items():
            case = ET.fromstring(result.to_junit()).find("testcase")
            case.set("name", _xml_text(name))
            case.set("classname", "proofml.slices")
            suite.append(case)
        if self.issues:
            case = ET.SubElement(suite, "testcase", classname="proofml.suite", name="required_reports")
            error = ET.SubElement(case, "error", message="Suite evidence incomplete")
            error.text = _xml_text("\n".join(i.message for i in self.issues))
        cases = list(suite)
        suite.set("tests", str(len(cases)))
        suite.set("failures", str(sum(c.find("failure") is not None for c in cases)))
        suite.set("errors", str(sum(c.find("error") is not None for c in cases)))
        return ET.tostring(suite, encoding="unicode", xml_declaration=True)

    def save_junit(self, path, *, overwrite=False):
        return write_text(path, self.to_junit() + "\n", overwrite=overwrite)


@dataclass(frozen=True)
class SuitePolicy:
    """Apply a default to every supplied report; overrides replace it, not merge.

    Every override is implicitly required, so misspelled names cannot silently
    disable protection. A supplied baseline must contain exactly the same names.
    """
    default: AuditPolicy = field(default_factory=AuditPolicy)
    overrides: Mapping[str, AuditPolicy] = field(default_factory=dict)
    require_reports: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.default, AuditPolicy) or not isinstance(self.overrides, Mapping):
            raise TypeError("SuitePolicy needs an AuditPolicy default and a mapping of overrides")
        if len(self.overrides) > MAX_REPORTS:
            raise ValueError("Too many policy overrides")
        _names(self.overrides)
        if any(not isinstance(value, AuditPolicy) for value in self.overrides.values()):
            raise TypeError("Overrides must be AuditPolicy instances")
        if not isinstance(self.require_reports, (tuple, list)) or len(self.require_reports) > MAX_REPORTS:
            raise ValueError("require_reports must be a bounded list/tuple of report names")
        _names(self.require_reports)
        if len(set(self.require_reports)) != len(self.require_reports):
            raise ValueError("Duplicate required report names")
        object.__setattr__(self, "overrides", MappingProxyType(dict(self.overrides)))
        object.__setattr__(self, "require_reports", tuple(self.require_reports))

    def to_dict(self):
        return {"schema_version": "1.0", "kind": "suite_policy", "default": self.default.to_dict(),
                "overrides": {name: p.to_dict() for name, p in self.overrides.items()}, "require_reports": list(self.require_reports)}

    @classmethod
    def from_file(cls, path, *, max_bytes=1_000_000):
        value = read_json(path, max_bytes=max_bytes)
        allowed = {"schema_version", "kind", "default", "overrides", "require_reports"}
        if not isinstance(value, dict) or set(value) - allowed or value.get("schema_version") != "1.0" or value.get("kind") != "suite_policy":
            raise ValueError("Unsupported or malformed suite policy")
        overrides = value.get("overrides", {})
        if not isinstance(overrides, dict) or len(overrides) > MAX_REPORTS:
            raise ValueError("Invalid suite policy overrides")
        return cls(default=AuditPolicy.from_dict(value.get("default", {"schema_version": "1.0"})),
                   overrides={name: AuditPolicy.from_dict(p) for name, p in overrides.items()},
                   require_reports=value.get("require_reports", ()))

    def save(self, path, *, overwrite=False):
        return write_json(path, self.to_dict(), overwrite=overwrite)

    def evaluate(self, suite: AuditSuite, *, baseline: AuditSuite | None = None) -> SuiteResult:
        if not isinstance(suite, AuditSuite) or (baseline is not None and not isinstance(baseline, AuditSuite)):
            raise TypeError("Suite policies require AuditSuite instances")
        results, issues = {}, []
        required = set(self.require_reports) | set(self.overrides)
        for name in sorted(required - set(suite.reports)):
            issues.append(GateIssue("required_report_missing", "insufficient", f"Required report {name} is absent from the candidate.", {"report": name}))
        if baseline is not None:
            for name in sorted((required | set(suite.reports)) - set(baseline.reports)):
                issues.append(GateIssue("baseline_report_missing", "insufficient", f"Report {name} is absent from the baseline.", {"report": name}))
            for name in sorted(set(baseline.reports) - set(suite.reports)):
                issues.append(GateIssue("candidate_report_removed", "insufficient", f"Baseline report {name} is absent from the candidate.", {"report": name}))
        for name, report in suite.reports.items():
            policy = self.overrides.get(name, self.default)
            before = baseline.reports.get(name) if baseline is not None else None
            results[name] = policy.evaluate(report, baseline=before)
        return SuiteResult(self, results, tuple(issues))

    def enforce(self, suite, *, baseline=None):
        return self.evaluate(suite, baseline=baseline).raise_for_issues()
