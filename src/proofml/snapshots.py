"""Bounded, strict JSON snapshots. Input files are data, never executable objects."""
from dataclasses import fields
import json
import os
from pathlib import Path
import re
import tempfile

from .models import AuditReport, CheckResult, Coverage, Finding


def read_json(path, *, max_bytes=100_000_000):
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    def invalid_constant(_):
        raise ValueError("Nonfinite JSON numbers are not supported")
    with Path(path).open("rb") as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("JSON snapshot exceeds max_bytes")
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=unique, parse_constant=invalid_constant)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds supported depth") from error


def write_text(path, payload, *, overwrite=False):
    """Atomic single-file publication with the same overwrite policy as reports."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".proofml-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            os.link(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def write_json(path, payload, *, overwrite=False):
    return write_text(path, json.dumps(payload, allow_nan=False, indent=2) + "\n", overwrite=overwrite)


def _object(value, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("Invalid or unknown fields in report snapshot")


def report_from_dict(value):
    """Validate structure and recompute derived fields; support pre-coverage reports."""
    _object(value, {f.name for f in fields(AuditReport)} | {"summary", "metrics"})
    if value.get("schema_version") != "1.0":
        raise ValueError("Unsupported report schema_version")
    for name in ("tool_version", "created_at"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise ValueError("Report version and timestamp must be nonempty strings")
    if not isinstance(value.get("config"), dict) or not isinstance(value.get("datasets"), dict) or not value["datasets"]:
        raise ValueError("Report requires configuration and dataset metadata")
    for name, data in value["datasets"].items():
        if not isinstance(name, str) or not name or not isinstance(data, dict):
            raise ValueError("Invalid dataset metadata")
        if type(data.get("rows")) is not int or data["rows"] < 0:
            raise ValueError("Invalid dataset row count")
        columns = data.get("columns")
        if not isinstance(columns, list) or any(not isinstance(c, str) for c in columns):
            raise ValueError("Invalid dataset columns")
        if not isinstance(data.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", data["sha256"]):
            raise ValueError("Invalid dataset fingerprint")
    if not isinstance(value.get("checks"), (list, tuple)):
        raise ValueError("Report checks must be an array")
    checks = []
    for record in value["checks"]:
        _object(record, {f.name for f in fields(CheckResult)})
        if not isinstance(record.get("check_id"), str) or not record["check_id"] or not isinstance(record.get("reason", ""), str):
            raise ValueError("Invalid check identity or reason")
        findings = []
        if not isinstance(record.get("findings", []), (list, tuple)):
            raise ValueError("Check findings must be an array")
        for finding in record.get("findings", []):
            _object(finding, {f.name for f in fields(Finding)})
            for field in ("code", "severity", "confidence", "title", "explanation", "recommendation"):
                if not isinstance(finding.get(field), str):
                    raise ValueError("Invalid finding text")
            columns = finding.get("columns", [])
            if not isinstance(columns, (list, tuple)) or any(not isinstance(c, str) for c in columns) or not isinstance(finding.get("evidence", {}), dict):
                raise ValueError("Invalid finding evidence or columns")
            findings.append(Finding(**{**finding, "columns": tuple(columns)}))
        coverage = record.get("coverage")
        if coverage is not None:
            _object(coverage, {f.name for f in fields(Coverage)})
            coverage = Coverage(**coverage)
        if not isinstance(record.get("metrics", {}), dict):
            raise ValueError("Metrics must be an object")
        checks.append(CheckResult(**{**record, "findings": tuple(findings), "coverage": coverage}))
    ids = [check.check_id for check in checks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate check IDs in report")
    config = dict(value["config"])
    for key in ("disabled_checks", "unavailable_features"):
        if isinstance(config.get(key), list):
            config[key] = tuple(config[key])
    report = AuditReport(**{**{k: value[k] for k in ("schema_version", "tool_version", "created_at", "datasets")},
                           "config": config, "checks": tuple(checks)})
    # Numeric overflow such as 1e999 is valid JSON syntax but not valid evidence.
    json.dumps(report.to_dict(), allow_nan=False)
    for derived in ("summary", "metrics"):
        if derived in value and value[derived] != report.to_dict()[derived]:
            raise ValueError("Derived report fields disagree with check results")
    return report


def load_report(path, *, max_bytes=100_000_000):
    try:
        return report_from_dict(read_json(path, max_bytes=max_bytes))
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Malformed report snapshot") from error
