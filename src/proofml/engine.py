"""Deterministic orchestration. A failed check never becomes a passing audit."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from typing import Any
from ._version import __version__
from .checks import Check, default_checks
from .config import AuditConfig
from .context import AuditContext
from .data import load_dataset
from .models import AuditReport, CheckResult
from .inputs import attach_target


def audit(train: Any, test: Any = None, *, target: str | None = None,
          y: Any = None, y_test: Any = None,
          task: str | None = None, config: AuditConfig | None = None,
          checks: Iterable[Check] | None = None, **options: Any) -> AuditReport:
    """Audit files, pandas DataFrames, or dense 2D arrays without changing them.

    Examples::

        report = audit("train.csv", target="churn")
        report = audit(train_df, test_df, target="price", task="regression")
        report = audit(X_train, X_test, y=y_train, y_test=y_test)
        report.save("audit-report")

    ``target`` names a column in training data; test labels may be omitted.
    Separate ``y``/``y_test`` labels require in-memory features and the pandas
    extra. Series indexes must match DataFrame indexes exactly; array labels
    attach positionally. Neither feature columns nor labels are overwritten.
    ``task`` defaults to classification. All other AuditConfig fields may be
    supplied directly as keywords, e.g. ``series_id="store"``. Direct values
    override config fields; omitted/None target/task preserve the config value.
    Unknown keywords fail early. No task, target, or business intent is guessed.

    ``checks`` replaces the default task-specific registry.

    To extend it, pass `[*default_checks(), MyCheck()]`. Plugins run as trusted
    Python code. Exceptions are isolated and reported without raw exception
    messages, which can contain private data. Input/config errors raise.
    """
    values = config.to_dict() if config is not None else {}
    values.update(options)
    if target is not None:
        values["target"] = target
    if task is not None:
        values["task"] = task
    if y_test is not None and test is None:
        raise ValueError("y_test requires test features")
    if y is not None:
        values.setdefault("target", "__proofml_target__")
        if values["target"] is None:
            values["target"] = "__proofml_target__"
    if y_test is not None and not values.get("target"):
        raise ValueError("y_test requires training labels or an explicit target column")
    config = AuditConfig(**values)
    train = attach_target(train, y, config.target, config)
    test = attach_target(test, y_test, config.target, config)
    registry = tuple(default_checks(config.task) if checks is None else checks)
    ids = [check.id for check in registry]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Check IDs must be nonempty and unique")
    if set(config.disabled_checks) - set(ids):
        raise ValueError("disabled_checks contains an unknown check ID")
    train_data = load_dataset(train, config)
    test_data = load_dataset(test, config) if test is not None else None
    for column in (config.target, config.entity_id, config.time_column, config.series_id,
                   config.label_available_column, *config.unavailable_features):
        if column and column not in train_data.columns:
            raise ValueError(f"Declared column absent from training data: {column}")
    ctx = AuditContext(train_data, test_data, config)
    results = []
    for check in registry:
        if check.id in config.disabled_checks:
            results.append(CheckResult(check.id, "skipped", reason="Disabled explicitly in configuration."))
            continue
        try:
            result = check.run(ctx)
            if not isinstance(result, CheckResult) or result.check_id != check.id:
                raise ValueError("Invalid plugin result")
            # Reject non-JSON evidence and nonfinite numbers before rendering.
            from dataclasses import asdict
            json.dumps(asdict(result), allow_nan=False)
            results.append(result)
        except Exception as error:
            results.append(CheckResult(check.id, "error", reason=f"Check raised {type(error).__name__}; run it directly to debug with your data."))
    datasets = {"train": train_data.metadata()}
    if test_data is not None:
        datasets["test"] = test_data.metadata()
    return AuditReport("1.0", __version__, datetime.now(timezone.utc).isoformat(), config.to_dict(), datasets, tuple(results))
