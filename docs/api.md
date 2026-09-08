# Public API: start small, add context when needed

## Files or DataFrames

```python
from proofml import audit

report = audit("train.csv", target="churn")
report.save("audit-report")
```

`audit(train, test=None, *, target=None, y=None, y_test=None, task=None, config=None, checks=None,
**options)` accepts paths or pandas DataFrames, including one of each.
`target` names a column; alternatively provide a separate `y` array/Series. Omit both for data-quality-only
assessment; target checks then explicitly skip. The task defaults to
classification, so declare regression for continuous numeric targets.

```python
report = audit(train_df, test_df, target="price", task="regression")
```

Inputs are not modified. DataFrame indexes are ignored; call `reset_index()` if
the index holds a time or entity key you need to audit. DataFrame column names
must be unique strings. DataFrame support requires pandas, but plain CSV use
does not import it or require any additional package.

## Separate features and labels

For separate features and labels, use:

```python
report = audit(X_train, X_test, y=y_train, y_test=y_test, task="regression")
```

Dense 2D NumPy arrays and pandas DataFrames work through the pandas extra.
Labels must be one-dimensional and equal the feature row count. Pandas Series
labels must have exactly the same index/order as DataFrame features; array/list
labels attach positionally. Separate labels with file paths are rejected.
Array columns become `feature_0`, `feature_1`, etc. The default label column is
`__proofml_target__`; override it with `target="outcome"`. Existing features
cannot be overwritten. Without y_test, the holdout is unlabelled.

## Forecasting

For variable horizons, use `label_available_column="label_ready_at"` instead of
`label_horizon_seconds`. Training rows supply actual availability timestamps;
the column is excluded from candidate predictors and may be absent in test.

```python
report = audit(
    train_df, test_df,
    target="demand", task="forecasting",
    time_column="origin", series_id="store",
    expected_interval_seconds=86400,
    label_horizon_seconds=172800,
)
```

The example assumes daily origins and labels available two days later. Those
values are project facts, not universal defaults. Omit `series_id` for a single
series. See [the forecasting contract](forecasting.md).

## Work with the report

```python
print(report)                         # Summary includes skipped and errored checks
report.findings                      # Severity-sorted Finding objects
report.checks                        # Executed, skipped, and failed check results
report.to_dict()                     # JSON-compatible report
report.to_frame()                    # Optional pandas table, one row per finding
report.save("run-001")               # Writes report.html and report.json; returns both paths
report.save("run-001", overwrite=True)
```

In a notebook, display `report` to see the isolated offline HTML view. Findings
contain codes, severity, confidence, evidence, and suggested next steps.
An empty findings table does not imply every check was applicable.

## Fail a pipeline deliberately

```python
report.raise_for_issues(
    severity="high",
    require_checks=("split_overlap", "target_health"),
)
```

The method raises `proofml.AuditFailed` for findings at/above the threshold,
any errored check, or any required check that did not complete. The exception
retains `error.report`. Unknown required IDs also fail the gate. Skips are
permitted unless required explicitly. Calling `audit` alone never raises for
findings; invalid input/configuration still raises ValueError/TypeError or an
underlying I/O/format exception.

## Reusable configuration and custom checks

Every `AuditConfig` field is accepted directly as an `audit` keyword. Unknown
keywords fail; none are silently ignored. Direct arguments override config.
Omitted/None `target` and `task` preserve the config values. To remove a target
from an existing config, create a replacement `AuditConfig` with `target=None`.

```python
from proofml import AuditConfig, audit

policy = AuditConfig(task="regression", target="price", missing_threshold=0.1)
report = audit(train_df, test_df, config=policy)
```

The earlier `config=AuditConfig(...)` API remains supported. A supplied
`checks` collection replaces the task-specific registry. See the
[plugin guide](architecture.md). ProofML is an audit function, not a fitted
scikit-learn estimator; it does not implement `fit`, `predict`, or estimator cloning.
