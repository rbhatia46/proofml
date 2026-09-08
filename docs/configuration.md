# Configuration reference

```json
{
  "task": "classification",
  "target": "churn",
  "entity_id": "customer_id",
  "time_column": "snapshot_date",
  "require_disjoint_entities": true,
  "expect_temporal_split": true,
  "unavailable_features": ["cancellation_recorded"],
  "missing_threshold": 0.2,
  "imbalance_threshold": 0.1,
  "association_threshold": 0.98,
  "drift_threshold": 0.25,
  "min_samples": 30,
  "max_rows": 200000,
  "max_bytes": 100000000,
  "disabled_checks": []
}
```

All fields are optional; omit semantic fields when unknown. `task` defaults to
classification and accepts regression or forecasting. A target is not guessed. Column names
are case sensitive after stripping surrounding whitespace. Declared columns
must exist in training data. CLI target/task/entity/time flags override JSON.
Paths are supplied on the command line, not embedded in configuration.

Declare `require_disjoint_entities` only when evaluating unseen entities.
Declare `expect_temporal_split` only for a strictly future holdout. Missing
context results in skipped checks or contextual findings, not invented intent.
An empty unavailable-features list means availability was not assessed.

`disabled_checks` accepts IDs from `proofml checks`. Unknown fields, duplicate
JSON keys, unknown check IDs, invalid thresholds, and malformed datasets fail
with exit code 2. Row and byte limits refuse the audit instead of sampling.

Forecasting additionally accepts `series_id`, `expected_interval_seconds`,
`label_horizon_seconds`, and `embargo_seconds`. See the
[forecasting contract](forecasting.md) before choosing these settings.

## CI exit codes

| Code | Meaning |
| --- | --- |
| 0 | Audit completed, no finding at the configured failure threshold |
| 1 | At least one finding meets/exceeds `--fail-on` (default: high) |
| 2 | Invalid input/config, failed check, or report-write failure |

`--fail-on none` disables severity gating, never check-error detection. Low,
medium, high, critical are ordered thresholds. Skipped checks do not cause a
failure; inspect coverage and require expected checks in your own integration.
Findings, including suspicious findings, can fail CI at the configured severity.

Outputs are `report.html` and `report.json`. Existing reports are protected by
default; use `--overwrite` for intentional replacement. Each file is atomic,
but the two-file output is not a transaction. Use one unique output directory
per concurrent job. A partial failure may leave one complete report.
