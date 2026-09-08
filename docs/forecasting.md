# Forecasting audits: fixed-horizon holdouts

Use this mode for numeric forecasts such as daily demand per store or hourly
sensor measurements. Supply one train/test fold at a time. The audit assumes
one model fitting cutoff shared across all series: the earliest test origin.
It reads data without fitting a model or changing rows.

## Try the examples

```bash
proofml demo --problem forecasting --output reports/forecast-faulty --fail-on none
proofml demo --problem forecasting --clean --output reports/forecast-clean
```

The faulty synthetic example has duplicate store/timestamps with different
targets, backwards row order, missing daily observations, and training labels
that reach the holdout. The corrected example resolves these conditions and
produces zero findings. Skips remain visible; this is not a performance benchmark.

## Configuration

Each row represents one **forecast origin** for one series, with a numeric
target observed later. `time_column` is when the prediction is made, not when
the outcome occurs. Prepare this supervised representation before using the
label-boundary check. Save the following as `forecast.json`:

```json
{
  "task": "forecasting",
  "target": "demand",
  "time_column": "origin",
  "series_id": "store",
  "expected_interval_seconds": 86400,
  "label_horizon_seconds": 172800,
  "embargo_seconds": 0
}
```

```bash
proofml audit train.csv --test test.csv --config forecast.json --output reports/demand
proofml checks --task forecasting
```

Omit `series_id` for one series. Repeated series across splits are normal;
declare `entity_id` separately only if entity-overlap review is intended.
Series IDs are excluded from target-association and drift checks.

## Three additional modules

| ID | Evidence | Meaning |
| --- | --- | --- |
| `forecast_index` | Duplicate normalized times within series, backwards row order, invalid/missing index | Duplicates are confirmed; unsorted input is a contextual risk for positional lag calculations |
| `forecast_cadence` | Intervals differ from declared sampling duration; exact-multiple missing slots | Violation of a user-declared grid, not an inferred schedule |
| `forecast_label_boundary` | Training labels plus embargo reach the earliest test origin | Violation of a declared single-cutoff availability contract |

The nine common checks still run. Forecast targets use regression validation,
not class-imbalance rules. Forecasting enables strict chronological holdout checking.

With a two-day label horizon, a January 9 origin has a label available January
11. Testing on January 10 is too early despite the earlier origin. The rule is:

```text
train origin + label_horizon_seconds + embargo_seconds < earliest test origin
```

Equality fails deliberately: availability must be strictly before prediction.
Include reporting delay in the horizon. Zero means labels are available
immediately. No horizon means an explicit skip. Embargo requires a horizon.

## Coverage and limits

- Target and time column are required. Forecast-specific settings are rejected
  for other tasks rather than silently ignored. Durations are integer elapsed
  seconds, capped at 315576000 (ten Julian years); cadence must be positive.
- ISO 8601 timestamps normalize to UTC; naive values mean UTC. Equivalent
  timezone spellings represent the same timestamp.
- Cadence is fixed elapsed time, not months, business calendars, holidays, or
  local days across DST. Omit cadence when that contract does not fit.
- Cadence checks splits separately, excluding deliberate gaps between train and
  test. Single-timestamp groups cannot establish cadence and appear in coverage.
- Invalid times/keys produce index findings; dependent checks skip explicitly.
- The global cutoff suits one pooled model. Independent per-series models,
  rolling retraining, variable horizons, and event-time labels need separate
  audits/contracts. This does not inspect lag-feature code, centered windows,
  future covariates, or model performance. No automatic repairs are made.
- Evidence omits raw series names and timestamps, retaining counts and column
  names. Grouping is O(n); cadence sorting is O(n log n) worst case with O(n)
  extra memory. Existing input limits apply; CSV needs no extra dependencies.

For custom plugins use `checks=[*default_checks("forecasting"), MyCheck()]`.
A custom registry replaces automatic task-based selection.
