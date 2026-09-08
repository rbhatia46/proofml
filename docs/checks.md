# Checks, evidence, and limitations

Audit the actual candidate model inputs after choosing columns. An unused
column in a raw export may produce findings irrelevant to your model.

| Check ID | Detects | Needs | Limits |
| --- | --- | --- | --- |
| `data_quality` | Repeated rows, missingness, constant columns, mixed numeric strings | Train; optionally test | Missing CSV values mean empty/whitespace, not arbitrary sentinels |
| `target_health` | Missing/constant labels, nonfinite regression labels, minority class share | Declared train target | Unlabelled test sets allowed; no model metric validation |
| `split_schema` | Non-target column-name mismatches | Both splits | Column order may differ; type contracts are not enforced |
| `split_overlap` | Exact normalized non-target row overlap | Both splits, matching schemas | Includes ID/time; identical features can be legitimate; no fuzzy/entity matching |
| `entity_overlap` | Shared nonempty entity IDs | Both splits, declared ID | High/confirmed only with disjoint-entity requirement; otherwise needs context |
| `temporal_order` | Test timestamps at or before training end | Both splits, declared time, future-only holdout | ISO 8601; naive times treated as UTC; does not inspect label windows |
| `feature_availability` | Declared unavailable predictors present | User-supplied unavailable feature list | Availability cannot be inferred from a CSV |
| `target_association` | Near-exact target copies and strong Pearson association | Target; minimum paired rows | Pearson only regression/binary numeric labels; no arbitrary categorical or nonlinear leakage detection |
| `feature_drift` | Empirical numeric KS / categorical total variation distance | Both splits; minimum nonmissing rows per feature | Categorical union <=50 values; skips high-cardinality text; no p-values or multiple-testing guarantees |

Forecasting adds `forecast_index`, `forecast_cadence`, and `forecast_label_boundary`.
See [forecast methods](forecasting.md). Run `proofml checks` (or
`proofml checks --task forecasting`) to list IDs. `disabled_checks` records explicit skips;
misspelled IDs fail configuration validation.

## Interpretation

- **Confirmed** means the reported observation or explicitly declared contract
  violation is established. Exact overlap is confirmed; whether it is leakage
  still depends on how the data was collected.
- **Suspicious** means a heuristic warrants investigation. Strong correlation
  may reflect an excellent legitimate feature. No predictive accuracy is inferred.
- **Needs context** means project intent determines whether intervention is needed.

Severity controls review priority and CI, not statistical confidence. A clean
report only says enabled, applicable checks found no threshold violations. It
does not certify a model. Coverage lists skips and errors separately.

## Statistical settings

Defaults: missing fraction >=0.20, minority class share <0.10, target agreement
or absolute Pearson r >=0.98, drift distance >=0.25, minimum 30 paired/observed
rows. Drift excludes missing values; missingness is reported separately.
These are configurable review heuristics, not calibrated error probabilities.
Reaching 30 observations does not establish statistical power or independence.
Repeated measurements, small subgroups, and many checked features need expert
interpretation. Numeric class encodings with more than two classes never
receive Pearson-based association findings.

## Explicitly outside version 0.2

No notebook execution, AST inspection, scikit-learn estimator introspection,
preprocessing-order detection, fairness assessment, p-value validity audit,
NLP/image checks, automatic fixes, model training, or LLM explanation layer.
Forecasting checks cover fixed durations and a shared cutoff, not rolling
retraining or feature-generation code.
Future checks should earn inclusion through faulty and clean counterexamples.
