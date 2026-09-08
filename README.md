# ProofML

**Catch data problems before you trust your model's results.**

ProofML audits tabular ML datasets for split overlap, unavailable predictors,
suspicious target associations, missing labels, and distribution shifts. Every
finding includes evidence and a next step. Every skipped check explains why.

**No API key. No LLM download. No account. No runtime dependencies for CSV.**

## Three lines to an audit

```python
from proofml import audit
report = audit("train.csv", target="churn")
report.save("audit-report")
```

Already using pandas? Pass your DataFrames directly:

```python
report = audit(train_df, test_df, target="price", task="regression")
report.to_frame()              # Findings as a pandas table
report.raise_for_issues()      # Optional pipeline gate; raises on high/critical findings or check errors
```

No config file is required. Display `report` in a notebook for an embedded HTML
report, or `print(report)` for findings and check coverage. Start with
[the short API guide](docs/api.md) and [the compatibility matrix](docs/compatibility.md).

**Publication status:** a public PyPI release is not yet verified. Install from
this checkout with `python -m pip install .`, or from the supplied `.whl` file.
The intended post-publication command is `python -m pip install proofml`.
Until then, install directly from GitHub:

```bash
python -m pip install git+https://github.com/rbhatia46/proofml.git
```

See [release instructions](docs/releasing.md) for the remaining account setup.

## See the problem in one minute

From a checkout of this repository:

```bash
python -m pip install .
proofml demo --output proofml-report --fail-on none
```

Open `proofml-report/report.html`. The bundled synthetic churn example produces:

```text
ProofML: 8 findings | 2 passed | 7 findings | 0 skipped | 0 error

[CRITICAL / confirmed] Unavailable predictor present: cancellation_recorded
[HIGH / confirmed] Entities appear in both splits
[HIGH / suspicious] Predictor closely matches target: cancellation_recorded
[HIGH / confirmed] Test data is not strictly after training
[HIGH / confirmed] Test observations also occur in training
[MEDIUM / confirmed] Repeated rows in train
[MEDIUM / suspicious] Distribution differs: monthly_charge
[MEDIUM / confirmed] Missing values in monthly_charge
```

The critical finding uses explicitly declared prediction-time availability.
ProofML does not magically infer what your business knows at prediction time.
The demo data and its assumptions are bundled in `src/proofml/demo_data/`.

Compare a corrected example with unique customers, a chronological holdout,
complete inputs, and the unavailable predictor removed:

```bash
proofml demo --clean --output proofml-clean
```

This example has zero findings and explicitly skips feature-availability review
because no unavailable columns remain declared. Both examples are synthetic
teaching fixtures, not a benchmark of real-world detection accuracy.

## Use your own data

```bash
# Training data only: target/data checks, explicit skips for unavailable split checks.
proofml audit train.csv --target churn --output reports/churn

# Add a holdout for split and distribution checks.
proofml audit train.csv --test test.csv --target churn --output reports/holdout

# Regression and project-specific assumptions.
proofml audit train.csv --test test.csv --task regression --target price \
  --config audit.json --output reports/prices
```

Audit the columns your model will actually receive. Unlabelled test datasets
are allowed. Use `--overwrite` to replace an existing report intentionally.

The CLI exits **1 for high/critical findings by default**, after writing both
reports. This is intentional CI behavior, not an execution failure. Use
`--fail-on none` for interactive exploration. Exit 2 means invalid input,
failed checks, or report-write errors.

For Parquet: `python -m pip install '.[parquet]'` from this checkout. The `pandas`
extra installs DataFrame support if you do not already have pandas. Python
3.10+ is required; supported pandas versions are 2.x. These extras add no cloud service.

## Different projects need different assumptions

```json
{
  "task": "classification",
  "target": "churn",
  "entity_id": "customer_id",
  "require_disjoint_entities": true,
  "time_column": "snapshot_date",
  "expect_temporal_split": true,
  "unavailable_features": ["cancellation_recorded"]
}
```

Shared customers are a violation when testing unseen customers. They can be
valid when predicting future behavior for existing customers. Your declared
intent determines the finding's interpretation. Unknown configuration fields
fail early so a typo cannot silently disable your intended check.

## What you get

| Output | Use |
| --- | --- |
| Self-contained HTML | Review findings, evidence, next steps, and coverage offline |
| Versioned JSON | Integrate with CI, notebooks, or an agent's tool interface |
| Source hashes and config | Identify which exact inputs and assumptions were audited |
| Independent check plugins | Add your team's rules without changing the engine |

Nine common modules cover data quality, target health, schema alignment, row overlap,
entity overlap, temporal order, feature availability, target association, and
feature drift. See [the check catalog](docs/checks.md) for thresholds and limits.

## Forecasting support (v0.2)

Three additional modules audit series/timestamp integrity, declared sampling
cadence, and label-horizon overlap at the training cutoff.

```bash
proofml demo --problem forecasting --output reports/forecast --fail-on none
proofml demo --problem forecasting --clean --output reports/forecast-clean
```

A chronological split can still leak future information when training labels
become available after testing starts. See the [forecasting guide](docs/forecasting.md)
for prediction-origin and label-horizon configuration, examples, and limits.

## Python API

```python
from proofml import audit

report = audit(
    "train.csv",
    test="test.csv",
    target="churn",
    entity_id="customer_id",
)
for finding in report.findings:
    print(finding.severity, finding.confidence, finding.title)

report.save("reports/run-001")
```

Use it before training, before merging a dataset change, or as a structured
tool for a data-science agent. [Agent tool example](examples/agent_tool.py).
The current release is a deterministic audit toolkit; it has no LLM planner.

## Honest scope

ProofML supports scalar CSV/Parquet classification, regression, and fixed-horizon
forecasting datasets with a shared holdout cutoff.
It does not certify a model or generate a pseudo-precise trust score. Strong
correlation is a suspicion, not proof of leakage. No-finding reports do not
imply that skipped checks passed.

Version 0.3 does not inspect notebooks, fitted pipelines, preprocessing order,
fairness, NLP/images, or arbitrary project code. It can audit exported inputs
from a scikit-learn workflow, but does not introspect the estimator. Those
capabilities require separate adapters and validated checks.

Inputs are read only; built-in reports omit raw row values. Column names,
statistics, and configuration remain visible. Default limits are 200,000 rows
and 100 MB per input, with no silent sampling. The engine is in-memory.

## Test and contribute

```bash
python -m pip install -e '.[parquet]'
python -m unittest discover -s tests -v
```

Tests cover faulty and clean data, project assumptions, malformed inputs,
plugin failures, privacy, HTML escaping, and CLI behavior. CI includes Linux,
macOS, Windows, and a separate optional Parquet job.

- [Configuration and CI contract](docs/configuration.md)
- [Check methods and limitations](docs/checks.md)
- [Architecture and plugin guide](docs/architecture.md)
- [Runnable custom check](examples/custom_check.py)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [MIT license](LICENSE)

Useful contributions start with a real failure case and a clean counterexample.
Help make an incorrect evaluation harder to ship.
