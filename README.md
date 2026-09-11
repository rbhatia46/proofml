# ProofML

**Catch data problems before you trust your model's results.**

ProofML audits tabular ML data, forecasting datasets, text corpora, and
search/RAG retrieval outputs. Catch contamination and broken evaluation
assumptions before trusting results. Every finding includes evidence and a next
step. Every skipped check explains why.

**No API key. No model download. No account. Zero runtime dependencies for CSV, text, and retrieval.**

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

Separate features and labels work too (DataFrames or dense NumPy arrays):

```python
report = audit(X_train, X_test, y=y_train, y_test=y_test)
```

No config file is required. Display `report` in a notebook for an embedded HTML
report, or `print(report)` for findings and check coverage. Start with
[the short API guide](docs/api.md) and [the compatibility matrix](docs/compatibility.md).

## One report interface across workflows

| Workflow | Entry point | Problems to review |
| --- | --- | --- |
| Tabular classification/regression | `audit(train, test, target="label")` | Split leakage, missing labels, schema/category changes, drift |
| Forecasting | `audit(..., task="forecasting")` | Series integrity, cadence, labels crossing the evaluation cutoff |
| Text/NLP corpora | `audit_text(train_texts, test_texts)` | Empty documents, exact/lexical duplicates, inconsistent labels |
| Search/RAG retrieval | `audit_retrieval(retrieved, relevant, k=5)` | Ranking quality, missing judgments, duplicate or unreachable IDs |

```python
from proofml import audit_text, audit_retrieval

text_report = audit_text(train_texts, test_texts)
retrieval_report = audit_retrieval(retrieved_ids, relevant_ids, k=5, min_recall=0.8)
retrieval_report.raise_for_issues(require_checks=("retrieval_ranking",))
```

All reports support `.save()`, `.findings`, `.checks`, `.metrics`, `.to_dict()`,
notebook display, and the same CI gate. Choose thresholds for your project;
the example's 0.8 recall is not a universal standard. Query-ID mappings are
recommended for retrieval to avoid accidental positional misalignment.

See the [text guide](docs/text.md), [retrieval guide](docs/retrieval.md), and
[runnable offline example](examples/text_and_retrieval.py). Text checks do not
understand meaning; retrieval metrics do not judge generated answers.

## Evidence-aware release gates

A high score on too few evaluated cases should not approve a release. Define
your evidence requirements once and reuse them in Python or CI:

```python
from proofml import AuditPolicy

policy = AuditPolicy(
    require_checks=("retrieval_ranking",),
    min_coverage={"retrieval_ranking": 0.95},
    min_evaluated={"retrieval_ranking": 100},
    min_metrics={"retrieval_ranking.recall@5": 0.8},
)
decision = policy.enforce(retrieval_report)
```

Thresholds are illustrative; choose them for your project. Policies work across
report types. Retrieval reports additionally support strict baseline comparisons
using stable query IDs and separate evaluation/output fingerprints. Missing
evidence and incompatible evaluations block the gate.

See [release policies, baseline comparisons, and JUnit CI output](docs/release-gates.md).
The earlier `report.raise_for_issues()` retains its simple severity/completion
semantics; it does not automatically require a sufficient evaluated population.

## Catch regressions hidden by an average

```python
from proofml import audit_retrieval_slices, AuditPolicy, SuitePolicy

suite = audit_retrieval_slices(retrieved_ids, relevant_ids, groups=query_to_language, k=5)
policy = SuitePolicy(
    default=AuditPolicy(min_coverage={"retrieval_ranking": 0.95},
                        min_evaluated={"retrieval_ranking": 20},
                        max_drop={"retrieval_ranking.recall@5": 0.02}),
    require_reports=("overall", "english", "spanish"),
)
decision = policy.enforce(suite, baseline=approved_suite)
```

Each slice has its own population, metrics, and compatibility checks. An overall
improvement cannot mask a slice failure under the declared policy. The thresholds
above are illustrative, not universal. General `AuditSuite` collections also
apply policies across tabular/text reports or cross-validation folds.

See [slice-aware evaluation and CI](docs/slices.md) and the
[runnable hidden-regression example](examples/slice_regression.py).

## Demonstrated on real data

**Start with the [executed notebook gallery](examples/README.md):**

- [Search/retrieval on BEIR SciFact](examples/notebooks/scifact_retrieval.ipynb):
  real local rankings, cohort regression gates, ID-integrity faults, and CI replay.
  On 300 test queries, removing abstracts from the index lowers recall@10 from
  0.7735 to 0.5476 under the documented TF-IDF recipe.
- [Bank Marketing feature availability](examples/notebooks/bank_marketing.ipynb):
  turn a prediction-time constraint into a persistent training guardrail.

Both notebooks include measured outputs and run without API keys. See their
source attribution and limits; the gallery includes local/offline setup.

On all 41,188 records of UCI's Bank Marketing additional dataset, our fixed-holdout
logistic-regression example scores ROC AUC **0.8211 with call duration** versus
**0.7483 without it**. Duration is unavailable before the call: ProofML blocks
that explicitly declared availability violation, regardless of the better score.
No synthetic leakage is injected. Other review findings remain after removal.

Read the [reproducible case study, measured results, and limitations](docs/case-studies/bank-marketing.md)
or run [the example](examples/bank_marketing.py). This is one experiment, not a
claim of general detection accuracy or production readiness.

## Install

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
ProofML: 8 findings | 3 passed | 7 findings | 0 skipped | 0 error

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

Ten common modules cover data quality, target health, schema alignment, row overlap,
entity overlap, temporal order, feature availability, target association, and
feature drift, and train/test compatibility. Compatibility checks flag unseen test
classes, new feature categories, numeric parsing changes, and increased missingness.
See [the check catalog](docs/checks.md) for thresholds and limits.

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

ProofML supports scalar tabular classification/regression, forecasting with a
shared holdout cutoff, text corpus integrity, and binary-judgment retrieval evaluation.
It does not certify a model or generate a pseudo-precise trust score. Strong
correlation is a suspicion, not proof of leakage. No-finding reports do not
imply that skipped checks passed.

Version 0.7 does not inspect notebooks, fitted pipelines, preprocessing order,
fairness, images/audio/video, text semantics, or arbitrary project code. It can audit exported inputs
from a scikit-learn workflow, but does not introspect the estimator. Those
capabilities require separate adapters and validated checks.

This is a pre-1.0 package, not a substitute for scikit-learn, a model-quality
certification, or a claim of community adoption. See the
[design principles and roadmap](docs/design.md) for how new domains earn support.

Inputs are read only; built-in reports omit raw row/document/ID values. Column names,
statistics, and configuration remain visible. Default limits are 200,000 rows
and 100 MB per tabular/text input, with no silent sampling. Retrieval and text
similarity have additional [documented budgets](docs/compatibility.md). The engine is in-memory.

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
