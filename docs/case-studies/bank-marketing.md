# Real case study: a higher score with a feature you cannot use

**Question:** before placing a marketing call, could a model prioritize likely
subscribers? An exported training table can contain information recorded after
that decision. A strong holdout score does not make such a feature usable.

This example uses the actual [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing),
specifically `bank-additional-full.csv`: 41,188 records, 20 inputs, and outcome
`y`. The source describes this variant as date-ordered. Its accompanying
`bank-additional-names.txt` explicitly warns that `duration` is unavailable before
the call. No synthetic feature, duplicated row, or label corruption is injected.

## What we actually measured

Run on 2026-09-11 using ProofML 0.7.1 and scikit-learn 1.9.0:

| Candidate inputs | ROC AUC | Average precision | Duration-availability gate |
| --- | ---: | ---: | --- |
| Original predictors, including duration | 0.8211 | 0.5981 | Failed |
| Same predictors, duration removed | 0.7483 | 0.5268 | Passed for this contract only |

Both models use exactly the same 32,950 training rows and 8,238 held-out rows.
The difference is 0.0728 ROC-AUC units. This is one measured comparison, **not** a
confidence interval, a generalization claim, or a measured financial loss.
Average precision is scikit-learn's metric, not trapezoidal PR AUC.

The higher-scoring model answers a different information-availability question.
ProofML does not improve the model's score: it makes the evaluation's declared
assumptions enforceable before an unsuitable model is accepted.

Full metrics, dependency versions, feature lists, checksum provenance, and
aggregate findings are in [the recorded results](bank-marketing-results.json).
The [runnable script](../../examples/bank_marketing.py) generates JSON/HTML audit
reports and JSON/JUnit decisions; raw customer rows and trained models are not
exported or committed.

## Reproduce it

From a checkout containing this example:

```bash
python -m pip install -e '.[pandas]' 'scikit-learn>=1.4,<2' certifi
python examples/bank_marketing.py --download --output artifacts/bank-marketing
```

No Kaggle account, API token, LLM, or hosted evaluator is needed. Downloading is
explicit and uses UCI's public HTTPS archive. The script verifies both the archive
and inner CSV against recorded SHA-256 values, bounds download/decompression
sizes, and never extracts arbitrary ZIP paths. A changed source fails instead
of silently substituting data. Use a new output directory for each run.
The example optionally adds certifi's public certificate roots for Python
installations lacking a configured CA bundle; HTTPS verification stays enabled.
certifi and scikit-learn are example dependencies, not core package requirements.

For an offline run after downloading the official outer archive:

```bash
python examples/bank_marketing.py --archive /path/to/bank+marketing.zip \
  --output artifacts/bank-marketing-offline
```

Dependency versions and numerical implementations may slightly change scores;
the recorded run lists its environment. Unit tests use small synthetic harness
fixtures with no network access. They do not claim to rerun the real dataset in CI.

## The useful integration is small

Here `train` and `test` are the actual candidate input DataFrames including `y`:

```python
from proofml import audit, AuditPolicy

policy = AuditPolicy(severity="critical", require_checks=("feature_availability",))
report = audit(train, test, target="y", unavailable_features=("duration",))
policy.enforce(report)  # raises AuditFailed while duration is a predictor
```

Remove `duration` from **both** candidate inputs and repeat the same call with
the same rule. Keep the rule in version control to catch later reintroduction.
Rules use exact column names: renamed or derived post-call features still need
lineage review. The rule checks absence, not semantic feature derivations.

The example demonstrates three states:

| Input/contract state | Outcome | Why |
| --- | --- | --- |
| Duration present, no availability declaration | Insufficient | Required availability check has no evidence |
| Duration present, declared unavailable | Failed | Confirmed violation of the supplied contract |
| Duration absent, same declaration retained | Passed for availability | Forbidden predictor is absent |

Without a declaration, ProofML cannot infer that call duration happens after
the decision. Its association heuristic is not a substitute for domain context.
The critical-only policy above intentionally isolates availability, rather than
pretending every finding has been resolved. A separate high-severity review
policy in the example remains failed after removal.

In a production job, enforce the gate **before fitting**. This teaching script
deliberately fits the rejected variant to quantify the misleading score; that
is an explicit demonstration exception, not the recommended pipeline behavior.

## Other real findings we did not hide

On the duration-removed candidate, ProofML reports:

- **Two of 8,238 test feature rows match training feature rows.** Removing a
  distinguishing attribute can make different observations identical. Without
  stable customer IDs, this is not proof of leakage and not a reason to blindly
  deduplicate. This high-severity review finding blocks the broader gate.
- **1,637 extra duplicate training rows and 146 extra duplicate test rows** on
  the selected duration-removed schema. These counts include the label; they
  are not claims that those customers are duplicated in the source system.
- **Nine feature distribution shifts** exceed the configured default heuristic,
  including `euribor3m` (empirical KS distance 0.9879). These are distances, not
  p-values or evidence that drift caused the model's errors.
- **570 test rows contain a month category absent from training.** The model's
  encoder handles unknown categories, but the evaluation coverage gap is still
  worth reviewing.

Separately computed outcome prevalence changes from **6.37% in training to
30.83% in the holdout**. ProofML flags training class imbalance; this example's
prevalence comparison is descriptive analysis, not a new automatic label-shift
check. A majority-class training accuracy of 93.63% would be a poor headline.

## Experimental design and boundaries

- First 80% of rows train, final 20% test, preserving the published ordering.
  We do not manufacture ISO dates from month/weekday or claim a verified strict
  temporal boundary. The CSV lacks exact timestamps and stable customer IDs;
  within-date ordering and repeat-customer independence remain unverified.
- One logistic-regression recipe: `lbfgs`, `C=1`, `max_iter=4000`, seed 42,
  one numerical thread. No search or tuning against the holdout.
- Numeric standardization and categorical one-hot encoding are fitted only on
  training rows through a scikit-learn `Pipeline`. The `unknown` string remains
  a category, and `pdays=999` remains the published sentinel. This is a transparent
  baseline, not an optimized feature-engineering recipe.
- Removing duration does not prove all remaining inputs are available at a
  particular scoring time. Campaign/contact fields and publication lags of
  economic indicators require additional feature-lineage review.
- No confidence intervals, fairness conclusions, causal claims, prospective
  validation, or production readiness are implied. Metrics come from
  scikit-learn; ProofML supplies the audit and policy evidence.

## A usability bug this experiment uncovered

Before 0.7.1, ProofML required every `unavailable_features` name to exist in the
training table. Correcting the table therefore required deleting its guardrail.
The case study exposed this contradiction. The fix treats these names as a
persistent denylist, checks both train and test predictors, and preserves the
rule in the clean bundled demo. Regression tests cover removal, reintroduction,
test-only violations, and missing declarations. Other required semantic columns
still must exist; this does not weaken target/entity/time validation.

## Next steps justified by this evidence

1. Have a second practitioner reproduce the case and review feature availability
   at an explicitly chosen scoring time; collect false positives as well as wins.
2. Add a separate real retrieval benchmark to exercise cohort/baseline policies
   without relying only on synthetic retrieval examples.
3. Design a framework-neutral saved-prediction adapter, then uncertainty estimates
   with an explicit resampling unit. Repeated customers make naive row-bootstrap
   confidence intervals potentially misleading; do not bolt on arbitrary intervals.

These are next steps, not implemented capabilities. One successful case study
does not establish performance across arbitrary datasets.

## Attribution

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine
Learning Repository. [DOI:10.24432/C5K306](https://doi.org/10.24432/C5K306).
UCI lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the data.
The experiment uses the original full additional CSV, selects a row-order split,
and removes duration for one candidate; it does not redistribute the raw data.

Related paper: Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to
Predict the Success of Bank Telemarketing*. Decision Support Systems.
[DOI:10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).
