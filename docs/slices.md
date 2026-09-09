# Slice-aware evaluation and reusable audit suites

An overall metric can improve while a language, cohort, or query type regresses.
ProofML keeps each population's evidence separate and lets one release policy
require all of them to meet their own evidence and quality contracts.

## Retrieval slices in a few lines

```python
from proofml import audit_retrieval_slices

suite = audit_retrieval_slices(
    retrieved_ids, relevant_ids,
    groups={"query-1": "english", "query-2": "spanish"},
    k=5,
)
suite.save("slice-report")
print(suite.metrics["spanish"]["retrieval_ranking"])
```

Both inputs must be query-ID mappings. The `groups` mapping must contain every
input query ID exactly once, with no unknown IDs. Group values are public
report names: nonempty strings of at most 256 characters. `overall` is reserved.
No language, demographic characteristic, or cohort is inferred from text.

The helper always includes `overall`, scored on every original query exactly
once. It then audits each group with the same checks/configuration. Retrieved
IDs, judgments, and the corpus iterator are loaded once. Corpus storage and its
fingerprint are reused across subsets; no model, retriever, or service is called.
Custom checks execute once per report and remain trusted local Python code.

## Explicit overlapping slices

For more than one dimension, supply named sets of query IDs instead of groups:

```python
suite = audit_retrieval_slices(
    retrieved_ids, relevant_ids, k=5,
    slices={
        "spanish": {"query-2", "query-3"},
        "long_queries": {"query-1", "query-3"},
        "required_but_empty": set(),
    },
)
```

Use exactly one of `groups` or `slices`. Overlap between slices is permitted;
duplicate IDs within one slice are rejected. Unknown IDs never silently drop.
Queries may belong to no explicit slice but still contribute to `overall`.
Slice scores are **not averaged to produce the overall score**, and their
sample counts must not be added as though the populations were disjoint.

Empty requested slices remain in the suite with zero rows. Their ranking check
skips with 0/0 coverage and no ranking metrics. A required-check, coverage, or
minimum-evaluated rule blocks them as insufficient evidence, not as zero or
perfect performance. Automatically grouped slices disappear when no matching
queries exist, so require expected names in the policy.

## Gate every slice against a baseline

```python
from proofml import AuditPolicy, AuditSuite, SuitePolicy

baseline = AuditSuite.load("approved/suite.json")
policy = SuitePolicy(
    default=AuditPolicy(
        require_checks=("retrieval_ranking",),
        min_coverage={"retrieval_ranking": 0.95},
        min_evaluated={"retrieval_ranking": 20},
        max_drop={"retrieval_ranking.recall@5": 0.02},
    ),
    require_reports=("overall", "english", "spanish"),
)
decision = policy.evaluate(suite, baseline=baseline)
decision.save("decision.json")
decision.save_junit("decision.xml")
decision.raise_for_issues()
```

These thresholds are illustrative project choices, not recommended sample
sizes or proof of statistical power. Use `policy.enforce(...)` for the short
raise-or-return form. `decision.results[name]` contains each report's normal
`GateResult`, including before/after metric values, rule evidence, and precise
compatibility reasons. `AuditFailed.decision` exposes the complete suite
decision; its `.report` is an `AuditSuite` of the assessed report snapshots.

Rules apply as follows:

- `default` applies to **every** supplied report, including unexpected new slices.
- `overrides={name: AuditPolicy(...)}` replaces the default policy for that report;
  it does not merge requirements. Use `dataclasses.replace(default_policy, ...)`
  if you want to inherit rules while changing one setting.
- Every override is implicitly required. A misspelled name cannot silently turn
  off a guardrail. `require_reports` can require additional names.
- With a baseline, candidate and baseline names must match exactly. Adding,
  deleting, or renaming a slice blocks until the reviewed baseline is updated.
- Each named report uses strict baseline compatibility. Moving cases between
  same-named slices changes their evaluation identities and blocks comparison,
  even when overall identities are unchanged.
- Small/unjudged slices remain insufficient; errors and incompatible reports
  cannot disappear into an overall passing average. Status precedence is
  incompatible, insufficient, failed, passed; all individual outcomes are retained.

Baseline comparison inherits the [release gate contract](release-gates.md):
matching tool/config/check definitions, stable query IDs and judgments, and
matching declared corpus identities. It currently supports built-in retrieval
reports only. Query content is not supplied to this API; use versioned case IDs
if that content changes. A saved fingerprint is not an authenticity signature.

## The failure this prevents

The runnable [slice regression example](../examples/slice_regression.py) produces:

```text
overall: 25% -> 75%; passed
english: 0% -> 100%; passed
spanish: 100% -> 0%; failed
Release decision: failed
```

This is a synthetic eight-query teaching fixture, not an empirical performance
benchmark or evidence of a real demographic disparity.

## Suites also support tabular, text, and fold workflows

`AuditSuite` is a general collection of existing reports, not a retrieval-only
container. It does not choose splits, run arbitrary model code, or combine fold
metrics under assumptions about weighting/independence:

```python
from proofml import AuditSuite, SuitePolicy, AuditPolicy, audit

suite = AuditSuite({
    "fold_0": audit(train_0, test_0, target="label"),
    "fold_1": audit(train_1, test_1, target="label"),
})
SuitePolicy(
    default=AuditPolicy(require_checks=("data_quality", "split_overlap"),
                        min_rows={"train": 1000, "test": 200}),
    require_reports=("fold_0", "fold_1"),
).enforce(suite)
```

Manual suites may contain independent stages or different modalities. Their
absolute policies work normally; supplying a baseline for unsupported domains
remains incompatible. Suites cannot be nested. Policy definitions and report
name maps are copied and exposed read-only; report payloads retain the existing
mutable-dictionary contract, so do not mutate them during evaluation.

## Reports, pandas, and CI

- `suite.reports[name]`: the standard `AuditReport`, which can also be saved alone.
- `suite.metrics`: nested report/check/metric measurements.
- `suite.to_frame()`: optional pandas table with report, check, metric, value,
  evaluated count, total count, and population unit. Missing/skipped metrics do
  not become zero rows; inspect `suite.reports[name].checks` for complete coverage.
- `suite.save(directory)`: `suite.json` plus an escaped offline `suite.html`
  summary. No filenames are derived from slice names. Notebook display is sandboxed.
- `SuitePolicy.save(path)` / `.from_file(path)`: strict, versioned JSON policy.
- `decision.save_junit(path)`: one test case per assessed report, plus a separate
  contract-error case when required report names are missing. Numeric failure
  evidence is included directly in JUnit, not only in the JSON artifact.

```bash
proofml gate candidate/suite.json --suite --policy suite-policy.json \
  --baseline approved/suite.json --output decision.json --junit decision.xml
```

Exit 0 means passed; 1 means measured failure; 2 means insufficient/incompatible
evidence or invalid input/output. Existing files are protected unless overwrite
is explicit. Each file is atomic; multi-file saves are not transactions.

## Limits and interpretation

Slice helper defaults: at most 50 slices plus overall, and 1,000,000 query-to-slice
memberships. Both are configurable with `max_slices` and `max_memberships`;
`max_slices` must remain below 1,000. Limits and invalid grouping reject the
audit before any check runs. Existing per-input retrieval limits also apply.
Generic suites contain 1–1,000 reports; JSON loading defaults to 100 MB for a
suite and 1 MB for a policy. All execution is in-memory.

Loading sorts query IDs so equivalent mappings accumulate floating-point metrics
in a reproducible order. This adds O(Q log Q) query ordering; checks and subset
fingerprints scale with the selected data and membership count. The shared
corpus is not duplicated, but rankings, judgments, report objects, serialization,
and custom checks can still consume substantial memory. These caps are not a
hardened resource sandbox or out-of-core execution engine.

Raw query/document IDs do not enter built-in suite reports. **Slice names are
visible** in Python/JSON/HTML/JUnit: use coarse labels, not personal IDs. Small
cohort aggregates can still reveal sensitive information. Coverage does not
establish representativeness, independence, fairness, or statistical significance.
This release deliberately does not add confidence intervals or automatic slice
discovery. Comparing many selected slices needs appropriate statistical review.
