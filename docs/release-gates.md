# Evidence-aware release gates

An audit measures observations; a release policy states what evidence and
quality are required before proceeding. A perfect score on one judged query
out of 100 is not adequate evidence merely because a ranking check completed.
Policies make that distinction explicit, without changing the existing API.

## Start with coverage

```python
from proofml import AuditPolicy, audit_retrieval

report = audit_retrieval(retrieved_ids, relevant_ids, k=5)
policy = AuditPolicy(
    require_checks=("retrieval_ranking",),
    min_coverage={"retrieval_ranking": 0.95},
    min_evaluated={"retrieval_ranking": 100},
    min_metrics={"retrieval_ranking.recall@5": 0.8},
)
decision = policy.enforce(report)
decision.save("decision.json")
```

The example thresholds are **project choices**, not universal recommendations
or guarantees of statistical power. `enforce` returns a passing `GateResult` or
raises `AuditFailed`; the exception holds `.report` and `.decision`. For review
without exceptions, use `policy.evaluate(report)` and inspect `.issues`.

| Status | Meaning | Blocks release? |
| --- | --- | --- |
| `passed` | All configured requirements met | No |
| `failed` | A measured quality/severity/regression requirement was violated | Yes |
| `insufficient` | Required checks, measurements, rows, coverage, or baseline are unavailable/inadequate | Yes |
| `incompatible` | Baseline and candidate do not establish a comparable evaluation | Yes |

All issues remain visible. When several kinds occur, overall status precedence
is incompatible, insufficient, failed, passed. Errors are never hidden by a
high severity threshold. Policies snapshot report inputs when evaluated; they
do not mutate the source reports.

## Policy rules

| Field | Keys / behavior |
| --- | --- |
| `name` | Human-readable CI test name; default `release` |
| `severity` | Candidate finding threshold, default `high`; check errors always block |
| `require_checks` | Check IDs that must complete, even with findings |
| `min_rows` | Dataset names (`train`, `test`, `evaluation`) and minimum integer row counts |
| `min_coverage` | Check IDs and minimum evaluated/total fractions in (0, 1] |
| `min_evaluated` | Check IDs and minimum positive integer assessed counts |
| `min_metrics`, `max_metrics` | Explicit `check_id.metric_name` paths and numeric bounds |
| `max_drop` | Metric paths and maximum allowed baseline minus candidate; for higher-is-better metrics |
| `max_increase` | Metric paths and maximum allowed candidate minus baseline; for lower-is-better metrics |

Missing, ambiguous, skipped, or errored metric paths block the gate. Typos
cannot silently disable requirements. Coverage is not inferred from sample-like
metric names. Zero denominators are unknown, not 100%. Metric bounds and
regression tolerances are inclusive. Tolerances use **absolute metric units**:
0.02 on a recall fraction means two percentage points, not two percent relative.
Nonzero floating-point boundaries allow a 1e-12 relative rounding tolerance;
zero regression tolerance does not excuse a small real regression.

Policy dictionaries are copied and exposed read-only. Unknown fields, nonfinite
numbers, booleans as numeric requirements, contradictory bounds, and duplicate
required IDs fail validation. There are no automatic severity suppressions.

## What coverage measures

Checks may return `Coverage(total=..., evaluated=..., unit=...)`. In 0.6:

- `retrieval_ranking`: total query count versus queries with positive relevance
  judgments. Empty rankings with judgments are evaluated and score zero.
- `retrieval_inputs`: all queries inspected for structural issues.
- `retrieval_corpus`: all queries inspected when a corpus snapshot is supplied.
- `data_quality`: all rows inspected across supplied tabular splits.
- `text_quality`: all documents inspected across supplied text splits.

Other checks do not yet declare this population contract. Requiring their
numeric coverage blocks as insufficient; `require_checks` can still require
completion. Custom checks can declare coverage using the exported `Coverage`
type. Definitions are check-specific: document coverage is not semantic
understanding, and row coverage does not count independent statistical samples.

For a tabular pipeline, use the same policy interface:

```python
policy = AuditPolicy(
    require_checks=("data_quality", "target_health", "split_overlap"),
    min_rows={"train": 1000, "test": 200},
)
policy.enforce(tabular_report)
```

## Compare an approved retrieval baseline

Use query-ID mappings in both runs. Persist an approved audit with
`baseline.save("approved")`; loading it never reruns a model or retriever.

```python
from proofml import AuditReport, AuditPolicy

baseline = AuditReport.load("approved/report.json")
policy = AuditPolicy(
    require_checks=("retrieval_ranking",),
    min_coverage={"retrieval_ranking": 0.95},
    min_evaluated={"retrieval_ranking": 100},
    min_metrics={"retrieval_ranking.recall@5": 0.8},
    max_drop={"retrieval_ranking.recall@5": 0.02},
)
decision = policy.evaluate(candidate, baseline=baseline)
decision.save("decision.json")
decision.save_junit("decision.xml")
decision.raise_for_issues()
```

`candidate.compare(baseline)` returns compatibility reasons and individual
metric changes (candidate minus baseline), without applying a policy. Missing
measurements have `None` values/deltas, never invented zeros. An incompatible
comparison returns no numerical changes. Finite values whose difference
overflows also make the comparison incompatible.

Compatibility requires:

1. Identical report schema/tool versions and audit configurations, including k,
   thresholds, and budgets. Re-evaluate the baseline after an upgrade or a change
   to the audit contract.
2. Matching built-in retrieval check registries and definition contracts.
   Custom check metrics are usable in absolute policies, but their code/config
   semantics are not yet verified for strict baseline comparisons.
3. Matching stable query IDs and positive-judgment sets, independent of mapping
   insertion order. Positional lists are still auditable but cannot establish
   cross-run identity, so comparisons reject them.
4. Matching declared corpus snapshot fingerprints. If both omit a corpus,
   comparisons remain possible but make no corpus-integrity guarantee. Require
   `retrieval_corpus` when that guarantee is part of your workflow.

The evaluation fingerprint excludes retrieved results; a separate output
fingerprint changes when results do. **Stable IDs are a caller contract**:
query content is not supplied to this API. Use versioned case IDs when query
inputs change; reusing an ID cannot prove that hidden content stayed unchanged.
Hashes are identifiers, not signatures, and reports can be forged. Only use
trusted, reviewed baseline artifacts. No baseline is automatically promoted.

Evidence requirements (required checks, row counts, coverage, assessed counts)
apply to both runs. Absolute metric/severity requirements apply to the candidate:
an inadequate old score should not prevent a valid improvement. A supplied
baseline is always compatibility-checked, even with no regression rule.

Strict metric comparisons currently support **retrieval only**. Policies work
across all report types. Version 0.7 adds [slice/suite policies](slices.md) on top
of this contract. Statistical significance, per-case debugging, saved-prediction
scoring, and agent trace checks remain outside the implemented scope.

## Version control and CI

```python
policy.save("release-policy.json")
policy = AuditPolicy.from_file("release-policy.json")
```

Policy JSON includes `schema_version: "1.0"`; unknown fields and duplicate JSON
keys are rejected. Snapshot loading is bounded (100 MB for reports, 1 MB for
policies by default), uses JSON rather than pickle, and validates check/metric/
coverage structure. Legacy reports can load but cannot supply absent identity
or coverage. Derived summaries are recomputed and checked for consistency.

```bash
proofml gate candidate/report.json --policy release-policy.json \
  --baseline approved/report.json --output decision.json --junit decision.xml
```

Exit codes: **0** passed, **1** measured failure, **2** insufficient/incompatible
evidence or input/write error. JUnit emits one testcase per policy: quality
failures are failures; insufficient/incompatible decisions are errors. Input
files cannot be overwritten by CLI outputs, including with `--overwrite`.
Output files are individually atomic, not a multi-file transaction.

Default decisions contain rule evidence, metric differences, and whole-report
fingerprints—not raw queries, documents, or labels. Policies, field names, and
hashes can still be sensitive. Do not treat hashing as anonymization.

Try the [offline end-to-end example](../examples/release_gate.py). A policy is
a reproducible release requirement, not model certification or proof of safety.
