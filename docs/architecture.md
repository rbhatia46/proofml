# Architecture and extension guide

ProofML is a deterministic audit engine for candidate tabular model inputs. Its
specialist checks are reusable tools for data scientists, CI, and AI agents.
There is no LLM planner or autonomous code execution in version 0.1. Calling a
fixed sequence of checks “intelligent agents” would overstate its capabilities.

## Module boundaries

| Module | Responsibility | Does not do |
| --- | --- | --- |
| `config.py` | Validate explicit task semantics and thresholds | Guess business context |
| `data.py` | Bounded CSV/Parquet loading and normalization | Modify inputs or execute code |
| `context.py` | Immutable data/config interface | Know about rendering or CLI |
| `checks/` | Independent evidence-generating checks | Print, persist, or call a model |
| `engine.py` | Validate inputs, run registry, isolate failures | Hide errors as passes |
| `models.py` | Versioned report contracts | Assign an arbitrary trust score |
| `reporting.py` | HTML/JSON presentation and atomic file writes | Send data to a service |
| `cli.py` | Argument handling and CI exit policy | Contain check algorithms |

## Add a custom check

Use a stable unique ID, accept `AuditContext`, and return `CheckResult`.

```python
from proofml import CheckResult, Finding, audit
from proofml.checks import default_checks

class MinimumRows:
    id = "myteam.minimum_rows"

    def run(self, ctx):
        if len(ctx.train.rows) >= 100:
            return CheckResult.complete(self.id, [])
        return CheckResult.complete(self.id, [Finding(
            code="myteam.small_training_set",
            severity="medium",
            confidence="needs_context",
            title="Small training dataset",
            explanation="This project normally requires at least 100 observations.",
            recommendation="Review sampling and expected evaluation uncertainty.",
            evidence={"rows": len(ctx.train.rows), "threshold": 100},
        )])

report = audit("train.csv", checks=[*default_checks(), MinimumRows()])
```

Use `CheckResult(id, "skipped", reason="...")` for missing prerequisites. Do not
return a pass when a check could not assess its intended condition. For a
partial check, document exactly which columns/types it supports. Plugin
exceptions become `error` results; the CLI exits 2 even with `--fail-on none`.
For debugging, run `MyCheck().run(context)` directly to obtain a traceback.

Plugins are trusted Python code; there is no sandbox. They must avoid raw data
in evidence and must return JSON-compatible values without NaN or infinity.
Plugin imports are explicit: ProofML never scans and executes user repositories.

## Add an adapter

Implement the same `Dataset` contract: immutable string tuples, unique column
names, a SHA-256 source fingerprint, and format metadata. Enforce limits before
large allocations when the source format permits. Add normalization tests and
document any source-to-string conversions. Version 0.1 dispatches adapters
explicitly in `load_dataset`; there is no automatic entry-point discovery.

## Determinism and provenance

For unchanged source bytes/config/check implementations, findings and evidence
are deterministic. Reports include versions, file fingerprints, shapes, and
configuration; timestamps differ between runs. SHA-256 identifies an input,
not a signed attestation. Inputs must remain stable while being read. Store
reports with the code revision and dependency lockfile in your own workflow.

Default limits: 200,000 rows and 100 MB on-disk per dataset. Both splits reside
in memory; string conversion, sets, and sorting can require several times the
source size. Parquet can expand substantially. This is a bounded local tool,
not an out-of-core engine. Increase limits only after measuring memory.

## Report contract

JSON `schema_version` is `1.0`. Consumers should tolerate added fields and
finding codes. Breaking structural changes require a schema-version change.
`tool_version` identifies behavior; threshold changes can alter results.
The HTML report has no external fonts, assets, JavaScript, or requests.
All text is escaped. Native details controls expose supporting evidence.
