# Search and RAG retrieval audits

Evaluate the **retrieval stage** without calling an LLM or connecting to a vector
database. Export ranked document/chunk IDs and supply known relevant IDs. This
works across retriever implementations because it evaluates their outputs, not
framework internals. No third-party packages are required.

```python
from proofml import audit_retrieval

report = audit_retrieval(
    retrieved={"q1": ["doc-7", "doc-2"], "q2": ["doc-9"]},
    relevant={"q2": {"doc-9"}, "q1": {"doc-2"}},
    k=2, min_recall=0.8,
)
print(report.metrics["retrieval_ranking"])
report.save("reports/retrieval")
report.raise_for_issues(require_checks=("retrieval_ranking",))
```

The minimum recall is **your policy**, not a universal quality standard. Defaults
compute measurements without inventing a pass threshold. `min_recall` and
`min_ndcg` create high/confirmed findings when their macro means are below the
declared minimum. Equality passes.

## Input contract

- Prefer matching query-ID mappings; query IDs must be identical across both
  mappings and values align by key, not insertion order. Query IDs are nonempty strings.
- Alternatively supply two nested ordered iterables; query rows align by position.
  Different row counts, mixing mapping/sequence styles, or empty evaluations fail.
- Retrieved IDs must be ranked ordered iterables. Relevant IDs and `corpus_ids`
  may be sets or iterables. Duplicate judgments have set semantics.
- All document IDs are nonempty strings, matched exactly without stripping,
  case normalization, fuzzy matching, or document/chunk namespace conversion.
- Empty rankings and empty relevance sets are accepted and reported, not hidden.
- Inputs are not modified. Generators are consumed once. IDs and query text are
  never put in built-in evidence or manifests; only aggregate hashes are included.

## Measurements and edge cases

All ranking metrics use **binary** relevance and macro averaging over queries
with at least one known positive. Unlisted IDs count as nonrelevant; incomplete
judgments can underestimate quality. A relevance set is not inferred from text.

For each query, only the first `k` rank positions count. An ID earns relevance
credit once; duplicates still consume positions. Short result lists implicitly
have nonrelevant missing slots. With `hits` distinct relevant IDs retrieved:

| Metric | Per-query definition |
| --- | --- |
| `precision@k` | `hits / k`, even for short lists |
| `recall@k` | `hits / number_of_known_relevant_ids` |
| `hit_rate@k` | 1 if at least one hit, otherwise 0 |
| `mrr@k` | Reciprocal rank of the first hit, or 0 |
| `ndcg@k` | Sum of `1 / log2(rank + 1)` at first relevant occurrences, divided by the ideal binary ranking's gain |

The ranking definitions follow standard
[ranked retrieval evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html).
This implementation does not support graded judgments. It records
`evaluated_queries` and `excluded_queries`; no positives means ranking metrics
and thresholds **skip**, never a fabricated zero or perfect score. Always use
`require_checks=("retrieval_ranking",)` to require completion. For a deployment
gate, also declare minimum judged coverage and assessed counts with
[`AuditPolicy`](release-gates.md). Completion alone can succeed with only one
judged query. Skips are allowed
by the general report gate unless explicitly required.

## Checks

| ID | Purpose |
| --- | --- |
| `retrieval_inputs` | Report empty results, missing positive judgments, duplicates, short-list coverage |
| `retrieval_corpus` | Validate retrieved and relevant IDs against optional `corpus_ids`; skips without the snapshot |
| `retrieval_ranking` | Compute metrics and enforce user-declared recall/nDCG minima |

Corpus findings do not automatically suppress ranking metrics: measurements
remain conditional on supplied judgments and should not be interpreted as valid
deployment evidence when other checks fail. Require `retrieval_corpus` as well
when corpus integrity is part of your gate.

This is **not answer evaluation**: it does not assess groundedness, factual
correctness, prompt injection, generation safety, or language-model quality.
It cannot detect contamination from arbitrary hidden training corpora.

## Budgets and extension

Reuse `RetrievalConfig`, or supply its fields as direct keywords. Defaults:
`k=10`, `max_queries=100000`, `max_ids_per_query=10000` for each ranking/judgment
list, `max_corpus_ids=1000000`, and `max_bytes=100000000` summed over ASCII-escaped
JSON query-ID/document-ID values across the evaluation and corpus. Count/byte
limits reject the whole input without sampling. All data stays in memory; Python
objects, sets, and serialization buffers can require substantially more memory.

Metrics cost O(sum of top-k ranking lengths plus ideal gain lengths); ingestion
and integrity checks read all supplied IDs. Sorted set fingerprints add sorting
cost for judgments/corpus. Budgets are limits, not distributed execution.

Use `checks=[*default_retrieval_checks(), MyCheck()]` from `proofml.retrieval`
for custom checks. They receive `RetrievalContext(data, config)`. Set
`disabled_checks` to record explicit skips. Every domain shares `AuditReport`,
HTML/JSON output, exception isolation, and pipeline gate behavior.
