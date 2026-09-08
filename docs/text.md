# Text corpus audits

Catch document duplication, inconsistent single-label annotations, and lexical
train/test contamination before training an NLP model or evaluating on a corpus.
This is a data audit, not text understanding. It needs no third-party packages.

```python
from proofml import audit_text

report = audit_text(train_texts, test_texts)
report.save("reports/text")
report.raise_for_issues(require_checks=("text_overlap", "text_near_duplicates"))
```

Use ordered iterables of strings: lists, tuples, or generators. One string is
one document. A scalar string, unordered set, mapping, non-string document, or
empty corpus is rejected. Files are not automatically opened by this API;
extract their text explicitly with your own loader. Blank documents are accepted
and flagged. Inputs are not modified; generators are consumed once.

## Labels and normalization

```python
report = audit_text(train_texts, test_texts, y=train_labels, y_test=test_labels)
```

Labels are optional single-class strings or Python integers. `None` and `""`
mean missing labels; booleans, floats, and multi-label collections are rejected.
String `"1"` and integer `1` are distinct classes. Labels attach **positionally**;
pandas indexes are not aligned. `y_test` requires `y` and test documents.
Labels and document counts must match exactly.

Default `normalization="unicode"` applies Unicode NFKC, case folding, then
whitespace collapsing. For case/whitespace-sensitive inputs use
`normalization="exact"`, which preserves document strings. These policies can
change what counts as an exact duplicate: identical normalized text need not
have identical bytes. This is particularly important for source code, formulas,
case-sensitive entities, and contextual labels. See Python's
[Unicode normalization reference](https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize).

## Checks

| ID | What it establishes | Limits |
| --- | --- | --- |
| `text_quality` | Empty documents and repeated nonempty documents within each split | Repetition may be intentional |
| `text_labels` | Missing labels, equivalent documents with conflicting labels, unseen test classes | Single-label classification only; conflicts need context |
| `text_overlap` | Test documents also present in training under configured normalization | Does not prove improper leakage |
| `text_near_duplicates` | Lexically similar train/test documents | Not semantic similarity or paraphrase detection |

Near duplicates use sets of consecutive word-token shingles and Jaccard
similarity (intersection size / union size), default shingle size 3 and threshold
0.8. Tokenization uses Unicode `\w+`, not a language-aware tokenizer. Punctuation
is ignored even in exact-normalization mode for this lexical check. Short or
wordless documents are excluded; if neither split has an eligible pair, the
check skips. Exact document matches are handled separately, not counted twice.
No guarantee is made for scripts that require language-specific segmentation.

The inverted index considers every candidate sharing at least one shingle.
No random sampling or approximate nearest-neighbor search is used. Successful
runs expose eligible unique-document counts and candidate visits under
`report.metrics["text_near_duplicates"]`. A resource-budget exceedance discards
partial near-duplicate results and returns a visible skip. Other checks still run.

## Configuration and budgets

`TextConfig` can be reused or its fields passed directly to `audit_text`.
Unknown options fail, and direct options override config.

| Setting | Default | Meaning |
| --- | --- | --- |
| `max_documents` | 200,000 | Documents per split; exceedance rejects input |
| `max_bytes` | 100,000,000 | Sum of ASCII-escaped JSON document/label payload bytes per split; exceedance rejects input |
| `max_tokens_per_document` | 10,000 | Near-duplicate token budget per document |
| `max_shingles` | 500,000 | Total distinct shingles per unique document, summed across both splits |
| `max_candidate_visits` | 100,000 | Inverted-index posting visits, including repeated candidates |
| `disabled_checks` | `()` | Explicitly skipped check IDs |

Near-duplicate memory and runtime can exceed raw input size. Index construction
scales with generated shingles; candidate work is bounded by posting visits,
and comparisons cost the sizes of the two shingle sets. This is not a distributed
or streaming audit engine. Check limits are not a hardened resource sandbox.

Use `checks=[*default_text_checks(), MyCheck()]` from `proofml.text` to extend
the registry. Plugins receive `TextContext` with normalized immutable documents,
encoded labels, and config. Report hashes cover original strings and labels;
normalization settings are recorded separately. Built-in reports omit documents,
labels, query content, and per-document hashes, but aggregate fingerprints can
still disclose identity for known corpora. See [security](../SECURITY.md).
