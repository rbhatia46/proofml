# Real retrieval benchmark: BEIR SciFact

**Start with the [executed notebook](../../examples/notebooks/scifact_retrieval.ipynb).**
It explains the experiment, runs actual local retrieval, applies ProofML policies,
and saves replayable CI evidence. [Setup instructions](../../examples/README.md).

## The workflow problem

A search-index migration drops abstracts and retains only titles. The same
documents still exist, so ID membership checks can pass while retrieval quality
changes. Comparing a few manually selected queries is not sufficient evidence
for releasing that change.

We build both title-plus-abstract and title-only indexes using the full
[BEIR SciFact archive](https://github.com/beir-cellar/beir). The authors' benchmark
supplies a scientific-claim retrieval task. We evaluate only the 300 BEIR test
queries and their 339 positive judgments against all 5,183 documents. The
archive includes additional queries/train judgments, which are not mixed into
the test population. No synthetic ranking degradation is injected into this
comparison; the candidate actually indexes different fields.

## Recorded results

Measured 2026-09-11 with ProofML 0.7.1, scikit-learn 1.9.0, and Python 3.13:

| Cohort | Queries | Full-index recall@10 | Title-only recall@10 | Full-index nDCG@10 | Title-only nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 300 | 0.7735 | 0.5476 | 0.6286 | 0.4166 |
| Short queries | 163 | 0.7727 | 0.5322 | 0.6385 | 0.4094 |
| Long queries | 137 | 0.7745 | 0.5659 | 0.6167 | 0.4251 |

The declared policy **fails for all three reports**. It allows at most a 0.05
absolute decline in each metric, requires every cohort, full positive-judgment
coverage, and at least 20 assessed queries per report. These illustrative rules
and the 12-word cohort boundary were defined before measuring this candidate.
They are not confidence intervals or calibrated significance thresholds.

This is an aggregate **and** cohort regression, not an example of aggregate
improvement hiding a cohort failure. The earlier synthetic slice example
demonstrates that separate pattern. No latency/storage benefit is measured.

[Machine-readable results and environment](scifact-results.json) ·
[Complete retrieval helper](../../examples/scifact_retrieval.py)

## Reproducible recipe

Both indexes use scikit-learn `TfidfVectorizer`: unigram tokens of at least two
word characters, lowercase, Unicode accent stripping, sublinear TF, default
smoothed IDF, L2 normalization, no stop-word list. Vocabulary and IDF are fit on
the searchable corpus only. No query or qrel fitting, no hyperparameter search,
and no embedding downloads. Corpus statistics are normal index construction,
not fitting a supervised model on test relevance labels.

Ranking is cosine similarity, evaluated one query at a time. Ties use
lexicographic document ID; zero-score results are omitted. This is a simple
educational TF-IDF recipe, not a reproduction of BEIR's BM25 or leaderboard
models. Recall is independently recalculated in the notebook and checked
against ProofML. nDCG uses ProofML's binary-relevance implementation.

Short queries have at most 12 Unicode word tokens (`\b\w+\b`); long queries
have more. These groups are defined from query text without relevance outcomes
and are not demographic or clinical subpopulations.

The script bounds archive/member sizes, verifies SHA-256 on the ZIP and each
required file, never extracts arbitrary paths, and validates IDs and binary
test judgments. Source pins are retained separately from ProofML report hashes:
the latter reflect stable IDs/judgments/corpus membership, not document contents.
Version content and case IDs in your own application when those change.

## What the notebook teaches beyond metrics

1. Use `audit_retrieval_slices` on ordinary ranked-ID mappings from any retriever.
2. Compare the same population with a saved baseline and explicit `SuitePolicy`.
3. Inspect a real measured regression and stop deployment with `AuditFailed`.
4. Detect an **explicitly injected** document-ID namespace mismatch.
5. Block an **explicitly injected** missing cohort as insufficient evidence.
6. Save JSON/HTML, per-cohort JUnit, and a policy that the existing CLI can replay.

The unchanged baseline is a passing control. Expected gate failures are caught
for teaching, so Restart/Run All succeeds. In production, do not catch and ignore
the exception at the deployment boundary.

## Boundaries

- Binary, positive-only qrels are not exhaustive judgments. Full query coverage
  does not mean every document's relevance is known.
- A relevant abstract can support **or refute** a claim. Retrieval metrics do not
  assess generated answers, factual correctness, medical validity, or safety.
- This small, single-domain benchmark does not establish performance on a
  production corpus, multilingual search, or larger-scale infrastructure.
- No statistical significance, fairness conclusion, operational savings, or
  production adoption is claimed. Repeated tuning on test judgments would
  invalidate treating them as an untouched test set.
- The 20-query floor is an illustrative evidence rule, not proof of adequate
  statistical power or independence.

## Attribution and data rights

Thakur et al. (2021), *BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation
of Information Retrieval Models*, [benchmark source](https://github.com/beir-cellar/beir).
Wadden et al. (2020), *Fact or Fiction: Verifying Scientific Claims*,
[SciFact source](https://github.com/allenai/scifact).

The [upstream SciFact component license](https://github.com/allenai/scifact/blob/master/LICENSE.md)
lists CC BY 4.0 for claims/evidence annotations and ODC-By 1.0 for corpus
abstracts. We download the BEIR archive directly; mirror metadata may differ.
Do not treat ProofML's MIT code license as relicensing the dataset. Raw text is
not redistributed in this repository; notebook outputs contain aggregate
measurements and audit evidence.
