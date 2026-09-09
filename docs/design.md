# Design principles and roadmap

ProofML's role is **evidence before trusting an evaluation**, not training every
kind of model. Its adoption case is a small, offline audit layer that fits an
existing workflow and supplies reviewable findings and CI contracts. It is a
pre-1.0 project; popularity, institutional endorsement, and production maturity
cannot be established by adding more APIs.

## Decisions in 0.5

1. **Separate domain semantics, share infrastructure.** Tabular, text, and
   retrieval contexts differ. They share findings, check execution, reports, and
   gates. An adapter must not erase meaningful structure just to fit a generic
   table abstraction.
2. **Evaluate outputs before adding framework integrations.** Document strings
   and ranked IDs work without depending on a particular agent framework or
   retriever. Native integrations should earn their maintenance cost by removing
   demonstrated friction, not by adding badges to a README.
3. **Keep the copyable path short.** One import and one audit call; named config
   objects are optional. Data is never silently corrected. Unknown options fail.
4. **No hidden evaluator.** Lexical overlap is not semantic equivalence; retrieval
   is not answer correctness. No API key, downloaded model, or external service
   is required for the supported dependency-free paths.
5. **Coverage is part of the result.** Skips and errors remain visible. Budgets
   cannot silently turn partial computations into passes. Deployment gates must
   require the checks their workflow depends on.
6. **Evidence has a privacy budget too.** Built-in findings expose aggregates,
   not raw text or IDs. Hashes/configuration may still be sensitive. Plugins are
   trusted local Python, not sandboxed or magically privacy-preserving.

## A practical architecture workflow

- Before a dataset merge: audit data integrity and split assumptions.
- Before text-model training: check document grouping, duplicates, and labels.
- Before a retriever release: evaluate a fixed judged query set, set explicit
  quality minima, verify corpus membership, and store reports with the code and
  corpus versions. Evaluate multiple slices separately rather than assuming an
  aggregate average represents every language or user cohort.
- Before trusting CI: require expected checks and review judgment coverage.
  Passing thresholds does not establish statistical significance or safety.

For cross-validation, audit each fold independently and retain one report per
fold. For heterogeneous projects, compose the explicit audit functions in your
pipeline; no workflow scheduler or autonomous agent is required.

## What must happen before a stable 1.0

Version 0.6 adds evidence-aware policies and strict retrieval baseline comparison.
Version 0.7 adds explicit retrieval slices and named audit suites, keeping each
cohort's or fold's evidence separate. Missing cohorts and insufficient evaluated
populations can block a release even when the overall score improves. See
[the policy contract](release-gates.md) and [slice evaluation](slices.md).
These features do not establish production maturity or statistical significance.

- Gather reproducible failure/clean cases from independent users and domains;
  measure false positives and missed failures, not just unit-test counts.
- Benchmark runtime/memory on documented corpus sizes and publish methodology.
- Exercise package installation, required-check gates, and report consumers in
  downstream projects; define deprecation and schema-compatibility policies.
- Complete public PyPI publication and verify installation from the registry.

Potential next domains include image dataset integrity and exported agent trace
contracts. They are **not implemented**: each needs a useful input schema,
privacy/resource limits, independently checkable evidence, legitimate
counterexamples, and an optional-dependency strategy. Audio/video understanding,
LLM judgment, and arbitrary project certification are not implied by this roadmap.

Useful bug reports, accurate documentation, reproducible examples, and stable
contracts are the adoption strategy. Stars are an outcome, not a product feature.
