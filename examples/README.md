# Start with a real workflow

These notebooks contain executed outputs, explanations, and runnable code. Read
them directly on GitHub, or run them locally with no API keys. They are tutorials,
not evidence of production adoption or general detection accuracy.

| Notebook | Real data | What you learn |
| --- | --- | --- |
| [Search/retrieval release gates](notebooks/scifact_retrieval.ipynb) | BEIR SciFact: 5,183 documents, 300 test queries | Compare actual local retrievers, audit cohorts, catch ID mismatches/missing cohorts, save CI evidence |
| [Prediction-time feature availability](notebooks/bank_marketing.ipynb) | UCI Bank Marketing: 41,188 rows | Detect declared unavailable predictors, measure misleading holdout scores, retain guardrails after correction |

## Run locally

```bash
git clone https://github.com/rbhatia46/proofml.git
cd proofml
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r examples/requirements-notebooks.txt
python -m jupyterlab examples/notebooks
```

Select the kernel from this environment, then **Restart Kernel and Run All**.
The loading cells explicitly download public archives over verified HTTPS.
Checksums pin the source, and changed archives fail rather than silently changing
the experiment. No embedding models, GPUs, API tokens, or cloud databases are
needed. Dependencies here are example tooling; the core package remains separate.

Read source/license notices in each notebook. Data licenses are not replaced by
ProofML's code license. Raw corpora, customer records, and trained models are not
committed; checked-in outputs contain aggregate evidence. New runs create new
directories under ignored `artifacts/`. Do not blindly delete matching rows or
interpret retrieval of an abstract as verification of a claim.

## Offline and headless execution

Download the official archives once and set `PROOFML_SCIFACT_ARCHIVE` and
`PROOFML_BANK_ARCHIVE` to their paths before launching Jupyter. These point to
the original ZIPs, not extracted directories. Both hashes are still verified.

For an automated, offline **fresh-kernel** check of both notebooks:

```bash
python scripts/verify_notebooks.py \
  --scifact-archive /path/to/scifact.zip \
  --bank-archive /path/to/bank+marketing.zip \
  --output artifacts/notebook-check
```

Alternatively, explicitly permit downloads with:

```bash
python scripts/verify_notebooks.py --download --output artifacts/notebook-check
```

Choose a new output directory. The verifier fails on a cell error, validates
notebook structure, and writes executed copies without modifying source
notebooks or installing a global kernel. The manually dispatched **Benchmark
notebooks** GitHub workflow runs this same integration check and uploads its
outputs. Ordinary tests stay network-free, with synthetic adapter/structure
fixtures; they do not substitute for a real-data notebook execution.

If downloading fails, check connectivity/certificates or supply an offline
archive. Never disable TLS verification or bypass a changed checksum.

## Prefer scripts?

```bash
python examples/scifact_retrieval.py --download --output artifacts/scifact
python examples/bank_marketing.py --download --output artifacts/bank-marketing
```

The notebooks share these helpers so examples do not maintain a second evaluator.
The integration cells call ProofML directly; substitute your own ranked-ID exports
or candidate DataFrames there. See the [SciFact case study](../docs/case-studies/scifact.md)
and [Bank Marketing case study](../docs/case-studies/bank-marketing.md) for measured
results, provenance, assumptions, and interpretation.

Smaller synthetic examples remain available: [slice regression](slice_regression.py),
[text and retrieval](text_and_retrieval.py), and [release policies](release_gate.py).
