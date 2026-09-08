# Contributing to ProofML

Start with a reproducible failure in a data-science workflow. Include a small
synthetic dataset and the expected finding. Include a legitimate counterexample
that must *not* be flagged. Avoid sharing private data.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
proofml demo --output artifacts/demo --fail-on none
```

Windows activation: `.venv\Scripts\activate`. The CSV suite can also run without
installation: `PYTHONPATH=src python -m unittest discover -s tests -v` on POSIX.
Install `python -m pip install -e '.[parquet]'` to test the optional adapter.
Use `python -m pip install -e '.[dev]'` for the complete release suite, including
reference-dataset tests and build/metadata verification tools. scikit-learn is
used only by those compatibility tests, never required for normal audits.
CI runs Linux, macOS, Windows, Python 3.10/3.13, plus a separate Parquet job.

## Design rules

1. Keep algorithms in independent checks, never in the CLI or renderer.
2. Describe evidence before assigning severity. Distinguish observation from interpretation.
3. Missing prerequisites must be explained, not silently treated as success.
4. Keep reports free of raw records. Escape all user-provided report content.
5. Add regression tests for both failures and clean data, plus malformed input.
6. Document complexity, thresholds, applicability, and false-positive cases.
7. Avoid network dependencies in the default audit path.

See [architecture](docs/architecture.md) for plugin contracts and
[check catalog](docs/checks.md) for current coverage. Tests use unittest so
contributors do not need another runner. Before submission, install the package
in a fresh environment and run the bundled demo from outside the repository.

The project is an initial release. Production deployment should validate the
checks against your own labelled failure cases and operational constraints.
