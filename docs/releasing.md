# Publishing the standard pip package

## Status and remaining ownership setup

The source builds an ordinary wheel and source distribution. A public PyPI
release has not been verified. On 2026-09-08 the ProofML project page returned
404; this does not reserve the name or guarantee PyPI will accept it. The local
repository is hosted at https://github.com/rbhatia46/proofml. Do not advertise `pip install
proofml` as available until an upload and a clean install from PyPI succeed.

The GitHub owner is `rbhatia46` and repository is `proofml`. A PyPI account must
still own the package and register its trusted publisher. Repository, documentation,
and issue URLs are configured in pyproject.toml. Do not paste API tokens into source or chat.

## Build and verify locally

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python -m build --outdir artifacts/release-0.7.1
python -m twine check --strict artifacts/release-0.7.1/*
python scripts/check_distribution.py artifacts/release-0.7.1
```

Use a fresh output directory per version. The build command creates an sdist
and builds a wheel from it, testing that the source archive contains the inputs
needed for a build. The distribution verifier installs the wheel in a fresh
environment outside the checkout, without an index or dependency downloads,
then runs both demos and the short Python API.

Version is declared once in `src/proofml/_version.py`; package metadata, CLI,
and report versions derive from it. Update CHANGELOG.md for each release.

## Configure Trusted Publishing

Follow the [Python Packaging Authority guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).
Register a pending publisher on TestPyPI and PyPI using:

- Project name: `proofml` (subject to registry acceptance)
- Owner: `rbhatia46`
- Repository: `proofml`
- Workflow filename: `publish.yml`
- Environment: `testpypi` or `pypi`, matching the chosen registry

The configured workflow uses short-lived identity tokens instead of a stored
API key. Configure the GitHub `pypi` environment with required reviewers as
described in the PyPA guide. No external account settings are changed by adding
the workflow file; that setup must actually be completed by an account owner.

## Release sequence

1. Push the reviewed package and workflow to the chosen GitHub repository.
2. Dispatch **Build and publish package** with destination `testpypi`.
3. Verify from a fresh environment:
   `python -m pip install --index-url https://test.pypi.org/simple/ --no-deps proofml==0.7.1`.
4. Create/push the reviewed `v0.7.1` tag. Dispatch the same workflow from that
   tag with destination `pypi`. The workflow checks that tag and version match.
5. Verify with `python -m pip install --index-url https://pypi.org/simple/ proofml==0.7.1`,
   then run the demos and Python quick start outside the repository.
6. Only then switch the README installation instructions to the public command.

Publishing is manual: ordinary pushes and PRs never upload a release. Builds,
tests, metadata validation, and installed-wheel verification precede the upload.
PyPI versions cannot be overwritten; fix a bad published release in a new version.
