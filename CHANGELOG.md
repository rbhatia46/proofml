# Changelog

## 0.4.0 — Unreleased

- Added dense NumPy and separate X/y inputs with explicit pandas index checks.
- Added train/test compatibility checks for unseen classes/categories, numeric
  parsing changes, and missingness increases.
- Added per-row forecast label-availability timestamps for variable horizons
  and reporting delays, excluded from candidate features and schema matching.
- Expanded the regression suite to cover generalization and input alignment.

## 0.3.0 — Unreleased

- Added direct `audit(data, target=...)` arguments with optional reusable config.
- Added pandas DataFrame inputs alongside CSV/Parquet paths.
- Added report saving, findings tables, isolated notebook display, and explicit
  pipeline gates with required-check coverage.
- Unified package, CLI, and report versions; added source-distribution metadata,
  installed-wheel verification, and a manual Trusted Publishing workflow.
- Documented input compatibility separately from detection limits and publication status.

## 0.2.0 — 2026-09-08

- Added numeric forecasting as an explicit task with three modular checks:
  index integrity, fixed-interval cadence, and label-horizon/embargo boundaries.
- Added separate series metadata, shared-cutoff semantics, and strict validation
  of forecast-only settings. Existing classification/regression registries retain
  their nine checks and default behavior.
- Added faulty/corrected multi-series demos and a forecasting configuration guide.
- Added 17 forecasting tests and CI demo/package smoke checks.
- JSON remains schema 1.0; config fields and task-specific check IDs are additive.

## 0.1.0 — 2026-09-06

- Initial local CSV/Parquet audit engine, nine checks, extensible Python API,
  CLI severity gating, HTML/JSON reports, and 41 behavior tests.
