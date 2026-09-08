"""Fixed-duration forecasting checks for a single shared training cutoff.

Time denotes forecast origin, not the date of the future outcome. Series keys
remain internal; evidence only exposes aggregates. No resampling or imputation
is performed. Variable label availability is supported through declared timestamps.
"""
from collections import defaultdict
from datetime import datetime

from ..context import AuditContext
from ..data import Dataset
from ..models import CheckResult, Finding
from .splits import parse_time


def group_times(data: Dataset, ctx: AuditContext) -> dict[str, list[datetime]]:
    """Fail the entire split on invalid keys/times; never silently drop rows."""
    config = ctx.config
    if config.time_column not in data.columns:
        raise ValueError("Declared time column is absent")
    if config.series_id and config.series_id not in data.columns:
        raise ValueError("Declared series column is absent")
    keys = data.column(config.series_id) if config.series_id else ("__single__",) * len(data.rows)
    groups = defaultdict(list)
    for key, value in zip(keys, data.column(config.time_column)):
        if not key:
            raise ValueError("Missing series key")
        groups[key].append(parse_time(value))
    return dict(groups)


class ForecastIndexCheck:
    id = "forecast_index"

    def run(self, ctx: AuditContext) -> CheckResult:
        findings = []
        for split, data in (("train", ctx.train), ("test", ctx.test)):
            if data is None:
                continue
            try:
                groups = group_times(data, ctx)
            except (ValueError, OverflowError):
                findings.append(Finding(
                    "invalid_forecast_index", "high", "confirmed", f"Invalid forecast index in {split}",
                    "The declared time/series column is absent, a series key is empty, or a timestamp is invalid.",
                    "Supply complete series keys and ISO 8601 forecast-origin timestamps.",
                    evidence={"split": split},
                ))
                continue
            duplicates = sum(len(times) - len(set(times)) for times in groups.values())
            if duplicates:
                findings.append(Finding(
                    "duplicate_series_timestamp", "high", "confirmed", f"Repeated series timestamps in {split}",
                    "More than one row has the same normalized forecast origin within a series, even if other values differ.",
                    "Resolve the observation grain or aggregate intentionally before constructing lag features.",
                    evidence={"split": split, "extra_rows_at_duplicate_times": duplicates},
                ))
            unordered = sum(any(b < a for a, b in zip(times, times[1:])) for times in groups.values())
            if unordered:
                findings.append(Finding(
                    "unsorted_forecast_series", "medium", "suspicious", f"Series are out of time order in {split}",
                    "Some series have backwards timestamps in input row order. Positional lag operations may then be incorrect.",
                    "Sort within each series before rolling/shift operations; existing preprocessing may already do so.",
                    evidence={"split": split, "unordered_series": unordered},
                ))
        return CheckResult.complete(self.id, findings)


class ForecastCadenceCheck:
    id = "forecast_cadence"

    def run(self, ctx: AuditContext) -> CheckResult:
        interval = ctx.config.expected_interval_seconds
        if interval is None:
            return CheckResult(self.id, "skipped", reason="Declare expected_interval_seconds for a fixed-duration sampling grid.")
        findings = []
        pairs = 0
        singletons = 0
        for split, data in (("train", ctx.train), ("test", ctx.test)):
            if data is None:
                continue
            try:
                groups = group_times(data, ctx)
            except (ValueError, OverflowError):
                return CheckResult(self.id, "skipped", reason="Invalid forecast index; resolve forecast_index findings first.")
            irregular = missing_slots = 0
            for times in groups.values():
                unique = sorted(set(times))
                if len(unique) < 2:
                    singletons += 1
                for a, b in zip(unique, unique[1:]):
                    pairs += 1
                    delta = (b - a).total_seconds()
                    if delta != interval:
                        irregular += 1
                        # Only exact multiples establish a count of missing slots.
                        if delta > interval and delta % interval == 0:
                            missing_slots += int(delta // interval) - 1
            if irregular:
                findings.append(Finding(
                    "irregular_forecast_cadence", "medium", "confirmed", f"Sampling grid differs in {split}",
                    "Adjacent unique timestamps within series differ from the declared fixed interval.",
                    "Inspect missing or off-grid observations. Do not fill gaps without considering their meaning.",
                    evidence={"split": split, "irregular_intervals": irregular,
                              "missing_slots_in_exact_multiple_gaps": missing_slots, "expected_interval_seconds": interval},
                ))
        if not pairs:
            return CheckResult(self.id, "skipped", reason="No series has two distinct timestamps within a split.")
        result = CheckResult.complete(self.id, findings)
        return CheckResult(self.id, result.status, result.findings,
                           reason=f"Checked {pairs} within-split intervals; {singletons} single-timestamp groups unassessed. Split-boundary gaps are excluded.")


class ForecastBoundaryCheck:
    id = "forecast_label_boundary"

    def run(self, ctx: AuditContext) -> CheckResult:
        horizon = ctx.config.label_horizon_seconds
        available_column = ctx.config.label_available_column
        if horizon is None and not available_column:
            return CheckResult(self.id, "skipped", reason="Declare label_horizon_seconds or label_available_column to establish training label availability.")
        if ctx.test is None:
            return CheckResult(self.id, "skipped", reason="A test dataset is required to establish the shared cutoff.")
        try:
            train = group_times(ctx.train, ctx)
            test = group_times(ctx.test, ctx)
        except (ValueError, OverflowError):
            return CheckResult(self.id, "skipped", reason="Invalid forecast index; resolve forecast_index findings first.")
        cutoff = min(t for times in test.values() for t in times)
        # Compare timedeltas rather than adding a horizon to datetime.max.
        # Equality fails by contract: labels must be available strictly BEFORE
        # the first test origin, including the optional embargo margin.
        if available_column:
            try:
                available = [parse_time(v) for v in ctx.train.column(available_column)]
                origins = [parse_time(v) for v in ctx.train.column(ctx.config.time_column)]
            except (ValueError, OverflowError):
                return CheckResult.complete(self.id, [Finding(
                    "invalid_label_availability", "high", "confirmed", "Invalid label-availability timestamps",
                    "At least one training label availability is missing or not ISO 8601.",
                    "Provide the actual availability timestamp for every training label.", (available_column,),
                )])
            backwards = sum(a < o for a, o in zip(available, origins))
            if backwards:
                return CheckResult.complete(self.id, [Finding(
                    "label_available_before_origin", "high", "confirmed", "Forecast labels precede their origins",
                    "The declared future-outcome labels have availability timestamps earlier than their forecast origins.",
                    "Check timestamp semantics and joins before assessing cutoff overlap.", (available_column,),
                    {"invalid_rows": backwards},
                )])
            affected = sum((cutoff - t).total_seconds() <= ctx.config.embargo_seconds for t in available)
        else:
            required = horizon + ctx.config.embargo_seconds
            affected = sum((cutoff - t).total_seconds() <= required for times in train.values() for t in times)
        findings = []
        if affected:
            findings.append(Finding(
                "forecast_label_boundary_overlap", "high", "confirmed", "Training labels cross the forecast cutoff",
                "Given the declared label availability and embargo, some training labels are not available strictly before the earliest test origin across all series.",
                "Purge affected training origins or move the holdout later; fit once using only labels available before that cutoff.",
                evidence={"affected_training_rows": affected, "training_rows": len(ctx.train.rows),
                          "label_horizon_seconds": horizon, "embargo_seconds": ctx.config.embargo_seconds,
                          "label_available_column": available_column,
                          "cutoff_scope": "global_first_test_origin", "boundary": "strictly_before"},
            ))
        return CheckResult.complete(self.id, findings)
