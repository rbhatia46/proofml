"""Split integrity: exact observations, entities, schemas, and temporal ordering."""
from datetime import datetime, timezone
from ..context import AuditContext
from ..models import CheckResult, Finding


class SchemaCheck:
    id = "split_schema"

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.test is None:
            return CheckResult(self.id, "skipped", reason="A test dataset is required.")
        train = set(ctx.train.columns) - {ctx.config.target}
        test = set(ctx.test.columns) - {ctx.config.target}
        findings = []
        if train != test:
            findings.append(Finding("schema_mismatch", "high", "confirmed", "Train and test schemas differ",
                "Non-target columns must be aligned before comparable feature checks can run.",
                "Align feature schemas; an omitted test target is allowed.", evidence={"missing_in_test": sorted(train - test), "extra_in_test": sorted(test - train)}))
        return CheckResult.complete(self.id, findings)


class OverlapCheck:
    id = "split_overlap"

    def run(self, ctx: AuditContext) -> CheckResult:
        if ctx.test is None:
            return CheckResult(self.id, "skipped", reason="A test dataset is required.")
        columns = tuple(c for c in ctx.train.columns if c != ctx.config.target)
        if not columns or set(columns) != set(ctx.test.columns) - {ctx.config.target}:
            return CheckResult(self.id, "skipped", reason="Matching non-target schemas are required.")
        train_indices = [ctx.train.columns.index(c) for c in columns]
        test_indices = [ctx.test.columns.index(c) for c in columns]
        fingerprints = {tuple(row[i] for i in train_indices) for row in ctx.train.rows}
        matches = sum(tuple(row[i] for i in test_indices) in fingerprints for row in ctx.test.rows)
        findings = []
        if matches:
            findings.append(Finding("train_test_overlap", "high", "confirmed", "Test observations also occur in training",
                "Exact normalized non-target rows match, including any declared ID/time columns. Repeated valid observations are possible; this does not prove contamination.",
                "Trace shared observations to their source and check the evaluation independence assumption.",
                evidence={"matching_test_rows": matches, "test_rows": len(ctx.test.rows), "fraction": matches / len(ctx.test.rows)}))
        return CheckResult.complete(self.id, findings)


class EntityCheck:
    id = "entity_overlap"

    def run(self, ctx: AuditContext) -> CheckResult:
        column = ctx.config.entity_id
        if not column or ctx.test is None or column not in ctx.test.columns:
            return CheckResult(self.id, "skipped", reason="An entity_id present in both splits is required.")
        train = set(ctx.train.column(column)) - {""}
        test = set(ctx.test.column(column)) - {""}
        shared = len(train & test)
        findings = []
        if shared:
            strict = ctx.config.require_disjoint_entities
            findings.append(Finding("shared_entities", "high" if strict else "medium", "confirmed" if strict else "needs_context",
                "Entities appear in both splits",
                "This violates the declared disjoint-entity requirement." if strict else "Shared entities can be valid for future predictions on existing customers, but not for evaluating unseen customers.",
                "Use a group-aware split if evaluation targets unseen entities.", (column,),
                {"shared_entities": shared, "test_entities": len(test), "disjoint_required": strict}))
        return CheckResult.complete(self.id, findings)


def parse_time(value: str) -> datetime:
    """ISO 8601 only. Naive timestamps are consistently interpreted as UTC."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class TemporalCheck:
    id = "temporal_order"

    def run(self, ctx: AuditContext) -> CheckResult:
        column = ctx.config.time_column
        if not ctx.config.expect_temporal_split:
            return CheckResult(self.id, "skipped", reason="Set expect_temporal_split for a future-only holdout.")
        if ctx.test is None or column not in ctx.test.columns:
            return CheckResult(self.id, "skipped", reason="The declared time column must be in both splits.")
        try:
            train = [parse_time(v) for v in ctx.train.column(column)]
            test = [parse_time(v) for v in ctx.test.column(column)]
        except (ValueError, OverflowError):
            return CheckResult.complete(self.id, [Finding("invalid_timestamp", "high", "confirmed", "Cannot validate temporal order",
                "At least one timestamp is missing or not parseable as ISO 8601.", "Provide complete ISO 8601 timestamps; dates and timezone offsets are supported.", (column,))])
        boundary = max(train)
        violations = sum(v <= boundary for v in test)
        findings = []
        if violations:
            findings.append(Finding("temporal_overlap", "high", "confirmed", "Test data is not strictly after training",
                "The declared future-only holdout requires every test timestamp to be after the last training timestamp.",
                "Split chronologically and consider a gap for delayed labels or rolling features.", (column,),
                {"test_rows_at_or_before_training_end": violations, "test_rows": len(test)}))
        return CheckResult.complete(self.id, findings)
