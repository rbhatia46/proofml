"""Strict JSON configuration: unknown fields fail rather than silently doing less."""
from __future__ import annotations
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class AuditConfig:
    task: str = "classification"
    target: str | None = None
    entity_id: str | None = None
    time_column: str | None = None
    require_disjoint_entities: bool = False
    expect_temporal_split: bool = False
    unavailable_features: tuple[str, ...] = ()
    missing_threshold: float = 0.2
    imbalance_threshold: float = 0.1
    association_threshold: float = 0.98
    drift_threshold: float = 0.25
    min_samples: int = 30
    max_rows: int = 200_000
    max_bytes: int = 100_000_000
    disabled_checks: tuple[str, ...] = ()
    series_id: str | None = None
    expected_interval_seconds: int | None = None
    label_horizon_seconds: int | None = None
    embargo_seconds: int = 0

    def __post_init__(self) -> None:
        if self.task not in {"classification", "regression", "forecasting"}:
            raise ValueError("task must be classification, regression, or forecasting")
        for name in ("target", "entity_id", "time_column", "series_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a nonempty column name")
        for name in ("missing_threshold", "imbalance_threshold", "association_threshold", "drift_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("min_samples", "max_rows", "max_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (3 if name == "min_samples" else 1):
                raise ValueError(f"{name} must be a positive integer (min_samples >= 3)")
        for name in ("require_disjoint_entities", "expect_temporal_split"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("unavailable_features", "disabled_checks"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or any(not isinstance(v, str) or not v for v in value):
                raise ValueError(f"{name} must be a list of nonempty strings")
            object.__setattr__(self, name, tuple(value))
        if self.require_disjoint_entities and not self.entity_id:
            raise ValueError("require_disjoint_entities needs entity_id")
        if self.expect_temporal_split and not self.time_column:
            raise ValueError("expect_temporal_split needs time_column")
        for name in ("expected_interval_seconds", "label_horizon_seconds", "embargo_seconds"):
            value = getattr(self, name)
            minimum = 1 if name == "expected_interval_seconds" else 0
            if value is not None and (type(value) is not int or not minimum <= value <= 315_576_000):
                raise ValueError(f"{name} must be an integer between {minimum} and 315576000 seconds")
        if self.embargo_seconds is None:
            raise ValueError("embargo_seconds must be an integer")
        if self.task == "forecasting":
            if not self.time_column or not self.target:
                raise ValueError("forecasting requires target and time_column")
            object.__setattr__(self, "expect_temporal_split", True)
            if self.series_id in {self.time_column, self.target} or self.time_column == self.target:
                raise ValueError("Forecast target, time_column, and series_id must be distinct")
            if self.embargo_seconds and self.label_horizon_seconds is None:
                raise ValueError("embargo_seconds requires label_horizon_seconds")
        elif (self.series_id is not None or self.expected_interval_seconds is not None
              or self.label_horizon_seconds is not None or self.embargo_seconds):
            raise ValueError("Forecast settings require task='forecasting'")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_file(cls, path: str | Path) -> AuditConfig:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate config field: {key}")
                result[key] = value
            return result
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle, object_pairs_hook=unique_object)
        if not isinstance(data, dict):
            raise ValueError("Config must be a JSON object")
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
        return cls(**data)
