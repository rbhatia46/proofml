"""Read-only check context: declared semantics control applicability."""
from dataclasses import dataclass
from .config import AuditConfig
from .data import Dataset


@dataclass(frozen=True)
class AuditContext:
    train: Dataset
    test: Dataset | None
    config: AuditConfig

    @property
    def features(self) -> tuple[str, ...]:
        excluded = {self.config.target, self.config.entity_id, self.config.time_column,
                    self.config.series_id, self.config.label_available_column}
        return tuple(c for c in self.train.columns if c not in excluded)
