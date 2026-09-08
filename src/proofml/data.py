"""Bounded adapters and immutable normalized data.

CSV missing means empty/whitespace only; 'NA' may be a legitimate category.
All scalar values are stripped strings. Inputs are never executed or changed.
"""
from __future__ import annotations
import csv
import hashlib
import math
import json
from os import PathLike
from dataclasses import dataclass
from pathlib import Path
from .config import AuditConfig


@dataclass(frozen=True)
class Dataset:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    sha256: str
    format: str

    def column(self, name: str) -> tuple[str, ...]:
        index = self.columns.index(name)
        return tuple(row[index] for row in self.rows)

    def metadata(self) -> dict:
        return {"sha256": self.sha256, "rows": len(self.rows), "columns": list(self.columns), "format": self.format}


def number(value: str) -> float | None:
    try:
        result = float(value)
    except (ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _headers(values) -> tuple[str, ...]:
    columns = tuple(str(v).strip() for v in values)
    if not columns or any(not c for c in columns) or len(set(columns)) != len(columns):
        raise ValueError("Input must have nonempty, unique column names")
    return columns


def load_dataset(path, config: AuditConfig) -> Dataset:
    if not isinstance(path, (str, PathLike)):
        return from_dataframe(path, config)
    path = Path(path)
    if not path.is_file():
        raise ValueError("Dataset must be an existing regular file")
    if path.stat().st_size > config.max_bytes:
        raise ValueError("Dataset exceeds max_bytes; increase the limit explicitly")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    rows = []
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            columns = _headers(next(reader, []))
            for row in reader:
                if len(rows) >= config.max_rows:
                    raise ValueError("Dataset exceeds max_rows; no silent sampling is performed")
                if len(row) != len(columns):
                    raise ValueError(f"Malformed CSV record at physical line {reader.line_num}: width differs from header")
                rows.append(tuple(value.strip() for value in row))
    elif suffix == ".parquet":
        try:
            import pandas as pd
            import pyarrow.parquet as pq
        except ImportError as error:
            raise ValueError("Parquet requires the optional extra: pip install '.[parquet]'") from error
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows > config.max_rows:
            raise ValueError("Dataset exceeds max_rows; no silent sampling is performed")
        columns = _headers(parquet.schema_arrow.names)
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pandas().itertuples(index=False, name=None):
                normalized = []
                for value in row:
                    if not pd.api.types.is_scalar(value):
                        raise ValueError("Nested Parquet values are unsupported; use scalar columns")
                    normalized.append("" if pd.isna(value) else str(value).strip())
                rows.append(tuple(normalized))
    else:
        raise ValueError("Supported dataset formats are .csv and .parquet")
    if not rows:
        raise ValueError("Dataset contains no data rows")
    return Dataset(columns, tuple(rows), digest.hexdigest(), suffix[1:])


def from_dataframe(frame, config: AuditConfig) -> Dataset:
    """Snapshot a pandas frame without using its index as a model feature.

    The fingerprint covers normalized columns/values (not pandas dtype/index).
    It is not comparable with a file's byte fingerprint. Missing scalars become
    empty strings; infinity remains a nonfinite numeric string for validation.
    """
    try:
        import pandas as pd
    except ImportError as error:
        raise TypeError("Expected a CSV/Parquet path or pandas DataFrame; pandas is not installed") from error
    import numpy as np
    if isinstance(frame, np.ndarray):
        if frame.ndim != 2:
            raise ValueError("Feature arrays must be two-dimensional")
        if len(frame) > config.max_rows or frame.nbytes > config.max_bytes:
            raise ValueError("Array exceeds configured input limits")
        frame = pd.DataFrame(frame, columns=[f"feature_{i}" for i in range(frame.shape[1])])
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected a CSV/Parquet path, pandas DataFrame, or dense NumPy array")
    if frame.empty:
        raise ValueError("DataFrame must contain columns and data rows")
    if len(frame) > config.max_rows:
        raise ValueError("DataFrame exceeds max_rows; no silent sampling is performed")
    if int(frame.memory_usage(index=True, deep=True).sum()) > config.max_bytes:
        raise ValueError("DataFrame exceeds max_bytes based on pandas deep memory usage")
    if any(not isinstance(c, str) for c in frame.columns):
        raise ValueError("DataFrame columns must be strings; rename them before auditing")
    columns = _headers(frame.columns)
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, ensure_ascii=True).encode("utf-8"))
    rows = []
    for row in frame.itertuples(index=False, name=None):
        normalized = []
        for value in row:
            if not pd.api.types.is_scalar(value):
                raise ValueError("DataFrame cells must be scalars; nested values are unsupported")
            normalized.append("" if pd.isna(value) else str(value).strip())
        record = tuple(normalized)
        digest.update(b"\n")
        digest.update(json.dumps(record, ensure_ascii=True).encode("utf-8"))
        rows.append(record)
    return Dataset(columns, tuple(rows), digest.hexdigest(), "dataframe")
