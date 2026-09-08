"""Optional pandas/NumPy interoperability, isolated from the CSV-only path."""
from os import PathLike


def attach_target(source, labels, target, config):
    """Combine separate labels with in-memory features without index alignment.

    Pandas pairs must already have exactly matching indexes; implicit alignment
    could silently change row-label associations. NumPy/list labels are positional.
    """
    if labels is None:
        return source
    if isinstance(source, (str, PathLike)):
        raise TypeError("Separate labels require in-memory features; put file labels in a target column")
    try:
        import numpy as np
        import pandas as pd
    except ImportError as error:
        raise ImportError("Separate X/y inputs require the pandas extra: pip install 'proofml[pandas]'") from error

    if isinstance(source, pd.DataFrame):
        if len(source) > config.max_rows or source.memory_usage(index=True, deep=True).sum() > config.max_bytes:
            raise ValueError("Feature data exceeds configured input limits")
        if isinstance(labels, pd.Series) and not source.index.equals(labels.index):
            raise ValueError("X and y pandas indexes must match exactly, including order")
        frame = source.copy(deep=False)
    elif isinstance(source, np.ndarray):
        if source.ndim != 2:
            raise ValueError("Feature arrays must be two-dimensional")
        if len(source) > config.max_rows or source.nbytes > config.max_bytes:
            raise ValueError("Feature data exceeds configured input limits")
        frame = pd.DataFrame(source, columns=[f"feature_{i}" for i in range(source.shape[1])])
    else:
        raise TypeError("Separate labels require a pandas DataFrame or dense NumPy array")
    if frame.empty:
        raise ValueError("Features must contain at least one row and one column")
    if target in [str(c).strip() for c in frame.columns]:
        raise ValueError("Target name collides with a feature column; choose a different target name")
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) != len(frame):
        raise ValueError("Labels must be one-dimensional and match the feature row count")
    # A fresh frame prevents modifications to the caller's data even when pandas
    # copy-on-write is disabled. Index alignment has been validated above.
    return frame.assign(**{target: values})
