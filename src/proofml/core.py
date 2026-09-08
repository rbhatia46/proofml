"""Modality-independent execution; adapters own input semantics, checks own evidence."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Iterable, Protocol, TypeVar

from .models import CheckResult

ContextT = TypeVar("ContextT", contravariant=True)


class Check(Protocol[ContextT]):
    """Explicit, trusted plugin contract. Context types are domain-specific."""

    id: str

    def run(self, ctx: ContextT) -> CheckResult: ...


def validate_checks(checks: Iterable[Check[Any]], disabled: Iterable[str] = ()) -> tuple:
    registry = tuple(checks)
    ids = [check.id for check in registry]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Check IDs must be nonempty and unique")
    if isinstance(disabled, str) or set(disabled) - set(ids):
        raise ValueError("disabled_checks must contain known check IDs")
    return registry


def run_checks(ctx: Any, registry: Iterable[Check[Any]], disabled: Iterable[str] = ()) -> tuple[CheckResult, ...]:
    """Never turn an exception, invalid result, or nonfinite metric into a pass.

    Exceptions expose only their type, because their message may contain input
    content. Plugins themselves are trusted code, not a security sandbox.
    """
    results = []
    for check in registry:
        if check.id in disabled:
            results.append(CheckResult(check.id, "skipped", reason="Disabled explicitly in configuration."))
            continue
        try:
            result = check.run(ctx)
            if not isinstance(result, CheckResult) or result.check_id != check.id:
                raise ValueError("Invalid plugin result")
            result.__post_init__()  # Revalidate mutable metric dictionaries after plugin execution.
            json.dumps(asdict(result), allow_nan=False)
            results.append(result)
        except Exception as error:
            results.append(CheckResult(check.id, "error", reason=f"Check raised {type(error).__name__}; run it directly to debug with your data."))
    return tuple(results)
