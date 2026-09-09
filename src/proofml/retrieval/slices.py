"""Explicit, bounded slice definitions over stable query IDs."""
from collections.abc import Mapping, Set
from dataclasses import asdict

from .._validation import ordered_iterable, positive_integer
from ..core import validate_checks
from ..suite import AuditSuite, MAX_REPORTS
from .checks import default_retrieval_checks
from .config import RetrievalConfig
from .data import load_retrieval, select_retrieval
from .engine import build_report


def audit_retrieval_slices(retrieved, relevant, *, groups=None, slices=None,
                           corpus_ids=None, config: RetrievalConfig | None = None, checks=None,
                           max_slices=50, max_memberships=1_000_000, **options) -> AuditSuite:
    """Audit overall retrieval and named slices using one validated input load.

    Supply exactly one of groups={query_id: group_name} for a full partition or
    slices={slice_name: iterable_of_query_ids} for explicit, possibly overlapping
    subsets. Names are public report metadata; never use personal identifiers.
    Unknown/missing IDs and exceeded budgets reject the audit before checks run.
    """
    positive_integer("max_slices", max_slices)
    positive_integer("max_memberships", max_memberships)
    if max_slices >= MAX_REPORTS:
        raise ValueError(f"max_slices must be less than {MAX_REPORTS}")
    if (groups is None) == (slices is None):
        raise ValueError("Supply exactly one of groups or slices")
    if not isinstance(retrieved, Mapping) or not isinstance(relevant, Mapping):
        raise TypeError("Slice audits require query-ID mappings for retrieved and relevant")
    config = RetrievalConfig(**{**(asdict(config) if config is not None else {}), **options})
    registry = validate_checks(default_retrieval_checks() if checks is None else checks, config.disabled_checks)
    data = load_retrieval(retrieved, relevant, corpus_ids, config)
    positions = {key: i for i, key in enumerate(data.query_ids)}
    selected = {}
    memberships = 0

    def check_name(name):
        if not isinstance(name, str) or not name.strip() or len(name) > 256 or name == "overall":
            raise ValueError("Slice names must be nonempty strings up to 256 characters; 'overall' is reserved")

    if groups is not None:
        if not isinstance(groups, Mapping) or groups.keys() != positions.keys():
            raise ValueError("groups must map every query ID exactly once, without missing or unknown IDs")
        for key in data.query_ids:
            name = groups[key]
            check_name(name)
            if name not in selected and len(selected) >= max_slices:
                raise ValueError("Slice count exceeds max_slices; no partial audit was run")
            selected.setdefault(name, []).append(positions[key])
            memberships += 1
            if memberships > max_memberships:
                raise ValueError("Slice memberships exceed max_memberships; no partial audit was run")
    else:
        if not isinstance(slices, Mapping) or not 1 <= len(slices) <= max_slices:
            raise ValueError("slices must be a nonempty, bounded mapping of names to query IDs")
        for name, keys in slices.items():
            check_name(name)
            seen = set()
            iterator = iter(keys) if isinstance(keys, Set) else ordered_iterable(keys, "slice query IDs")
            for key in iterator:
                memberships += 1
                if memberships > max_memberships:
                    raise ValueError("Slice memberships exceed max_memberships; no partial audit was run")
                if not isinstance(key, str) or key not in positions:
                    raise ValueError("Slice contains an unknown or invalid query ID")
                if key in seen:
                    raise ValueError("Repeated query ID within a slice; duplicates cannot weight its metrics")
                seen.add(key)
            # Follow input order rather than hash-set order: checks and metric
            # accumulation stay deterministic across Python hash seeds.
            selected[name] = sorted(positions[key] for key in seen)
    reports = {"overall": build_report(data, config, registry)}
    for name in sorted(selected):
        reports[name] = build_report(select_retrieval(data, tuple(selected[name])), config, registry)
    return AuditSuite(reports)
