"""Small dependency-free validators shared by non-tabular adapters."""
from collections.abc import Mapping, Set
from numbers import Real


def positive_integer(name, value):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def fraction(name, value, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not (0 <= value <= 1 if allow_zero else 0 < value <= 1):
        raise ValueError(f"{name} must be a finite fraction in {'[0, 1]' if allow_zero else '(0, 1]'}")


def ordered_iterable(value, name):
    """Reject accidental scalar strings, mappings, and unordered row collections."""
    if isinstance(value, (str, bytes, Mapping, Set)):
        raise TypeError(f"{name} must be an ordered iterable, not a scalar, mapping, or set")
    try:
        return iter(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an ordered iterable") from error


def disabled_ids(value):
    values = tuple(ordered_iterable(value, "disabled_checks"))
    if any(not isinstance(v, str) or not v for v in values):
        raise ValueError("disabled_checks must contain nonempty strings")
    return values
