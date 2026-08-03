"""Text normalization and stable hashing shared across the data pipelines."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable
from typing import Any

import numpy as np

# Metadata placeholders that carry no information and must never become a constraint.
EMPTY_VALUES = frozenset({"", "n a", "na", "none", "not available", "unknown"})

_SPLIT_PATTERN = re.compile(r"[,;|]")
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w]+", flags=re.UNICODE)


def sha256_text(value: str) -> str:
    """Return the SHA-256 hex digest of *value* encoded as UTF-8."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def collapse_whitespace(value: str) -> str:
    """Collapse runs of whitespace into single spaces and strip the result."""
    return _WHITESPACE.sub(" ", value).strip()


def normalize_phrase(value: Any) -> str:
    """Normalize one metadata phrase for conservative literal matching.

    Unicode is NFKC-folded, case is removed, and every non-word run becomes a
    single space, so ``"Amp-R"`` and ``"amp r"`` compare equal.
    """
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(_NON_WORD.sub(" ", text).split())


def normalize_values(value: Any) -> frozenset[str]:
    """Normalize a scalar, list, or semicolon-flattened metadata field.

    Returns the set of informative normalized phrases; placeholders such as
    ``"unknown"`` and missing values collapse to the empty set.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return frozenset()
    if isinstance(value, str):
        raw_values: Iterable[Any] = value.split(";")
    elif isinstance(value, list | tuple | set | frozenset | np.ndarray):
        raw_values = value
    else:
        raw_values = (value,)
    normalized = {normalize_phrase(item) for item in raw_values if item is not None}
    return frozenset(item for item in normalized if item not in EMPTY_VALUES)


def split_delimited(value: Any) -> list[str]:
    """Flatten a nested string/list/dict metadata value into trimmed strings.

    Strings are split on ``,``, ``;`` and ``|`` because Addgene stores several
    multi-valued fields as delimited text.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in _SPLIT_PATTERN.split(value) if item.strip()]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in split_delimited(entry)]
    if isinstance(value, list | tuple):
        return [item for entry in value for item in split_delimited(entry)]
    return [str(value)]


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    """De-duplicate case-insensitively while keeping first-seen order and casing."""
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def as_list(value: Any) -> list[Any]:
    """Coerce a possibly-missing list-like cell into a plain list.

    A left join leaves absent list columns as NaN and Parquet returns them as
    NumPy arrays, so callers cannot rely on either ``is None`` or truthiness.
    """
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list | tuple | set | frozenset):
        return list(value)
    return [value]
