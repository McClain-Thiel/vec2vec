"""Stable JSON conversion for scientific records and content identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def to_jsonable(value: Any) -> Any:
    """Convert one nested table value into JSON-compatible built-in values."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def stable_json(value: Any) -> str:
    """Serialize nested table values with stable ordering and compact whitespace."""
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def json_content_sha256(value: Any) -> str:
    """Hash finite JSON-compatible content without coercing scientific values."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("content is not finite, JSON-compatible data") from error
    return hashlib.sha256(payload).hexdigest()


def dataframe_content_sha256(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    value_columns: Sequence[str] | None = None,
) -> str:
    """Hash a stable row representation for provenance checks after reload."""
    missing_sort = set(sort_columns).difference(frame.columns)
    if missing_sort:
        raise ValueError(f"hash sort columns are missing: {sorted(missing_sort)}")
    columns = list(value_columns) if value_columns is not None else sorted(frame.columns)
    missing_values = set(columns).difference(frame.columns)
    if missing_values:
        raise ValueError(f"hash value columns are missing: {sorted(missing_values)}")
    ordered = frame.sort_values(list(sort_columns), kind="stable").loc[:, columns]
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, separators=(",", ":")).encode())
    digest.update(b"\n")
    for row in ordered.itertuples(index=False, name=None):
        values = [_json_scalar(value) for value in row]
        digest.update(json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple | np.ndarray):
        return [_json_scalar(item) for item in value]
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(format(value, ".17g"))
    return value
