"""Stable JSON conversion for scientific records and content identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
