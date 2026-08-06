"""Stable JSON conversion for scientific records and content identities."""

from __future__ import annotations

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
