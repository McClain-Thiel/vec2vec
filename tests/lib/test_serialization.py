"""Tests for stable scientific-record serialization."""

import numpy as np
import pandas as pd
import pytest

from vec2vec.lib.serialization import json_content_sha256, stable_json, to_jsonable
from vec2vec.lib.text import exact_metadata_key


def test_stable_json_normalizes_nested_table_values() -> None:
    value = {
        "missing": pd.NA,
        "array": np.array([np.int64(2), np.int64(1)]),
        "tuple": ("β", None),
    }

    assert to_jsonable(value) == {
        "missing": None,
        "array": [2, 1],
        "tuple": ["β", None],
    }
    assert stable_json(value) == '{"array":[2,1],"missing":null,"tuple":["β",null]}'


def test_exact_metadata_key_preserves_scientific_punctuation() -> None:
    assert exact_metadata_key("  AMP + Chl (12.5 μg/mL) ") == "amp + chl (12.5 μg/ml)"
    assert exact_metadata_key(None) is None


def test_json_content_hash_is_order_independent_and_rejects_nonfinite_values() -> None:
    assert json_content_sha256({"b": 2, "a": 1}) == json_content_sha256({"a": 1, "b": 2})
    with pytest.raises(ValueError, match="finite, JSON-compatible"):
        json_content_sha256({"value": float("nan")})
