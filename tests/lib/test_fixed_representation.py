from __future__ import annotations

from hashlib import sha256

import numpy as np

from vec2vec.lib import fixed_representation


def test_short_circular_window_wraps_to_the_tokenizer_unit() -> None:
    windows = fixed_representation.circular_window_plan(
        10,
        maximum_content_bp=48,
        tokenizer_unit_bp=6,
        overlap_fraction=0.25,
    )

    assert windows == [
        fixed_representation.CircularWindow(
            index=0,
            start_bp=0,
            input_base_count=12,
            newly_covered_base_count=10,
            wrapped_input_base_count=2,
        )
    ]
    assert fixed_representation.circular_subsequence("ACGTACGTAA", windows[0]) == "ACGTACGTAAAC"


def test_long_circular_windows_cover_each_base_once_by_weight() -> None:
    windows = fixed_representation.circular_window_plan(
        100,
        maximum_content_bp=48,
        tokenizer_unit_bp=6,
        overlap_fraction=0.25,
    )

    assert [window.start_bp for window in windows] == [0, 36, 72]
    assert [window.newly_covered_base_count for window in windows] == [48, 36, 16]
    assert sum(window.newly_covered_base_count for window in windows) == 100
    assert windows[-1].wrapped_input_base_count == 20


def test_embedding_hash_uses_little_endian_float32_bytes() -> None:
    vector = np.asarray([1.0, -2.0], dtype=np.float64)
    expected = sha256(np.asarray(vector, dtype="<f4").tobytes(order="C")).hexdigest()
    assert fixed_representation.embedding_sha256(vector) == expected
