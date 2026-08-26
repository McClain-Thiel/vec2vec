from __future__ import annotations

from hashlib import sha256

import numpy as np

from vec2vec.lib import fixed_representation


def test_embedding_hash_uses_little_endian_float32_bytes() -> None:
    vector = np.asarray([1.0, -2.0], dtype=np.float64)
    expected = sha256(np.asarray(vector, dtype="<f4").tobytes(order="C")).hexdigest()
    assert fixed_representation.embedding_sha256(vector) == expected
