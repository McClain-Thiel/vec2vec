"""Stable identities for selected encoder outputs."""

from __future__ import annotations

import hashlib

import numpy as np


def embedding_sha256(vector: np.ndarray) -> str:
    """Hash a normalized embedding in a fixed little-endian float32 representation."""
    array = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()
