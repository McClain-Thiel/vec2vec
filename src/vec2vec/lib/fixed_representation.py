"""Circular-window planning and embedding identities for fixed representations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircularWindow:
    """One model input window over a circular DNA sequence."""

    index: int
    start_bp: int
    input_base_count: int
    newly_covered_base_count: int
    wrapped_input_base_count: int


def circular_window_plan(
    sequence_length_bp: int,
    *,
    maximum_content_bp: int,
    tokenizer_unit_bp: int,
    overlap_fraction: float,
) -> list[CircularWindow]:
    """Plan token-aligned windows that cover each circular base at least once."""
    if sequence_length_bp < 1:
        raise ValueError("sequence_length_bp must be positive")
    if tokenizer_unit_bp < 1:
        raise ValueError("tokenizer_unit_bp must be positive")
    if maximum_content_bp < tokenizer_unit_bp:
        raise ValueError("maximum_content_bp must fit one tokenizer unit")
    if maximum_content_bp % tokenizer_unit_bp:
        raise ValueError("maximum_content_bp must be a multiple of tokenizer_unit_bp")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")

    input_base_count = min(
        maximum_content_bp,
        _round_up(sequence_length_bp, tokenizer_unit_bp),
    )
    if sequence_length_bp <= maximum_content_bp:
        return [
            CircularWindow(
                index=0,
                start_bp=0,
                input_base_count=input_base_count,
                newly_covered_base_count=sequence_length_bp,
                wrapped_input_base_count=max(0, input_base_count - sequence_length_bp),
            )
        ]

    stride = math.floor(maximum_content_bp * (1.0 - overlap_fraction))
    stride -= stride % tokenizer_unit_bp
    if stride < tokenizer_unit_bp:
        raise ValueError("overlap_fraction leaves a zero-length token-aligned stride")
    starts = list(range(0, sequence_length_bp - maximum_content_bp + 1, stride)) or [0]
    covered = np.zeros(sequence_length_bp, dtype=bool)
    windows: list[CircularWindow] = []

    while True:
        start = starts[len(windows)]
        positions = (start + np.arange(maximum_content_bp, dtype=np.int64)) % sequence_length_bp
        new_count = int((~covered[positions]).sum())
        covered[positions] = True
        windows.append(
            CircularWindow(
                index=len(windows),
                start_bp=start,
                input_base_count=maximum_content_bp,
                newly_covered_base_count=new_count,
                wrapped_input_base_count=max(0, start + maximum_content_bp - sequence_length_bp),
            )
        )
        if covered.all():
            break
        if len(windows) == len(starts):
            starts.append(starts[-1] + stride)
        if len(windows) > math.ceil(sequence_length_bp / stride) + 1:
            raise RuntimeError("circular window planner failed to cover the sequence")

    if sum(window.newly_covered_base_count for window in windows) != sequence_length_bp:
        raise RuntimeError("circular window weights do not sum to the sequence length")
    return windows


def circular_subsequence(sequence: str, window: CircularWindow) -> str:
    """Return one window, wrapping across the recorded origin when necessary."""
    if not sequence:
        raise ValueError("sequence must not be empty")
    return "".join(
        sequence[(window.start_bp + offset) % len(sequence)]
        for offset in range(window.input_base_count)
    )


def embedding_sha256(vector: np.ndarray) -> str:
    """Hash a normalized embedding in a fixed little-endian float32 representation."""
    array = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _round_up(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit
