"""DNA sequence normalization and identity."""

from __future__ import annotations

import hashlib
import re

# IUPAC nucleotide codes. Anything outside this alphabet is a data error.
DNA_ALPHABET = frozenset("ACGTRYSWKMBDHVN")

_NON_SEQUENCE = re.compile(r"[\s\\]+")


def normalize_sequence(sequence: str) -> str:
    """Uppercase a sequence and drop whitespace and stray escape characters."""
    return _NON_SEQUENCE.sub("", sequence).upper()


def clean_sequence(sequence: str) -> str:
    """Normalize a sequence and drop characters outside the IUPAC alphabet.

    Used at ingestion, where source payloads occasionally carry annotation
    artefacts inside the sequence string.
    """
    return "".join(char for char in normalize_sequence(sequence) if char in DNA_ALPHABET)


def validate_sequence(sequence: str) -> str:
    """Normalize a sequence and reject empty or non-IUPAC input."""
    normalized = normalize_sequence(sequence)
    if not normalized:
        raise ValueError("sequence must be non-empty")
    invalid = sorted(set(normalized) - DNA_ALPHABET)
    if invalid:
        raise ValueError(f"sequence contains unsupported DNA symbols: {''.join(invalid)}")
    return normalized


def sequence_sha256(sequence: str) -> str:
    """Return a stable exact-sequence family identifier."""
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()
