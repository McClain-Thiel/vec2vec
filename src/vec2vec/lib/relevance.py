"""Constraint-aware relevance for many-to-many plasmid retrieval.

A generated description states requirements that several distinct plasmids can
legitimately satisfy, so treating only the paired sequence as relevant
mislabels genuine matches as errors.

The approach here is deliberately conservative. Only metadata values that appear
*literally* in a description become constraints, a candidate satisfies one only
when its own metadata records the required value, and metadata a candidate does
not record is treated as unknown rather than as a contradiction. Exact sequence
duplicates are always one family.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vec2vec.lib.text import normalize_phrase, normalize_values

#: Fields describing what a plasmid *does*, used for retrieval relevance.
FUNCTIONAL_FIELDS = (
    "vector_types",
    "bacterial_resistance",
    "insert_species",
    "plasmid_copy",
    "growth_strain",
    "growth_temp",
)
#: Fields describing the cargo a plasmid carries.
PAYLOAD_FIELDS = (
    "insert_names",
    "insert_alt_names",
    "insert_genes",
    "insert_gene_aliases",
    "insert_mutations",
    "insert_tags",
    "insert_promoters",
)
#: Everything eligible to become a constraint, including the two fields that
#: identify a specific construct rather than describing its function.
STRUCTURED_FIELDS = ("name", "backbone", *PAYLOAD_FIELDS, *FUNCTIONAL_FIELDS)

_LENGTH_PATTERN = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(bp|base pairs?|kb|kilobases?)(?!\w)",
    flags=re.IGNORECASE,
)


def _surfaced_length(description: str, expected_length: int | None) -> int | None:
    """Return *expected_length* when the description states a matching size."""
    if expected_length is None or expected_length <= 0:
        return None
    tolerance = max(100.0, expected_length * 0.05)
    for raw_value, unit in _LENGTH_PATTERN.findall(description):
        value = float(raw_value.replace(",", ""))
        if unit.casefold().startswith("k"):
            value *= 1000
        if abs(value - expected_length) <= tolerance:
            return expected_length
    return None


@dataclass(frozen=True)
class SurfaceConstraints:
    """Constraints visibly supported by one description."""

    fields: tuple[tuple[str, frozenset[str]], ...]
    features: frozenset[str]
    length_bp: int | None

    @property
    def group_count(self) -> int:
        """Number of independent metadata groups the query constrains."""
        return len(self.fields) + bool(self.features) + (self.length_bp is not None)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, order-stable view for persistence."""
        return {
            "fields": {field: sorted(values) for field, values in self.fields},
            "features": sorted(self.features),
            "length_bp": self.length_bp,
            "group_count": self.group_count,
        }


def extract_surface_constraints(
    description: str,
    field_values: Mapping[str, frozenset[str]],
    features: frozenset[str],
    length_bp: int | None,
) -> SurfaceConstraints:
    """Keep only the source metadata values that appear literally in *description*.

    The description is normalized once and space-padded, so a plain substring
    test acts as a whole-phrase match. That matters for cost: each row is checked
    against tens of field values and annotation names, and normalizing is the
    most expensive step in building the dataset.
    """
    haystack = f" {normalize_phrase(description)} "
    surfaced = (
        (field, frozenset(value for value in values if f" {value} " in haystack))
        for field, values in field_values.items()
    )
    return SurfaceConstraints(
        fields=tuple((field, values) for field, values in surfaced if values),
        features=frozenset(feature for feature in features if f" {feature} " in haystack),
        length_bp=_surfaced_length(description, length_bp),
    )


def constraints_from_values(values: Mapping[str, str]) -> SurfaceConstraints:
    """Build explicit constraints from normalized field/value pairs."""
    return SurfaceConstraints(
        fields=tuple((field, frozenset({value})) for field, value in values.items()),
        features=frozenset(),
        length_bp=None,
    )


class RelevanceIndex:
    """Inverted index over structured metadata, plus exact-sequence families.

    Answers two questions cheaply across the whole dataset: which rows record a
    given metadata value, and which rows carry a given sequence. Everything that
    reasons about queries is built on those two primitives.

    Args:
        sequence_hashes: One exact-sequence identifier per row.
        columns: Metadata columns keyed by name; must cover *fields*.
        fields: Metadata fields eligible to become constraints.
    """

    def __init__(
        self,
        sequence_hashes: Sequence[str],
        columns: Mapping[str, Sequence[Any]],
        *,
        fields: Sequence[str] = FUNCTIONAL_FIELDS,
    ) -> None:
        size = len(sequence_hashes)
        self.fields = tuple(fields)
        if not self.fields:
            raise ValueError("fields cannot be empty")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("fields cannot contain duplicates")
        missing = set(self.fields).difference(columns)
        if missing:
            raise ValueError(f"missing relevance columns: {sorted(missing)}")
        for name, values in columns.items():
            if len(values) != size:
                raise ValueError(f"column {name} has {len(values)} rows, expected {size}")

        self.sequence_hashes = tuple(str(value) for value in sequence_hashes)
        self.field_values = {
            field: tuple(normalize_values(value) for value in columns[field])
            for field in self.fields
        }

        self._exact: dict[str, set[int]] = defaultdict(set)
        self._inverted: dict[str, dict[str, set[int]]] = {
            field: defaultdict(set) for field in self.fields
        }
        self._known: dict[str, set[int]] = {field: set() for field in self.fields}
        for index in range(size):
            self._exact[self.sequence_hashes[index]].add(index)
            for field in self.fields:
                values = self.field_values[field][index]
                if values:
                    self._known[field].add(index)
                for value in values:
                    self._inverted[field][value].add(index)

    @classmethod
    def from_frame(cls, frame: Any, *, fields: Sequence[str] = FUNCTIONAL_FIELDS) -> RelevanceIndex:
        """Build an index from the retrieval dataset's columns."""
        missing = {"sequence_sha256", *fields}.difference(frame.columns)
        if missing:
            raise ValueError(f"missing relevance columns: {sorted(missing)}")
        return cls(
            frame["sequence_sha256"].astype(str).tolist(),
            {field: frame[field].tolist() for field in fields},
            fields=fields,
        )

    def exact_candidates(self, index: int) -> set[int]:
        """Return every row carrying this row's exact sequence."""
        return set(self._exact[self.sequence_hashes[index]])

    def candidates_with_field_values(self, field: str, required: Collection[str]) -> set[int]:
        """Return rows known to carry every requested normalized value."""
        if field not in self._inverted:
            raise ValueError(f"field is not indexed: {field}")
        if not required:
            raise ValueError("required values cannot be empty")
        candidates: set[int] | None = None
        for value in required:
            matches = self._inverted[field].get(value, set())
            candidates = set(matches) if candidates is None else candidates & matches
        return candidates or set()

    def partition_candidates_by_field(
        self,
        field: str,
        required: Collection[str],
        candidate_indices: Collection[int],
    ) -> tuple[set[int], set[int], set[int]]:
        """Split a candidate subset into ``(matches, contradictions, unknowns)``.

        The third bucket is the point: a candidate recording nothing for this
        field has contradicted nothing, and must not be counted as a negative.
        """
        if field not in self._inverted:
            raise ValueError(f"field is not indexed: {field}")
        if not required:
            raise ValueError("required values cannot be empty")
        candidates = set(candidate_indices)
        known = self._known[field] & candidates
        matches = set(known)
        for value in required:
            matches &= self._inverted[field].get(value, set())
        return matches, known - matches, candidates - known
