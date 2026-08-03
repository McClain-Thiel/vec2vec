"""Constraint-aware relevance for many-to-many plasmid retrieval.

A generated description states requirements that several distinct plasmids can
legitimately satisfy, so treating only the paired sequence as relevant
mislabels genuine matches as errors.

The approach here is deliberately conservative. Only metadata values that appear
*literally* in a description become constraints, a candidate is relevant when it
satisfies all of them, and metadata a candidate simply does not record is
treated as unknown rather than as a contradiction. Exact sequence duplicates are
always one positive family.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from vec2vec.lib.sequences import sequence_sha256
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
#: Fields identifying a specific construct rather than its function.
IDENTITY_FIELDS = ("name", "backbone")
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
STRUCTURED_FIELDS = (*IDENTITY_FIELDS, *PAYLOAD_FIELDS, *FUNCTIONAL_FIELDS)

_LENGTH_PATTERN = re.compile(
    r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(bp|base pairs?|kb|kilobases?)(?!\w)",
    flags=re.IGNORECASE,
)


def _is_surfaced(description: str, value: str) -> bool:
    """True when a normalized phrase appears as whole words in a description."""
    return f" {value} " in f" {normalize_phrase(description)} "


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


@dataclass(frozen=True)
class CandidateJudgment:
    """Constraint-level explanation of one query/candidate pair."""

    exact: bool
    acceptable: bool
    grade: float
    matched_groups: tuple[str, ...]
    contradicted_groups: tuple[str, ...]
    unknown_groups: tuple[str, ...]


def extract_surface_constraints(
    description: str,
    field_values: Mapping[str, frozenset[str]],
    features: frozenset[str],
    length_bp: int | None,
) -> SurfaceConstraints:
    """Keep only the source metadata values that appear literally in *description*."""
    surfaced = (
        (field, frozenset(value for value in values if _is_surfaced(description, value)))
        for field, values in field_values.items()
    )
    return SurfaceConstraints(
        fields=tuple((field, values) for field, values in surfaced if values),
        features=frozenset(feature for feature in features if _is_surfaced(description, feature)),
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
    """Inverted index over structured metadata for constraint-based relevance.

    Args:
        descriptions: One generated description per row.
        sequences: The paired DNA sequence per row.
        columns: Metadata columns keyed by name. Must cover *fields*; may also
            supply ``annotations``, ``length_bp`` and ``sequence_sha256``.
        fields: Metadata fields eligible to become constraints.
        min_constraint_groups: Queries constraining fewer groups than this are
            too vague to admit non-exact positives.
        length_tolerance_fraction: Relative length tolerance for a match.
        length_tolerance_bp: Absolute floor on the length tolerance.
    """

    def __init__(
        self,
        descriptions: Sequence[str],
        sequences: Sequence[str],
        columns: Mapping[str, Sequence[Any]],
        *,
        fields: Sequence[str] = FUNCTIONAL_FIELDS,
        min_constraint_groups: int = 2,
        length_tolerance_fraction: float = 0.15,
        length_tolerance_bp: int = 500,
    ) -> None:
        size = len(descriptions)
        if size != len(sequences):
            raise ValueError("description and sequence counts differ")
        if min_constraint_groups < 1:
            raise ValueError("min_constraint_groups must be positive")
        if not 0.0 <= length_tolerance_fraction <= 1.0:
            raise ValueError("length_tolerance_fraction must be between 0 and 1")
        if length_tolerance_bp < 0:
            raise ValueError("length_tolerance_bp cannot be negative")

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

        self.min_constraint_groups = min_constraint_groups
        self.length_tolerance_fraction = length_tolerance_fraction
        self.length_tolerance_bp = length_tolerance_bp

        hashes = columns.get("sequence_sha256")
        if hashes is None:
            hashes = [sequence_sha256(sequence) for sequence in sequences]
        self.sequence_hashes = tuple(str(value) for value in hashes)
        self.field_values = {
            field: tuple(normalize_values(value) for value in columns[field])
            for field in self.fields
        }
        self.features = tuple(
            normalize_values(value) for value in columns.get("annotations", [()] * size)
        )
        self.lengths = tuple(
            None
            if value is None or (isinstance(value, float) and math.isnan(value))
            else int(value)
            for value in columns.get("length_bp", [None] * size)
        )
        self.constraints = tuple(
            extract_surface_constraints(
                str(descriptions[index]),
                {field: self.field_values[field][index] for field in self.fields},
                self.features[index],
                self.lengths[index],
            )
            for index in range(size)
        )

        # Sorted length index, so a tolerance window is two binary searches.
        pairs = sorted(
            (length, index) for index, length in enumerate(self.lengths) if length is not None
        )
        self._sorted_lengths = np.array([length for length, _ in pairs], dtype=np.int64)
        self._length_order = np.array([index for _, index in pairs], dtype=np.int64)

        self._exact: dict[str, set[int]] = defaultdict(set)
        self._field_inverted: dict[str, dict[str, set[int]]] = {
            field: defaultdict(set) for field in self.fields
        }
        self._field_known: dict[str, set[int]] = {field: set() for field in self.fields}
        self._feature_inverted: dict[str, set[int]] = defaultdict(set)
        self._feature_known: set[int] = set()
        for index in range(size):
            self._exact[self.sequence_hashes[index]].add(index)
            for field in self.fields:
                values = self.field_values[field][index]
                if values:
                    self._field_known[field].add(index)
                for value in values:
                    self._field_inverted[field][value].add(index)
            if self.features[index]:
                self._feature_known.add(index)
            for feature in self.features[index]:
                self._feature_inverted[feature].add(index)

    @classmethod
    def from_frame(
        cls,
        frame: Any,
        *,
        fields: Sequence[str] = FUNCTIONAL_FIELDS,
        **kwargs: Any,
    ) -> RelevanceIndex:
        """Build an index from the paired dataset's DataFrame columns."""
        missing = {"description", "sequence", *fields}.difference(frame.columns)
        if missing:
            raise ValueError(f"missing relevance columns: {sorted(missing)}")
        optional = {"annotations", "length_bp", "sequence_sha256"}.intersection(frame.columns)
        columns = {name: frame[name].tolist() for name in (*fields, *sorted(optional))}
        return cls(
            frame["description"].astype(str).tolist(),
            frame["sequence"].astype(str).tolist(),
            columns,
            fields=fields,
            **kwargs,
        )

    def _length_tolerance(self, query_length: int) -> int:
        return max(self.length_tolerance_bp, round(query_length * self.length_tolerance_fraction))

    def _length_matches(self, query_length: int, candidate_index: int) -> bool:
        candidate_length = self.lengths[candidate_index]
        if candidate_length is None:
            return False
        return abs(candidate_length - query_length) <= self._length_tolerance(query_length)

    def _length_candidates(self, query_length: int) -> set[int]:
        tolerance = self._length_tolerance(query_length)
        start = int(np.searchsorted(self._sorted_lengths, query_length - tolerance, "left"))
        stop = int(np.searchsorted(self._sorted_lengths, query_length + tolerance, "right"))
        return set(self._length_order[start:stop].tolist())

    def compatible_candidates(self, constraints: SurfaceConstraints) -> set[int]:
        """Return candidates satisfying every supplied structured constraint."""
        if constraints.group_count < self.min_constraint_groups:
            return set()

        compatible: set[int] | None = None
        for field, required in constraints.fields:
            if field not in self._field_inverted:
                raise ValueError(f"constraint field is not indexed: {field}")
            for value in required:
                matches = self._field_inverted[field].get(value, set())
                compatible = set(matches) if compatible is None else compatible & matches
        for feature in constraints.features:
            matches = self._feature_inverted.get(feature, set())
            compatible = set(matches) if compatible is None else compatible & matches
        if constraints.length_bp is not None:
            matches = self._length_candidates(constraints.length_bp)
            compatible = matches if compatible is None else compatible & matches
        return compatible or set()

    def exact_candidates(self, query_index: int) -> set[int]:
        """Return every row carrying the query's exact sequence."""
        return set(self._exact[self.sequence_hashes[query_index]])

    def positive_candidates(self, query_index: int) -> set[int]:
        """Return exact-sequence and structurally compatible candidates."""
        return self.exact_candidates(query_index) | self.compatible_candidates(
            self.constraints[query_index]
        )

    def candidates_with_field_values(
        self, field: str, required_values: Collection[str]
    ) -> set[int]:
        """Return candidates known to carry every requested normalized value."""
        if field not in self._field_inverted:
            raise ValueError(f"field is not indexed: {field}")
        if not required_values:
            raise ValueError("required_values cannot be empty")
        candidates: set[int] | None = None
        for value in required_values:
            matches = self._field_inverted[field].get(value, set())
            candidates = set(matches) if candidates is None else candidates & matches
        return candidates or set()

    def partition_candidates_by_field(
        self,
        field: str,
        required_values: Collection[str],
        candidate_indices: Collection[int],
    ) -> tuple[set[int], set[int], set[int]]:
        """Split a candidate subset into ``(matches, contradictions, unknowns)``."""
        if field not in self._field_inverted:
            raise ValueError(f"field is not indexed: {field}")
        if not required_values:
            raise ValueError("required_values cannot be empty")
        candidates = set(candidate_indices)
        known = self._field_known[field] & candidates
        matches = set(known)
        for value in required_values:
            matches &= self._field_inverted[field].get(value, set())
        return matches, known - matches, candidates - known

    def judge_candidate(self, query_index: int, candidate_index: int) -> CandidateJudgment:
        """Explain whether a candidate satisfies each surfaced query constraint."""
        return self.judge_constraints(
            self.constraints[query_index],
            candidate_index,
            exact=self.sequence_hashes[query_index] == self.sequence_hashes[candidate_index],
        )

    def judge_constraints(
        self,
        constraints: SurfaceConstraints,
        candidate_index: int,
        *,
        exact: bool = False,
    ) -> CandidateJudgment:
        """Explain whether one candidate satisfies explicit structured constraints."""
        matched: list[str] = []
        contradicted: list[str] = []
        unknown: list[str] = []

        def classify(
            group: str, candidate_values: frozenset[str], required: frozenset[str]
        ) -> None:
            if not candidate_values:
                unknown.append(group)
            elif required.issubset(candidate_values):
                matched.append(group)
            else:
                contradicted.append(group)

        for field, required in constraints.fields:
            if field not in self.field_values:
                raise ValueError(f"constraint field is not indexed: {field}")
            classify(field, self.field_values[field][candidate_index], required)
        if constraints.features:
            classify("annotations", self.features[candidate_index], constraints.features)
        if constraints.length_bp is not None:
            if self.lengths[candidate_index] is None:
                unknown.append("length_bp")
            elif self._length_matches(constraints.length_bp, candidate_index):
                matched.append("length_bp")
            else:
                contradicted.append("length_bp")

        specific = constraints.group_count >= self.min_constraint_groups
        acceptable = exact or (specific and not contradicted and not unknown)
        if acceptable:
            grade = 1.0
        elif specific and not contradicted:
            grade = len(matched) / constraints.group_count
        else:
            grade = 0.0
        return CandidateJudgment(
            exact=exact,
            acceptable=acceptable,
            grade=grade,
            matched_groups=tuple(matched),
            contradicted_groups=tuple(contradicted),
            unknown_groups=tuple(unknown),
        )
