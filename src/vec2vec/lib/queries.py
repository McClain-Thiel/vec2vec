"""Deterministic structured queries and metadata-proven hard negatives.

Retrieval training needs negatives that are hard but *provably* wrong. Sampling
them at random gives easy negatives; sampling by embedding similarity risks
labelling genuine matches as negatives. Instead, same-backbone peers are
partitioned using the source metadata itself: a peer is a hard negative only
when its recorded metadata contradicts an explicit query requirement, never
because the metadata is absent.

Every choice is seeded and content-addressed, so a given ``(seed, epoch, row)``
always yields the same queries and the same negatives.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from vec2vec.lib.relevance import RelevanceIndex, SurfaceConstraints, constraints_from_values

#: Fields eligible to become query requirements, each with a rendering template.
QUERY_TEMPLATES: Mapping[str, str] = {
    "vector_types": "usable as a {value} vector",
    "bacterial_resistance": "supporting {value} bacterial selection",
    "insert_species": "with an insert from {value}",
    "plasmid_copy": "with {value} copy number",
    "growth_strain": "propagated in {value}",
    "insert_genes": "carrying the gene {value}",
    "insert_mutations": "carrying the mutation {value}",
    "insert_tags": "carrying a {value} tag",
    "insert_promoters": "using the {value} promoter",
}
QUERY_FIELDS = tuple(QUERY_TEMPLATES)

QUERY_VARIANTS = ("direct", "requirements")


@dataclass(frozen=True)
class StructuredQuery:
    """One deterministic query derived only from a source row's metadata."""

    query_id: str
    source_index: int
    text: str
    constraints: SurfaceConstraints
    field_names: tuple[str, ...]
    order: int
    variant: str


@dataclass(frozen=True)
class HardNegativePools:
    """Same-backbone candidates partitioned by explicit query evidence."""

    same_backbone: tuple[int, ...]
    alternative_positives: tuple[int, ...]
    known_hard_negatives: tuple[int, ...]
    strict_near_misses: tuple[int, ...]


def render_query(values: Mapping[str, str], variant: str) -> str:
    """Render one controlled query without adding unsupported requirements."""
    if not values:
        raise ValueError("a structured query needs at least one requirement")
    try:
        clauses = [QUERY_TEMPLATES[field].format(value=value) for field, value in values.items()]
    except KeyError as exc:
        raise ValueError(f"unsupported structured-query field: {exc.args[0]}") from exc
    if variant == "direct":
        return "Find a plasmid " + ", ".join(clauses) + "."
    if variant == "requirements":
        return (
            "Retrieve a construct satisfying these requirements: "
            + "; ".join(reversed(clauses))
            + "."
        )
    raise ValueError(f"unknown query variant: {variant}; expected one of {QUERY_VARIANTS}")


def _stable_key(seed: int, epoch: int, source_index: int, value: str) -> bytes:
    """Content-addressed sort key, so shuffles are reproducible without RNG state."""
    payload = f"{seed}\0{epoch}\0{source_index}\0{value}".encode()
    return hashlib.blake2b(payload, digest_size=16).digest()


def build_query_family(
    relevance: RelevanceIndex,
    source_index: int,
    *,
    max_order: int,
    seed: int,
    epoch: int = 0,
    variants: Sequence[str] = ("direct",),
) -> list[StructuredQuery]:
    """Build a reproducible nested query curriculum for one source row.

    Order *k* constrains the first *k* selected fields, so order 1 is broad and
    each successive order narrows the same query.
    """
    if max_order < 1:
        raise ValueError("max_order must be positive")
    if not variants:
        raise ValueError("variants cannot be empty")

    available = sorted(
        (
            field
            for field in QUERY_FIELDS
            if field in relevance.field_values and relevance.field_values[field][source_index]
        ),
        key=lambda field: _stable_key(seed, epoch, source_index, field),
    )[:max_order]
    chosen = {
        field: min(
            relevance.field_values[field][source_index],
            key=lambda value: _stable_key(seed, epoch, source_index, f"{field}:{value}"),
        )
        for field in available
    }

    queries: list[StructuredQuery] = []
    for order in range(1, len(available) + 1):
        values = {field: chosen[field] for field in available[:order]}
        constraints = constraints_from_values(values)
        for variant in variants:
            queries.append(
                StructuredQuery(
                    query_id=f"source-{source_index}-epoch-{epoch}-order-{order}-{variant}",
                    source_index=source_index,
                    text=render_query(values, variant),
                    constraints=constraints,
                    field_names=tuple(values),
                    order=order,
                    variant=variant,
                )
            )
    return queries


class SourceNegatives:
    """Hard-negative pools for one source row, reused across its query orders.

    A source row's query family is a set of nested prefixes, so every order asks
    about the same same-backbone pool and re-asks about fields the shorter orders
    already covered. Computing the pool once and memoizing each field partition
    turns that quadratic re-work into one pass per field.
    """

    def __init__(
        self,
        relevance: RelevanceIndex,
        source_index: int,
        eligible: AbstractSet[int],
    ) -> None:
        if "backbone" not in relevance.field_values:
            raise ValueError("backbone must be indexed to construct same-backbone negatives")
        self._relevance = relevance
        backbone = relevance.field_values["backbone"][source_index]
        self.same_backbone: set[int] = (
            (relevance.candidates_with_field_values("backbone", backbone) & eligible)
            - relevance.exact_candidates(source_index)
            if backbone
            else set()
        )
        self._partitions: dict[tuple[str, frozenset[str]], tuple[set[int], set[int]]] = {}

    def _partition(self, field: str, required: frozenset[str]) -> tuple[set[int], set[int]]:
        key = (field, required)
        if key not in self._partitions:
            matches, mismatches, _ = self._relevance.partition_candidates_by_field(
                field, required, self.same_backbone
            )
            self._partitions[key] = (matches, mismatches)
        return self._partitions[key]

    def pools(self, query: StructuredQuery) -> HardNegativePools:
        """Partition same-backbone peers for one query of this source row."""
        if not self.same_backbone:
            return HardNegativePools((), (), (), ())
        matches_by_field: dict[str, set[int]] = {}
        mismatches_by_field: dict[str, set[int]] = {}
        for field, required in query.constraints.fields:
            matches_by_field[field], mismatches_by_field[field] = self._partition(field, required)
        return _combine(self.same_backbone, matches_by_field, mismatches_by_field)


def same_backbone_hard_negatives(
    relevance: RelevanceIndex,
    query: StructuredQuery,
    eligible_indices: Collection[int],
) -> HardNegativePools:
    """Partition same-backbone peers without calling missing metadata negative.

    ``strict_near_misses`` are the most valuable negatives: peers that match
    every requirement except exactly one.

    Computes the source row's pool from scratch. To evaluate a whole query
    family, build one :class:`SourceNegatives` instead and reuse it.
    """
    eligible = (
        eligible_indices if isinstance(eligible_indices, AbstractSet) else set(eligible_indices)
    )
    return SourceNegatives(relevance, query.source_index, eligible).pools(query)


def _combine(
    same_backbone: set[int],
    matches_by_field: dict[str, set[int]],
    mismatches_by_field: dict[str, set[int]],
) -> HardNegativePools:
    """Fold per-field partitions into positive, negative and near-miss pools."""
    positives = set(same_backbone)
    for matches in matches_by_field.values():
        positives &= matches
    hard_negatives: set[int] = set()
    for mismatches in mismatches_by_field.values():
        hard_negatives |= mismatches

    near_misses: set[int] = set()
    for contradicted, mismatches in mismatches_by_field.items():
        candidates = set(mismatches)
        for field, matches in matches_by_field.items():
            if field != contradicted:
                candidates &= matches
        near_misses |= candidates

    return HardNegativePools(
        same_backbone=tuple(sorted(same_backbone)),
        alternative_positives=tuple(sorted(positives)),
        known_hard_negatives=tuple(sorted(hard_negatives)),
        strict_near_misses=tuple(sorted(near_misses)),
    )
