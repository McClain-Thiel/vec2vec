"""Tests for structured query construction and hard-negative partitioning."""

from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.lib import queries
from vec2vec.lib.relevance import RelevanceIndex

FIELDS = ("backbone", *queries.QUERY_FIELDS)


@pytest.fixture
def index() -> RelevanceIndex:
    frame = pd.DataFrame(
        {
            "sequence_sha256": ["seq-a", "seq-b", "seq-c", "seq-d"],
            "backbone": ["pET28a"] * 4,
            "vector_types": [["Bacterial Expression"]] * 4,
            "bacterial_resistance": ["Ampicillin", "Ampicillin", "Kanamycin", None],
            "insert_species": [[], [], [], []],
            "plasmid_copy": ["High Copy", "Low Copy", "High Copy", "High Copy"],
            "growth_strain": [None] * 4,
            "insert_genes": [["GFP"], ["GFP"], ["GFP"], ["GFP"]],
            "insert_mutations": [[]] * 4,
            "insert_tags": [[]] * 4,
            "insert_promoters": [[]] * 4,
        }
    )
    return RelevanceIndex.from_frame(frame, fields=FIELDS)


def test_render_query_uses_only_supplied_requirements():
    values = {"bacterial_resistance": "ampicillin", "plasmid_copy": "high copy"}
    assert queries.render_query(values) == (
        "Find a plasmid supporting ampicillin bacterial selection, with high copy copy number."
    )


def test_render_query_rejects_unknown_fields_and_empty_requirements():
    with pytest.raises(ValueError, match="unsupported structured-query field"):
        queries.render_query({"nope": "x"})
    with pytest.raises(ValueError, match="at least one requirement"):
        queries.render_query({})


def test_query_family_is_nested_and_reproducible(index):
    first = queries.build_query_family(index, 0, max_order=3, seed=5)
    second = queries.build_query_family(index, 0, max_order=3, seed=5)
    assert [query.query_id for query in first] == [query.query_id for query in second]
    assert [query.order for query in first] == [1, 2, 3]
    # Each order extends the previous one rather than replacing it.
    for earlier, later in zip(first, first[1:], strict=False):
        assert later.field_names[: earlier.order] == earlier.field_names


def test_query_family_changes_with_the_seed(index):
    seeds = {
        tuple(query.field_names)
        for seed in range(8)
        for query in queries.build_query_family(index, 0, max_order=1, seed=seed)
    }
    assert len(seeds) > 1


def test_hard_negatives_require_contradicting_evidence(index):
    query = next(
        query
        for query in queries.build_query_family(index, 0, max_order=5, seed=5)
        if "bacterial_resistance" in query.field_names and query.order == 1
    )
    pools = queries.SourceNegatives(index, query.source_index, {0, 1, 2, 3}).pools(query)
    # Row 2 records a different resistance; row 3 records none and stays unknown.
    assert 2 in pools.known_hard_negatives
    assert 3 not in pools.known_hard_negatives
    assert 1 in pools.alternative_positives


def test_rows_without_a_backbone_yield_no_pools(index):
    query = queries.build_query_family(index, 0, max_order=1, seed=5)[0]
    empty = RelevanceIndex.from_frame(
        pd.DataFrame({"sequence_sha256": ["seq-a"], **{field: [None] for field in FIELDS}}),
        fields=FIELDS,
    )
    assert queries.SourceNegatives(empty, 0, {0}).pools(query) == queries.HardNegativePools(
        (), (), (), ()
    )
