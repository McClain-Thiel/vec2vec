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
            "description": ["ignored"] * 4,
            "sequence": ["ACGT", "TTTT", "GGGG", "CCCC"],
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
    return RelevanceIndex.from_frame(frame, fields=FIELDS, min_constraint_groups=1)


def test_render_query_uses_only_supplied_requirements():
    values = {"bacterial_resistance": "ampicillin", "plasmid_copy": "high copy"}
    direct = queries.render_query(values, "direct")
    assert direct == (
        "Find a plasmid supporting ampicillin bacterial selection, with high copy copy number."
    )
    assert queries.render_query(values, "requirements").startswith("Retrieve a construct")


def test_render_query_rejects_unknown_fields_and_variants():
    with pytest.raises(ValueError, match="unsupported structured-query field"):
        queries.render_query({"nope": "x"}, "direct")
    with pytest.raises(ValueError, match="unknown query variant"):
        queries.render_query({"plasmid_copy": "high copy"}, "poetic")
    with pytest.raises(ValueError, match="at least one requirement"):
        queries.render_query({}, "direct")


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
    pools = queries.same_backbone_hard_negatives(index, query, {0, 1, 2, 3})
    # Row 2 records a different resistance; row 3 records none and stays unknown.
    assert 2 in pools.known_hard_negatives
    assert 3 not in pools.known_hard_negatives
    assert 1 in pools.alternative_positives


def test_sampling_prefers_strict_near_misses():
    pools = queries.HardNegativePools(
        same_backbone=(1, 2, 3),
        alternative_positives=(),
        known_hard_negatives=(1, 2, 3),
        strict_near_misses=(3,),
    )
    assert queries.sample_hard_negatives(pools, count=1, seed=1, epoch=0, source_index=0) == (3,)
    assert len(queries.sample_hard_negatives(pools, count=3, seed=1, epoch=0, source_index=0)) == 3


def test_rows_without_a_backbone_yield_no_pools(index):
    query = queries.build_query_family(index, 0, max_order=1, seed=5)[0]
    empty = RelevanceIndex.from_frame(
        pd.DataFrame(
            {
                "description": ["x"],
                "sequence": ["ACGT"],
                **{field: [None] for field in FIELDS},
            }
        ),
        fields=FIELDS,
        min_constraint_groups=1,
    )
    assert queries.same_backbone_hard_negatives(empty, query, {0}) == queries.HardNegativePools(
        (), (), (), ()
    )
