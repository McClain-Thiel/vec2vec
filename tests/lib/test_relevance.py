"""Tests for constraint extraction and the relevance index."""

from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.lib import relevance
from vec2vec.lib.text import normalize_values


def constraints(description: str, **fields):
    return relevance.extract_surface_constraints(
        description,
        {name: normalize_values(value) for name, value in fields.items()},
        frozenset(),
        None,
    )


def test_only_values_present_in_the_text_become_constraints():
    surfaced = constraints(
        "An ampicillin-resistant bacterial expression vector.",
        bacterial_resistance="Ampicillin",
        plasmid_copy="High Copy",
    )
    assert dict(surfaced.fields) == {"bacterial_resistance": frozenset({"ampicillin"})}
    assert surfaced.group_count == 1


def test_matching_is_whole_phrase_not_substring():
    assert constraints("carries KanR", bacterial_resistance="Kan").fields == ()
    assert constraints("carries Kan resistance", bacterial_resistance="Kan").fields != ()


def test_length_is_surfaced_only_when_the_stated_size_agrees():
    fields = {"vector_types": frozenset()}
    assert (
        relevance.extract_surface_constraints(
            "A 5.0 kb plasmid.", fields, frozenset(), 5000
        ).length_bp
        == 5000
    )
    assert (
        relevance.extract_surface_constraints(
            "A 9000 bp plasmid.", fields, frozenset(), 5000
        ).length_bp
        is None
    )


@pytest.fixture
def index() -> relevance.RelevanceIndex:
    frame = pd.DataFrame(
        {
            "description": [
                "An ampicillin high copy vector.",
                "An ampicillin high copy vector.",
                "A kanamycin low copy vector.",
                "A vector.",
            ],
            "sequence": ["ACGT", "TTTT", "GGGG", "ACGT"],
            "bacterial_resistance": ["Ampicillin", "Ampicillin", "Kanamycin", None],
            "plasmid_copy": ["High Copy", "High Copy", "Low Copy", "High Copy"],
            "vector_types": [["Bacterial Expression"]] * 4,
            "insert_species": [[], [], [], []],
            "growth_strain": [None] * 4,
            "growth_temp": [None] * 4,
            "length_bp": [4, 4, 4, 4],
        }
    )
    return relevance.RelevanceIndex.from_frame(frame)


def test_structurally_identical_rows_are_mutual_positives(index):
    assert index.positive_candidates(0) == {0, 1, 3}  # 3 shares row 0's exact sequence
    assert 2 not in index.positive_candidates(0)


def test_a_vague_query_admits_only_its_exact_sequence_family(index):
    # Row 3's description surfaces nothing, so it falls below min_constraint_groups.
    assert index.positive_candidates(3) == {0, 3}


def test_judgement_separates_contradiction_from_missing_metadata(index):
    contradicted = index.judge_candidate(0, 2)
    assert not contradicted.acceptable
    assert set(contradicted.contradicted_groups) == {"bacterial_resistance", "plasmid_copy"}

    unknown = index.judge_candidate(2, 3)
    assert "bacterial_resistance" in unknown.unknown_groups
    assert not unknown.acceptable


def test_exact_sequence_matches_are_always_acceptable(index):
    assert index.judge_candidate(3, 0).exact is True
    assert index.judge_candidate(3, 0).acceptable is True


def test_partition_splits_candidates_three_ways(index):
    matches, contradictions, unknowns = index.partition_candidates_by_field(
        "bacterial_resistance", {"ampicillin"}, {0, 1, 2, 3}
    )
    assert (matches, contradictions, unknowns) == ({0, 1}, {2}, {3})


def test_missing_columns_are_rejected_up_front():
    with pytest.raises(ValueError, match="missing relevance columns"):
        relevance.RelevanceIndex.from_frame(
            pd.DataFrame({"description": ["x"], "sequence": ["ACGT"]})
        )
