"""Tests for constraint extraction and the metadata index."""

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


def test_annotation_features_and_length_count_as_their_own_groups():
    surfaced = relevance.extract_surface_constraints(
        "A 5.0 kb ampicillin plasmid carrying AmpR.",
        {"bacterial_resistance": normalize_values("Ampicillin")},
        normalize_values(["AmpR", "GFP"]),
        5000,
    )
    assert surfaced.features == frozenset({"ampr"})
    assert surfaced.length_bp == 5000
    assert surfaced.group_count == 3


def test_length_is_surfaced_only_when_the_stated_size_agrees():
    fields = {"vector_types": frozenset()}
    assert (
        relevance.extract_surface_constraints(
            "A 9000 bp plasmid.", fields, frozenset(), 5000
        ).length_bp
        is None
    )


def test_constraints_serialize_order_stably():
    surfaced = constraints("An ampicillin plasmid.", bacterial_resistance="Ampicillin")
    assert surfaced.to_dict() == {
        "fields": {"bacterial_resistance": ["ampicillin"]},
        "features": [],
        "length_bp": None,
        "group_count": 1,
    }


@pytest.fixture
def index() -> relevance.RelevanceIndex:
    frame = pd.DataFrame(
        {
            # Rows 0 and 3 share a sequence; row 3 records no resistance.
            "sequence_sha256": ["seq-a", "seq-b", "seq-c", "seq-a"],
            "bacterial_resistance": ["Ampicillin", "Ampicillin", "Kanamycin", None],
            "plasmid_copy": ["High Copy", "High Copy", "Low Copy", "High Copy"],
            "vector_types": [["Bacterial Expression"]] * 4,
            "insert_species": [[], [], [], []],
            "growth_strain": [None] * 4,
            "growth_temp": [None] * 4,
        }
    )
    return relevance.RelevanceIndex.from_frame(frame)


def test_exact_candidates_group_identical_sequences(index):
    assert index.exact_candidates(0) == {0, 3}
    assert index.exact_candidates(1) == {1}


def test_candidates_with_field_values_intersects_requirements(index):
    assert index.candidates_with_field_values("bacterial_resistance", {"ampicillin"}) == {0, 1}
    assert index.candidates_with_field_values("plasmid_copy", {"high copy"}) == {0, 1, 3}
    assert index.candidates_with_field_values("bacterial_resistance", {"nonesuch"}) == set()


def test_partition_separates_contradiction_from_missing_metadata(index):
    matches, contradictions, unknowns = index.partition_candidates_by_field(
        "bacterial_resistance", {"ampicillin"}, {0, 1, 2, 3}
    )
    # Row 2 records a different resistance; row 3 records none, so it is unknown.
    assert (matches, contradictions, unknowns) == ({0, 1}, {2}, {3})


def test_placeholder_metadata_never_counts_as_known(index):
    frame = pd.DataFrame(
        {
            "sequence_sha256": ["seq-a"],
            "bacterial_resistance": ["Unknown"],
            "plasmid_copy": ["n/a"],
            "vector_types": [[]],
            "insert_species": [[]],
            "growth_strain": [None],
            "growth_temp": [None],
        }
    )
    single = relevance.RelevanceIndex.from_frame(frame)
    _, contradictions, unknowns = single.partition_candidates_by_field(
        "bacterial_resistance", {"ampicillin"}, {0}
    )
    assert (contradictions, unknowns) == (set(), {0})


def test_missing_columns_are_rejected_up_front():
    with pytest.raises(ValueError, match="missing relevance columns"):
        relevance.RelevanceIndex.from_frame(pd.DataFrame({"sequence_sha256": ["a"]}))


def test_ragged_columns_and_bad_fields_are_rejected():
    with pytest.raises(ValueError, match="expected 2"):
        relevance.RelevanceIndex(["a", "b"], {"backbone": ["x"]}, fields=["backbone"])
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        relevance.RelevanceIndex(["a"], {"backbone": ["x"]}, fields=["backbone", "backbone"])


@pytest.mark.parametrize("identity", [None, 1, ""])
def test_missing_or_non_string_sequence_hashes_are_rejected(identity):
    with pytest.raises(ValueError, match="sequence_hashes must contain non-empty strings"):
        relevance.RelevanceIndex([identity], {"backbone": ["x"]}, fields=["backbone"])


def test_unindexed_fields_and_empty_requirements_are_rejected(index):
    with pytest.raises(ValueError, match="not indexed"):
        index.candidates_with_field_values("backbone", {"x"})
    with pytest.raises(ValueError, match="cannot be empty"):
        index.candidates_with_field_values("bacterial_resistance", set())
