"""Tests for E00 field, split, and pLannotate profiles."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vec2vec.lib import constraint_semantics


@pytest.fixture
def retrieval() -> pd.DataFrame:
    """Small retrieval frame with known, unknown, repeated, and multi-valued fields."""
    return pd.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3", "s4"],
            "sequence_sha256": ["a", "b", "c", "d"],
            "family_key": ["backbone::x", "backbone::x", "id::s3", "id::s4"],
            "leakage_component": [10, 10, 20, 30],
            "split_grouped": ["train", "train", "val", "test"],
            "length_bp": [100, 100, 80, 120],
            "plasmid_copy": ["High Copy", "High Copy", "Low Copy", None],
            "vector_types": [
                ["Bacterial Expression", "CRISPR"],
                ["Bacterial Expression"],
                [],
                None,
            ],
        }
    )


def test_constraint_profile_keeps_unknowns_and_split_support(retrieval):
    fields, values = constraint_semantics.profile_constraint_fields(
        retrieval,
        fields=("plasmid_copy", "vector_types"),
        split_labels=("train", "val", "test"),
        minimum_rows=2,
        minimum_components=1,
    )

    copy = fields.set_index("field").loc["plasmid_copy"]
    assert copy["known_rows"] == 3
    assert copy["unknown_rows"] == 1
    assert copy["normalized_value_count"] == 2
    assert copy["values_meeting_train_support"] == 1

    high = values.set_index(["field", "normalized_value"]).loc[("plasmid_copy", "high copy")]
    assert high["row_support"] == 2
    assert high["component_support"] == 1
    assert high["train_row_support"] == 2
    assert high["test_row_support"] == 0
    assert json.loads(high["raw_variants_json"]) == [{"raw_cell_json": '"High Copy"', "rows": 2}]


def test_constraint_profile_rejects_duplicate_identity_and_unknown_splits(retrieval):
    duplicate = pd.concat([retrieval, retrieval.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate sequence_id"):
        constraint_semantics.profile_constraint_fields(
            duplicate,
            fields=("plasmid_copy",),
            split_labels=("train", "val", "test"),
            minimum_rows=1,
            minimum_components=1,
        )

    unexpected = retrieval.assign(split_grouped=["train", "train", "val", "holdout"])
    with pytest.raises(ValueError, match="unexpected splits"):
        constraint_semantics.profile_constraint_fields(
            unexpected,
            fields=("plasmid_copy",),
            split_labels=("train", "val", "test"),
            minimum_rows=1,
            minimum_components=1,
        )


def test_component_profile_reports_concentration_without_changing_split(retrieval):
    components, summary = constraint_semantics.profile_split_components(retrieval)

    train = summary["by_split"]["train"]
    assert train == {
        "rows": 2,
        "components": 1,
        "singleton_components": 0,
        "largest_component_rows": 2,
        "largest_component_row_fraction": 1.0,
        "ten_largest_component_rows": 2,
        "ten_largest_component_row_fraction": 1.0,
    }
    assert summary["components_crossing_grouped_split"] == 0
    assert summary["family_key_source_rows"] == {"backbone": 2, "id": 2}
    assert components["rows"].sum() == len(retrieval)


def test_component_profile_rejects_an_impure_component(retrieval):
    impure = retrieval.copy()
    impure.loc[1, "split_grouped"] = "test"
    with pytest.raises(ValueError, match="1 leakage components cross"):
        constraint_semantics.profile_split_components(impure)


def test_plannotate_profile_reports_coverage_raw_coordinates_and_missing_provenance(retrieval):
    annotations = pd.DataFrame(
        {
            "sequence_id": ["s1", "s1", "outside"],
            "source": ["plannotate"] * 3,
            "feature": ["AmpR", "ori", "other"],
            "feature_type": ["CDS", "rep_origin", "CDS"],
            "start": [90, 0, 1],
            "end": [10, 101, 5],
            "strand": ["-", "+", "+"],
            "confidence": [0.99, 1.0, 0.5],
        }
    )
    result = constraint_semantics.profile_plannotate(
        retrieval,
        annotations,
        expected_source="plannotate",
        provenance={
            "software_version": None,
            "database_version": None,
            "circular_setting": None,
            "coordinate_convention": "source_qstart_qend_preserved_uninterpreted",
        },
    )

    assert result["annotation_rows_in_retrieval"] == 2
    assert result["annotation_rows_outside_retrieval"] == 1
    assert result["retrieval_sequences_with_annotations"] == 1
    assert result["retrieval_sequences_without_annotations"] == 3
    assert result["raw_coordinate_checks"]["start_greater_than_end"] == 1
    assert result["raw_coordinate_checks"]["end_greater_than_sequence_length"] == 1
    assert result["provenance_complete"] is False
    assert result["missing_provenance_fields"] == [
        "software_version",
        "database_version",
        "circular_setting",
    ]


def test_plannotate_profile_rejects_a_mixed_source(retrieval):
    annotations = pd.DataFrame(
        {
            "sequence_id": ["s1", "s2"],
            "source": ["plannotate", "plasmidkit"],
            "feature": ["AmpR", "GFP"],
            "feature_type": ["CDS", "CDS"],
            "start": [1, 1],
            "end": [10, 10],
            "strand": ["+", "+"],
            "confidence": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="expected only annotation source"):
        constraint_semantics.profile_plannotate(
            retrieval,
            annotations,
            expected_source="plannotate",
            provenance={},
        )


def test_plannotate_profile_matches_string_equivalent_sequence_ids(retrieval):
    retrieval = retrieval.copy()
    retrieval["sequence_id"] = [1, 2, 3, 4]
    annotations = pd.DataFrame(
        {
            "sequence_id": ["1"],
            "source": ["plannotate"],
            "feature": ["AmpR"],
            "feature_type": ["CDS"],
            "start": [1],
            "end": [10],
            "strand": ["+"],
            "confidence": [1.0],
        }
    )

    profile = constraint_semantics.profile_plannotate(
        retrieval,
        annotations,
        expected_source="plannotate",
        provenance={},
    )

    assert profile["annotation_rows_in_retrieval"] == 1
    assert profile["retrieval_sequences_with_annotations"] == 1
