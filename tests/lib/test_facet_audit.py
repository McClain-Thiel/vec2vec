"""Tests for deterministic, label-free facet-audit sampling."""

from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.lib import facet_audit


def audit_params() -> dict:
    """Small protocol that preserves the structure of the frozen base configuration."""
    return {
        "audit_version": "audit-v2",
        "sampling_key": "audit-v2",
        "input_retrieval_version": "fixture-v1",
        "eligible_splits": ["train", "val"],
        "second_review_modulus": 5,
        "copy_class": {
            "rule_id": "copy.v1",
            "facet": "addgene_copy_class",
            "relation": "reported_as",
            "included": {"High Copy": ["high"], "Low Copy": ["low"]},
            "missing_exact": ["Unknown"],
            "target_per_value": 2,
            "missing_target": 2,
        },
        "growth_temperature": {
            "rule_id": "temperature.v2",
            "facet": "addgene_growth_temperature",
            "relation": "reported_for_propagation_at",
            "included": {"30": ["30_c"], "37": ["37_c"]},
            "reviewed_mappings": {
                "23": {
                    "canonical_values": ["room_temperature"],
                    "interpretation": "A category, not an exact Celsius measurement.",
                }
            },
            "held_out": {},
            "missing_exact": [],
            "target_per_value": 2,
            "sample_all_held_out": True,
            "sample_all_missing": True,
        },
        "bacterial_selection": {
            "rule_id": "resistance.v2",
            "facet": "bacterial_selection_marker",
            "relation": "reported_selection_includes",
            "included": {
                "Ampicillin": ["ampicillin"],
                "Kanamycin": ["kanamycin"],
                "Ampicillin and Kanamycin": ["ampicillin", "kanamycin"],
            },
            "reviewed_mappings": {
                "Kan + DAP": {
                    "canonical_values": ["kanamycin"],
                    "interpretation": (
                        "Kan means kanamycin. DAP is a growth requirement and is not mapped."
                    ),
                }
            },
            "excluded": {},
            "missing_exact": ["None"],
            "included_single_target": 2,
            "included_combination_target": 2,
            "minimum_per_canonical": 1,
            "excluded_target": 2,
            "sample_all_missing": True,
        },
        "intended_use": {
            "rule_id": "use.v1",
            "expression_facet": "addgene_expression_context",
            "expression_relation": "tagged_for_expression_in",
            "expression_included": {
                "Bacterial Expression": ["bacterial"],
                "Mammalian Expression": ["mammalian"],
            },
            "use_facet": "addgene_use_category",
            "use_relation": "tagged_for",
            "use_included": {"CRISPR": ["crispr"], "AAV": ["aav"]},
            "excluded_exact": {"Other": "No free-text category."},
            "missing_exact": ["N/A"],
            "default_exclusion_reason": "Free text is excluded.",
            "expression_target": 2,
            "use_target": 2,
            "minimum_per_canonical": 1,
            "excluded_target": 2,
            "missing_target": 2,
        },
    }


@pytest.fixture
def retrieval() -> pd.DataFrame:
    """Rows cover duplicate components, held-out values, exclusions, missingness, and test."""
    return pd.DataFrame(
        {
            "sequence_id": [f"s{index}" for index in range(1, 8)],
            "sequence_sha256": [f"hash{index}" for index in range(1, 8)],
            "addgene_id": list(range(1, 8)),
            "url": [f"https://www.addgene.org/{index}/" for index in range(1, 8)],
            "description": [f"generated {index}" for index in range(1, 8)],
            "source_description": [f"source {index}" for index in range(1, 8)],
            "leakage_component": [10, 10, 20, 30, 40, 50, 60],
            "split_grouped": ["train", "train", "val", "val", "test", "train", "val"],
            "plasmid_copy": [
                "High Copy",
                "High Copy",
                "Low Copy",
                "Unknown",
                "Low Copy",
                "High Copy",
                "Low Copy",
            ],
            "growth_temp": ["37", "37", "30", "23", "30", "37", None],
            "bacterial_resistance": [
                "Ampicillin",
                "Ampicillin",
                "Kanamycin",
                "Kan + DAP",
                "Kanamycin",
                "Ampicillin and Kanamycin",
                "None",
            ],
            "vector_types": [
                ["Bacterial Expression"],
                ["Bacterial Expression"],
                ["CRISPR"],
                ["Other", "Gateway destination vector"],
                ["CRISPR"],
                ["Mammalian Expression", "AAV", "free note"],
                ["N/A"],
            ],
        }
    )


def test_audit_sample_is_order_invariant_component_aware_and_test_sealed(retrieval):
    sample, vocabulary, manifest = facet_audit.build_facet_audit_sample(retrieval, audit_params())
    shuffled, shuffled_vocabulary, shuffled_manifest = facet_audit.build_facet_audit_sample(
        retrieval.sample(frac=1, random_state=4).reset_index(drop=True), audit_params()
    )

    comparison_columns = [
        "audit_row_id",
        "selection_hash",
        "stratum",
        "sequence_id",
        "source_value_json",
    ]
    pd.testing.assert_frame_equal(sample[comparison_columns], shuffled[comparison_columns])
    pd.testing.assert_frame_equal(vocabulary, shuffled_vocabulary)
    assert manifest["input_population_sha256"] == shuffled_manifest["input_population_sha256"]

    assert set(sample["split_grouped"]) <= {"train", "val"}
    assert not sample["sequence_id"].eq("s5").any()
    assert sample.groupby(["stratum", "leakage_component"]).size().max() == 1
    assert manifest["accepted_labels_created"] is False
    assert manifest["test_metadata_used_for_sampling"] is False


def test_audit_sample_preserves_reviewed_mappings_and_review_fields(retrieval):
    sample, vocabulary, manifest = facet_audit.build_facet_audit_sample(retrieval, audit_params())

    room_temperature = sample.loc[sample["source_value_json"] == '"23"'].iloc[0]
    assert room_temperature["stratum"] == "growth_temperature:reviewed_exact_mapping"
    assert room_temperature["proposed_evidence_state"] == "verified"
    assert room_temperature["canonical_values_json"] == '["room_temperature"]'
    assert "not an exact Celsius" in room_temperature["mapping_note"]

    compound = sample.loc[sample["source_value_json"] == '"Ampicillin and Kanamycin"'].iloc[0]
    assert compound["canonical_values_json"] == '["ampicillin","kanamycin"]'
    assert compound["generated_description"] == "generated 6"
    assert compound["source_description"] == "source 6"

    reviewed = vocabulary.loc[
        (vocabulary["source_field"] == "bacterial_resistance")
        & (vocabulary["exact_key"] == "kan + dap")
    ].iloc[0]
    assert reviewed["mapping_status"] == "proposed_include"
    assert reviewed["canonical_values_json"] == '["kanamycin"]'
    assert "Kan means kanamycin" in reviewed["mapping_note"]
    placeholders = vocabulary.loc[vocabulary["exact_key"].isin(["unknown", "none", "n/a"])]
    assert set(placeholders["mapping_status"]) == {"missing"}
    assert manifest["strata"]["growth_temperature:reviewed_exact_mapping"]["sample_all"] is True


def test_audit_sample_rejects_unclassified_closed_vocabulary(retrieval):
    changed = retrieval.copy()
    changed.loc[0, "bacterial_resistance"] = "Ampicillin + 1% Glucose"
    with pytest.raises(ValueError, match="unclassified bacterial_resistance"):
        facet_audit.build_facet_audit_sample(changed, audit_params())


def test_audit_sample_rejects_test_as_an_eligible_split(retrieval):
    params = audit_params()
    params["eligible_splits"] = ["train", "val", "test"]
    with pytest.raises(ValueError, match="test cannot be an eligible"):
        facet_audit.build_facet_audit_sample(retrieval, params)
