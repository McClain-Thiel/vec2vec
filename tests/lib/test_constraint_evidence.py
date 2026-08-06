"""Tests for rule-derived training evidence and compact benchmark sampling."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vec2vec.lib import constraint_evidence


def evidence_params() -> dict:
    return {
        "evidence_version": "evidence-v1",
        "benchmark_sample_version": "benchmark-v1",
        "input_retrieval_version": "retrieval-v1",
        "training_split": "train",
        "benchmark_split": "val",
        "sampling_key": "benchmark-v1",
        "enabled_sections": {
            "copy_class": ["included"],
            "growth_temperature": ["included"],
            "bacterial_selection": ["included", "reviewed_mappings"],
            "intended_use": ["expression_included", "use_included"],
        },
        "benchmark": {"target_applications": 5, "minimum_per_facet": 1},
    }


def facet_params() -> dict:
    return {
        "copy_class": {
            "rule_id": "copy.v1",
            "facet": "copy_class",
            "relation": "reported_as",
            "included": {"High Copy": ["high"], "Low Copy": ["low"]},
        },
        "growth_temperature": {
            "rule_id": "growth.v2",
            "facet": "growth_temperature",
            "relation": "reported_at",
            "included": {"30": ["30_c"], "37": ["37_c"]},
            "reviewed_mappings": {
                "23": {
                    "canonical_values": ["room_temperature"],
                    "interpretation": "A categorical value, not an exact Celsius measurement.",
                }
            },
        },
        "bacterial_selection": {
            "rule_id": "selection.v2",
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
                    "interpretation": "DAP is a growth requirement and is not mapped.",
                }
            },
        },
        "intended_use": {
            "rule_id": "use.v1",
            "expression_facet": "expression_context",
            "expression_relation": "tagged_for_expression_in",
            "expression_included": {
                "Bacterial Expression": ["bacterial"],
                "Mammalian Expression": ["mammalian"],
            },
            "use_facet": "use_category",
            "use_relation": "tagged_for",
            "use_included": {"CRISPR": ["crispr"]},
        },
    }


@pytest.fixture
def retrieval() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3", "s4"],
            "sequence_sha256": ["a", "b", "c", "d"],
            "addgene_id": [1, 2, 3, 4],
            "url": [f"https://example.org/{value}" for value in range(1, 5)],
            "source_description": ["one", "two", "three", "four"],
            "leakage_component": [10, 20, 30, 40],
            "split_grouped": ["train", "train", "val", "val"],
            "plasmid_copy": ["High Copy", "Low Copy", "High Copy", "Low Copy"],
            "growth_temp": ["37", "23", "30", "37"],
            "bacterial_resistance": [
                "Kan + DAP",
                "Ampicillin",
                "Kanamycin",
                "Ampicillin and Kanamycin",
            ],
            "vector_types": [
                ["Bacterial Expression", "free text"],
                ["Other"],
                ["CRISPR"],
                ["Mammalian Expression"],
            ],
        }
    )


@pytest.fixture
def plannotate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["s1", "s3"],
            "source": ["plannotate", "plannotate"],
            "feature": ["aphA1", "Cas9"],
            "feature_type": ["CDS", "CDS"],
            "start": [10, 20],
            "end": [100, 200],
            "strand": ["+", "+"],
            "confidence": [0.99, 0.98],
        }
    )


def test_builds_noisy_training_labels_and_label_free_validation_sample(retrieval, plannotate):
    training, benchmark, manifest = constraint_evidence.build_constraint_evidence(
        retrieval, plannotate, evidence_params(), facet_params()
    )

    assert set(training["split_grouped"]) == {"train"}
    assert training["training_label_created"].all()
    assert not training["benchmark_label_created"].any()
    assert set(benchmark["split_grouped"]) == {"val"}
    assert not benchmark["benchmark_label_created"].any()
    assert set(benchmark["judge_status"]) == {"not_run"}

    dap = training.loc[training["source_value_json"].eq('"Kan + DAP"')]
    assert set(dap["canonical_value"]) == {"kanamycin"}
    assert not training["canonical_value"].str.contains("dap", case=False).any()
    assert not training["canonical_value"].eq("room_temperature").any()
    assert manifest["source_field_coverage"]["growth_temp"]["unlabeled_units"] == 1
    assert manifest["test_rows_loaded"] == 0
    assert manifest["benchmark_labels_created"] is False

    s3 = benchmark.loc[benchmark["sequence_id"].eq("s3")]
    assert s3["plannotate_feature_count"].eq(1).all()
    assert json.loads(s3.iloc[0]["plannotate_features_json"])[0]["feature"] == "Cas9"


def test_outputs_are_order_invariant(retrieval, plannotate):
    first = constraint_evidence.build_constraint_evidence(
        retrieval, plannotate, evidence_params(), facet_params()
    )
    second = constraint_evidence.build_constraint_evidence(
        retrieval.sample(frac=1, random_state=9).reset_index(drop=True),
        plannotate.sample(frac=1, random_state=4).reset_index(drop=True),
        evidence_params(),
        facet_params(),
    )

    assert list(first[0]["evidence_id"]) == list(second[0]["evidence_id"])
    assert list(first[1]["mapping_application_id"]) == list(second[1]["mapping_application_id"])
    assert first[2]["input_population_sha256"] == second[2]["input_population_sha256"]


def test_rejects_test_rows_and_non_plannotate_annotations(retrieval, plannotate):
    with_test = pd.concat(
        [retrieval, retrieval.iloc[[0]].assign(sequence_id="s5", split_grouped="test")],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="unexpected splits"):
        constraint_evidence.build_constraint_evidence(
            with_test, plannotate, evidence_params(), facet_params()
        )

    mixed = pd.concat([plannotate, plannotate.iloc[[0]].assign(source="plasmidkit")])
    with pytest.raises(ValueError, match="non-pLannotate"):
        constraint_evidence.build_constraint_evidence(
            retrieval, mixed, evidence_params(), facet_params()
        )
