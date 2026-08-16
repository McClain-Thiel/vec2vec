"""Tests for the sparse verified/contradicted constraint-state product."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from vec2vec.lib import constraint_state
from vec2vec.lib.constraint_rules import build_mapping_contract


def evidence_params() -> dict:
    return {
        "enabled_sections": {
            "copy_class": ["included"],
            "growth_temperature": ["included"],
            "bacterial_selection": ["included", "reviewed_mappings"],
            "intended_use": ["expression_included", "use_included"],
        }
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
                    "interpretation": "A category, not an exact Celsius measurement.",
                }
            },
        },
        "bacterial_selection": {
            "rule_id": "selection.v2",
            "facet": "selection_marker",
            "relation": "reported_selection_includes",
            "included": {
                "Ampicillin": ["ampicillin"],
                "Kanamycin": ["kanamycin"],
                "Ampicillin and Kanamycin": ["ampicillin", "kanamycin"],
            },
            "reviewed_mappings": {
                "Kan + DAP": {
                    "canonical_values": ["kanamycin"],
                    "interpretation": "DAP is not an antibiotic marker.",
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


def state_params(
    *,
    expected_contract: str,
    expected_population: str,
    expected_state_input: str,
) -> dict:
    return {
        "state_version": "state-v1",
        "input_retrieval_version": "retrieval-v1",
        "expected_input_population_sha256": expected_population,
        "expected_state_input_sha256": expected_state_input,
        "expected_rule_contract_sha256": expected_contract,
        "allowed_splits": ["train", "val", "test"],
        "training_split": "train",
        "conflict_groups": [
            {
                "name": "copy_high_vs_low",
                "conflict_rule_id": "copy_conflict.v1",
                "facet": "copy_class",
                "relation": "reported_as",
                "rule_id": "copy.v1",
                "values": ["high", "low"],
            },
            {
                "name": "growth_30_vs_37",
                "conflict_rule_id": "growth_conflict.v1",
                "facet": "growth_temperature",
                "relation": "reported_at",
                "rule_id": "growth.v2",
                "values": ["30_c", "37_c"],
            },
        ],
    }


@pytest.fixture
def retrieval() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3", "s4"],
            "sequence_sha256": ["h1", "h2", "h3", "h4"],
            "leakage_component": ["c1", "c2", "c3", "c4"],
            "split_grouped": ["train", "train", "val", "test"],
            "plasmid_copy": ["High Copy", "Low Copy", "Unknown", "High Copy"],
            "growth_temp": ["37", "30", "23", "30"],
            "bacterial_resistance": [
                "Ampicillin",
                "Kanamycin",
                "Ampicillin",
                "Ampicillin and Kanamycin",
            ],
            "vector_types": [
                ["Bacterial Expression"],
                ["CRISPR"],
                [],
                ["Mammalian Expression"],
            ],
        }
    )


def build(retrieval: pd.DataFrame):
    evidence = evidence_params()
    facets = facet_params()
    _, contract_hash = build_mapping_contract(facets, evidence["enabled_sections"])
    population_hash = constraint_state.retrieval_population_sha256(retrieval)
    state_input_hash = constraint_state.retrieval_state_input_sha256(retrieval)
    return constraint_state.build_constraint_state_tables(
        retrieval,
        state_params(
            expected_contract=contract_hash,
            expected_population=population_hash,
            expected_state_input=state_input_hash,
        ),
        evidence,
        facets,
    )


def test_builds_verified_conflicted_and_implicit_unknown_states(retrieval):
    vocabulary, states, manifest = build(retrieval)

    by_state = states.set_index(["sequence_id", "facet", "canonical_value"])["state"]
    assert by_state["s1", "copy_class", "high"] == "verified"
    assert by_state["s1", "copy_class", "low"] == "contradicted"
    assert by_state["s1", "growth_temperature", "37_c"] == "verified"
    assert by_state["s1", "growth_temperature", "30_c"] == "contradicted"
    assert by_state["s1", "selection_marker", "ampicillin"] == "verified"

    # Selection markers are positive-only. A recorded ampicillin value does not
    # contradict kanamycin, and the disabled room-temperature value stays unknown.
    assert ("s1", "selection_marker", "kanamycin") not in by_state.index
    assert (
        not states.loc[states["sequence_id"].eq("s3"), "facet"]
        .isin({"copy_class", "growth_temperature"})
        .any()
    )
    assert "test" in set(states["split_grouped"])

    high = vocabulary.loc[
        vocabulary["facet"].eq("copy_class") & vocabulary["canonical_value"].eq("high")
    ].iloc[0]
    assert high["train_row_support"] == 1
    assert high["total_verified_row_support"] == 2
    assert high["train_contradicted_row_support"] == 1
    assert high["has_reviewed_conflict_rule"]
    assert manifest["unknown_policy"] == "absence_from_sparse_table"
    assert manifest["pair_state_conflicts"] == 0
    assert manifest["test_metadata_used_for_rule_selection"] is False

    contradiction = states.loc[
        states["sequence_id"].eq("s1")
        & states["facet"].eq("copy_class")
        & states["canonical_value"].eq("low")
    ].iloc[0]
    evidence_record = json.loads(contradiction["evidence_json"])[0]
    assert evidence_record["evidence_type"] == "reviewed_conflict_rule"
    assert evidence_record["source_value_json"] == '"High Copy"'


def test_outputs_are_content_addressed_and_order_invariant(retrieval):
    first = build(retrieval)
    second = build(retrieval.sample(frac=1, random_state=7).reset_index(drop=True))

    assert list(first[0]["constraint_id"]) == list(second[0]["constraint_id"])
    assert list(first[1]["state_id"]) == list(second[1]["state_id"])
    assert list(first[1]["evidence_json"]) == list(second[1]["evidence_json"])
    assert first[2]["input_population_sha256"] == second[2]["input_population_sha256"]


def test_rejects_contract_drift_and_unmapped_conflicts(retrieval):
    evidence = evidence_params()
    facets = facet_params()
    _, contract_hash = build_mapping_contract(facets, evidence["enabled_sections"])
    population_hash = constraint_state.retrieval_population_sha256(retrieval)
    state_input_hash = constraint_state.retrieval_state_input_sha256(retrieval)

    with pytest.raises(ValueError, match="accepted rule contract changed"):
        constraint_state.build_constraint_state_tables(
            retrieval,
            state_params(
                expected_contract="wrong",
                expected_population=population_hash,
                expected_state_input=state_input_hash,
            ),
            evidence,
            facets,
        )

    with pytest.raises(ValueError, match="retrieval population changed"):
        constraint_state.build_constraint_state_tables(
            retrieval,
            state_params(
                expected_contract=contract_hash,
                expected_population="wrong",
                expected_state_input=state_input_hash,
            ),
            evidence,
            facets,
        )

    changed_source = retrieval.copy()
    changed_source.loc[0, "plasmid_copy"] = "Low Copy"
    with pytest.raises(ValueError, match="constraint source data changed"):
        constraint_state.build_constraint_state_tables(
            changed_source,
            state_params(
                expected_contract=contract_hash,
                expected_population=population_hash,
                expected_state_input=state_input_hash,
            ),
            evidence,
            facets,
        )

    params = state_params(
        expected_contract=contract_hash,
        expected_population=population_hash,
        expected_state_input=state_input_hash,
    )
    params["conflict_groups"][0]["values"] = ["high", "medium"]
    with pytest.raises(ValueError, match="unmapped constraints"):
        constraint_state.build_constraint_state_tables(retrieval, params, evidence, facets)
