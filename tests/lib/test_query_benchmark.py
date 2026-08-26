"""Tests for the frozen query catalog, galleries, labels, and controls."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from vec2vec.lib import (
    query_benchmark,
    similarity_graph,
    similarity_split,
)


def _fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    dict,
    dict,
]:
    sequence_ids = [f"s{index:02d}" for index in range(12)]
    split_labels = ["train"] * 6 + ["val"] * 3 + ["test"] * 3
    retrieval = pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "sequence_sha256": [f"h{index:02d}" for index in range(12)],
            "family_key": [f"f{index:02d}" for index in range(12)],
        }
    )
    mapping = pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "similarity_component_primary": [f"p{index:02d}" for index in range(12)],
            "leakage_component_v2": [f"p{index:02d}" for index in range(12)],
            "split_grouped_v2": split_labels,
        }
    )
    graph_edges = pd.DataFrame(columns=["sequence_a", "sequence_b", "primary_near_duplicate"])
    vocabulary = pd.DataFrame(
        {
            "constraint_id": ["a", "b", "c"],
            "facet": ["facet_a", "facet_b", "facet_c"],
            "relation": ["has", "has", "has"],
            "canonical_value": ["alpha", "beta", "gamma"],
            "rule_id": ["rule_a.v1", "rule_b.v1", "rule_c.v1"],
            "rule_version": ["v1", "v1", "v1"],
            "has_reviewed_conflict_rule": [True, True, False],
        }
    )
    state_values = {
        "a": {
            "verified": ["s00", "s01", "s02", "s06", "s07", "s09", "s10"],
            "contradicted": ["s03", "s04", "s08", "s11"],
        },
        "b": {
            "verified": ["s00", "s01", "s03", "s06", "s08", "s09", "s11"],
            "contradicted": ["s02", "s04", "s07", "s10"],
        },
        "c": {
            "verified": ["s00", "s01", "s05", "s06", "s09"],
            "contradicted": [],
        },
    }
    state_records = []
    for constraint_id, by_state in state_values.items():
        for state, members in by_state.items():
            state_records.extend(
                {
                    "sequence_id": sequence_id,
                    "constraint_id": constraint_id,
                    "state": state,
                }
                for sequence_id in members
            )
    states = pd.DataFrame.from_records(state_records)
    edge_hash = similarity_graph.dataframe_content_sha256(
        graph_edges, sort_columns=["sequence_a", "sequence_b"]
    )
    mapping_hash = similarity_graph.dataframe_content_sha256(
        mapping.loc[:, list(similarity_split.MAPPING_COLUMNS)],
        sort_columns=["sequence_id"],
    )
    graph_manifest = {
        "output_content_hashes": {"edges_sha256": edge_hash},
        "decision": {"edge_enumeration_complete_under_configured_caps": True},
    }
    split_manifest = {
        "input_graph_artifact_version": "graph-v1",
        "build": {"mapping_sha256": mapping_hash},
        "decision": {"status": "accepted_strict_similarity_closed_split"},
    }
    state_manifest = {
        "input_population_sha256": "population-hash",
        "pair_state_conflicts": 0,
        "output_content_hashes": {
            "vocabulary_sha256": similarity_graph.dataframe_content_sha256(
                vocabulary, sort_columns=["constraint_id"]
            ),
            "states_sha256": similarity_graph.dataframe_content_sha256(
                states, sort_columns=["sequence_id", "constraint_id", "state"]
            ),
        },
    }
    params = {
        "benchmark_version": "benchmark-v1",
        "canonical_text_revision": "text-v1",
        "expected_input_population_sha256": "population-hash",
        "expected_constraint_artifact_hashes": dict(state_manifest["output_content_hashes"]),
        "minimum_atomic_train_verified_rows": 2,
        "minimum_atomic_train_verified_components": 2,
        "minimum_pair_train_verified_rows": 2,
        "minimum_pair_train_verified_components": 2,
        "minimum_train_contradicted_rows": 2,
        "minimum_train_contradicted_components": 2,
        "minimum_gallery_verified_rows_for_measurement": 1,
        "minimum_gallery_verified_components_for_measurement": 1,
        "minimum_gallery_contradicted_rows_for_control": 1,
        "minimum_gallery_contradicted_components_for_control": 1,
        "minimum_usable_atomic_queries_each_closed_eval": 1,
        "minimum_usable_pair_queries_each_closed_eval": 1,
        "minimum_usable_pair_contradiction_controls_each_closed_eval": 1,
        "evaluation_splits": ["val", "test"],
        "top_k": [1, 2],
        "random_seed": 42,
    }
    return (
        retrieval,
        mapping,
        graph_edges,
        graph_manifest,
        split_manifest,
        vocabulary,
        states,
        state_manifest,
        params,
    )


def _build():
    return query_benchmark.build_query_benchmark(*_fixture())


def test_builds_disjoint_sets_normalized_masses_and_expected_controls():
    catalog, galleries, query_states, masses, rankings, metrics, manifest = _build()

    assert set(catalog["query_kind"]) == {"atomic", "pair_conjunction"}
    assert catalog["query_id"].is_unique
    assert not query_states.duplicated(["semantic_query_id", "sequence_id"]).any()
    assert set(galleries["gallery_kind"]) == {"closed_grouped_v2", "open_all"}
    mass_sums = (
        masses.assign(mass=masses["log_base_mass"].map(__import__("math").exp))
        .groupby(["gallery_id", "base_measure"])["mass"]
        .sum()
    )
    assert all(value == pytest.approx(1.0, abs=1e-12) for value in mass_sums)

    enough_valid = metrics.loc[
        metrics["control"].eq("verified_first_oracle")
        & metrics["query_id"].map(catalog.set_index("query_id")["answer_set_size"]).ge(metrics["k"])
    ]
    assert enough_valid["verified_at_k"].eq(1.0).all()
    enough_negative = metrics.loc[
        metrics["control"].eq("contradiction_first")
        & metrics["query_id"]
        .map(catalog.set_index("query_id")["contradiction_set_size"])
        .ge(metrics["k"])
    ]
    assert enough_negative["contradicted_at_k"].eq(1.0).all()
    assert not rankings.duplicated(["query_id", "control", "sequence_id"]).any()
    assert manifest["checks"]["base_measures_normalized"] is True
    assert manifest["decision"]["status"] == "accepted_gate0_data"
    assert manifest["decision"]["gate0_data_ready"] is True


def test_outputs_are_invariant_to_input_row_order():
    first = _build()
    values = list(_fixture())
    for index in (0, 1, 2, 5, 6):
        values[index] = values[index].sample(frac=1, random_state=index + 7).reset_index(drop=True)
    second = query_benchmark.build_query_benchmark(*values)

    assert first[-1]["output_content_hashes"] == second[-1]["output_content_hashes"]
    assert first[-1]["decision"] == second[-1]["decision"]


def test_accepted_pre_consolidation_state_manifest_uses_frozen_table_hashes():
    values = list(_fixture())
    values[7] = copy.deepcopy(values[7])
    values[7].pop("output_content_hashes")

    result = query_benchmark.build_query_benchmark(*values)

    assert result[-1]["decision"]["status"] == "accepted_gate0_data"


def test_rejects_content_mismatch_and_crossing_components():
    values = list(_fixture())
    values[7] = copy.deepcopy(values[7])
    values[7]["output_content_hashes"]["states_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="tables differ from their manifest"):
        query_benchmark.build_query_benchmark(*values)

    values = list(_fixture())
    mapping = values[1].copy()
    mapping.loc[0, "leakage_component_v2"] = mapping.loc[6, "leakage_component_v2"]
    values[1] = mapping
    values[4] = copy.deepcopy(values[4])
    values[4]["build"]["mapping_sha256"] = similarity_graph.dataframe_content_sha256(
        mapping.loc[:, list(similarity_split.MAPPING_COLUMNS)],
        sort_columns=["sequence_id"],
    )
    with pytest.raises(RuntimeError, match="component crosses splits"):
        query_benchmark.build_query_benchmark(*values)
