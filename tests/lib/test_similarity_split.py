from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.lib import similarity_graph, similarity_split, split_audit


def test_similarity_split_keeps_primary_components_whole_and_is_reproducible():
    retrieval, nodes, edges, graph_manifest = _fixture()

    first, profile, summary = similarity_split.build_similarity_grouped_split(
        retrieval,
        nodes,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=1,
        expected_population_sha256=split_audit.retrieval_population_sha256(retrieval),
    )
    second, _, _ = similarity_split.build_similarity_grouped_split(
        retrieval,
        nodes.sample(frac=1, random_state=7),
        train_fraction=0.6,
        val_fraction=0.2,
        seed=1,
        expected_population_sha256=split_audit.retrieval_population_sha256(retrieval),
    )

    assert first.equals(second)
    assert first.groupby("leakage_component_v2")["split_grouped_v2"].nunique().max() == 1
    assert profile["rows"].sum() == len(retrieval)
    assert summary["rows"] == len(retrieval)

    cross_edges, audit = similarity_split.audit_similarity_grouped_split(
        retrieval,
        nodes,
        edges,
        graph_manifest,
        first,
        summary,
    )

    assert not cross_edges["primary_near_duplicate"].any()
    assert audit["primary_cross_split_edges"] == 0
    assert not any(audit["crossing_groups"].values())


def test_similarity_split_rejects_missing_graph_node():
    retrieval, nodes, _, _ = _fixture()

    with pytest.raises(ValueError, match="sequence IDs differ"):
        similarity_split.build_similarity_grouped_split(
            retrieval,
            nodes.iloc[:-1],
            train_fraction=0.6,
            val_fraction=0.2,
            seed=1,
            expected_population_sha256=split_audit.retrieval_population_sha256(retrieval),
        )


def test_similarity_split_audit_rejects_changed_component_mapping():
    retrieval, nodes, edges, graph_manifest = _fixture()
    mapping, _, summary = similarity_split.build_similarity_grouped_split(
        retrieval,
        nodes,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=1,
        expected_population_sha256=split_audit.retrieval_population_sha256(retrieval),
    )
    changed = mapping.copy()
    changed.loc[0, "similarity_component_primary"] = "changed"
    changed.loc[0, "leakage_component_v2"] = "changed"

    with pytest.raises(RuntimeError, match="component IDs differ"):
        similarity_split.audit_similarity_grouped_split(
            retrieval,
            nodes,
            edges,
            graph_manifest,
            changed,
            summary,
        )


def test_similarity_split_audit_rejects_unaccepted_graph_manifest():
    retrieval, nodes, edges, graph_manifest = _fixture()
    mapping, _, summary = similarity_split.build_similarity_grouped_split(
        retrieval,
        nodes,
        train_fraction=0.6,
        val_fraction=0.2,
        seed=1,
        expected_population_sha256=split_audit.retrieval_population_sha256(retrieval),
    )
    graph_manifest["decision"]["no_final_query_saturated"] = False

    with pytest.raises(RuntimeError, match="graph manifest is not accepted"):
        similarity_split.audit_similarity_grouped_split(
            retrieval,
            nodes,
            edges,
            graph_manifest,
            mapping,
            summary,
        )


def _fixture():
    sequences = ["ACGT" * (index + 1) for index in range(8)]
    retrieval = pd.DataFrame(
        {
            "sequence_id": [f"s{index}" for index in range(8)],
            "sequence": sequences,
            "sequence_sha256": [split_audit.sequence_sha256(value) for value in sequences],
            "family_key": ["f0", "f0", "f1", "f2", "f3", "f4", "f5", "f6"],
            "leakage_component": ["old0", "old0", "old1", "old2", "old3", "old4", "old5", "old6"],
            "split_grouped": ["train", "train", "val", "test", "train", "val", "test", "train"],
            "length_bp": [4 * (index + 1) for index in range(8)],
        }
    )
    raw_edges = pd.DataFrame(
        {
            "query_sequence_id": ["s1", "s4"],
            "subject_sequence_id": ["s2", "s5"],
            "query_length_bp": [8, 20],
            "subject_length_bp": [12, 24],
            "identity": [0.995, 0.96],
            "query_coverage": [0.98, 0.92],
            "subject_coverage": [0.98, 0.92],
            "length_ratio": [0.96, 0.92],
            "orientation": ["same", "same"],
            "alignment_block_length": [8, 20],
            "matching_bases": [8, 19],
            "cap": [1_000, 10_000],
        }
    )
    edges = similarity_graph.canonicalize_similarity_edges(
        raw_edges,
        primary_rule=split_audit.SimilarityRule(0.99, 0.95, 0.95, 0.95),
        sensitivity_rule=split_audit.SimilarityRule(0.95, 0.90, 0.90, 0.90),
    )
    nodes, _, _ = similarity_graph.build_similarity_components(retrieval, edges)
    graph_manifest = {
        "graph_version": "global_similarity_graph_v0.1",
        "decision": {
            "all_queries_have_final_exact_search": True,
            "no_final_query_saturated": True,
            "edge_enumeration_complete_under_configured_caps": True,
        },
        "output_content_hashes": {
            "edges_sha256": similarity_graph.dataframe_content_sha256(
                edges, sort_columns=["sequence_a", "sequence_b"]
            ),
            "nodes_sha256": similarity_graph.dataframe_content_sha256(
                nodes, sort_columns=["sequence_id"]
            ),
        },
    }
    return retrieval, nodes, edges, graph_manifest
