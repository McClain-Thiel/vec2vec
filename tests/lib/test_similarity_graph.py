from __future__ import annotations

import pandas as pd

from vec2vec.lib.similarity_graph import (
    build_similarity_components,
    canonicalize_similarity_edges,
    dataframe_content_sha256,
)
from vec2vec.lib.split_audit import SimilarityRule


def test_canonical_edges_swap_coverage_and_keep_the_best_direction():
    directional = pd.DataFrame.from_records(
        [
            _directional_edge("b", "a", query_coverage=0.96, subject_coverage=0.97),
            _directional_edge(
                "a",
                "b",
                query_coverage=1.0,
                subject_coverage=1.0,
                identity=0.995,
            ),
        ]
    )

    edges = canonicalize_similarity_edges(
        directional,
        primary_rule=SimilarityRule(0.99, 0.95, 0.95, 0.95),
        sensitivity_rule=SimilarityRule(0.95, 0.90, 0.90, 0.90),
    )

    assert len(edges) == 1
    assert edges.loc[0, "sequence_a"] == "a"
    assert edges.loc[0, "sequence_b"] == "b"
    assert edges.loc[0, "coverage_a"] == 1.0
    assert edges.loc[0, "coverage_b"] == 1.0
    assert edges.loc[0, "detection_directions"] == 2
    assert bool(edges.loc[0, "primary_near_duplicate"])


def test_components_keep_prior_groups_whole_and_add_similarity_bridges():
    retrieval = pd.DataFrame(
        {
            "sequence_id": ["a", "b", "c", "d"],
            "sequence_sha256": ["ha", "hb", "hc", "hd"],
            "family_key": ["f1", "f1", "f2", "f3"],
            "leakage_component": ["old1", "old1", "old2", "old3"],
            "split_grouped": ["train", "train", "val", "test"],
            "length_bp": [100, 100, 100, 100],
        }
    )
    edges = pd.DataFrame(
        {
            "sequence_a": ["b", "c"],
            "sequence_b": ["c", "d"],
            "primary_near_duplicate": [True, False],
            "sensitivity_near_duplicate": [True, True],
        }
    )

    nodes, profiles, summary = build_similarity_components(retrieval, edges)
    by_id = nodes.set_index("sequence_id")

    assert (
        by_id.loc["a", "similarity_component_primary"]
        == by_id.loc["c", "similarity_component_primary"]
    )
    assert (
        by_id.loc["c", "similarity_component_primary"]
        != by_id.loc["d", "similarity_component_primary"]
    )
    assert len(set(by_id["similarity_component_sensitivity"])) == 1
    assert summary["primary_components"] == 2
    assert summary["sensitivity_components"] == 1
    assert set(profiles["threshold"]) == {"primary_99", "sensitivity_95"}


def test_table_content_hash_is_stable_to_row_order_and_changes_with_content():
    frame = pd.DataFrame({"id": ["b", "a"], "value": [2, 1]})
    first = dataframe_content_sha256(frame, sort_columns=["id"])
    second = dataframe_content_sha256(frame.iloc[::-1], sort_columns=["id"])
    changed = frame.copy()
    changed.loc[changed["id"].eq("a"), "value"] = 3

    assert first == second
    assert first != dataframe_content_sha256(changed, sort_columns=["id"])


def _directional_edge(
    query: str,
    subject: str,
    *,
    identity: float = 0.99,
    query_coverage: float = 0.95,
    subject_coverage: float = 0.95,
) -> dict:
    return {
        "query_sequence_id": query,
        "subject_sequence_id": subject,
        "query_length_bp": 100,
        "subject_length_bp": 100,
        "identity": identity,
        "query_coverage": query_coverage,
        "subject_coverage": subject_coverage,
        "length_ratio": 1.0,
        "orientation": "same",
        "alignment_block_length": 100,
        "matching_bases": int(identity * 100),
        "cap": 1_000,
    }
