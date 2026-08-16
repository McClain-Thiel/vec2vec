from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.lib import similarity_graph
from vec2vec.lib.similarity_split_validation import validate_similarity_split_outputs


def test_split_readback_accepts_consistent_outputs():
    tables, manifest = _fixture()

    report = validate_similarity_split_outputs(
        *tables,
        manifest,
        expected_rows=4,
        expected_graph_artifact_version="graph-version",
    )

    assert report["status"] == "accepted_independent_s3_readback"
    assert report["primary_cross_split_edges"] == 0


def test_split_readback_rejects_strict_cross_edge():
    tables, manifest = _fixture()
    tables[2].loc[0, "primary_near_duplicate"] = True

    with pytest.raises(RuntimeError, match="strict crossings"):
        validate_similarity_split_outputs(
            *tables,
            manifest,
            expected_rows=4,
            expected_graph_artifact_version="graph-version",
        )


def test_split_readback_rejects_changed_mapping():
    tables, manifest = _fixture()
    tables[0].loc[0, "split_grouped_v2"] = "test"

    with pytest.raises(RuntimeError, match="component crosses"):
        validate_similarity_split_outputs(
            *tables,
            manifest,
            expected_rows=4,
            expected_graph_artifact_version="graph-version",
        )


def _fixture():
    mapping = pd.DataFrame(
        {
            "sequence_id": ["a", "b", "c", "d"],
            "similarity_component_primary": ["p1", "p1", "p2", "p3"],
            "leakage_component_v2": ["p1", "p1", "p2", "p3"],
            "split_grouped_v2": ["train", "train", "val", "test"],
        }
    )
    components = pd.DataFrame(
        {
            "leakage_component_v2": ["p1", "p2", "p3"],
            "split_grouped_v2": ["train", "val", "test"],
            "rows": [2, 1, 1],
        }
    )
    cross_edges = pd.DataFrame(
        {
            "sequence_a": ["b"],
            "sequence_b": ["d"],
            "primary_near_duplicate": [False],
            "split_a_v2": ["train"],
            "split_b_v2": ["test"],
        }
    )
    mapping_hash = similarity_graph.dataframe_content_sha256(mapping, sort_columns=["sequence_id"])
    component_hash = similarity_graph.dataframe_content_sha256(
        components, sort_columns=["leakage_component_v2"]
    )
    cross_hash = similarity_graph.dataframe_content_sha256(
        cross_edges, sort_columns=["sequence_a", "sequence_b"]
    )
    manifest = {
        "input_graph_artifact_version": "graph-version",
        "build": {
            "mapping_sha256": mapping_hash,
            "component_profile_sha256": component_hash,
            "split_counts": {"test": 1, "train": 2, "val": 1},
        },
        "audit": {
            "cross_edges_sha256": cross_hash,
            "concentration": {},
            "concentration_warning_splits": [],
        },
        "decision": {
            "status": "accepted_strict_similarity_closed_split",
            "strict_group_crossings": 0,
            "strict_primary_edge_crossings": 0,
            "current_split_overwritten": False,
        },
    }
    return [mapping, components, cross_edges], manifest
