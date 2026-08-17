from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.lib import similarity_graph
from vec2vec.lib.similarity_graph_validation import validate_similarity_graph_outputs


def test_readback_validation_accepts_consistent_outputs():
    tables, manifest = _fixture()

    report = validate_similarity_graph_outputs(
        *tables,
        manifest,
        expected_rows=2,
        expected_population_sha256="population",
        expected_retrieval_version="retrieval-version",
    )

    assert report["status"] == "accepted_independent_s3_readback"
    assert report["final_exact_profiles"] == 2
    assert report["final_saturated_queries"] == 0


def test_readback_validation_rejects_saturated_final_query():
    tables, manifest = _fixture()
    tables[3].loc[0, "potentially_saturated"] = True

    with pytest.raises(RuntimeError, match="final exact query remains saturated"):
        validate_similarity_graph_outputs(
            *tables,
            manifest,
            expected_rows=2,
            expected_population_sha256="population",
            expected_retrieval_version="retrieval-version",
        )


def test_readback_validation_rejects_changed_persisted_content():
    tables, manifest = _fixture()
    tables[0].loc[0, "identity"] = 0.996

    with pytest.raises(RuntimeError, match="content hashes differ"):
        validate_similarity_graph_outputs(
            *tables,
            manifest,
            expected_rows=2,
            expected_population_sha256="population",
            expected_retrieval_version="retrieval-version",
        )


def _fixture():
    edges = pd.DataFrame(
        {
            "sequence_a": ["a"],
            "sequence_b": ["b"],
            "identity": [0.995],
            "coverage_a": [0.98],
            "coverage_b": [0.98],
            "length_ratio": [0.98],
            "primary_near_duplicate": [True],
            "sensitivity_near_duplicate": [True],
        }
    )
    nodes = pd.DataFrame(
        {
            "sequence_id": ["a", "b"],
            "leakage_component": ["old-a", "old-b"],
            "similarity_component_primary": ["p", "p"],
            "similarity_component_sensitivity": ["s", "s"],
        }
    )
    components = pd.DataFrame(
        {
            "threshold": ["primary_99", "sensitivity_95"],
            "similarity_component": ["p", "s"],
            "rows": [2, 2],
        }
    )
    profiles = pd.DataFrame(
        {
            "token": ["ta", "tb"],
            "sequence_id": ["a", "b"],
            "stage": ["exact_normal", "exact_adaptive"],
            "cap": [1_000, 10_000],
            "final_for_query": [True, True],
            "potentially_saturated": [False, False],
        }
    )
    runs = pd.DataFrame(
        {
            "stage": ["exact_normal", "exact_adaptive"],
            "cap": [1_000, 10_000],
            "shard_id": [0, 1],
            "query_count": [1, 1],
            "cpu_seconds": [1.0, 2.0],
            "paf_bytes": [10, 20],
        }
    )
    tables = [edges, nodes, components, profiles, runs]
    hashes = {
        "edges_sha256": similarity_graph.dataframe_content_sha256(
            edges, sort_columns=["sequence_a", "sequence_b"]
        ),
        "nodes_sha256": similarity_graph.dataframe_content_sha256(
            nodes, sort_columns=["sequence_id"]
        ),
        "components_sha256": similarity_graph.dataframe_content_sha256(
            components, sort_columns=["threshold", "similarity_component"]
        ),
        "query_profile_sha256": similarity_graph.dataframe_content_sha256(
            profiles, sort_columns=["stage", "cap", "token"]
        ),
        "runs_sha256": similarity_graph.dataframe_content_sha256(
            runs, sort_columns=["stage", "cap", "shard_id"]
        ),
    }
    manifest = {
        "input_retrieval_version": "retrieval-version",
        "input_validation": {"input_population_sha256": "population"},
        "decision": {
            "all_queries_have_final_exact_search": True,
            "no_final_query_saturated": True,
            "edge_enumeration_complete_under_configured_caps": True,
            "split_grouped_v2_assigned": False,
            "model_outcomes_inspected": False,
        },
        "output_content_hashes": hashes,
        "graph_summary": {
            "nodes": 2,
            "canonical_edges": 1,
            "primary_edges": 1,
            "sensitivity_edges": 1,
            "primary_components": 1,
            "sensitivity_components": 1,
        },
        "search_summary": {
            "by_stage": {
                "exact_normal": {
                    "shards": 1,
                    "queries": 1,
                    "cpu_seconds": 1.0,
                    "paf_bytes": 10,
                },
                "exact_adaptive": {
                    "shards": 1,
                    "queries": 1,
                    "cpu_seconds": 2.0,
                    "paf_bytes": 20,
                },
            }
        },
        "runtime": {
            "wall_seconds": 10.0,
            "observed_child_cpu_hours": 3.0 / 3_600,
            "observed_raw_paf_bytes": 30,
        },
        "resolved_configuration": {
            "execution": {
                "full_run_wall_limit_seconds": 100,
                "maximum_cpu_hours": 1.0,
                "maximum_persisted_bytes": 100,
            }
        },
    }
    return tables, manifest
