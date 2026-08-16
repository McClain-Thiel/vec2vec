"""Independent read-back validation for persisted global similarity-graph outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vec2vec.lib import similarity_graph


def validate_similarity_graph_outputs(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    components: pd.DataFrame,
    profiles: pd.DataFrame,
    runs: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    expected_rows: int,
    expected_population_sha256: str,
    expected_retrieval_version: str,
) -> dict[str, Any]:
    """Validate reloaded graph tables against invariants and the persisted manifest."""
    _require_columns(
        edges,
        {
            "sequence_a",
            "sequence_b",
            "identity",
            "coverage_a",
            "coverage_b",
            "length_ratio",
            "primary_near_duplicate",
            "sensitivity_near_duplicate",
        },
        name="edges",
    )
    _require_columns(
        nodes,
        {
            "sequence_id",
            "leakage_component",
            "similarity_component_primary",
            "similarity_component_sensitivity",
        },
        name="nodes",
    )
    _require_columns(
        components,
        {"threshold", "similarity_component", "rows"},
        name="components",
    )
    _require_columns(
        profiles,
        {
            "token",
            "sequence_id",
            "stage",
            "cap",
            "final_for_query",
            "potentially_saturated",
        },
        name="query profile",
    )
    _require_columns(
        runs,
        {"stage", "cap", "shard_id", "query_count", "cpu_seconds", "paf_bytes"},
        name="runs",
    )

    if len(nodes) != expected_rows or nodes["sequence_id"].nunique() != expected_rows:
        raise RuntimeError(
            f"graph nodes do not contain {expected_rows} unique rows: "
            f"rows={len(nodes)}, unique={nodes['sequence_id'].nunique()}"
        )
    if edges.duplicated(["sequence_a", "sequence_b"]).any():
        raise RuntimeError("canonical graph edge keys are not unique")
    if edges["sequence_a"].astype(str).ge(edges["sequence_b"].astype(str)).any():
        raise RuntimeError("canonical graph edges are not ordered or contain self edges")
    known_ids = set(nodes["sequence_id"].astype(str))
    edge_ids = set(edges["sequence_a"].astype(str)).union(edges["sequence_b"].astype(str))
    if edge_ids - known_ids:
        raise RuntimeError(f"graph edges contain {len(edge_ids - known_ids)} unknown endpoints")

    sensitivity_rule = (
        edges["identity"].ge(0.95)
        & edges["coverage_a"].ge(0.90)
        & edges["coverage_b"].ge(0.90)
        & edges["length_ratio"].ge(0.90)
    )
    primary_rule = (
        edges["identity"].ge(0.99)
        & edges["coverage_a"].ge(0.95)
        & edges["coverage_b"].ge(0.95)
        & edges["length_ratio"].ge(0.95)
    )
    if not sensitivity_rule.all() or not edges["sensitivity_near_duplicate"].all():
        raise RuntimeError("a persisted graph edge fails the sensitivity rule")
    if not edges["primary_near_duplicate"].astype(bool).equals(primary_rule.astype(bool)):
        raise RuntimeError("persisted primary edge flags differ from the fixed primary rule")
    if (edges["primary_near_duplicate"] & ~edges["sensitivity_near_duplicate"]).any():
        raise RuntimeError("a primary edge is not a sensitivity edge")

    final_profiles = profiles.loc[
        profiles["stage"].isin(["exact_normal", "exact_adaptive"])
        & profiles["final_for_query"].astype(bool)
    ]
    final_counts = final_profiles.groupby("sequence_id", sort=False).size()
    if len(final_profiles) != expected_rows or not final_counts.eq(1).all():
        raise RuntimeError("not every persisted sequence has exactly one final exact profile")
    if set(final_profiles["sequence_id"].astype(str)) != known_ids:
        raise RuntimeError("final exact profile identifiers differ from graph nodes")
    if final_profiles["potentially_saturated"].astype(bool).any():
        raise RuntimeError("a persisted final exact query remains saturated")

    threshold_columns = {
        "primary_99": "similarity_component_primary",
        "sensitivity_95": "similarity_component_sensitivity",
    }
    component_checks: dict[str, dict[str, int]] = {}
    for threshold, node_column in threshold_columns.items():
        table = components.loc[components["threshold"].eq(threshold)]
        expected_components = int(nodes[node_column].nunique())
        if len(table) != expected_components:
            raise RuntimeError(f"{threshold} component table count differs from graph nodes")
        if int(table["rows"].sum()) != expected_rows:
            raise RuntimeError(f"{threshold} component rows do not sum to the population")
        if set(table["similarity_component"].astype(str)) != set(nodes[node_column].astype(str)):
            raise RuntimeError(f"{threshold} component identifiers differ from graph nodes")
        old_component_crossings = int(
            nodes.groupby("leakage_component", sort=False)[node_column].nunique().gt(1).sum()
        )
        if old_component_crossings:
            raise RuntimeError(
                f"{old_component_crossings} old leakage components split across {threshold}"
            )
        component_checks[threshold] = {
            "components": expected_components,
            "largest_component_rows": int(table["rows"].max()),
            "old_component_crossings": old_component_crossings,
        }

    decision = manifest.get("decision", {})
    required_decisions = {
        "all_queries_have_final_exact_search": True,
        "no_final_query_saturated": True,
        "edge_enumeration_complete_under_configured_caps": True,
        "split_grouped_v2_assigned": False,
        "model_outcomes_inspected": False,
    }
    failed_decisions = {
        key: decision.get(key)
        for key, expected in required_decisions.items()
        if decision.get(key) != expected
    }
    if failed_decisions:
        raise RuntimeError(f"persisted graph decision is not acceptable: {failed_decisions}")
    input_validation = manifest.get("input_validation", {})
    if input_validation.get("population_sha256") != expected_population_sha256:
        raise RuntimeError("graph manifest population hash differs from the pinned input")
    if manifest.get("input_retrieval_version") != expected_retrieval_version:
        raise RuntimeError("graph manifest retrieval version differs from the pinned input")

    observed_hashes = {
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
    persisted_hashes = manifest.get("output_content_hashes", {})
    hash_failures = {
        key: {"expected": persisted_hashes.get(key), "observed": value}
        for key, value in observed_hashes.items()
        if persisted_hashes.get(key) != value
    }
    if hash_failures:
        raise RuntimeError(f"reloaded graph content hashes differ: {hash_failures}")

    graph_summary = manifest.get("graph_summary", {})
    observed_summary = {
        "nodes": int(len(nodes)),
        "canonical_edges": int(len(edges)),
        "primary_edges": int(edges["primary_near_duplicate"].sum()),
        "sensitivity_edges": int(edges["sensitivity_near_duplicate"].sum()),
        "primary_components": int(nodes["similarity_component_primary"].nunique()),
        "sensitivity_components": int(nodes["similarity_component_sensitivity"].nunique()),
    }
    summary_failures = {
        key: {"expected": graph_summary.get(key), "observed": value}
        for key, value in observed_summary.items()
        if graph_summary.get(key) != value
    }
    if summary_failures:
        raise RuntimeError(f"graph summary differs from reloaded tables: {summary_failures}")

    stage_checks: dict[str, dict[str, float | int]] = {}
    manifest_stages = manifest.get("search_summary", {}).get("by_stage", {})
    for stage, group in runs.groupby("stage", sort=True):
        observed_stage = {
            "shards": int(len(group)),
            "queries": int(group["query_count"].sum()),
            "cpu_seconds": float(group["cpu_seconds"].sum()),
            "paf_bytes": int(group["paf_bytes"].sum()),
        }
        expected_stage = manifest_stages.get(str(stage), {})
        for key, value in observed_stage.items():
            if expected_stage.get(key) != value:
                raise RuntimeError(f"run-table {stage} {key} differs from the manifest")
        stage_checks[str(stage)] = observed_stage

    runtime = manifest.get("runtime", {})
    execution = manifest.get("resolved_configuration", {}).get("execution", {})
    resource_checks = {
        "wall_seconds": float(runtime["wall_seconds"]),
        "child_cpu_hours": float(runtime["observed_child_cpu_hours"]),
        "raw_paf_bytes": int(runtime["observed_raw_paf_bytes"]),
    }
    if resource_checks["wall_seconds"] > float(execution["full_run_wall_limit_seconds"]):
        raise RuntimeError("persisted graph crossed the fixed wall-time limit")
    if resource_checks["child_cpu_hours"] > float(execution["maximum_cpu_hours"]):
        raise RuntimeError("persisted graph crossed the fixed CPU-hour limit")
    if resource_checks["raw_paf_bytes"] > int(execution["maximum_persisted_bytes"]):
        raise RuntimeError("persisted graph crossed the fixed output-byte limit")

    return {
        "status": "accepted_independent_s3_readback",
        "rows": expected_rows,
        "final_exact_profiles": int(len(final_profiles)),
        "final_saturated_queries": 0,
        "graph_summary": observed_summary,
        "component_checks": component_checks,
        "stage_checks": stage_checks,
        "resource_checks": resource_checks,
        "content_hashes": observed_hashes,
    }


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
