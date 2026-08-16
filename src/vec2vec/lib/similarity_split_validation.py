"""Independent read-back validation for persisted similarity-closed split outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vec2vec.lib import similarity_graph, similarity_split, splits


def validate_similarity_split_outputs(
    mapping: pd.DataFrame,
    components: pd.DataFrame,
    cross_edges: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    expected_rows: int,
    expected_graph_artifact_version: str,
) -> dict[str, Any]:
    """Validate reloaded v2 split outputs against their manifest and strict closure rules."""
    _require_columns(mapping, set(similarity_split.MAPPING_COLUMNS), name="v2 mapping")
    _require_columns(
        components,
        {"leakage_component_v2", "split_grouped_v2", "rows"},
        name="v2 components",
    )
    _require_columns(
        cross_edges,
        {"sequence_a", "sequence_b", "primary_near_duplicate", "split_a_v2", "split_b_v2"},
        name="v2 cross edges",
    )
    if len(mapping) != expected_rows or mapping["sequence_id"].nunique() != expected_rows:
        raise RuntimeError("v2 mapping row identity differs from the expected population")
    if mapping["sequence_id"].duplicated().any():
        raise RuntimeError("v2 mapping contains duplicate sequence IDs")
    invalid_splits = set(mapping["split_grouped_v2"].astype(str)) - set(splits.SPLIT_LABELS)
    if invalid_splits:
        raise RuntimeError(f"v2 mapping contains invalid split labels: {sorted(invalid_splits)}")
    if set(splits.SPLIT_LABELS) - set(mapping["split_grouped_v2"].astype(str)):
        raise RuntimeError("v2 mapping does not contain all three split labels")
    if (
        not mapping["leakage_component_v2"]
        .astype(str)
        .equals(mapping["similarity_component_primary"].astype(str))
    ):
        raise RuntimeError("v2 leakage components differ from primary graph components")
    if mapping.groupby("leakage_component_v2")["split_grouped_v2"].nunique().gt(1).any():
        raise RuntimeError("a v2 component crosses persisted split labels")
    if components["leakage_component_v2"].duplicated().any():
        raise RuntimeError("v2 component profile contains duplicate component IDs")
    if int(components["rows"].sum()) != expected_rows:
        raise RuntimeError("v2 component profile rows do not sum to the population")
    if set(components["leakage_component_v2"].astype(str)) != set(
        mapping["leakage_component_v2"].astype(str)
    ):
        raise RuntimeError("v2 component profile identifiers differ from the mapping")
    if cross_edges["split_a_v2"].astype(str).eq(cross_edges["split_b_v2"].astype(str)).any():
        raise RuntimeError("v2 cross-edge table contains an edge within one split")
    primary_crossings = int(cross_edges["primary_near_duplicate"].sum())
    if primary_crossings:
        raise RuntimeError(f"v2 cross-edge table contains {primary_crossings} strict crossings")

    if manifest.get("input_graph_artifact_version") != expected_graph_artifact_version:
        raise RuntimeError("v2 manifest graph artifact version differs from the pinned graph")
    decision = manifest.get("decision", {})
    if decision.get("status") != "accepted_strict_similarity_closed_split":
        raise RuntimeError("v2 manifest does not contain an accepted decision")
    if decision.get("strict_group_crossings") != 0:
        raise RuntimeError("v2 manifest reports a strict group crossing")
    if decision.get("strict_primary_edge_crossings") != 0:
        raise RuntimeError("v2 manifest reports a strict primary edge crossing")
    if decision.get("current_split_overwritten") is not False:
        raise RuntimeError("v2 manifest does not preserve the current split")

    observed_hashes = {
        "mapping_sha256": similarity_graph.dataframe_content_sha256(
            mapping, sort_columns=["sequence_id"]
        ),
        "component_profile_sha256": similarity_graph.dataframe_content_sha256(
            components, sort_columns=["leakage_component_v2"]
        ),
        "cross_edges_sha256": similarity_graph.dataframe_content_sha256(
            cross_edges, sort_columns=["sequence_a", "sequence_b"]
        ),
    }
    expected_hashes = {
        "mapping_sha256": manifest.get("build", {}).get("mapping_sha256"),
        "component_profile_sha256": manifest.get("build", {}).get("component_profile_sha256"),
        "cross_edges_sha256": manifest.get("audit", {}).get("cross_edges_sha256"),
    }
    hash_failures = {
        key: {"expected": expected_hashes[key], "observed": observed}
        for key, observed in observed_hashes.items()
        if expected_hashes[key] != observed
    }
    if hash_failures:
        raise RuntimeError(f"reloaded v2 content hashes differ: {hash_failures}")

    split_counts = {
        str(split): int(count)
        for split, count in mapping["split_grouped_v2"].value_counts().sort_index().items()
    }
    if split_counts != manifest.get("build", {}).get("split_counts"):
        raise RuntimeError("reloaded v2 split counts differ from the manifest")
    return {
        "status": "accepted_independent_s3_readback",
        "rows": expected_rows,
        "components": int(mapping["leakage_component_v2"].nunique()),
        "split_counts": split_counts,
        "primary_cross_split_edges": primary_crossings,
        "sensitivity_only_cross_split_edges": int(len(cross_edges)),
        "content_hashes": observed_hashes,
        "concentration": manifest.get("audit", {}).get("concentration"),
        "concentration_warning_splits": manifest.get("audit", {}).get(
            "concentration_warning_splits"
        ),
    }


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} table is missing columns: {sorted(missing)}")
