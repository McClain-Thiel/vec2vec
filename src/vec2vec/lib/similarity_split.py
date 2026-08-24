"""Deterministic split assignment and independent audit from similarity components."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import similarity_graph, split_audit, splits

MAPPING_COLUMNS = (
    "sequence_id",
    "similarity_component_primary",
    "leakage_component_v2",
    "split_grouped_v2",
)


def build_similarity_grouped_split(
    retrieval: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
    expected_population_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Assign complete primary similarity components to a separately named split."""
    validation = split_audit.validate_retrieval(
        retrieval,
        expected_population_sha256=expected_population_sha256,
    )
    required_nodes = {
        "sequence_id",
        "similarity_component_primary",
        "similarity_component_sensitivity",
    }
    missing_nodes = required_nodes.difference(graph_nodes.columns)
    if missing_nodes:
        raise ValueError(f"graph nodes are missing columns: {sorted(missing_nodes)}")
    _validate_unique_complete_ids(retrieval, graph_nodes, name="graph nodes")

    fractions = splits.SplitFractions(train=train_fraction, val=val_fraction)
    nodes = graph_nodes.loc[:, sorted(required_nodes)].copy()
    nodes["sequence_id"] = nodes["sequence_id"].astype(str)
    nodes["similarity_component_primary"] = nodes["similarity_component_primary"].astype(str)
    nodes = nodes.sort_values("sequence_id", kind="stable").reset_index(drop=True)
    labels = _assign_stable_component_split(
        nodes["similarity_component_primary"],
        fractions=fractions,
        seed=seed,
    )
    mapping = nodes.loc[:, ["sequence_id", "similarity_component_primary"]].copy()
    mapping["leakage_component_v2"] = mapping["similarity_component_primary"]
    mapping["split_grouped_v2"] = labels
    mapping = mapping.loc[:, MAPPING_COLUMNS]

    joined = retrieval.merge(mapping, on="sequence_id", how="inner", validate="one_to_one")
    profile = (
        joined.groupby("leakage_component_v2", sort=True)
        .agg(
            split_grouped_v2=("split_grouped_v2", "first"),
            rows=("sequence_id", "size"),
            total_length_bp=("length_bp", "sum"),
            family_count=("family_key", "nunique"),
            exact_sequence_count=("sequence_sha256", "nunique"),
            old_component_count=("leakage_component", "nunique"),
            old_split_count=("split_grouped", "nunique"),
        )
        .reset_index()
    )
    old_split_labels = joined.groupby("leakage_component_v2", sort=True)["split_grouped"].agg(
        lambda values: ",".join(sorted(set(str(value) for value in values)))
    )
    profile = profile.join(old_split_labels.rename("old_split_labels"), on="leakage_component_v2")
    profile = profile.sort_values(
        ["split_grouped_v2", "rows", "leakage_component_v2"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    targets = {
        "train": int(len(mapping) * fractions.train),
        "val": int(len(mapping) * fractions.val),
    }
    targets["test"] = len(mapping) - targets["train"] - targets["val"]
    split_counts = {
        str(label): int(count)
        for label, count in mapping["split_grouped_v2"].value_counts().sort_index().items()
    }
    empty_splits = set(splits.SPLIT_LABELS).difference(split_counts)
    if empty_splits:
        raise RuntimeError(
            f"whole-component assignment produced empty splits: {sorted(empty_splits)}"
        )
    summary = {
        "input_validation": validation,
        "rows": int(len(mapping)),
        "components": int(mapping["leakage_component_v2"].nunique()),
        "split_counts": split_counts,
        "split_targets": targets,
        "split_deviation_rows": {
            label: int(split_counts.get(label, 0) - target) for label, target in targets.items()
        },
        "mapping_sha256": similarity_graph.dataframe_content_sha256(
            mapping,
            sort_columns=["sequence_id"],
        ),
        "component_profile_sha256": similarity_graph.dataframe_content_sha256(
            profile,
            sort_columns=["leakage_component_v2"],
        ),
    }
    return mapping, profile, summary


def audit_similarity_grouped_split(
    retrieval: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    graph_edges: pd.DataFrame,
    graph_manifest: dict[str, Any],
    mapping: pd.DataFrame,
    build_summary: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Independently audit v2 identifiers, group purity, graph edges, and concentration."""
    missing_mapping = set(MAPPING_COLUMNS).difference(mapping.columns)
    if missing_mapping:
        raise ValueError(f"v2 mapping is missing columns: {sorted(missing_mapping)}")
    mapping = mapping.loc[:, list(MAPPING_COLUMNS)].copy()
    mapping["sequence_id"] = mapping["sequence_id"].astype(str)
    mapping = mapping.sort_values("sequence_id", kind="stable").reset_index(drop=True)
    _validate_unique_complete_ids(retrieval, mapping, name="v2 mapping")
    _validate_unique_complete_ids(retrieval, graph_nodes, name="graph nodes")
    if set(mapping["split_grouped_v2"].astype(str)) - set(splits.SPLIT_LABELS):
        raise ValueError("v2 mapping contains an invalid split label")
    if (
        not mapping["leakage_component_v2"]
        .astype(str)
        .equals(mapping["similarity_component_primary"].astype(str))
    ):
        raise ValueError("v2 leakage component differs from the primary similarity component")

    required_edge_columns = {
        "sequence_a",
        "sequence_b",
        "primary_near_duplicate",
        "sensitivity_near_duplicate",
    }
    missing_edges = required_edge_columns.difference(graph_edges.columns)
    if missing_edges:
        raise ValueError(f"graph edges are missing columns: {sorted(missing_edges)}")
    if graph_edges.duplicated(["sequence_a", "sequence_b"]).any():
        raise ValueError("graph edge keys are not unique")
    if (graph_edges["primary_near_duplicate"] & ~graph_edges["sensitivity_near_duplicate"]).any():
        raise ValueError("a primary graph edge is not a sensitivity edge")

    graph_decision = graph_manifest.get("decision", {})
    required_graph_decisions = {
        "all_queries_have_final_exact_search": True,
        "no_final_query_saturated": True,
        "edge_enumeration_complete_under_configured_caps": True,
    }
    failed_graph_decisions = {
        key: graph_decision.get(key)
        for key, expected in required_graph_decisions.items()
        if graph_decision.get(key) is not expected
    }
    if failed_graph_decisions:
        raise RuntimeError(
            f"graph manifest is not accepted for splitting: {failed_graph_decisions}"
        )

    graph_hashes = graph_manifest.get("output_content_hashes", {})
    observed_graph_hashes = {
        "edges_sha256": similarity_graph.dataframe_content_sha256(
            graph_edges,
            sort_columns=["sequence_a", "sequence_b"],
        ),
        "nodes_sha256": similarity_graph.dataframe_content_sha256(
            graph_nodes,
            sort_columns=["sequence_id"],
        ),
    }
    for key, observed in observed_graph_hashes.items():
        if graph_hashes.get(key) != observed:
            raise RuntimeError(f"graph {key} differs from its accepted manifest")

    node_components = graph_nodes.loc[:, ["sequence_id", "similarity_component_primary"]].copy()
    node_components["sequence_id"] = node_components["sequence_id"].astype(str)
    node_components = node_components.sort_values("sequence_id", kind="stable").reset_index(
        drop=True
    )
    component_check = mapping.loc[:, ["sequence_id", "similarity_component_primary"]].merge(
        node_components,
        on="sequence_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_mapping", "_graph"),
    )
    if (
        not component_check["similarity_component_primary_mapping"]
        .astype(str)
        .equals(component_check["similarity_component_primary_graph"].astype(str))
    ):
        raise RuntimeError("v2 mapping component IDs differ from graph nodes")

    joined = retrieval.merge(mapping, on="sequence_id", how="inner", validate="one_to_one")
    crossing_groups = {
        key: int(joined.groupby(key, sort=False)["split_grouped_v2"].nunique().gt(1).sum())
        for key in (
            "family_key",
            "sequence_sha256",
            "leakage_component",
            "leakage_component_v2",
        )
    }
    if any(crossing_groups.values()):
        raise RuntimeError(f"v2 split has groups crossing splits: {crossing_groups}")

    split_by_sequence = mapping.set_index("sequence_id")["split_grouped_v2"].astype(str)
    known_ids = set(split_by_sequence.index.astype(str))
    edge_ids = set(graph_edges["sequence_a"].astype(str)).union(
        graph_edges["sequence_b"].astype(str)
    )
    unknown_edges = edge_ids - known_ids
    if unknown_edges:
        examples = sorted(unknown_edges)[:5]
        raise RuntimeError(f"graph edges contain unknown v2 identifiers: {examples}")
    edge_a = graph_edges["sequence_a"].astype(str).map(split_by_sequence)
    edge_b = graph_edges["sequence_b"].astype(str).map(split_by_sequence)
    cross_mask = edge_a.ne(edge_b)
    cross_edges = graph_edges.loc[cross_mask].copy()
    cross_edges["split_a_v2"] = edge_a.loc[cross_mask].to_numpy()
    cross_edges["split_b_v2"] = edge_b.loc[cross_mask].to_numpy()
    cross_edges = cross_edges.sort_values(["sequence_a", "sequence_b"], kind="stable").reset_index(
        drop=True
    )
    primary_crossings = int(cross_edges["primary_near_duplicate"].sum())
    if primary_crossings:
        raise RuntimeError(f"v2 split has {primary_crossings} strict graph edges crossing splits")

    concentration = _split_concentration(joined)
    warnings = [
        split
        for split in ("val", "test")
        if concentration[split]["largest_component_fraction"] > 0.25
    ]
    mapping_hash = similarity_graph.dataframe_content_sha256(
        mapping,
        sort_columns=["sequence_id"],
    )
    if mapping_hash != build_summary["mapping_sha256"]:
        raise RuntimeError("v2 mapping content changed between construction and audit")
    audit = {
        "rows": int(len(joined)),
        "crossing_groups": crossing_groups,
        "primary_cross_split_edges": primary_crossings,
        "sensitivity_only_cross_split_edges": int((~cross_edges["primary_near_duplicate"]).sum()),
        "concentration": concentration,
        "concentration_warning_splits": warnings,
        "graph_content_hashes": observed_graph_hashes,
        "mapping_sha256": mapping_hash,
        "cross_edges_sha256": similarity_graph.dataframe_content_sha256(
            cross_edges,
            sort_columns=["sequence_a", "sequence_b"],
        ),
    }
    return cross_edges, audit


def _assign_stable_component_split(
    components: pd.Series,
    *,
    fractions: splits.SplitFractions,
    seed: int,
) -> np.ndarray:
    component_sizes = components.astype(str).value_counts(sort=False).to_dict()
    order = np.asarray(sorted(component_sizes), dtype=object)
    np.random.default_rng(seed).shuffle(order)
    train_target = int(len(components) * fractions.train)
    val_target = train_target + int(len(components) * fractions.val)
    split_by_component: dict[str, str] = {}
    placed = 0
    for component in order.tolist():
        label = (
            splits.TRAIN
            if placed < train_target
            else (splits.VAL if placed < val_target else splits.TEST)
        )
        split_by_component[str(component)] = label
        placed += int(component_sizes[str(component)])
    return components.astype(str).map(split_by_component).to_numpy(dtype=object)


def _validate_unique_complete_ids(
    retrieval: pd.DataFrame,
    other: pd.DataFrame,
    *,
    name: str,
) -> None:
    if "sequence_id" not in retrieval or "sequence_id" not in other:
        raise ValueError(f"{name} and retrieval must contain sequence_id")
    if retrieval["sequence_id"].duplicated().any():
        raise ValueError("retrieval sequence IDs are not unique")
    if other["sequence_id"].duplicated().any():
        raise ValueError(f"{name} sequence IDs are not unique")
    retrieval_ids = set(retrieval["sequence_id"].astype(str))
    other_ids = set(other["sequence_id"].astype(str))
    if retrieval_ids != other_ids:
        raise ValueError(
            f"{name} sequence IDs differ from retrieval: "
            f"missing={len(retrieval_ids - other_ids)}, extra={len(other_ids - retrieval_ids)}"
        )


def _split_concentration(joined: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for split in splits.SPLIT_LABELS:
        subset = joined.loc[joined["split_grouped_v2"].eq(split)]
        sizes = (
            subset.groupby("leakage_component_v2", sort=True).size().sort_values(ascending=False)
        )
        rows = int(len(subset))
        weights = sizes.to_numpy(dtype=float) / rows
        result[split] = {
            "rows": rows,
            "components": int(len(sizes)),
            "effective_components": float(1.0 / np.square(weights).sum()),
            "largest_component_rows": int(sizes.iloc[0]),
            "largest_component_fraction": float(sizes.iloc[0] / rows),
            "ten_largest_component_rows": int(sizes.iloc[:10].sum()),
            "ten_largest_component_fraction": float(sizes.iloc[:10].sum() / rows),
        }
    return result
