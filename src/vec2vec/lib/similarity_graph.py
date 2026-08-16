"""Canonical edges and stable components for the global plasmid similarity graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

from vec2vec.lib.split_audit import SimilarityRule

EDGE_COLUMNS = (
    "sequence_a",
    "sequence_b",
    "length_a_bp",
    "length_b_bp",
    "identity",
    "coverage_a",
    "coverage_b",
    "length_ratio",
    "orientation",
    "alignment_block_length",
    "matching_bases",
    "primary_near_duplicate",
    "sensitivity_near_duplicate",
    "similarity_class",
    "detection_directions",
    "exact_search_cap",
)


def canonicalize_similarity_edges(
    directional_edges: pd.DataFrame,
    *,
    primary_rule: SimilarityRule,
    sensitivity_rule: SimilarityRule,
) -> pd.DataFrame:
    """Return one best, validated, undirected row for each qualifying sequence pair."""
    required = {
        "query_sequence_id",
        "subject_sequence_id",
        "query_length_bp",
        "subject_length_bp",
        "identity",
        "query_coverage",
        "subject_coverage",
        "length_ratio",
        "orientation",
        "alignment_block_length",
        "matching_bases",
        "cap",
    }
    missing = required.difference(directional_edges.columns)
    if missing:
        raise ValueError(f"directional edge table is missing columns: {sorted(missing)}")
    if directional_edges.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    frame = directional_edges.copy()
    self_edges = frame["query_sequence_id"].eq(frame["subject_sequence_id"])
    if self_edges.any():
        raise ValueError(f"directional edge table contains {int(self_edges.sum())} self edges")

    query_first = (
        frame["query_sequence_id"].astype(str).lt(frame["subject_sequence_id"].astype(str))
    )
    frame["sequence_a"] = frame["query_sequence_id"].where(
        query_first, frame["subject_sequence_id"]
    )
    frame["sequence_b"] = frame["subject_sequence_id"].where(
        query_first, frame["query_sequence_id"]
    )
    frame["length_a_bp"] = frame["query_length_bp"].where(query_first, frame["subject_length_bp"])
    frame["length_b_bp"] = frame["subject_length_bp"].where(query_first, frame["query_length_bp"])
    frame["coverage_a"] = frame["query_coverage"].where(query_first, frame["subject_coverage"])
    frame["coverage_b"] = frame["subject_coverage"].where(query_first, frame["query_coverage"])
    frame["minimum_coverage"] = frame[["coverage_a", "coverage_b"]].min(axis=1)
    direction_counts = (
        frame.groupby(["sequence_a", "sequence_b"], sort=True)["query_sequence_id"]
        .nunique()
        .rename("detection_directions")
    )
    frame = frame.sort_values(
        [
            "sequence_a",
            "sequence_b",
            "minimum_coverage",
            "identity",
            "alignment_block_length",
            "cap",
        ],
        ascending=[True, True, False, False, False, False],
        kind="stable",
    ).drop_duplicates(["sequence_a", "sequence_b"], keep="first")
    frame = frame.join(direction_counts, on=["sequence_a", "sequence_b"])
    frame["primary_near_duplicate"] = _meets_rule(frame, primary_rule)
    frame["sensitivity_near_duplicate"] = _meets_rule(frame, sensitivity_rule)
    if (~frame["sensitivity_near_duplicate"]).any():
        raise ValueError("directional input contains an edge that fails the sensitivity rule")
    if (frame["primary_near_duplicate"] & ~frame["sensitivity_near_duplicate"]).any():
        raise ValueError("primary edge is not a sensitivity edge")
    frame["similarity_class"] = frame["primary_near_duplicate"].map(
        {True: "primary", False: "sensitivity_only"}
    )
    frame["exact_search_cap"] = frame["cap"].astype("int64")
    result = (
        frame.loc[:, EDGE_COLUMNS]
        .sort_values(["sequence_a", "sequence_b"], kind="stable")
        .reset_index(drop=True)
    )
    if result.duplicated(["sequence_a", "sequence_b"]).any():
        raise RuntimeError("canonical similarity edge keys are not unique")
    return result


def build_similarity_components(
    retrieval: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Union existing leakage groups with primary and sensitivity similarity edges."""
    required = {
        "sequence_id",
        "sequence_sha256",
        "family_key",
        "leakage_component",
        "split_grouped",
        "length_bp",
    }
    missing = required.difference(retrieval.columns)
    if missing:
        raise ValueError(f"retrieval dataset is missing columns: {sorted(missing)}")
    if retrieval["sequence_id"].duplicated().any():
        raise ValueError("retrieval sequence IDs are not unique")
    missing_edge_columns = {
        "sequence_a",
        "sequence_b",
        "primary_near_duplicate",
        "sensitivity_near_duplicate",
    }.difference(edges.columns)
    if missing_edge_columns:
        raise ValueError(f"similarity edges are missing columns: {sorted(missing_edge_columns)}")
    known = set(retrieval["sequence_id"].astype(str))
    unknown = set(edges["sequence_a"].astype(str)).union(edges["sequence_b"].astype(str)) - known
    if unknown:
        raise ValueError(f"similarity graph contains unknown sequence IDs: {sorted(unknown)[:5]}")

    nodes = retrieval.loc[:, sorted(required)].copy()
    nodes["sequence_id"] = nodes["sequence_id"].astype(str)
    primary = _component_assignment(
        nodes,
        edges.loc[edges["primary_near_duplicate"]],
        prefix="sim99",
    )
    sensitivity = _component_assignment(
        nodes,
        edges.loc[edges["sensitivity_near_duplicate"]],
        prefix="sim95",
    )
    nodes = nodes.merge(primary, on="sequence_id", validate="one_to_one")
    nodes = nodes.merge(sensitivity, on="sequence_id", validate="one_to_one")
    nodes = nodes.sort_values("sequence_id", kind="stable").reset_index(drop=True)

    primary_profile = _component_profile(
        nodes,
        edges.loc[edges["primary_near_duplicate"]],
        component_column="similarity_component_primary",
        threshold="primary_99",
    )
    sensitivity_profile = _component_profile(
        nodes,
        edges.loc[edges["sensitivity_near_duplicate"]],
        component_column="similarity_component_sensitivity",
        threshold="sensitivity_95",
    )
    profiles = pd.concat([primary_profile, sensitivity_profile], ignore_index=True)
    profiles = profiles.sort_values(
        ["threshold", "rows", "similarity_component"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    summary = {
        "nodes": int(len(nodes)),
        "canonical_edges": int(len(edges)),
        "primary_edges": int(edges["primary_near_duplicate"].sum()),
        "sensitivity_edges": int(edges["sensitivity_near_duplicate"].sum()),
        "primary_components": int(nodes["similarity_component_primary"].nunique()),
        "sensitivity_components": int(nodes["similarity_component_sensitivity"].nunique()),
        "largest_primary_component_rows": int(primary_profile["rows"].max()),
        "largest_sensitivity_component_rows": int(sensitivity_profile["rows"].max()),
        "primary_components_crossing_old_splits": int(
            primary_profile["old_split_count"].gt(1).sum()
        ),
        "sensitivity_components_crossing_old_splits": int(
            sensitivity_profile["old_split_count"].gt(1).sum()
        ),
    }
    return nodes, profiles, summary


def dataframe_content_sha256(
    frame: pd.DataFrame,
    *,
    sort_columns: Sequence[str],
    value_columns: Sequence[str] | None = None,
) -> str:
    """Hash a stable row representation for provenance checks after reload."""
    missing_sort = set(sort_columns).difference(frame.columns)
    if missing_sort:
        raise ValueError(f"hash sort columns are missing: {sorted(missing_sort)}")
    columns = list(value_columns) if value_columns is not None else sorted(frame.columns)
    missing_values = set(columns).difference(frame.columns)
    if missing_values:
        raise ValueError(f"hash value columns are missing: {sorted(missing_values)}")
    ordered = frame.sort_values(list(sort_columns), kind="stable").loc[:, columns]
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, separators=(",", ":")).encode())
    digest.update(b"\n")
    for row in ordered.itertuples(index=False, name=None):
        values = [_json_scalar(value) for value in row]
        digest.update(json.dumps(values, separators=(",", ":"), ensure_ascii=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _component_assignment(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    sequence_ids = nodes["sequence_id"].astype(str).tolist()
    positions = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    parent = list(range(len(sequence_ids)))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != root:
            parent[index], index = root, parent[index]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    for _, group in nodes.groupby("leakage_component", sort=True):
        members = [positions[str(value)] for value in group["sequence_id"]]
        first = members[0]
        for member in members[1:]:
            union(first, member)
    for row in edges.itertuples(index=False):
        union(positions[str(row.sequence_a)], positions[str(row.sequence_b)])

    by_root: dict[int, list[str]] = {}
    for sequence_id in sequence_ids:
        by_root.setdefault(find(positions[sequence_id]), []).append(sequence_id)
    component_by_root = {
        root: f"{prefix}_{_member_hash(members)}" for root, members in by_root.items()
    }
    if len(component_by_root) != len(set(component_by_root.values())):
        raise RuntimeError("stable similarity component hash collision")
    column = (
        "similarity_component_primary" if prefix == "sim99" else "similarity_component_sensitivity"
    )
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            column: [component_by_root[find(positions[value])] for value in sequence_ids],
        }
    )


def _component_profile(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    component_column: str,
    threshold: str,
) -> pd.DataFrame:
    grouped = nodes.groupby(component_column, sort=True)
    profile = grouped.agg(
        rows=("sequence_id", "size"),
        total_length_bp=("length_bp", "sum"),
        minimum_length_bp=("length_bp", "min"),
        maximum_length_bp=("length_bp", "max"),
        family_count=("family_key", "nunique"),
        exact_sequence_count=("sequence_sha256", "nunique"),
        prior_component_count=("leakage_component", "nunique"),
        old_split_count=("split_grouped", "nunique"),
    ).reset_index(names="similarity_component")
    split_labels = grouped["split_grouped"].agg(
        lambda values: ",".join(sorted(set(str(value) for value in values)))
    )
    profile = profile.join(split_labels.rename("old_split_labels"), on="similarity_component")
    component_by_sequence = nodes.set_index("sequence_id")[component_column].to_dict()
    edge_counts: dict[str, int] = {}
    for row in edges.itertuples(index=False):
        left = component_by_sequence[str(row.sequence_a)]
        right = component_by_sequence[str(row.sequence_b)]
        if left != right:
            raise RuntimeError("qualifying edge endpoints were assigned to different components")
        edge_counts[left] = edge_counts.get(left, 0) + 1
    profile["similarity_edge_count"] = (
        profile["similarity_component"].map(edge_counts).fillna(0).astype("int64")
    )
    profile.insert(0, "threshold", threshold)
    return profile


def _meets_rule(frame: pd.DataFrame, rule: SimilarityRule) -> pd.Series:
    return (
        frame["identity"].ge(rule.minimum_identity)
        & frame["coverage_a"].ge(rule.minimum_query_coverage)
        & frame["coverage_b"].ge(rule.minimum_subject_coverage)
        & frame["length_ratio"].ge(rule.minimum_length_ratio)
    )


def _member_hash(members: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for member in sorted(str(value) for value in members):
        digest.update(member.encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_scalar(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_json_scalar(item) for item in value]
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(format(value, ".17g"))
    return value
