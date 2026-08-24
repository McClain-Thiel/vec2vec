"""Kedro nodes for the separately named similarity-closed grouped split."""

from __future__ import annotations

import hashlib
import subprocess
from typing import Any

import pandas as pd

from vec2vec.lib import similarity_split


def build_similarity_split(
    retrieval: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the deterministic v2 mapping and component profile."""
    _require_graph_artifact_version(params)
    return similarity_split.build_similarity_grouped_split(
        retrieval,
        graph_nodes,
        train_fraction=float(params["train_fraction"]),
        val_fraction=float(params["val_fraction"]),
        seed=int(params["seed"]),
        expected_population_sha256=str(params["expected_input_population_sha256"]),
    )


def audit_similarity_split(
    retrieval: pd.DataFrame,
    graph_nodes: pd.DataFrame,
    graph_edges: pd.DataFrame,
    graph_manifest: dict[str, Any],
    mapping: pd.DataFrame,
    build_summary: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit v2 independently and return cross-split sensitivity evidence."""
    cross_edges, audit = similarity_split.audit_similarity_grouped_split(
        retrieval,
        graph_nodes,
        graph_edges,
        graph_manifest,
        mapping,
        build_summary,
    )
    graph_artifact_version = _require_graph_artifact_version(params)
    manifest = {
        "protocol_version": str(params["protocol_version"]),
        "protocol": (
            "studies/set_valued_compositional_embeddings/experiments/E00_split_grouped_v2.md"
        ),
        "input_retrieval_version": str(params["input_retrieval_version"]),
        "input_graph_artifact_version": graph_artifact_version,
        "input_graph_protocol_version": graph_manifest["graph_version"],
        "resolved_configuration": params,
        "build": build_summary,
        "audit": audit,
        "git": _git_provenance(),
        "decision": {
            "status": "accepted_strict_similarity_closed_split",
            "strict_group_crossings": 0,
            "strict_primary_edge_crossings": 0,
            "concentration_warning": bool(audit["concentration_warning_splits"]),
            "current_split_overwritten": False,
            "model_outcomes_inspected": False,
        },
        "known_limitations": [
            "Strict closure is conditional on the accepted heuristic minimap2 graph.",
            "Single-linkage components can connect rows through similarity chains.",
            "Sequence dissimilarity does not establish functional independence.",
        ],
    }
    return cross_edges, manifest


def _require_graph_artifact_version(params: dict[str, Any]) -> str:
    graph_artifact_version = params["input_graph_artifact_version"]
    if not isinstance(graph_artifact_version, str) or not graph_artifact_version.strip():
        raise ValueError("similarity_split input_graph_artifact_version must be pinned")
    return graph_artifact_version


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], check=True, capture_output=True, text=True
    ).stdout
    lines = [line for line in status.splitlines() if line]
    return {
        "commit": commit,
        "worktree_dirty": bool(lines),
        "worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "changed_paths": [line[3:] for line in lines],
    }
