"""Kedro nodes for the frozen symbolic query benchmark."""

from __future__ import annotations

import hashlib
import subprocess
from typing import Any

import pandas as pd

from vec2vec.lib import query_benchmark


def build_benchmark(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    graph_edges: pd.DataFrame,
    graph_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    vocabulary: pd.DataFrame,
    states: pd.DataFrame,
    state_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Build the benchmark only from accepted, explicitly pinned inputs."""
    outputs = query_benchmark.build_query_benchmark(
        retrieval,
        split_mapping,
        graph_edges,
        graph_manifest,
        split_manifest,
        vocabulary,
        states,
        state_manifest,
        params,
    )
    *tables, manifest = outputs
    manifest["git"] = _git_provenance()
    return (*tables, manifest)


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
