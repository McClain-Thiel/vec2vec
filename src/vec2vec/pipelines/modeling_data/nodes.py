"""Small Kedro boundaries around the selected modeling-data functions."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import pandas as pd

from vec2vec.lib import (
    constraint_state,
    fixed_representation_bakeoff,
    fixed_representation_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
QWEN_CANDIDATE = "qwen3_embedding_0_6b"


def build_constraint_states(
    retrieval: pd.DataFrame,
    state_params: dict[str, Any],
    evidence_params: dict[str, Any],
    mapping_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply the frozen exact metadata mappings to the retrieval population."""
    return constraint_state.build_constraint_state_tables(
        retrieval, state_params, evidence_params, mapping_params
    )


def build_model_inputs(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    split_manifest: dict[str, Any],
    query_catalog: pd.DataFrame,
    query_states: pd.DataFrame,
    query_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the deterministic train panel, validation gallery, and queries."""
    outputs = fixed_representation_bakeoff.build_bakeoff_inputs(
        retrieval,
        split_mapping,
        split_manifest,
        query_catalog,
        query_states,
        query_manifest,
        params,
    )
    *tables, manifest = outputs
    expected = params["expected_input_artifact_hashes"]
    if manifest["output_hashes"] != expected:
        raise ValueError("model input tables changed from the accepted content hashes")
    manifest["protocol"] = "modeling_data_v1"
    manifest["runtime"] = _runtime_provenance()
    manifest["git"] = _git_provenance()
    return (*tables, manifest)


def fit_selected_dna_features(
    pairs: pd.DataFrame,
    input_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit the selected train-only 6-mer TF-IDF/SVD DNA encoder."""
    features, vocabulary, state, summary = fixed_representation_features.fit_tfidf_dna_features(
        pairs, input_manifest, params
    )
    expected = params["expected_feature_artifact_hashes"]["tfidf_6mer_svd_512"]
    if summary["output_hashes"] != expected:
        raise ValueError("TF-IDF feature artifacts changed from the accepted content hashes")
    return features, vocabulary, state, _feature_manifest(summary, params, compute=None)


def extract_selected_text_features(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    input_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract the selected Qwen document and query embeddings."""
    stage = f"text_features:{QWEN_CANDIDATE}"
    compute = fixed_representation_bakeoff.validated_compute_authorization(params, stage=stage)
    deadline = time.monotonic() + float(compute["instance_hour_limit"]) * 3600.0
    features, summary = fixed_representation_features.extract_text_features(
        pairs,
        queries,
        input_manifest,
        params,
        QWEN_CANDIDATE,
        deadline_monotonic=deadline,
    )
    expected = params["expected_feature_artifact_hashes"][QWEN_CANDIDATE]
    if summary["output_hashes"] != expected:
        raise ValueError("Qwen features changed from the accepted content hash")
    return features, _feature_manifest(summary, params, compute=compute)


def _feature_manifest(
    summary: dict[str, Any],
    params: dict[str, Any],
    *,
    compute: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "protocol_version": str(params["protocol_version"]),
        "protocol": "modeling_data_v1",
        **summary,
        "resolved_feature_configuration": {
            "device": str(params["device"]),
            "precision": str(params["precision"]),
            "seed": int(params["seed"]),
        },
        "compute_authorization": compute,
        "runtime": _runtime_provenance(),
        "git": _git_provenance(),
    }


def _runtime_provenance() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "hostname": platform.node(),
        "machine": platform.machine(),
    }


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    lines = [line for line in status.splitlines() if line]
    return {
        "commit": commit,
        "worktree_dirty": bool(lines),
        "worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "changed_paths": [line[3:] for line in lines],
    }
