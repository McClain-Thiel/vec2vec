"""Kedro nodes for the reduced-population Gate 1 bake-off."""

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
    fixed_representation_alignment,
    fixed_representation_bakeoff,
    fixed_representation_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def build_inputs(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    split_manifest: dict[str, Any],
    query_catalog: pd.DataFrame,
    query_states: pd.DataFrame,
    query_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build and provenance the deterministic E02b input products."""
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
    expected = params.get("expected_input_artifact_hashes")
    if expected is not None and manifest["output_hashes"] != expected:
        raise ValueError("E02b input artifact hashes changed from the frozen configuration")
    manifest["protocol"] = str(params["protocol_path"])
    manifest["runtime"] = _runtime_provenance()
    manifest["git"] = _git_provenance()
    return (*tables, manifest)


def extract_dna_features(
    pairs: pd.DataFrame,
    input_manifest: dict[str, Any],
    invariance_manifest: dict[str, Any],
    params: dict[str, Any],
    candidate_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Extract one accepted neural DNA representation."""
    compute = fixed_representation_bakeoff.validated_compute_authorization(
        params, stage=f"dna_features:{candidate_id}"
    )
    deadline = time.monotonic() + float(compute["instance_hour_limit"]) * 3600.0
    features, coverage, summary = fixed_representation_features.extract_neural_dna_features(
        pairs,
        input_manifest,
        invariance_manifest,
        params,
        candidate_id,
        deadline_monotonic=deadline,
    )
    return features, coverage, _feature_manifest(summary, params, compute=compute)


def extract_text_features(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    input_manifest: dict[str, Any],
    params: dict[str, Any],
    candidate_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract one frozen text representation for documents and queries."""
    compute = fixed_representation_bakeoff.validated_compute_authorization(
        params, stage=f"text_features:{candidate_id}"
    )
    deadline = time.monotonic() + float(compute["instance_hour_limit"]) * 3600.0
    features, summary = fixed_representation_features.extract_text_features(
        pairs,
        queries,
        input_manifest,
        params,
        candidate_id,
        deadline_monotonic=deadline,
    )
    return features, _feature_manifest(summary, params, compute=compute)


def fit_tfidf_features(
    pairs: pd.DataFrame,
    input_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit and persist the train-only TF-IDF/SVD baseline."""
    features, vocabulary, state, summary = fixed_representation_features.fit_tfidf_dna_features(
        pairs, input_manifest, params
    )
    return features, vocabulary, state, _feature_manifest(summary, params, compute=None)


def run_alignment(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    query_states: pd.DataFrame,
    input_manifest: dict[str, Any],
    dna_tfidf: pd.DataFrame,
    dna_carbon: pd.DataFrame,
    dna_generanno: pd.DataFrame,
    dna_generator_v2: pd.DataFrame,
    dna_tfidf_manifest: dict[str, Any],
    dna_carbon_manifest: dict[str, Any],
    dna_generanno_manifest: dict[str, Any],
    dna_generator_v2_manifest: dict[str, Any],
    text_bge: pd.DataFrame,
    text_gte: pd.DataFrame,
    text_qwen: pd.DataFrame,
    text_bge_manifest: dict[str, Any],
    text_gte_manifest: dict[str, Any],
    text_qwen_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[Any, ...]:
    """Run the complete validation-only E02b factorial after artifact freezing."""
    compute = fixed_representation_bakeoff.validated_compute_authorization(
        params, stage="alignment_probe"
    )
    deadline = time.monotonic() + float(compute["instance_hour_limit"]) * 3600.0
    started = time.perf_counter()
    outputs = fixed_representation_alignment.run_factorial_alignment(
        pairs,
        queries,
        query_states,
        input_manifest,
        {
            "tfidf_6mer_svd_512": dna_tfidf,
            "carbon_500m": dna_carbon,
            "generanno_prokaryote_500m": dna_generanno,
            "generator_v2_prokaryote_1_2b": dna_generator_v2,
        },
        {
            "tfidf_6mer_svd_512": dna_tfidf_manifest,
            "carbon_500m": dna_carbon_manifest,
            "generanno_prokaryote_500m": dna_generanno_manifest,
            "generator_v2_prokaryote_1_2b": dna_generator_v2_manifest,
        },
        {
            "bge_base_en_v1_5": text_bge,
            "gte_modernbert_base": text_gte,
            "qwen3_embedding_0_6b": text_qwen,
        },
        {
            "bge_base_en_v1_5": text_bge_manifest,
            "gte_modernbert_base": text_gte_manifest,
            "qwen3_embedding_0_6b": text_qwen_manifest,
        },
        params,
        deadline_monotonic=deadline,
    )
    elapsed_seconds = time.perf_counter() - started
    *tables, report = outputs
    report["protocol"] = str(params["protocol_path"])
    report["compute_authorization"] = compute
    report["runtime"] = _runtime_provenance()
    report["git"] = _git_provenance()
    report["alignment_elapsed_seconds"] = elapsed_seconds
    report["alignment_estimated_compute_cost_usd"] = (
        elapsed_seconds / 3600.0 * float(compute["observed_instance_price_usd_per_hour"])
    )
    return (*tables, report)


def _feature_manifest(
    summary: dict[str, Any],
    params: dict[str, Any],
    *,
    compute: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "protocol_version": str(params["protocol_version"]),
        "protocol": str(params["protocol_path"]),
        **summary,
        "resolved_feature_configuration": {
            "device": str(params["device"]),
            "precision": str(params["precision"]),
            "window_overlap_fraction": float(params["window_overlap_fraction"]),
            "seed": int(params["seed"]),
        },
        "compute_authorization": compute,
        "runtime": _runtime_provenance(),
        "git": _git_provenance(),
        "decision": {
            "status": "frozen_features_complete",
            "validation_rankings_computed": False,
            "candidate_selected": False,
            "current_test_split_contaminated_before_e02b": True,
        },
    }


def _runtime_provenance() -> dict[str, Any]:
    packages = {}
    for name in (
        "accelerate",
        "huggingface-hub",
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
    ):
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
