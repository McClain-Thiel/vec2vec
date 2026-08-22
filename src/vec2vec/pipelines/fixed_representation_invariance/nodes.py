"""Kedro nodes for the Gate 1 full-panel DNA invariance check."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from vec2vec.lib import fixed_representation, fixed_representation_invariance
from vec2vec.lib.dna_encoder import FrozenDnaEncoder
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def run_invariance_check(
    panel: pd.DataFrame,
    panel_manifest: dict[str, Any],
    numerical_smoke_manifest: dict[str, Any],
    smoke_params: dict[str, Any],
    params: dict[str, Any],
    candidate_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Encode all fixed variants and report invariance without reading model outcomes."""
    recipe = fixed_representation_invariance.validate_invariance_recipe(
        panel,
        panel_manifest,
        numerical_smoke_manifest,
        smoke_params,
        params,
        candidate_id,
    )
    accepted_smoke_artifact = fixed_representation_invariance.validated_accepted_smoke_artifact(
        params, candidate_id
    )
    _, smoke_runtime_version_source = fixed_representation_invariance.numerical_smoke_recipe(
        numerical_smoke_manifest
    )
    transforms = fixed_representation_invariance.validated_transforms(params)
    compute = fixed_representation_invariance.validated_compute_authorization(params)
    _configure_determinism(int(params["seed"]))

    features: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    variant_runs: dict[str, dict[str, float | int]] = {}
    started_at_utc = datetime.now(UTC).isoformat()
    run_started = time.perf_counter()
    authorized_seconds = float(compute["instance_hour_limit"]) * 3600.0
    encoder = FrozenDnaEncoder(
        recipe,
        precision="bfloat16",
        device=str(params["device"]),
        overlap_fraction=float(params["window_overlap_fraction"]),
    )
    try:
        encoder.load()
        encoder.reset_peak_device_memory()
        if encoder.maximum_content_bp is None:
            raise RuntimeError("encoder did not resolve its maximum content window")
        for variant_id, rotation_fraction in transforms:
            variant_started = time.perf_counter()
            variant_base_pairs = 0
            for row in panel.itertuples(index=False):
                elapsed = time.perf_counter() - run_started
                if elapsed >= authorized_seconds:
                    raise TimeoutError(
                        "fixed-representation invariance run reached its authorized instance-hour "
                        f"limit before {variant_id}:{row.sequence_id}"
                    )
                sequence = fixed_representation_invariance.transform_sequence(
                    str(row.sequence),
                    variant_id=variant_id,
                    rotation_fraction=rotation_fraction,
                    reverse_complement_variant=str(params["reverse_complement_variant"]),
                )
                result = encoder.encode_sequence(str(row.sequence_id), sequence)
                variant_base_pairs += int(row.length_bp)
                features.append(
                    {
                        "sequence_id": str(row.sequence_id),
                        "source_sequence_sha256": str(row.sequence_sha256),
                        "transformed_sequence_sha256": sequence_sha256(sequence),
                        "candidate_id": candidate_id,
                        "precision": "bfloat16",
                        "variant_id": variant_id,
                        "rotation_fraction": rotation_fraction,
                        "length_bp": int(row.length_bp),
                        "length_decile": int(row.length_decile),
                        "gc_fraction": fixed_representation.gc_fraction(str(row.sequence)),
                        "embedding_dimension": int(len(result.vector)),
                        "embedding": result.vector.tolist(),
                        "embedding_sha256": fixed_representation.embedding_sha256(result.vector),
                        "elapsed_seconds": float(result.elapsed_seconds),
                    }
                )
                coverage.extend(
                    {
                        **record,
                        "candidate_id": candidate_id,
                        "precision": "bfloat16",
                        "variant_id": variant_id,
                        "rotation_fraction": rotation_fraction,
                        "sequence_length_bp": int(row.length_bp),
                        "maximum_content_bp": int(encoder.maximum_content_bp),
                    }
                    for record in result.coverage
                )
            variant_elapsed = time.perf_counter() - variant_started
            variant_runs[variant_id] = {
                "elapsed_seconds": variant_elapsed,
                "plasmids_per_second": len(panel) / variant_elapsed,
                "base_pairs_per_second": variant_base_pairs / variant_elapsed,
                "input_base_pairs": variant_base_pairs,
            }
        peak_device_memory_bytes = encoder.peak_device_memory_bytes()
    finally:
        encoder.close()
        del encoder
        gc.collect()

    features_frame = (
        pd.DataFrame(features)
        .sort_values(["variant_id", "sequence_id"], kind="stable")
        .reset_index(drop=True)
    )
    coverage_frame = (
        pd.DataFrame(coverage)
        .sort_values(["variant_id", "sequence_id", "window_index"], kind="stable")
        .reset_index(drop=True)
    )
    similarities = fixed_representation_invariance.invariance_similarities(
        features_frame, transforms
    )
    diagnostics = fixed_representation_invariance.invariance_diagnostics(
        features_frame,
        coverage_frame,
        similarities,
        params=params,
        variant_runs=variant_runs,
        peak_device_memory_bytes=peak_device_memory_bytes,
    )
    elapsed_seconds = time.perf_counter() - run_started
    observed_cost_usd = (
        elapsed_seconds / 3600.0 * float(compute["observed_instance_price_usd_per_hour"])
    )
    output_hashes = {
        "features_sha256": dataframe_content_sha256(
            features_frame,
            sort_columns=["candidate_id", "variant_id", "sequence_id"],
        ),
        "coverage_sha256": dataframe_content_sha256(
            coverage_frame,
            sort_columns=["candidate_id", "variant_id", "sequence_id", "window_index"],
        ),
        "similarities_sha256": dataframe_content_sha256(
            similarities,
            sort_columns=["candidate_id", "variant_id", "sequence_id"],
        ),
    }
    manifest = {
        "protocol_version": str(params["protocol_version"]),
        "protocol": panel_manifest["protocol"],
        "candidate_id": candidate_id,
        "candidate": recipe.model_dump(mode="json"),
        "accepted_numerical_smoke_artifact": {
            "configured_version": accepted_smoke_artifact["version"],
            "observed_panel_manifest_sha256": (
                fixed_representation_invariance.json_content_sha256(panel_manifest)
            ),
            "observed_smoke_manifest_sha256": (
                fixed_representation_invariance.json_content_sha256(numerical_smoke_manifest)
            ),
            "transformers_version_source": smoke_runtime_version_source,
        },
        "sample": panel_manifest["summary"],
        "resolved_invariance_configuration": {
            key: params[key]
            for key in (
                "expected_panel_sha256",
                "expected_rows",
                "rotation_fractions",
                "reverse_complement_variant",
                "minimum_median_transform_cosine",
                "minimum_effective_rank_fraction",
                "window_overlap_fraction",
                "seed",
                "device",
            )
        },
        "compute_authorization": compute,
        "started_at_utc": started_at_utc,
        "elapsed_seconds": elapsed_seconds,
        "observed_node_cost_usd": observed_cost_usd,
        "variant_runs": variant_runs,
        "diagnostic_summary": diagnostics,
        "output_hashes": output_hashes,
        "runtime": _runtime_provenance(),
        "git": _git_provenance(),
        "decision": {
            "status": diagnostics["status"],
            "candidate_selected": False,
            "retrieval_metrics_computed": False,
            "validation_outcomes_read": False,
            "test_rows_read": False,
        },
        "known_limitations": [
            "This run uses 512 training rows and does not compare retrieval utility.",
            "Persisted Parquet byte counts require the independent post-run read-back.",
            "The observed EC2 price excludes storage and data-transfer charges.",
        ],
    }
    return features_frame, coverage_frame, similarities, diagnostics, manifest


def _configure_determinism(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _runtime_provenance() -> dict[str, Any]:
    packages = {}
    for name in ("accelerate", "huggingface-hub", "numpy", "pandas", "torch", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware: dict[str, Any] = {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    import torch

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        hardware["cuda"] = {
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "device_total_memory_bytes": int(properties.total_memory),
            "cuda_runtime": torch.version.cuda,
        }
    return {
        "python": platform.python_version(),
        "packages": packages,
        "hardware": hardware,
        "deterministic_algorithms": True,
        "tf32_allowed": False,
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
