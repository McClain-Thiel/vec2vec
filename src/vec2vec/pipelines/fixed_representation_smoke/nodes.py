"""Kedro nodes for frozen-panel DNA numerical smoke checks."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import platform
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import fixed_representation
from vec2vec.lib.dna_encoder import EncoderRecipe, FrozenDnaEncoder
from vec2vec.lib.similarity_graph import dataframe_content_sha256


def build_smoke_panel(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze the training-only invariance and numerical smoke panels."""
    panel, summary = fixed_representation.build_fixed_representation_panels(
        retrieval,
        split_mapping,
        expected_population_sha256=str(params["expected_input_population_sha256"]),
        invariance_rows=int(params["invariance_rows"]),
        numerical_smoke_rows=int(params["numerical_smoke_rows"]),
        length_strata=int(params["length_strata"]),
        selection_salt=str(params["selection_salt"]),
    )
    manifest = {
        "protocol_version": str(params["protocol_version"]),
        "protocol": (
            "studies/set_valued_compositional_embeddings/experiments/"
            "E02_fixed_representation_bakeoff.md"
        ),
        "input_retrieval_version": str(params["input_retrieval_version"]),
        "input_split_version": str(params["input_split_version"]),
        "resolved_sampling_configuration": {
            key: params[key]
            for key in (
                "expected_input_population_sha256",
                "invariance_rows",
                "numerical_smoke_rows",
                "length_strata",
                "selection_salt",
            )
        },
        "summary": summary,
        "git": _git_provenance(),
        "test_rows_read": False,
        "validation_outcomes_read": False,
    }
    return panel, manifest


def run_numerical_smoke(
    panel: pd.DataFrame,
    panel_manifest: dict[str, Any],
    params: dict[str, Any],
    candidate_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run bfloat16 and float32 extraction for one pinned DNA candidate."""
    candidates = params["candidates"]
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        raise ValueError(
            f"unknown fixed-representation candidate {candidate_id!r}; "
            f"expected one of {sorted(candidates)}"
        )
    recipe = EncoderRecipe.model_validate(candidates[candidate_id])
    smoke = panel.loc[panel["in_numerical_smoke_panel"]].sort_values(
        ["length_decile", "length_bp", "selection_sha256"], kind="stable"
    )
    expected_rows = int(params["numerical_smoke_rows"])
    if len(smoke) != expected_rows:
        raise ValueError(f"numerical smoke panel has {len(smoke)} rows, expected {expected_rows}")
    _configure_determinism(int(params["seed"]))

    features: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    precision_runs: dict[str, dict[str, Any]] = {}
    started_at_utc = datetime.now(UTC).isoformat()
    run_started = time.perf_counter()
    for precision in ("bfloat16", "float32"):
        encoder = FrozenDnaEncoder(
            recipe,
            precision=precision,
            device=str(params["device"]),
            overlap_fraction=float(params["window_overlap_fraction"]),
        )
        precision_started = time.perf_counter()
        try:
            encoder.load()
            encoder.reset_peak_device_memory()
            if encoder.maximum_content_bp is None:
                raise RuntimeError("encoder did not resolve its maximum content window")
            for row in smoke.itertuples(index=False):
                result = encoder.encode_sequence(str(row.sequence_id), str(row.sequence))
                features.append(
                    {
                        "sequence_id": str(row.sequence_id),
                        "sequence_sha256": str(row.sequence_sha256),
                        "candidate_id": candidate_id,
                        "precision": precision,
                        "length_bp": int(row.length_bp),
                        "length_decile": int(row.length_decile),
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
                        "precision": precision,
                        "sequence_length_bp": int(row.length_bp),
                        "maximum_content_bp": int(encoder.maximum_content_bp),
                    }
                    for record in result.coverage
                )
            precision_elapsed = time.perf_counter() - precision_started
            precision_runs[precision] = {
                "elapsed_seconds": precision_elapsed,
                "plasmids_per_second": len(smoke) / precision_elapsed,
                "base_pairs_per_second": int(smoke["length_bp"].sum()) / precision_elapsed,
                "peak_device_memory_bytes": encoder.peak_device_memory_bytes(),
                "maximum_content_bp": int(encoder.maximum_content_bp),
            }
        finally:
            encoder.close()
            del encoder
            gc.collect()

    features_frame = (
        pd.DataFrame(features)
        .sort_values(["precision", "sequence_id"], kind="stable")
        .reset_index(drop=True)
    )
    coverage_frame = (
        pd.DataFrame(coverage)
        .sort_values(["precision", "sequence_id", "window_index"], kind="stable")
        .reset_index(drop=True)
    )
    diagnostics = _numerical_diagnostics(
        features_frame,
        coverage_frame,
        minimum_cosine=float(params["minimum_bfloat16_float32_cosine"]),
    )
    status = (
        "passed_numerical_smoke"
        if diagnostics["passed_numerical_smoke"].all()
        else "rejected_by_numerical_smoke"
    )
    output_hashes = {
        "features_sha256": dataframe_content_sha256(
            features_frame,
            sort_columns=["candidate_id", "precision", "sequence_id"],
        ),
        "coverage_sha256": dataframe_content_sha256(
            coverage_frame,
            sort_columns=["candidate_id", "precision", "sequence_id", "window_index"],
        ),
        "diagnostics_sha256": dataframe_content_sha256(
            diagnostics,
            sort_columns=["candidate_id", "sequence_id"],
        ),
    }
    manifest = {
        "protocol_version": str(params["protocol_version"]),
        "protocol": panel_manifest["protocol"],
        "candidate_id": candidate_id,
        "candidate": recipe.model_dump(mode="json"),
        "sample": panel_manifest["summary"],
        "input_retrieval_version": panel_manifest["input_retrieval_version"],
        "input_split_version": panel_manifest["input_split_version"],
        "resolved_smoke_configuration": {
            key: params[key]
            for key in (
                "device",
                "window_overlap_fraction",
                "minimum_bfloat16_float32_cosine",
                "seed",
                "instance_type",
                "instance_hour_limit",
                "observed_instance_price_usd_per_hour",
            )
        },
        "started_at_utc": started_at_utc,
        "elapsed_seconds": time.perf_counter() - run_started,
        "precision_runs": precision_runs,
        "diagnostic_summary": {
            "status": status,
            "rows": int(len(diagnostics)),
            "minimum_bfloat16_float32_cosine": float(diagnostics["bfloat16_float32_cosine"].min()),
            "coverage_failure_rows": int(diagnostics["coverage_pass"].eq(False).sum()),
            "non_finite_rows": int(diagnostics["finite_pass"].eq(False).sum()),
        },
        "output_hashes": output_hashes,
        "runtime": _runtime_provenance(),
        "git": _git_provenance(),
        "decision": {
            "status": status,
            "candidate_selected": False,
            "retrieval_metrics_computed": False,
            "validation_outcomes_read": False,
            "test_rows_read": False,
        },
        "known_limitations": [
            "This run checks 32 training rows and cannot establish full-panel invariance.",
            "This run does not compare retrieval utility or select an encoder.",
            "The observed EC2 price excludes storage and data-transfer charges.",
        ],
    }
    return features_frame, coverage_frame, diagnostics, manifest


def _numerical_diagnostics(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    minimum_cosine: float,
) -> pd.DataFrame:
    if not 0.0 < minimum_cosine <= 1.0:
        raise ValueError("minimum_cosine must be in (0, 1]")
    rows: list[dict[str, Any]] = []
    candidate_ids = features["candidate_id"].unique().tolist()
    if len(candidate_ids) != 1:
        raise ValueError("numerical diagnostics require exactly one candidate")
    for sequence_id, group in features.groupby("sequence_id", sort=True):
        by_precision = {row.precision: row for row in group.itertuples(index=False)}
        if set(by_precision) != {"bfloat16", "float32"}:
            raise ValueError(f"sequence {sequence_id} is missing one precision")
        bfloat16 = np.asarray(by_precision["bfloat16"].embedding, dtype=np.float64)
        float32 = np.asarray(by_precision["float32"].embedding, dtype=np.float64)
        finite_pass = bool(np.isfinite(bfloat16).all() and np.isfinite(float32).all())
        if bfloat16.shape != float32.shape:
            raise ValueError(f"sequence {sequence_id} embedding dimensions differ by precision")
        denominator = float(np.linalg.norm(bfloat16) * np.linalg.norm(float32))
        cosine = (
            float(np.clip(np.dot(bfloat16, float32) / denominator, -1.0, 1.0))
            if finite_pass and denominator > 0.0
            else float("nan")
        )
        sequence_coverage = coverage.loc[coverage["sequence_id"].eq(sequence_id)]
        coverage_by_precision = sequence_coverage.groupby("precision", sort=True).agg(
            newly_covered_base_count=("newly_covered_base_count", "sum"),
            sequence_length_bp=("sequence_length_bp", "first"),
            out_of_vocabulary_token_count=("out_of_vocabulary_token_count", "sum"),
            window_count=("window_index", "count"),
        )
        if set(coverage_by_precision.index) != {"bfloat16", "float32"}:
            raise ValueError(f"sequence {sequence_id} coverage is missing one precision")
        coverage_pass = bool(
            coverage_by_precision["newly_covered_base_count"]
            .eq(coverage_by_precision["sequence_length_bp"])
            .all()
            and coverage_by_precision["out_of_vocabulary_token_count"].eq(0).all()
        )
        numerical_pass = finite_pass and coverage_pass and cosine >= minimum_cosine
        rows.append(
            {
                "sequence_id": str(sequence_id),
                "candidate_id": str(candidate_ids[0]),
                "length_bp": int(by_precision["bfloat16"].length_bp),
                "length_decile": int(by_precision["bfloat16"].length_decile),
                "embedding_dimension": int(by_precision["bfloat16"].embedding_dimension),
                "bfloat16_float32_cosine": cosine,
                "finite_pass": finite_pass,
                "coverage_pass": coverage_pass,
                "bfloat16_window_count": int(coverage_by_precision.loc["bfloat16", "window_count"]),
                "float32_window_count": int(coverage_by_precision.loc["float32", "window_count"]),
                "passed_numerical_smoke": numerical_pass,
            }
        )
    return pd.DataFrame(rows)


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
