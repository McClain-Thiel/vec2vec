"""Pure validation and diagnostics for the Gate 1 DNA invariance check."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import fixed_representation
from vec2vec.lib.dna_encoder import EncoderRecipe
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256


def validate_invariance_recipe(
    panel: pd.DataFrame,
    panel_manifest: dict[str, Any],
    numerical_smoke_manifest: dict[str, Any],
    smoke_params: dict[str, Any],
    params: dict[str, Any],
    candidate_id: str,
) -> EncoderRecipe:
    accepted_artifact = validated_accepted_smoke_artifact(params, candidate_id)
    candidates = smoke_params["candidates"]
    if not isinstance(candidates, dict) or candidate_id not in candidates:
        raise ValueError(f"candidate {candidate_id!r} has no configured encoder recipe")
    recipe = EncoderRecipe.model_validate(candidates[candidate_id])
    if numerical_smoke_manifest.get("candidate_id") != candidate_id:
        raise ValueError(
            "numerical smoke manifest candidate does not match the requested candidate"
        )
    smoke_recipe, version_source = numerical_smoke_recipe(numerical_smoke_manifest)
    if smoke_recipe != recipe:
        raise ValueError(
            "numerical smoke manifest candidate recipe does not match the configured recipe"
        )
    if version_source != accepted_artifact["transformers_version_source"]:
        raise ValueError(
            "numerical smoke manifest Transformers-version source differs from the accepted "
            f"artifact: observed={version_source}, "
            f"expected={accepted_artifact['transformers_version_source']}"
        )
    smoke_status = numerical_smoke_manifest.get("decision", {}).get("status")
    if smoke_status != "passed_numerical_smoke":
        raise ValueError(f"candidate {candidate_id!r} did not pass its numerical smoke check")
    observed_hashes = {
        "panel_manifest_sha256": json_content_sha256(panel_manifest),
        "smoke_manifest_sha256": json_content_sha256(numerical_smoke_manifest),
    }
    expected_hashes = {
        key: accepted_artifact[key] for key in ("panel_manifest_sha256", "smoke_manifest_sha256")
    }
    if observed_hashes != expected_hashes:
        raise ValueError(
            "loaded numerical smoke artifacts do not match the accepted content hashes: "
            f"expected={expected_hashes}, observed={observed_hashes}"
        )
    _validate_panel_sequence_identity(panel)
    observed_panel_hash = dataframe_content_sha256(
        panel,
        sort_columns=["sequence_id"],
        value_columns=fixed_representation.PANEL_HASH_COLUMNS,
    )
    expected_panel_hash = str(params["expected_panel_sha256"])
    manifest_panel_hash = str(panel_manifest.get("summary", {}).get("panel_sha256"))
    if observed_panel_hash != expected_panel_hash or manifest_panel_hash != expected_panel_hash:
        raise ValueError(
            "invariance panel hash mismatch: "
            f"expected={expected_panel_hash}, observed={observed_panel_hash}, "
            f"manifest={manifest_panel_hash}"
        )
    expected_rows = int(params["expected_rows"])
    if len(panel) != expected_rows:
        raise ValueError(f"invariance panel has {len(panel)} rows, expected {expected_rows}")
    if panel["sequence_id"].duplicated().any():
        raise ValueError("invariance panel repeats a sequence_id")
    if not panel["split_grouped_v2"].eq("train").all():
        raise ValueError("invariance panel contains a non-training row")
    return recipe


def numerical_smoke_recipe(manifest: dict[str, Any]) -> tuple[EncoderRecipe, str]:
    """Load a smoke recipe and expose the source of its runtime-version pin."""
    candidate = manifest.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("numerical smoke manifest has no candidate recipe mapping")
    candidate = dict(candidate)
    version_source = "candidate.transformers_version"
    if "transformers_version" not in candidate:
        runtime_version = manifest.get("runtime", {}).get("packages", {}).get("transformers")
        if not isinstance(runtime_version, str) or not runtime_version.strip():
            raise ValueError(
                "numerical smoke manifest has no candidate or runtime Transformers version"
            )
        candidate["transformers_version"] = runtime_version
        version_source = "runtime.packages.transformers"
    try:
        return EncoderRecipe.model_validate(candidate), version_source
    except (TypeError, ValueError) as error:
        raise ValueError("numerical smoke manifest has no valid candidate recipe") from error


def validated_accepted_smoke_artifact(params: dict[str, Any], candidate_id: str) -> dict[str, str]:
    """Return one candidate's immutable accepted-smoke identity."""
    accepted_artifacts = params.get("accepted_smoke_artifacts")
    if not isinstance(accepted_artifacts, dict) or candidate_id not in accepted_artifacts:
        accepted_ids = sorted(accepted_artifacts) if isinstance(accepted_artifacts, dict) else []
        raise ValueError(
            f"candidate {candidate_id!r} has no accepted numerical smoke artifact; "
            f"expected one of {accepted_ids}"
        )
    artifact = accepted_artifacts[candidate_id]
    if not isinstance(artifact, dict):
        raise ValueError(f"accepted smoke artifact for {candidate_id!r} must be a mapping")
    required = {
        "version",
        "panel_manifest_sha256",
        "smoke_manifest_sha256",
        "transformers_version_source",
    }
    missing = required.difference(artifact)
    if missing:
        raise ValueError(
            f"accepted smoke artifact for {candidate_id!r} is missing fields: {sorted(missing)}"
        )
    version = str(artifact["version"])
    if not version.strip():
        raise ValueError(f"accepted smoke artifact version for {candidate_id!r} must not be empty")
    transformers_version_source = str(artifact["transformers_version_source"])
    allowed_version_sources = {
        "candidate.transformers_version",
        "runtime.packages.transformers",
    }
    if transformers_version_source not in allowed_version_sources:
        raise ValueError(
            f"accepted smoke artifact for {candidate_id!r} has unsupported "
            f"Transformers-version source {transformers_version_source!r}"
        )
    return {
        "version": version,
        "panel_manifest_sha256": _validated_sha256(
            artifact["panel_manifest_sha256"],
            f"accepted panel manifest for {candidate_id!r}",
        ),
        "smoke_manifest_sha256": _validated_sha256(
            artifact["smoke_manifest_sha256"],
            f"accepted smoke manifest for {candidate_id!r}",
        ),
        "transformers_version_source": transformers_version_source,
    }


def json_content_sha256(value: Any) -> str:
    """Hash JSON-compatible content with stable key and separator rules."""
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("content is not finite, JSON-compatible data") from error
    return hashlib.sha256(payload).hexdigest()


def _validated_sha256(value: Any, name: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} SHA-256 must contain 64 lowercase hexadecimal characters")
    return digest


def _validate_panel_sequence_identity(panel: pd.DataFrame) -> None:
    required = {"sequence_id", "sequence", "sequence_sha256", "length_bp"}
    missing_columns = required.difference(panel.columns)
    if missing_columns:
        raise ValueError(f"invariance panel is missing columns: {sorted(missing_columns)}")
    missing_values = panel[list(sorted(required))].isna().any(axis=1)
    if missing_values.any():
        raise ValueError(
            "invariance panel has missing sequence identity values: "
            f"rows={int(missing_values.sum())}"
        )
    for row in panel.itertuples(index=False):
        sequence = str(row.sequence)
        if len(sequence) != int(row.length_bp):
            raise ValueError(
                f"sequence {row.sequence_id} length mismatch: "
                f"recorded={row.length_bp}, observed={len(sequence)}"
            )
        if sequence_sha256(sequence) != str(row.sequence_sha256):
            raise ValueError(f"sequence {row.sequence_id} SHA-256 mismatch")


def validated_transforms(params: dict[str, Any]) -> tuple[tuple[str, float | None], ...]:
    rotations = params["rotation_fractions"]
    if not isinstance(rotations, dict) or not rotations:
        raise ValueError("rotation_fractions must be a non-empty mapping")
    transforms: list[tuple[str, float | None]] = [("original", None)]
    for variant_id, fraction_value in rotations.items():
        fraction = float(fraction_value)
        if variant_id in {"original", str(params["reverse_complement_variant"])}:
            raise ValueError(f"duplicate or reserved variant identifier {variant_id!r}")
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"rotation {variant_id!r} must be strictly between zero and one")
        transforms.append((str(variant_id), fraction))
    transforms.append((str(params["reverse_complement_variant"]), None))
    return tuple(transforms)


def validated_compute_authorization(params: dict[str, Any]) -> dict[str, Any]:
    compute = params.get("compute_authorization")
    if not isinstance(compute, dict):
        raise ValueError("paid invariance execution requires explicit compute_authorization")
    required = {
        "approval_reference",
        "region",
        "instance_type",
        "instance_hour_limit",
        "batch_instance_hour_limit",
        "observed_instance_price_usd_per_hour",
    }
    missing = required.difference(compute)
    if missing:
        raise ValueError(f"compute_authorization is missing fields: {sorted(missing)}")
    if not str(compute["instance_type"]).strip():
        raise ValueError("compute_authorization instance_type must not be empty")
    if not str(compute["region"]).strip():
        raise ValueError("compute_authorization region must not be empty")
    if not str(compute["approval_reference"]).strip():
        raise ValueError("compute_authorization approval_reference must not be empty")
    instance_hour_limit = _positive_finite_number(
        compute["instance_hour_limit"], "compute_authorization instance_hour_limit"
    )
    batch_instance_hour_limit = _positive_finite_number(
        compute["batch_instance_hour_limit"],
        "compute_authorization batch_instance_hour_limit",
    )
    observed_price = _positive_finite_number(
        compute["observed_instance_price_usd_per_hour"],
        "compute_authorization observed price",
    )
    if instance_hour_limit <= 0.0:
        raise ValueError("compute_authorization instance_hour_limit must be positive")
    if batch_instance_hour_limit <= 0.0:
        raise ValueError("compute_authorization batch_instance_hour_limit must be positive")
    if instance_hour_limit > batch_instance_hour_limit:
        raise ValueError("candidate remaining hours cannot exceed the authorized batch hours")
    if observed_price <= 0.0:
        raise ValueError("compute_authorization observed price must be positive")
    maximum_instance_cost = instance_hour_limit * observed_price
    maximum_batch_instance_cost = batch_instance_hour_limit * observed_price
    if not math.isfinite(maximum_instance_cost) or not math.isfinite(maximum_batch_instance_cost):
        raise ValueError("compute_authorization maximum cost must be finite")
    return {
        "approval_reference": str(compute["approval_reference"]),
        "region": str(compute["region"]),
        "instance_type": str(compute["instance_type"]),
        "instance_hour_limit": instance_hour_limit,
        "batch_instance_hour_limit": batch_instance_hour_limit,
        "observed_instance_price_usd_per_hour": observed_price,
        "maximum_instance_cost_usd": maximum_instance_cost,
        "maximum_batch_instance_cost_usd": maximum_batch_instance_cost,
    }


def _positive_finite_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def transform_sequence(
    sequence: str,
    *,
    variant_id: str,
    rotation_fraction: float | None,
    reverse_complement_variant: str,
) -> str:
    if variant_id == "original":
        return sequence
    if variant_id == reverse_complement_variant:
        return fixed_representation.reverse_complement(sequence)
    if rotation_fraction is None:
        raise ValueError(f"rotation variant {variant_id!r} has no fraction")
    return fixed_representation.circular_rotate(sequence, rotation_fraction)


def invariance_similarities(
    features: pd.DataFrame,
    transforms: tuple[tuple[str, float | None], ...],
) -> pd.DataFrame:
    original = features.loc[features["variant_id"].eq("original")].set_index("sequence_id")
    if original.empty or original.index.duplicated().any():
        raise ValueError("invariance features need one original vector per sequence")
    rows: list[dict[str, Any]] = []
    for variant_id, rotation_fraction in transforms:
        if variant_id == "original":
            continue
        variant = features.loc[features["variant_id"].eq(variant_id)].set_index("sequence_id")
        if set(variant.index) != set(original.index) or variant.index.duplicated().any():
            raise ValueError(f"variant {variant_id!r} does not match the original sequence set")
        for sequence_id in sorted(original.index):
            left = np.asarray(original.at[sequence_id, "embedding"], dtype=np.float64)
            right = np.asarray(variant.at[sequence_id, "embedding"], dtype=np.float64)
            if (
                left.shape != right.shape
                or not np.isfinite(left).all()
                or not np.isfinite(right).all()
            ):
                raise ValueError(f"variant {variant_id!r} has an invalid vector for {sequence_id}")
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator == 0.0:
                raise ValueError(f"variant {variant_id!r} has a zero vector for {sequence_id}")
            rows.append(
                {
                    "sequence_id": str(sequence_id),
                    "candidate_id": str(original.at[sequence_id, "candidate_id"]),
                    "variant_id": variant_id,
                    "rotation_fraction": rotation_fraction,
                    "length_bp": int(original.at[sequence_id, "length_bp"]),
                    "length_decile": int(original.at[sequence_id, "length_decile"]),
                    "cosine_to_original": float(
                        np.clip(np.dot(left, right) / denominator, -1.0, 1.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def invariance_diagnostics(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    similarities: pd.DataFrame,
    *,
    params: dict[str, Any],
    variant_runs: dict[str, dict[str, float | int]],
    peak_device_memory_bytes: int | None,
) -> dict[str, Any]:
    minimum_cosine = float(params["minimum_median_transform_cosine"])
    minimum_rank_fraction = float(params["minimum_effective_rank_fraction"])
    if not 0.0 < minimum_cosine <= 1.0:
        raise ValueError("minimum_median_transform_cosine must be in (0, 1]")
    if not 0.0 < minimum_rank_fraction <= 1.0:
        raise ValueError("minimum_effective_rank_fraction must be in (0, 1]")
    coverage_summary = coverage.groupby(["variant_id", "sequence_id"], sort=True).agg(
        newly_covered_base_count=("newly_covered_base_count", "sum"),
        sequence_length_bp=("sequence_length_bp", "first"),
        out_of_vocabulary_token_count=("out_of_vocabulary_token_count", "sum"),
        window_count=("window_index", "count"),
    )
    if len(coverage_summary) != len(features):
        raise ValueError(
            "invariance coverage does not contain one sequence summary per feature row: "
            f"coverage={len(coverage_summary)}, features={len(features)}"
        )
    coverage_pass = bool(
        coverage_summary["newly_covered_base_count"]
        .eq(coverage_summary["sequence_length_bp"])
        .all()
        and coverage_summary["out_of_vocabulary_token_count"].eq(0).all()
    )
    original = features.loc[features["variant_id"].eq("original")].sort_values(
        "sequence_id", kind="stable"
    )
    matrix = np.vstack(original["embedding"].map(lambda value: np.asarray(value, dtype=np.float64)))
    geometry = fixed_representation.representation_geometry(
        matrix,
        lengths_bp=original["length_bp"].astype(int).tolist(),
        gc_fractions=original["gc_fraction"].astype(float).tolist(),
    )
    geometry["passed_effective_rank"] = bool(
        float(geometry["effective_rank_fraction"]) >= minimum_rank_fraction
    )
    transform_summaries: dict[str, dict[str, Any]] = {}
    for variant_id, group in similarities.groupby("variant_id", sort=True):
        values = group["cosine_to_original"].to_numpy(dtype=np.float64)
        median = float(np.median(values))
        transform_summaries[str(variant_id)] = {
            "rows": int(len(values)),
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "median": median,
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "passed_median_cosine": bool(median >= minimum_cosine),
        }
    transforms_pass = all(
        bool(summary["passed_median_cosine"]) for summary in transform_summaries.values()
    )
    output_memory_bytes = {
        "features": int(features.memory_usage(index=True, deep=True).sum()),
        "coverage": int(coverage.memory_usage(index=True, deep=True).sum()),
        "similarities": int(similarities.memory_usage(index=True, deep=True).sum()),
    }
    output_memory_bytes["total"] = int(sum(output_memory_bytes.values()))
    status = (
        "passed_invariance_check"
        if coverage_pass and bool(geometry["passed_effective_rank"]) and transforms_pass
        else "rejected_by_invariance_check"
    )
    return {
        "status": status,
        "coverage_pass": coverage_pass,
        "coverage_rows": int(len(coverage_summary)),
        "minimum_median_transform_cosine": minimum_cosine,
        "minimum_effective_rank_fraction": minimum_rank_fraction,
        "geometry": geometry,
        "transforms": transform_summaries,
        "variant_runs": variant_runs,
        "peak_device_memory_bytes": peak_device_memory_bytes,
        "output_dataframe_memory_bytes": output_memory_bytes,
    }
