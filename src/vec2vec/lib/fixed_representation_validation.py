"""Independent read-back checks for persisted Gate 1 invariance artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import fixed_representation_invariance
from vec2vec.lib.similarity_graph import dataframe_content_sha256

_RESOLVED_CONFIGURATION_KEYS = (
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


def validate_invariance_outputs(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    similarities: pd.DataFrame,
    diagnostics: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_configuration: dict[str, Any],
) -> dict[str, Any]:
    """Recompute persisted identities, coverage, and transform cosines."""
    candidate_id = str(manifest["candidate_id"])
    configuration = _validate_frozen_configuration(
        manifest,
        expected_configuration=expected_configuration,
        candidate_id=candidate_id,
    )
    expected_rows = int(configuration["expected_rows"])
    rotation_ids = list(configuration["rotation_fractions"].keys())
    reverse_id = str(configuration["reverse_complement_variant"])
    expected_variants = ["original", *rotation_ids, reverse_id]
    _validate_candidate(features, coverage, similarities, candidate_id)
    _validate_bfloat16_only(features, coverage)
    expected_feature_rows = expected_rows * len(expected_variants)
    if len(features) != expected_feature_rows:
        raise ValueError(
            f"persisted invariance features have {len(features)} rows, "
            f"expected {expected_feature_rows}"
        )
    observed_variants = sorted(features["variant_id"].unique().tolist())
    if observed_variants != sorted(expected_variants):
        raise ValueError(
            f"persisted invariance variants changed: observed {observed_variants}, "
            f"expected {sorted(expected_variants)}"
        )
    original = features.loc[features["variant_id"].eq("original")].set_index("sequence_id")
    if len(original) != expected_rows or original.index.duplicated().any():
        raise ValueError("persisted original features do not contain the frozen sequence set")

    maximum_cosine_error = 0.0
    recomputed_cosines: dict[str, list[float]] = {}
    for variant_id in expected_variants[1:]:
        variant = features.loc[features["variant_id"].eq(variant_id)].set_index("sequence_id")
        persisted = similarities.loc[similarities["variant_id"].eq(variant_id)].set_index(
            "sequence_id"
        )
        if variant.index.duplicated().any() or persisted.index.duplicated().any():
            raise ValueError(f"persisted variant {variant_id!r} repeats a sequence_id")
        if set(variant.index) != set(original.index) or set(persisted.index) != set(original.index):
            raise ValueError(f"persisted variant {variant_id!r} changed the frozen sequence set")
        for sequence_id in sorted(original.index):
            left = np.asarray(original.at[sequence_id, "embedding"], dtype=np.float64)
            right = np.asarray(variant.at[sequence_id, "embedding"], dtype=np.float64)
            if left.shape != right.shape:
                raise ValueError(
                    f"persisted feature dimensions differ for {variant_id}:{sequence_id}"
                )
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator == 0.0 or not np.isfinite(left).all() or not np.isfinite(right).all():
                raise ValueError(
                    f"persisted feature vector is invalid for {variant_id}:{sequence_id}"
                )
            recomputed = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
            observed = float(persisted.at[sequence_id, "cosine_to_original"])
            if not np.isfinite(observed):
                raise ValueError(f"persisted cosine is not finite for {variant_id}:{sequence_id}")
            recomputed_cosines.setdefault(variant_id, []).append(recomputed)
            error = abs(recomputed - observed)
            maximum_cosine_error = max(maximum_cosine_error, error)
            if error > 1e-12:
                raise ValueError(
                    f"persisted cosine changed for {variant_id}:{sequence_id}: "
                    f"observed={observed}, recomputed={recomputed}"
                )

    coverage_summary = coverage.groupby(["variant_id", "sequence_id"], sort=True).agg(
        newly_covered_base_count=("newly_covered_base_count", "sum"),
        sequence_length_bp=("sequence_length_bp", "first"),
        out_of_vocabulary_token_count=("out_of_vocabulary_token_count", "sum"),
    )
    if len(coverage_summary) != expected_feature_rows:
        raise ValueError("persisted coverage does not contain one summary per feature row")
    coverage_pass = bool(
        coverage_summary["newly_covered_base_count"]
        .eq(coverage_summary["sequence_length_bp"])
        .all()
        and coverage_summary["out_of_vocabulary_token_count"].eq(0).all()
    )
    if not coverage_pass:
        raise ValueError("persisted coverage contains a missing base or out-of-vocabulary token")

    recomputed_acceptance = _recompute_acceptance(
        original,
        recomputed_cosines,
        coverage_pass=coverage_pass,
        configuration=configuration,
    )
    _validate_persisted_acceptance(diagnostics, manifest, recomputed_acceptance)

    recomputed_hashes = {
        "features_sha256": dataframe_content_sha256(
            features,
            sort_columns=["candidate_id", "variant_id", "sequence_id"],
        ),
        "coverage_sha256": dataframe_content_sha256(
            coverage,
            sort_columns=["candidate_id", "variant_id", "sequence_id", "window_index"],
        ),
        "similarities_sha256": dataframe_content_sha256(
            similarities,
            sort_columns=["candidate_id", "variant_id", "sequence_id"],
        ),
    }
    if recomputed_hashes != manifest["output_hashes"]:
        raise ValueError(
            f"persisted output hashes changed: observed {recomputed_hashes}, "
            f"expected {manifest['output_hashes']}"
        )
    if diagnostics != manifest["diagnostic_summary"]:
        raise ValueError("persisted diagnostics differ from the manifest summary")
    decision = manifest["decision"]
    if decision.get("candidate_selected") is not False:
        raise ValueError("invariance manifest must not select a candidate")
    for key in ("retrieval_metrics_computed", "validation_outcomes_read", "test_rows_read"):
        if decision.get(key) is not False:
            raise ValueError(f"invariance manifest must record {key}=false")
    return {
        "status": "passed_independent_readback",
        "candidate_id": candidate_id,
        "feature_rows": int(len(features)),
        "coverage_window_rows": int(len(coverage)),
        "similarity_rows": int(len(similarities)),
        "maximum_cosine_absolute_error": maximum_cosine_error,
        "coverage_pass": coverage_pass,
        "recomputed_acceptance": recomputed_acceptance,
        "output_hashes": recomputed_hashes,
        "decision": decision,
    }


def _validate_frozen_configuration(
    manifest: dict[str, Any],
    *,
    expected_configuration: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    expected_resolved = {key: expected_configuration[key] for key in _RESOLVED_CONFIGURATION_KEYS}
    observed_resolved = manifest.get("resolved_invariance_configuration")
    if observed_resolved != expected_resolved:
        raise ValueError(
            "persisted invariance configuration differs from the frozen configuration: "
            f"observed={observed_resolved}, expected={expected_resolved}"
        )
    accepted = fixed_representation_invariance.validated_accepted_smoke_artifact(
        expected_configuration, candidate_id
    )
    expected_identity = {
        "configured_version": accepted["version"],
        "observed_panel_manifest_sha256": accepted["panel_manifest_sha256"],
        "observed_smoke_manifest_sha256": accepted["smoke_manifest_sha256"],
        "transformers_version_source": accepted["transformers_version_source"],
    }
    observed_identity = manifest.get("accepted_numerical_smoke_artifact")
    if observed_identity != expected_identity:
        raise ValueError(
            "persisted accepted numerical-smoke identity differs from frozen configuration: "
            f"observed={observed_identity}, expected={expected_identity}"
        )
    return expected_resolved


def _validate_bfloat16_only(features: pd.DataFrame, coverage: pd.DataFrame) -> None:
    for name, frame in (("features", features), ("coverage", coverage)):
        if "precision" not in frame:
            raise ValueError(f"persisted invariance {name} has no precision column")
        observed = sorted(frame["precision"].drop_duplicates().astype(str).tolist())
        if observed != ["bfloat16"]:
            raise ValueError(f"persisted invariance {name} is not BF16-only: observed {observed}")


def _recompute_acceptance(
    original: pd.DataFrame,
    cosines: dict[str, list[float]],
    *,
    coverage_pass: bool,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    geometry = _recompute_geometry(original)
    effective_rank = float(geometry["effective_rank"])
    effective_rank_fraction = float(geometry["effective_rank_fraction"])
    rank_threshold = float(configuration["minimum_effective_rank_fraction"])
    cosine_threshold = float(configuration["minimum_median_transform_cosine"])
    if not np.isfinite(rank_threshold) or not 0.0 < rank_threshold <= 1.0:
        raise ValueError("persisted effective-rank threshold must be finite and in (0, 1]")
    if not np.isfinite(cosine_threshold) or not 0.0 < cosine_threshold <= 1.0:
        raise ValueError("persisted transform-cosine threshold must be finite and in (0, 1]")
    transform_medians = {
        variant_id: float(np.median(np.asarray(values, dtype=np.float64)))
        for variant_id, values in sorted(cosines.items())
    }
    transform_passes = {
        variant_id: median >= cosine_threshold for variant_id, median in transform_medians.items()
    }
    effective_rank_pass = effective_rank_fraction >= rank_threshold
    status = (
        "passed_invariance_check"
        if coverage_pass and effective_rank_pass and all(transform_passes.values())
        else "rejected_by_invariance_check"
    )
    return {
        "status": status,
        "coverage_pass": coverage_pass,
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank_fraction,
        "passed_effective_rank": effective_rank_pass,
        "transform_medians": transform_medians,
        "transform_passes": transform_passes,
        "minimum_effective_rank_fraction": rank_threshold,
        "minimum_median_transform_cosine": cosine_threshold,
        "geometry": geometry,
    }


def _recompute_geometry(original: pd.DataFrame) -> dict[str, Any]:
    """Recompute every persisted geometry field from original-sequence features."""
    ordered = original.sort_index(kind="stable")
    required_columns = {"embedding", "length_bp", "gc_fraction"}
    missing = required_columns.difference(ordered.columns)
    if missing:
        raise ValueError(f"persisted originals are missing geometry columns: {sorted(missing)}")
    matrix = np.vstack(ordered["embedding"].map(lambda value: np.asarray(value, dtype=np.float64)))
    lengths = ordered["length_bp"].to_numpy(dtype=np.float64)
    gc_fractions = ordered["gc_fraction"].to_numpy(dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("persisted originals cannot support a representation-geometry check")
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(lengths).all()
        or not np.isfinite(gc_fractions).all()
    ):
        raise ValueError("persisted originals contain non-finite geometry inputs")
    if np.any(lengths <= 0.0):
        raise ValueError("persisted originals contain a non-positive sequence length")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("persisted originals contain a zero embedding")
    normalized = matrix / norms[:, None]
    pair_rows, pair_columns = np.triu_indices(len(normalized), k=1)
    pairwise_cosines = np.sum(normalized[pair_rows] * normalized[pair_columns], axis=1)
    length_differences = np.abs(np.log2(lengths[pair_rows] / lengths[pair_columns]))
    gc_differences = np.abs(gc_fractions[pair_rows] - gc_fractions[pair_columns])
    singular_values = np.linalg.svd(matrix - matrix.mean(axis=0, keepdims=True), compute_uv=False)
    singular_value_sum = float(singular_values.sum())
    if singular_value_sum == 0.0:
        effective_rank = 0.0
    else:
        probabilities = singular_values / singular_value_sum
        positive = probabilities > 0.0
        entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
        effective_rank = float(np.exp(entropy))
    length_correlation, length_status = _pearson(pairwise_cosines, length_differences)
    gc_correlation, gc_status = _pearson(pairwise_cosines, gc_differences)
    return {
        "rows": int(len(matrix)),
        "embedding_dimension": int(matrix.shape[1]),
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / matrix.shape[1],
        "mean_pairwise_cosine": float(pairwise_cosines.mean()),
        "median_pairwise_cosine": float(np.median(pairwise_cosines)),
        "pairwise_cosine_length_difference_pearson": length_correlation,
        "pairwise_cosine_length_difference_pearson_status": length_status,
        "pairwise_cosine_gc_difference_pearson": gc_correlation,
        "pairwise_cosine_gc_difference_pearson_status": gc_status,
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str]:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("persisted geometry needs at least two paired values")
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None, "undefined_constant_input"
    return float(np.corrcoef(left, right)[0, 1]), "calculated"


def _validate_persisted_acceptance(
    diagnostics: dict[str, Any],
    manifest: dict[str, Any],
    recomputed: dict[str, Any],
) -> None:
    if diagnostics.get("coverage_pass") is not recomputed["coverage_pass"]:
        raise ValueError("persisted coverage acceptance differs from independent read-back")
    for key in ("minimum_effective_rank_fraction", "minimum_median_transform_cosine"):
        observed_threshold = float(diagnostics.get(key, float("nan")))
        if observed_threshold != recomputed[key]:
            raise ValueError(f"persisted {key} differs from independent read-back")
    geometry = diagnostics.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("persisted diagnostics have no representation geometry")
    recomputed_geometry = recomputed["geometry"]
    for key in ("rows", "embedding_dimension"):
        if geometry.get(key) != recomputed_geometry[key]:
            raise ValueError(f"persisted {key} differs from independent read-back")
    for key in (
        "effective_rank",
        "effective_rank_fraction",
        "mean_pairwise_cosine",
        "median_pairwise_cosine",
    ):
        observed = float(geometry.get(key, float("nan")))
        if not np.isclose(observed, recomputed_geometry[key], rtol=1e-12, atol=1e-12):
            raise ValueError(f"persisted {key} differs from independent read-back")
    for prefix in (
        "pairwise_cosine_length_difference_pearson",
        "pairwise_cosine_gc_difference_pearson",
    ):
        observed_status = geometry.get(f"{prefix}_status")
        expected_status = recomputed_geometry[f"{prefix}_status"]
        if observed_status != expected_status:
            raise ValueError(f"persisted {prefix} status differs from independent read-back")
        observed_value = geometry.get(prefix)
        expected_value = recomputed_geometry[prefix]
        if observed_value is None or expected_value is None:
            if observed_value is not expected_value:
                raise ValueError(f"persisted {prefix} differs from independent read-back")
        elif not np.isclose(float(observed_value), float(expected_value), rtol=1e-12, atol=1e-12):
            raise ValueError(f"persisted {prefix} differs from independent read-back")
    if geometry.get("passed_effective_rank") is not recomputed["passed_effective_rank"]:
        raise ValueError("persisted effective-rank acceptance differs from independent read-back")
    transforms = diagnostics.get("transforms")
    if not isinstance(transforms, dict) or set(transforms) != set(recomputed["transform_medians"]):
        raise ValueError("persisted transform summaries differ from independent read-back")
    for variant_id, median in recomputed["transform_medians"].items():
        summary = transforms[variant_id]
        observed_median = float(summary.get("median", float("nan")))
        if not np.isclose(observed_median, median, rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"persisted transform median differs for {variant_id!r} from read-back"
            )
        if summary.get("passed_median_cosine") is not recomputed["transform_passes"][variant_id]:
            raise ValueError(
                f"persisted transform acceptance differs for {variant_id!r} from read-back"
            )
    if diagnostics.get("status") != recomputed["status"]:
        raise ValueError("persisted diagnostic status differs from independent read-back")
    if manifest.get("decision", {}).get("status") != recomputed["status"]:
        raise ValueError("persisted decision status differs from independent read-back")


def _validate_candidate(
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    similarities: pd.DataFrame,
    candidate_id: str,
) -> None:
    for name, frame in (
        ("features", features),
        ("coverage", coverage),
        ("similarities", similarities),
    ):
        if frame.empty:
            raise ValueError(f"persisted invariance {name} must not be empty")
        observed = frame["candidate_id"].drop_duplicates().astype(str).tolist()
        if observed != [candidate_id]:
            raise ValueError(
                f"persisted invariance {name} candidate changed: "
                f"observed {observed}, expected {[candidate_id]}"
            )
