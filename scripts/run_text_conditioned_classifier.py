#!/usr/bin/env python3
"""Fit and evaluate the E13 text-conditioned calibrated atomic classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_weak_annotation_classifier import (
    _composition_summary,
    _hardware,
    _interval,
    _load_accepted_classifier,
)
from run_weak_annotation_experiment import (
    PROJECT_ROOT,
    _array_sha256,
    _catalog_and_params,
    _file_sha256,
    _git_state,
    _load_annotations,
    _verify_frame_hash,
)

from vec2vec.lib import alignment_probe, fixed_representation_alignment, weak_annotations
from vec2vec.lib.serialization import dataframe_content_sha256, json_content_sha256
from vec2vec.lib.text_encoder import FrozenTextEncoder, TextEncoderRecipe

OUTPUT_DATASETS = (
    "e13_text_conditioned_checkpoint",
    "e13_text_conditioned_atomic_metrics",
    "e13_text_conditioned_compositional_metrics",
    "e13_text_conditioned_report",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "validate"), required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--approval-reference")
    parser.add_argument("--region")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-hour-limit", type=float)
    parser.add_argument("--price-usd-per-hour", type=float)
    args = parser.parse_args()

    catalog, context_params = _catalog_and_params()
    e10_params = dict(context_params["weak_annotation_experiment"])
    params = dict(context_params["text_conditioned_classifier_experiment"])
    if args.stage == "validate":
        if not args.version:
            parser.error("--version is required for validation")
        _validate_outputs(catalog, params, args.version)
        return
    if args.annotations is None:
        parser.error("--annotations is required for the run")

    pairs = catalog.load("e06_pairs", version=str(e10_params["inputs"]["panel_version"]))
    _verify_frame_hash(
        pairs,
        expected=str(e10_params["inputs"]["pairs_sha256"]),
        sort_columns=["panel_role", "sequence_id"],
        name="E06 pairs",
    )
    annotations = _load_annotations(args.annotations, e10_params)
    benchmark = weak_annotations.build_weak_annotation_benchmark(pairs, annotations, e10_params)
    query_hash = dataframe_content_sha256(
        benchmark.queries, sort_columns=["query_kind", "query_id"]
    )
    if query_hash != str(params["inputs"]["queries_sha256"]):
        raise RuntimeError("E13 query table differs from the frozen E10/E11 table")

    authorization = _validated_authorization(args, params)
    started = time.perf_counter()
    deadline = time.monotonic() + authorization["instance_hour_limit"] * 3600.0
    checkpoint, atomic_metrics, composition_metrics, result = _run(
        catalog,
        context_params,
        e10_params,
        params,
        pairs,
        benchmark,
        authorization,
        deadline=deadline,
    )
    elapsed = time.perf_counter() - started
    versions = {name: catalog.get(name).resolve_save_version() for name in OUTPUT_DATASETS}
    if len(set(versions.values())) != 1:
        raise RuntimeError(f"E13 output save versions differ: {versions}")
    output_hashes = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["model_id"]),
        "atomic_metrics_sha256": dataframe_content_sha256(
            atomic_metrics, sort_columns=["query_id", "k"]
        ),
        "compositional_metrics_sha256": dataframe_content_sha256(
            composition_metrics, sort_columns=["query_id", "k"]
        ),
    }
    report = {
        "protocol_version": str(params["protocol_version"]),
        "weak_label_semantics": (
            "positive means the pinned annotation list contains the normalized feature; "
            "every unreported feature is a weak negative"
        ),
        "resolved_configuration": params,
        "inputs": {
            **params["inputs"],
            "annotation_file_sha256": _file_sha256(args.annotations),
        },
        "population": {
            "training_rows": int(e10_params["inputs"]["training_rows"]),
            "validation_rows": int(e10_params["inputs"]["validation_rows"]),
            "atomic_queries": int(benchmark.queries["query_kind"].eq("atomic").sum()),
            "conjunction_queries": int(
                benchmark.queries["query_kind"].eq("pair_conjunction").sum()
            ),
            "test_rows_read": False,
        },
        **result,
        "output_hashes": output_hashes,
        "execution": {
            **authorization,
            "maximum_cost_usd": (
                authorization["instance_hour_limit"]
                * authorization["observed_instance_price_usd_per_hour"]
            ),
            "elapsed_seconds": elapsed,
            "cost_usd": elapsed / 3600.0 * authorization["observed_instance_price_usd_per_hour"],
            "hardware": _hardware(),
            "git": _git_state(),
            "source_sha256": {
                "script": _file_sha256(Path(__file__)),
                "weak_annotations_library": _file_sha256(
                    PROJECT_ROOT / "src/vec2vec/lib/weak_annotations.py"
                ),
                "text_encoder_library": _file_sha256(
                    PROJECT_ROOT / "src/vec2vec/lib/text_encoder.py"
                ),
                "parameters": _file_sha256(PROJECT_ROOT / "conf/base/parameters_modeling_data.yml"),
            },
            "artifact_versions": versions,
        },
        "known_limitations": [
            "Uncalled annotations are noisy weak negatives, not verified biological absences.",
            "Only 64 atomic names supervise the text-to-classifier mapping.",
            "Held-atom evaluation tests canonical names, not paraphrases or novel biology.",
            "The reused exploratory validation split is not a final holdout.",
            "Sequence-similarity components measure redundancy, not functional diversity.",
        ],
    }
    catalog.save("e13_text_conditioned_checkpoint", checkpoint)
    catalog.save("e13_text_conditioned_atomic_metrics", atomic_metrics)
    catalog.save("e13_text_conditioned_compositional_metrics", composition_metrics)
    catalog.save("e13_text_conditioned_report", report)
    _validate_outputs(catalog, params, next(iter(versions.values())))


def _run(
    catalog: Any,
    context_params: dict[str, Any],
    e10_params: dict[str, Any],
    params: dict[str, Any],
    pairs: pd.DataFrame,
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    authorization: dict[str, Any],
    *,
    deadline: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import wandb

    atomic_queries = benchmark.queries.loc[
        benchmark.queries["query_kind"].eq("atomic")
    ].reset_index(drop=True)
    pair_queries = benchmark.queries.loc[
        benchmark.queries["query_kind"].eq("pair_conjunction")
    ].reset_index(drop=True)
    source = _load_accepted_classifier(catalog, params)
    features = _prepare_features(
        catalog,
        context_params,
        e10_params,
        params,
        pairs,
        atomic_queries,
        deadline=deadline,
    )
    for name in ("dna_whitening_mean_sha256", "dna_whitening_matrix_sha256"):
        if features[name] != source["report"]["feature_preparation"][name]:
            raise RuntimeError(f"E13 {name} differs from accepted E11")
    state = source["state"]
    targets = np.column_stack([state["weight"], state["bias"] + state["log_prior_odds"]]).astype(
        np.float64
    )
    folds = _folds(atomic_queries["query_id"].astype(str).tolist(), params)
    sample_positions = _sample_positions(
        features["train_ids"],
        count=int(params["ridge"]["training_logit_sample_rows"]),
        salt=str(params["ridge"]["sample_salt"]),
    )
    oof_targets, fold_results = _nested_oof_heads(
        features["query"],
        targets,
        features["dna_train"][sample_positions],
        folds,
        tuple(map(float, params["ridge"]["alphas"])),
    )
    oof_logits = features["dna_validation"] @ oof_targets[:, :-1].T + oof_targets[:, -1][None, :]
    cutoffs = tuple(map(int, params["cutoffs"]))
    atomic_positions = np.flatnonzero(benchmark.queries["query_kind"].eq("atomic").to_numpy())
    atomic_metrics = weak_annotations.retrieval_metrics(
        oof_logits.T,
        atomic_queries,
        benchmark.validation_verified[atomic_positions],
        seed=0,
        representation="text_conditioned_oof",
        cutoffs=cutoffs,
    ).drop(columns=["query_kind", "seed", "representation"])
    pair_scores = weak_annotations.fuse_atomic_classifier_scores(
        oof_logits, oof_logits, atomic_queries, pair_queries
    )[str(params["score_representation"])]
    validation = pairs.loc[pairs["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    composition_metrics = weak_annotations.compositional_retrieval_metrics(
        pair_scores,
        pair_queries,
        atomic_queries,
        benchmark.validation_verified[atomic_positions],
        validation[str(params["gallery_component_column"])].to_numpy(),
        cutoffs=cutoffs,
    ).sort_values(["query_id", "k"], kind="stable", ignore_index=True)
    composition_summary = _composition_summary(composition_metrics, params)
    comparison = _comparison(catalog, params, composition_metrics)
    atomic_at_k = float(
        atomic_metrics.loc[atomic_metrics["k"].eq(int(params["primary_k"])), "utility"].mean()
    )
    deployment_alpha = _modal_alpha([row["selected_alpha"] for row in fold_results])
    deployment_map, deployment_intercept = weak_annotations.fit_ridge_map(
        features["query"], targets, alpha=deployment_alpha
    )
    checkpoint = _checkpoint(
        deployment_map,
        deployment_intercept,
        features,
        params,
        alpha=deployment_alpha,
    )

    run = wandb.init(
        project=str(params["tracking"]["project"]),
        entity=params["tracking"].get("entity"),
        group=str(params["tracking"]["group"]),
        name="e13-nested-held-atom-text-conditioned-head",
        tags=list(params["tracking"]["tags"]),
        config={
            "protocol_version": str(params["protocol_version"]),
            "ridge": params["ridge"],
            "folds": int(params["outer_folds"]),
            "authorization": authorization,
        },
        reinit="finish_previous",
    )
    try:
        for row in fold_results:
            run.log(
                {
                    "train/fold": row["fold"],
                    "train/selected_alpha": row["selected_alpha"],
                    "train/outer_normalized_logit_mse": row["outer_normalized_logit_mse"],
                },
                step=int(row["fold"]),
            )
        run.summary[f"validation/atomic_utility_at_{params['primary_k']}"] = atomic_at_k
        for metric, values in composition_summary["primary_k"].items():
            run.summary[f"validation/{metric}_at_{params['primary_k']}"] = values["mean"]
        run.summary["validation/utility_minus_e10"] = comparison["e13_minus_e10"]
        run.summary["validation/utility_minus_e11"] = comparison["e13_minus_e11"]
        run_id, run_url = str(run.id), str(run.url)
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return (
        checkpoint,
        atomic_metrics,
        composition_metrics,
        {
            "source_artifacts": source["artifacts"],
            "feature_preparation": {
                key: value for key, value in features.items() if key.endswith("sha256")
            }
            | {
                "query_embedding_elapsed_seconds": features["query_embedding_elapsed_seconds"],
                "query_embedding_peak_device_memory_bytes": features[
                    "query_embedding_peak_device_memory_bytes"
                ],
            },
            "training": {
                "target": "accepted_e11_calibrated_affine_atomic_heads",
                "outer_fold_results": fold_results,
                "deployment_alpha": deployment_alpha,
                "selection_uses_validation_retrieval": False,
            },
            "atomic_utility_at_10": atomic_at_k,
            "compositional_summary": composition_summary,
            "comparison": comparison,
            "tracking": {"run_id": run_id, "url": run_url, "status": "complete"},
            "decision": {
                "status": "exploratory_held_atom_result",
                "atom_targets_held_out": True,
                "confirmatory_claim": False,
                "test_rows_read": False,
            },
        },
    )


def _prepare_features(
    catalog: Any,
    context_params: dict[str, Any],
    e10_params: dict[str, Any],
    params: dict[str, Any],
    pairs: pd.DataFrame,
    atomic_queries: pd.DataFrame,
    *,
    deadline: float,
    query_texts: list[str] | None = None,
) -> dict[str, Any]:
    train = pairs.loc[pairs["panel_role"].eq("alignment_train")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    validation = pairs.loc[pairs["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    dna_features = catalog.load(
        "e06_dna_features_tfidf_6mer_svd_512",
        version=str(e10_params["inputs"]["dna_feature_version"]),
    )
    text_features = catalog.load(
        "e06_text_features_qwen3_embedding_0_6b",
        version=str(e10_params["inputs"]["text_feature_version"]),
    )
    _verify_frame_hash(
        dna_features,
        expected=str(e10_params["inputs"]["dna_features_sha256"]),
        sort_columns=["candidate_id", "sequence_sha256"],
        name="E06 DNA features",
    )
    _verify_frame_hash(
        text_features,
        expected=str(e10_params["inputs"]["text_features_sha256"]),
        sort_columns=["candidate_id", "text_role", "text_sha256"],
        name="E06 text features",
    )
    dna_train_raw = fixed_representation_alignment._join_embeddings(
        train, dna_features, key="sequence_sha256"
    )
    dna_validation_raw = fixed_representation_alignment._join_embeddings(
        validation, dna_features, key="sequence_sha256"
    )
    documents = text_features.loc[text_features["text_role"].eq("document")]
    text_train_raw = fixed_representation_alignment._join_embeddings(
        train,
        documents,
        key="description_sha256",
        feature_key="text_sha256",
    )
    epsilon = float(params["whitening_epsilon"])
    dna_whitening = alignment_probe.Whitening.fit(dna_train_raw, epsilon=epsilon)
    text_whitening = alignment_probe.Whitening.fit(text_train_raw, epsilon=epsilon)
    recipe = TextEncoderRecipe.model_validate(
        context_params["e06_modeling_features"]["text_candidates"]["qwen3_embedding_0_6b"]
    )
    encoder = FrozenTextEncoder(
        recipe,
        precision=str(params["text_feature_precision"]),
        device=str(params["device"]),
    )
    try:
        encoded = encoder.encode(
            (
                atomic_queries["canonical_query_text"].astype(str).tolist()
                if query_texts is None
                else query_texts
            ),
            role="query",
            deadline_monotonic=deadline,
        )
        peak_memory = encoder.peak_device_memory_bytes()
    finally:
        encoder.close()
    query = text_whitening.transform(encoded.vectors).astype(np.float64)
    norms = np.linalg.norm(query, axis=1, keepdims=True)
    if (norms <= 0.0).any() or not np.isfinite(norms).all():
        raise RuntimeError("E13 query embeddings cannot be L2 normalized")
    query /= norms
    return {
        "dna_train": dna_whitening.transform(dna_train_raw).astype(np.float64),
        "dna_validation": dna_whitening.transform(dna_validation_raw).astype(np.float64),
        "text_train": text_whitening.transform(text_train_raw).astype(np.float64),
        "query": query,
        "train_ids": train["sequence_id"].astype(str).tolist(),
        "dna_whitening_mean": dna_whitening.mean.astype(np.float32),
        "dna_whitening_matrix": dna_whitening.matrix.astype(np.float32),
        "dna_whitening_mean_sha256": _array_sha256(dna_whitening.mean),
        "dna_whitening_matrix_sha256": _array_sha256(dna_whitening.matrix),
        "text_whitening_mean": text_whitening.mean.astype(np.float32),
        "text_whitening_matrix": text_whitening.matrix.astype(np.float32),
        "text_whitening_mean_sha256": _array_sha256(text_whitening.mean),
        "text_whitening_matrix_sha256": _array_sha256(text_whitening.matrix),
        "query_embeddings_sha256": _array_sha256(query),
        "query_embedding_elapsed_seconds": encoded.elapsed_seconds,
        "query_embedding_peak_device_memory_bytes": peak_memory,
        "text_encoder_recipe": recipe.model_dump(mode="json"),
    }


def _folds(query_ids: list[str], params: dict[str, Any]) -> np.ndarray:
    count = int(params["outer_folds"])
    order = sorted(
        range(len(query_ids)),
        key=lambda position: hashlib.sha256(
            f"{params['fold_salt']}:{query_ids[position]}".encode()
        ).digest(),
    )
    folds = np.empty(len(query_ids), dtype=np.int64)
    folds[np.asarray(order)] = np.arange(len(query_ids)) % count
    if set(folds.tolist()) != set(range(count)):
        raise RuntimeError("E13 deterministic fold assignment is incomplete")
    return folds


def _sample_positions(ids: list[str], *, count: int, salt: str) -> np.ndarray:
    if count > len(ids):
        raise ValueError("E13 training-logit sample exceeds the training population")
    order = sorted(
        range(len(ids)),
        key=lambda position: hashlib.sha256(f"{salt}:{ids[position]}".encode()).digest(),
    )
    return np.asarray(order[:count], dtype=np.int64)


def _nested_oof_heads(
    text: np.ndarray,
    targets: np.ndarray,
    dna_sample: np.ndarray,
    folds: np.ndarray,
    alphas: tuple[float, ...],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    predicted = np.empty_like(targets)
    fold_results = []
    for outer in sorted(set(folds.tolist())):
        outer_train = np.flatnonzero(folds != outer)
        outer_test = np.flatnonzero(folds == outer)
        candidates = []
        for alpha in alphas:
            inner_predicted = np.empty_like(targets[outer_train])
            for inner in sorted(set(folds[outer_train].tolist())):
                inner_test_local = np.flatnonzero(folds[outer_train] == inner)
                inner_train_local = np.flatnonzero(folds[outer_train] != inner)
                mapping, intercept = weak_annotations.fit_ridge_map(
                    text[outer_train][inner_train_local],
                    targets[outer_train][inner_train_local],
                    alpha=alpha,
                )
                inner_predicted[inner_test_local] = (
                    text[outer_train][inner_test_local] @ mapping + intercept
                )
            candidates.append(
                {
                    "alpha": alpha,
                    "inner_normalized_logit_mse": _normalized_logit_mse(
                        inner_predicted, targets[outer_train], dna_sample
                    ),
                }
            )
        selected = min(
            candidates, key=lambda row: (row["inner_normalized_logit_mse"], row["alpha"])
        )
        mapping, intercept = weak_annotations.fit_ridge_map(
            text[outer_train], targets[outer_train], alpha=float(selected["alpha"])
        )
        predicted[outer_test] = text[outer_test] @ mapping + intercept
        fold_results.append(
            {
                "fold": outer,
                "training_atoms": len(outer_train),
                "held_atoms": len(outer_test),
                "selected_alpha": float(selected["alpha"]),
                "inner_normalized_logit_mse": float(selected["inner_normalized_logit_mse"]),
                "outer_normalized_logit_mse": _normalized_logit_mse(
                    predicted[outer_test], targets[outer_test], dna_sample
                ),
                "candidates": candidates,
            }
        )
    if not np.isfinite(predicted).all():
        raise FloatingPointError("E13 OOF head prediction produced non-finite values")
    return predicted, fold_results


def _normalized_logit_mse(
    predicted: np.ndarray, targets: np.ndarray, dna_sample: np.ndarray
) -> float:
    predicted_logits = dna_sample @ predicted[:, :-1].T + predicted[:, -1][None, :]
    target_logits = dna_sample @ targets[:, :-1].T + targets[:, -1][None, :]
    variance = target_logits.var(axis=0)
    if (variance <= 0.0).any():
        raise RuntimeError("E13 target classifier has zero training-logit variance")
    return float((((predicted_logits - target_logits) ** 2).mean(axis=0) / variance).mean())


def _modal_alpha(values: list[float]) -> float:
    counts = Counter(values)
    maximum = max(counts.values())
    return min(alpha for alpha, count in counts.items() if count == maximum)


def _comparison(
    catalog: Any, params: dict[str, Any], composition_metrics: pd.DataFrame
) -> dict[str, Any]:
    k = int(params["primary_k"])
    e13 = composition_metrics.loc[composition_metrics["k"].eq(k)].set_index("query_id")[
        "signed_strict_utility"
    ]
    e11_metrics = catalog.load(
        "e11_atomic_classifier_metrics", version=str(params["inputs"]["e11_version"])
    )
    e11 = e11_metrics.loc[
        e11_metrics["query_kind"].eq("pair_conjunction")
        & e11_metrics["representation"].eq(str(params["score_representation"]))
        & e11_metrics["k"].eq(k)
    ].set_index("query_id")["utility"]
    e10_metrics = catalog.load(
        "e10_weak_annotation_metrics", version=str(params["inputs"]["e10_metrics_version"])
    )
    _verify_frame_hash(
        e10_metrics,
        expected=str(params["inputs"]["e10_metrics_sha256"]),
        sort_columns=["query_kind", "query_id", "seed", "representation", "k"],
        name="E10 metrics",
    )
    e10 = (
        e10_metrics.loc[
            e10_metrics["query_kind"].eq("pair_conjunction")
            & e10_metrics["representation"].eq("atomic_sum")
            & e10_metrics["k"].eq(k)
        ]
        .groupby("query_id", sort=True)["utility"]
        .mean()
    )
    joined = pd.concat(
        [e13.rename("e13"), e11.rename("e11"), e10.rename("e10")], axis=1, join="inner"
    ).sort_index()
    if len(joined) != int(params["inputs"]["pair_queries"]) or joined.isna().any(axis=None):
        raise RuntimeError("E13, E11, and E10 query metrics do not align")
    generator = np.random.default_rng(int(params["bootstrap_seed"]))
    positions = generator.integers(
        0, len(joined), size=(int(params["bootstrap_draws"]), len(joined))
    )
    result = {name: float(joined[name].mean()) for name in ("e13", "e11", "e10")}
    for reference in ("e10", "e11"):
        difference = (joined["e13"] - joined[reference]).to_numpy()
        result[f"e13_minus_{reference}"] = float(difference.mean())
        result[f"e13_minus_{reference}_query_bootstrap_95_interval"] = _interval(
            difference[positions].mean(axis=1)
        )
    return result


def _checkpoint(
    mapping: np.ndarray,
    intercept: np.ndarray,
    features: dict[str, Any],
    params: dict[str, Any],
    *,
    alpha: float,
) -> pd.DataFrame:
    mapping = np.asarray(mapping, dtype=np.float32)
    intercept = np.asarray(intercept, dtype=np.float32)
    text_mean = np.asarray(features["text_whitening_mean"], dtype=np.float32)
    text_matrix = np.asarray(features["text_whitening_matrix"], dtype=np.float32)
    return pd.DataFrame(
        [
            {
                "model_id": "qwen_to_calibrated_atomic_affine_head",
                "alpha": alpha,
                "mapping": mapping.reshape(-1).tolist(),
                "mapping_rows": mapping.shape[0],
                "mapping_columns": mapping.shape[1],
                "mapping_sha256": _array_sha256(mapping),
                "intercept": intercept.tolist(),
                "intercept_sha256": _array_sha256(intercept),
                "text_whitening_mean": text_mean.tolist(),
                "text_whitening_mean_sha256": _array_sha256(text_mean),
                "text_whitening_matrix": text_matrix.reshape(-1).tolist(),
                "text_whitening_rows": text_matrix.shape[0],
                "text_whitening_columns": text_matrix.shape[1],
                "text_whitening_matrix_sha256": _array_sha256(text_matrix),
                "text_normalization": "l2_after_training_document_whitening",
                "text_encoder_recipe_json": json.dumps(
                    features["text_encoder_recipe"], sort_keys=True, separators=(",", ":")
                ),
                "target": "accepted_e11_calibrated_affine_atomic_heads",
                "source_e11_version": str(params["inputs"]["e11_version"]),
            }
        ]
    )


def _validate_outputs(catalog: Any, params: dict[str, Any], version: str) -> None:
    checkpoint = catalog.load("e13_text_conditioned_checkpoint", version=version)
    atomic = catalog.load("e13_text_conditioned_atomic_metrics", version=version)
    composition = catalog.load("e13_text_conditioned_compositional_metrics", version=version)
    report = catalog.load("e13_text_conditioned_report", version=version)
    observed = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["model_id"]),
        "atomic_metrics_sha256": dataframe_content_sha256(atomic, sort_columns=["query_id", "k"]),
        "compositional_metrics_sha256": dataframe_content_sha256(
            composition, sort_columns=["query_id", "k"]
        ),
    }
    if observed != report["output_hashes"]:
        raise RuntimeError("E13 persisted output hashes differ")
    expected_atomic = int(params["inputs"]["atomic_queries"]) * len(params["cutoffs"])
    expected_composition = int(params["inputs"]["pair_queries"]) * len(params["cutoffs"])
    if len(atomic) != expected_atomic or len(composition) != expected_composition:
        raise RuntimeError("E13 persisted metric row counts differ")
    row = checkpoint.iloc[0]
    arrays = {
        "mapping": np.asarray(row["mapping"], dtype=np.float32).reshape(
            int(row["mapping_rows"]), int(row["mapping_columns"])
        ),
        "intercept": np.asarray(row["intercept"], dtype=np.float32),
        "text_whitening_mean": np.asarray(row["text_whitening_mean"], dtype=np.float32),
        "text_whitening_matrix": np.asarray(row["text_whitening_matrix"], dtype=np.float32).reshape(
            int(row["text_whitening_rows"]), int(row["text_whitening_columns"])
        ),
    }
    for name, values in arrays.items():
        if _array_sha256(values) != str(row[f"{name}_sha256"]):
            raise RuntimeError(f"E13 checkpoint {name} hash differs")
    if _composition_summary(composition, params) != report["compositional_summary"]:
        raise RuntimeError("E13 compositional summary differs from recalculation")
    if report["tracking"].get("status") != "complete":
        raise RuntimeError("E13 W&B run is incomplete")
    if len(report["training"]["outer_fold_results"]) != int(params["outer_folds"]):
        raise RuntimeError("E13 outer-fold coverage is incomplete")
    print(
        json.dumps(
            {
                "status": "e13_text_conditioned_outputs_validated",
                "version": version,
                "report_sha256": json_content_sha256(report),
                "output_hashes": observed,
                "atomic_utility_at_10": report["atomic_utility_at_10"],
                "primary_k": report["compositional_summary"]["primary_k"],
                "comparison": report["comparison"],
                "tracking": report["tracking"],
            },
            sort_keys=True,
        )
    )


def _validated_authorization(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "approval_reference": args.approval_reference,
        "region": args.region,
        "instance_type": args.instance_type,
        "instance_hour_limit": args.instance_hour_limit,
        "observed_instance_price_usd_per_hour": args.price_usd_per_hour,
    }
    if observed != params["compute_authorization"]:
        raise ValueError(f"compute authorization differs from frozen E13 contract: {observed}")
    maximum_cost = float(args.instance_hour_limit) * float(args.price_usd_per_hour)
    if maximum_cost >= 20.0:
        raise ValueError(f"E13 authorization must remain below $20, observed ${maximum_cost:.2f}")
    return observed


if __name__ == "__main__":
    main()
