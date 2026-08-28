#!/usr/bin/env python3
"""Run E11 atomic classifiers and evaluate their compositional retrieval."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_weak_annotation_experiment import (
    PROJECT_ROOT,
    _array_sha256,
    _catalog_and_params,
    _file_sha256,
    _git_state,
    _load_annotations,
    _support_summary,
    _verify_frame_hash,
)

from vec2vec.lib import alignment_probe, fixed_representation_alignment, weak_annotations
from vec2vec.lib.serialization import dataframe_content_sha256, json_content_sha256

OUTPUT_DATASETS = (
    "e11_atomic_classifier_checkpoint",
    "e11_atomic_classifier_metrics",
    "e11_atomic_classifier_report",
)
COMPOSITION_OUTPUT_DATASETS = (
    "e12_compositional_metrics",
    "e12_compositional_report",
)
PRIMARY_REPRESENTATION = "calibrated_log_probability_sum"
COMPOSITION_METRICS = (
    "strict_adherence",
    "mean_clause_adherence",
    "partial_only_fraction",
    "zero_clause_fraction",
    "useful_component_fraction",
    "strict_component_diversity",
    "signed_strict_utility",
    "first_strict_rank",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("audit", "run", "validate", "composition", "validate-composition"),
        required=True,
    )
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
    params = dict(context_params["weak_annotation_classifier_experiment"])
    composition_params = dict(context_params["compositional_search_evaluation"])
    if str(params["fusion"]["primary_representation"]) != PRIMARY_REPRESENTATION:
        raise ValueError("E11 primary representation differs from the frozen implementation")
    if args.stage == "validate":
        if not args.version:
            parser.error("--version is required for validation")
        _validate_outputs(catalog, params, args.version)
        return
    if args.stage == "validate-composition":
        if not args.version:
            parser.error("--version is required for validation")
        _validate_composition_outputs(catalog, composition_params, args.version)
        return
    if args.annotations is None:
        parser.error("--annotations is required for audit, run, and composition")

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
        raise RuntimeError("E11 query table differs from the accepted E10 table")
    if args.stage == "audit":
        print(
            json.dumps(
                {
                    "status": "e11_atomic_classifier_audit_complete",
                    "queries_sha256": query_hash,
                    "query_counts": benchmark.queries["query_kind"].value_counts().to_dict(),
                    "training_positive_pairs": int(benchmark.train_verified.sum()),
                    "full_weak_negative_pairs": int(
                        benchmark.train_verified.size - benchmark.train_verified.sum()
                    ),
                    "annotation_support": _support_summary(benchmark.queries),
                },
                sort_keys=True,
            )
        )
        return
    if args.stage == "composition":
        _run_composition_stage(
            args,
            catalog,
            e10_params,
            composition_params,
            pairs,
            benchmark,
        )
        return

    authorization = _validated_authorization(args, params)
    started = time.perf_counter()
    deadline = time.monotonic() + authorization["instance_hour_limit"] * 3600.0
    checkpoint, metrics, run_report = _run(
        catalog,
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
        raise RuntimeError(f"E11 output save versions differ: {versions}")
    output_hashes = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["fit_id"]),
        "metrics_sha256": dataframe_content_sha256(
            metrics,
            sort_columns=["query_kind", "query_id", "representation", "k"],
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
            "atomic_queries": int(benchmark.train_verified.shape[0]),
            "held_out_conjunction_queries": int(
                benchmark.queries["query_kind"].eq("pair_conjunction").sum()
            ),
            "training_positive_pairs": int(benchmark.train_verified.sum()),
            "full_weak_negative_pairs": int(
                benchmark.train_verified.size - benchmark.train_verified.sum()
            ),
            "test_rows_read": False,
        },
        **run_report,
        "output_hashes": output_hashes,
        "execution": {
            **authorization,
            "maximum_cost_usd": (
                authorization["instance_hour_limit"]
                * authorization["observed_instance_price_usd_per_hour"]
            ),
            "elapsed_seconds": elapsed,
            "cost_usd": (elapsed / 3600.0 * authorization["observed_instance_price_usd_per_hour"]),
            "hardware": _hardware(),
            "git": _git_state(),
            "source_sha256": {
                "script": _file_sha256(Path(__file__)),
                "alignment_probe_library": _file_sha256(
                    PROJECT_ROOT / "src/vec2vec/lib/alignment_probe.py"
                ),
                "weak_annotations_library": _file_sha256(
                    PROJECT_ROOT / "src/vec2vec/lib/weak_annotations.py"
                ),
                "parameters": _file_sha256(PROJECT_ROOT / "conf/base/parameters_modeling_data.yml"),
            },
            "artifact_versions": versions,
        },
        "known_limitations": [
            "Uncalled annotations are noisy weak negatives, not verified biological absences.",
            "The reused exploratory validation split is not a final holdout.",
            "The direct atomic heads do not support unseen natural-language atoms.",
            "The fixed whole-plasmid representation may dilute localized sequence features.",
            "Confidence intervals resample queries, not annotation or gallery uncertainty.",
        ],
    }
    catalog.save("e11_atomic_classifier_checkpoint", checkpoint)
    catalog.save("e11_atomic_classifier_metrics", metrics)
    catalog.save("e11_atomic_classifier_report", report)
    _validate_outputs(catalog, params, next(iter(versions.values())))


def _run_composition_stage(
    args: argparse.Namespace,
    catalog: Any,
    e10_params: dict[str, Any],
    params: dict[str, Any],
    pairs: pd.DataFrame,
    benchmark: weak_annotations.WeakAnnotationBenchmark,
) -> None:
    authorization = _validated_composition_authorization(args, params)
    started = time.perf_counter()
    metrics, result = _evaluate_composition(catalog, e10_params, params, pairs, benchmark)
    elapsed = time.perf_counter() - started
    versions = {
        name: catalog.get(name).resolve_save_version() for name in COMPOSITION_OUTPUT_DATASETS
    }
    if len(set(versions.values())) != 1:
        raise RuntimeError(f"E12 output save versions differ: {versions}")
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
            "validation_rows": int(e10_params["inputs"]["validation_rows"]),
            "atomic_queries": int(benchmark.queries["query_kind"].eq("atomic").sum()),
            "conjunction_queries": int(
                benchmark.queries["query_kind"].eq("pair_conjunction").sum()
            ),
            "test_rows_read": False,
        },
        **result,
        "output_hashes": {
            "metrics_sha256": dataframe_content_sha256(metrics, sort_columns=["query_id", "k"])
        },
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
                "alignment_probe_library": _file_sha256(
                    PROJECT_ROOT / "src/vec2vec/lib/alignment_probe.py"
                ),
                "weak_annotations_library": _file_sha256(
                    PROJECT_ROOT / "src/vec2vec/lib/weak_annotations.py"
                ),
                "parameters": _file_sha256(PROJECT_ROOT / "conf/base/parameters_modeling_data.yml"),
            },
            "artifact_versions": versions,
        },
        "known_limitations": [
            "Uncalled annotations are noisy weak negatives, not verified biological absences.",
            "The reused exploratory validation split is not a final holdout.",
            "The direct atomic heads do not support unseen natural-language atoms.",
            "Sequence-similarity components measure redundancy, not functional diversity.",
            "Confidence intervals resample queries, not annotation or gallery uncertainty.",
        ],
    }
    catalog.save("e12_compositional_metrics", metrics)
    catalog.save("e12_compositional_report", report)
    _validate_composition_outputs(catalog, params, next(iter(versions.values())))


def _evaluate_composition(
    catalog: Any,
    e10_params: dict[str, Any],
    params: dict[str, Any],
    pairs: pd.DataFrame,
    benchmark: weak_annotations.WeakAnnotationBenchmark,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import wandb

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
    _verify_frame_hash(
        dna_features,
        expected=str(e10_params["inputs"]["dna_features_sha256"]),
        sort_columns=["candidate_id", "sequence_sha256"],
        name="E06 DNA features",
    )
    dna_train_raw = fixed_representation_alignment._join_embeddings(
        train, dna_features, key="sequence_sha256"
    )
    dna_validation_raw = fixed_representation_alignment._join_embeddings(
        validation, dna_features, key="sequence_sha256"
    )
    whitening = alignment_probe.Whitening.fit(
        dna_train_raw, epsilon=float(params["whitening_epsilon"])
    )
    dna_validation = whitening.transform(dna_validation_raw)
    source = _load_accepted_classifier(catalog, params)
    if (
        _array_sha256(whitening.mean)
        != source["report"]["feature_preparation"]["dna_whitening_mean_sha256"]
    ):
        raise RuntimeError("E12 recomputed whitening mean differs from E11")
    if (
        _array_sha256(whitening.matrix)
        != source["report"]["feature_preparation"]["dna_whitening_matrix_sha256"]
    ):
        raise RuntimeError("E12 recomputed whitening matrix differs from E11")

    _, calibrated = alignment_probe.atomic_logistic_scores(dna_validation, source["state"])
    atomic_queries = benchmark.queries.loc[
        benchmark.queries["query_kind"].eq("atomic")
    ].reset_index(drop=True)
    pair_queries = benchmark.queries.loc[
        benchmark.queries["query_kind"].eq("pair_conjunction")
    ].reset_index(drop=True)
    pair_scores = weak_annotations.fuse_atomic_classifier_scores(
        calibrated, calibrated, atomic_queries, pair_queries
    )[str(params["score_representation"])]
    atomic_positions = np.flatnonzero(benchmark.queries["query_kind"].eq("atomic").to_numpy())
    metrics = weak_annotations.compositional_retrieval_metrics(
        pair_scores,
        pair_queries,
        atomic_queries,
        benchmark.validation_verified[atomic_positions],
        validation[str(params["gallery_component_column"])].to_numpy(),
        cutoffs=tuple(map(int, params["cutoffs"])),
    ).sort_values(["query_id", "k"], kind="stable", ignore_index=True)
    summary = _composition_summary(metrics, params)
    expected = float(params["inputs"]["expected_signed_strict_utility_at_10"])
    observed = float(summary["primary_k"]["signed_strict_utility"]["mean"])
    if not np.isclose(observed, expected, atol=1e-12, rtol=0.0):
        raise RuntimeError(
            f"E12 signed strict utility does not reproduce E11: {observed} != {expected}"
        )

    run = wandb.init(
        project=str(params["tracking"]["project"]),
        entity=params["tracking"].get("entity"),
        group=str(params["tracking"]["group"]),
        name="e12-accepted-e11-compositional-evaluation",
        tags=list(params["tracking"]["tags"]),
        config={
            "protocol_version": str(params["protocol_version"]),
            "score_representation": str(params["score_representation"]),
            "cutoffs": list(map(int, params["cutoffs"])),
            "source_version": str(params["inputs"]["e11_version"]),
        },
        reinit="finish_previous",
    )
    try:
        for cutoff, values in summary["by_k"].items():
            for metric, value in values.items():
                run.summary[f"validation/{metric}_at_{cutoff}"] = value
        run_id, run_url = str(run.id), str(run.url)
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return metrics, {
        "source_artifacts": source["artifacts"],
        "summary": summary,
        "tracking": {"run_id": run_id, "url": run_url, "status": "complete"},
        "decision": {
            "status": "accepted_exploratory_measurement",
            "score_selected_before_evaluation": True,
            "combined_scalar_selected": False,
            "confirmatory_claim": False,
            "test_rows_read": False,
        },
    }


def _load_accepted_classifier(catalog: Any, params: dict[str, Any]) -> dict[str, Any]:
    version = str(params["inputs"]["e11_version"])
    checkpoint = catalog.load("e11_atomic_classifier_checkpoint", version=version)
    metrics = catalog.load("e11_atomic_classifier_metrics", version=version)
    report = catalog.load("e11_atomic_classifier_report", version=version)
    observed = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["fit_id"]),
        "metrics_sha256": dataframe_content_sha256(
            metrics, sort_columns=["query_kind", "query_id", "representation", "k"]
        ),
        "report_sha256": json_content_sha256(report),
    }
    expected = {
        name: str(params["inputs"][f"e11_{name}"])
        for name in ("checkpoint_sha256", "metrics_sha256", "report_sha256")
    }
    if observed != expected:
        raise RuntimeError(f"E12 source E11 artifact hashes differ: {observed}")
    row = checkpoint.iloc[0]
    state = {
        "weight": np.asarray(row["weight"], dtype=np.float32).reshape(
            int(row["weight_rows"]), int(row["weight_columns"])
        ),
        "bias": np.asarray(row["bias"], dtype=np.float32),
        "log_prior_odds": np.asarray(row["log_prior_odds"], dtype=np.float32),
    }
    for name, values in state.items():
        if _array_sha256(values) != str(row[f"{name}_sha256"]):
            raise RuntimeError(f"E12 source checkpoint {name} hash differs")
    return {
        "state": state,
        "report": report,
        "artifacts": {"version": version, **observed},
    }


def _composition_summary(metrics: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    by_k = {
        str(cutoff): {metric: float(group[metric].mean()) for metric in COMPOSITION_METRICS}
        for cutoff, group in metrics.groupby("k", sort=True)
    }
    primary = metrics.loc[metrics["k"].eq(int(params["primary_k"]))].sort_values("query_id")
    draws = int(params["bootstrap_draws"])
    generator = np.random.default_rng(int(params["bootstrap_seed"]))
    positions = generator.integers(0, len(primary), size=(draws, len(primary)))
    primary_summary = {}
    for metric in COMPOSITION_METRICS:
        values = primary[metric].to_numpy(dtype=np.float64)
        primary_summary[metric] = {
            "mean": float(values.mean()),
            "query_bootstrap_95_interval": _interval(values[positions].mean(axis=1)),
        }
    return {
        "by_k": by_k,
        "primary_k": primary_summary,
        "resampling_unit": "held_out_conjunction_query",
        "bootstrap_draws": draws,
    }


def _run(
    catalog: Any,
    e10_params: dict[str, Any],
    params: dict[str, Any],
    pairs: pd.DataFrame,
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    authorization: dict[str, Any],
    *,
    deadline: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    import wandb

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
    _verify_frame_hash(
        dna_features,
        expected=str(e10_params["inputs"]["dna_features_sha256"]),
        sort_columns=["candidate_id", "sequence_sha256"],
        name="E06 DNA features",
    )
    dna_train_raw = fixed_representation_alignment._join_embeddings(
        train, dna_features, key="sequence_sha256"
    )
    dna_validation_raw = fixed_representation_alignment._join_embeddings(
        validation, dna_features, key="sequence_sha256"
    )
    whitening = alignment_probe.Whitening.fit(
        dna_train_raw, epsilon=float(params["classifier"]["whitening_epsilon"])
    )
    dna_train = whitening.transform(dna_train_raw)
    dna_validation = whitening.transform(dna_validation_raw)
    e10_metrics = catalog.load(
        "e10_weak_annotation_metrics", version=str(params["inputs"]["e10_metrics_version"])
    )
    _verify_frame_hash(
        e10_metrics,
        expected=str(params["inputs"]["e10_metrics_sha256"]),
        sort_columns=["query_kind", "query_id", "seed", "representation", "k"],
        name="E10 metrics",
    )

    run = wandb.init(
        project=str(params["tracking"]["project"]),
        entity=params["tracking"].get("entity"),
        group=str(params["tracking"]["group"]),
        name="e11-deterministic-atomic-classifier",
        tags=list(params["tracking"]["tags"]),
        config={
            "protocol_version": str(params["protocol_version"]),
            "classifier": params["classifier"],
            "fusion": params["fusion"],
            "authorization": authorization,
        },
        reinit="finish_previous",
    )
    try:
        classifier = params["classifier"]
        state, history = alignment_probe.train_atomic_logistic_probe(
            dna_train,
            benchmark.train_verified,
            updates=int(classifier["updates"]),
            learning_rate=float(classifier["learning_rate"]),
            weight_decay=float(classifier["weight_decay"]),
            device=str(params["device"]),
            deadline_monotonic=deadline,
        )
        for row in history.itertuples(index=False):
            run.log(
                {
                    "train/loss": float(row.loss),
                    "train/weight_norm_mean": float(row.weight_norm_mean),
                    "train/weight_norm_max": float(row.weight_norm_max),
                    "train/bias_abs_max": float(row.bias_abs_max),
                },
                step=int(row.update),
            )
        stable = _is_stable(history, params["stability"])
        raw, calibrated = alignment_probe.atomic_logistic_scores(dna_validation, state)
        queries = benchmark.queries
        atomic_queries = queries.loc[queries["query_kind"].eq("atomic")].reset_index(drop=True)
        pair_queries = queries.loc[queries["query_kind"].eq("pair_conjunction")].reset_index(
            drop=True
        )
        fused = weak_annotations.fuse_atomic_classifier_scores(
            raw, calibrated, atomic_queries, pair_queries
        )
        cutoffs = tuple(map(int, params["fusion"]["cutoffs"]))
        metric_frames = [
            weak_annotations.retrieval_metrics(
                calibrated.T,
                atomic_queries,
                benchmark.validation_verified[: len(atomic_queries)],
                seed=0,
                representation="atomic_classifier",
                cutoffs=cutoffs,
            )
        ]
        for representation, scores in fused.items():
            metric_frames.append(
                weak_annotations.retrieval_metrics(
                    scores,
                    pair_queries,
                    benchmark.validation_verified[len(atomic_queries) :],
                    seed=0,
                    representation=representation,
                    cutoffs=cutoffs,
                )
            )
        metrics = pd.concat(metric_frames, ignore_index=True).sort_values(
            ["query_kind", "query_id", "representation", "k"],
            kind="stable",
            ignore_index=True,
        )
        comparison = _comparison(metrics, e10_metrics, params)
        primary_k = int(params["fusion"]["primary_k"])
        for representation, values in comparison["e11_representations"].items():
            run.summary[f"validation/{representation}/utility_at_{primary_k}"] = values[
                "utility_at_10"
            ]
        run.summary["validation/primary_minus_e10"] = comparison["primary_minus_e10"]
        run.summary["training/stable"] = stable
        run.summary["training/final_loss"] = float(history.iloc[-1]["loss"])
        run_id, run_url = str(run.id), str(run.url)
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise

    checkpoint = _checkpoint(state, history, stable=stable)
    return (
        checkpoint,
        metrics,
        {
            "feature_preparation": {
                "dna_candidate": "tfidf_6mer_svd_512",
                "dna_whitening_mean_sha256": _array_sha256(whitening.mean),
                "dna_whitening_matrix_sha256": _array_sha256(whitening.matrix),
            },
            "training": {
                "stable": stable,
                "initial_loss": float(history.iloc[0]["loss"]),
                "final_loss": float(history.iloc[-1]["loss"]),
                "maximum_weight_norm": float(history["weight_norm_max"].max()),
                "maximum_bias_abs": float(history["bias_abs_max"].max()),
            },
            "comparison": comparison,
            "tracking": {"run_id": run_id, "url": run_url, "status": "complete"},
            "decision": {
                "status": "exploratory_classifier_result" if stable else "unstable_rejected",
                "accepted_for_exploratory_iteration": stable,
                "confirmatory_claim": False,
                "test_rows_read": False,
            },
        },
    )


def _checkpoint(state: dict[str, Any], history: pd.DataFrame, *, stable: bool) -> pd.DataFrame:
    weight = np.asarray(state["weight"], dtype=np.float32)
    bias = np.asarray(state["bias"], dtype=np.float32)
    prior = np.asarray(state["log_prior_odds"], dtype=np.float32)
    return pd.DataFrame(
        [
            {
                "fit_id": "deterministic_zero_initialization",
                "objective": str(state["objective"]),
                "weight": weight.reshape(-1).tolist(),
                "weight_rows": int(weight.shape[0]),
                "weight_columns": int(weight.shape[1]),
                "weight_sha256": _array_sha256(weight),
                "bias": bias.tolist(),
                "bias_sha256": _array_sha256(bias),
                "log_prior_odds": prior.tolist(),
                "log_prior_odds_sha256": _array_sha256(prior),
                "updates": int(state["updates"]),
                "training_rows": int(state["training_rows"]),
                "positive_pairs": int(state["positive_pairs"]),
                "weak_negative_pairs": int(state["weak_negative_pairs"]),
                "initial_loss": float(history.iloc[0]["loss"]),
                "final_loss": float(history.iloc[-1]["loss"]),
                "stable": stable,
            }
        ]
    )


def _comparison(
    metrics: pd.DataFrame,
    e10_metrics: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    primary_k = int(params["fusion"]["primary_k"])
    pair = metrics.loc[metrics["query_kind"].eq("pair_conjunction") & metrics["k"].eq(primary_k)]
    e11 = pair.pivot(index="query_id", columns="representation", values="utility")
    baseline = (
        e10_metrics.loc[
            e10_metrics["query_kind"].eq("pair_conjunction")
            & e10_metrics["representation"].eq("atomic_sum")
            & e10_metrics["k"].eq(primary_k)
        ]
        .groupby("query_id", sort=True)["utility"]
        .mean()
        .rename("e10_atomic_sum")
    )
    joined = e11.join(baseline, how="inner", validate="one_to_one").sort_index()
    if len(joined) != int(params["inputs"]["pair_queries"]) or joined.isna().any(axis=None):
        raise RuntimeError("E10 and E11 paired query metrics do not align")
    draws = int(params["fusion"]["bootstrap_draws"])
    generator = np.random.default_rng(int(params["fusion"]["bootstrap_seed"]))
    positions = generator.integers(0, len(joined), size=(draws, len(joined)))
    representations = {}
    for representation in params["fusion"]["representations"]:
        values = joined[str(representation)].to_numpy()
        representations[str(representation)] = {
            "utility_at_10": float(values.mean()),
            "query_bootstrap_95_interval": _interval(values[positions].mean(axis=1)),
        }
    primary = joined[PRIMARY_REPRESENTATION].to_numpy()
    e10 = joined["e10_atomic_sum"].to_numpy()
    return {
        "e11_representations": representations,
        "e10_atomic_sum_utility_at_10": float(e10.mean()),
        "primary_representation": PRIMARY_REPRESENTATION,
        "primary_minus_e10": float(primary.mean() - e10.mean()),
        "primary_minus_e10_query_bootstrap_95_interval": _interval(
            (primary - e10)[positions].mean(axis=1)
        ),
        "resampling_unit": "held_out_conjunction_query",
        "bootstrap_draws": draws,
    }


def _validate_outputs(catalog: Any, params: dict[str, Any], version: str) -> None:
    checkpoint = catalog.load("e11_atomic_classifier_checkpoint", version=version)
    metrics = catalog.load("e11_atomic_classifier_metrics", version=version)
    report = catalog.load("e11_atomic_classifier_report", version=version)
    observed = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["fit_id"]),
        "metrics_sha256": dataframe_content_sha256(
            metrics,
            sort_columns=["query_kind", "query_id", "representation", "k"],
        ),
    }
    if observed != report["output_hashes"]:
        raise RuntimeError("E11 persisted output hashes differ")
    row = checkpoint.iloc[0]
    arrays = {
        "weight": np.asarray(row["weight"], dtype=np.float32).reshape(
            int(row["weight_rows"]), int(row["weight_columns"])
        ),
        "bias": np.asarray(row["bias"], dtype=np.float32),
        "log_prior_odds": np.asarray(row["log_prior_odds"], dtype=np.float32),
    }
    for name, values in arrays.items():
        if _array_sha256(values) != str(row[f"{name}_sha256"]):
            raise RuntimeError(f"E11 checkpoint {name} hash differs")
    expected_representations = {
        "atomic_classifier",
        *map(str, params["fusion"]["representations"]),
    }
    if set(metrics["representation"].astype(str)) != expected_representations:
        raise RuntimeError("E11 metric representation coverage is incomplete")
    if report["tracking"].get("status") != "complete":
        raise RuntimeError("E11 W&B run is incomplete")
    print(
        json.dumps(
            {
                "status": "e11_atomic_classifier_outputs_validated",
                "version": version,
                "report_sha256": json_content_sha256(report),
                "output_hashes": observed,
                "decision": report["decision"],
                "comparison": report["comparison"],
                "tracking": report["tracking"],
            },
            sort_keys=True,
        )
    )


def _validate_composition_outputs(catalog: Any, params: dict[str, Any], version: str) -> None:
    metrics = catalog.load("e12_compositional_metrics", version=version)
    report = catalog.load("e12_compositional_report", version=version)
    observed_hash = dataframe_content_sha256(metrics, sort_columns=["query_id", "k"])
    if observed_hash != str(report["output_hashes"]["metrics_sha256"]):
        raise RuntimeError("E12 persisted metric hash differs")
    expected_rows = int(params["inputs"]["pair_queries"]) * len(params["cutoffs"])
    if len(metrics) != expected_rows:
        raise RuntimeError(f"E12 metric row count differs: {len(metrics)} != {expected_rows}")
    if metrics["query_id"].nunique() != int(params["inputs"]["pair_queries"]):
        raise RuntimeError("E12 query coverage is incomplete")
    if set(metrics["k"].astype(int)) != set(map(int, params["cutoffs"])):
        raise RuntimeError("E12 cutoff coverage is incomplete")
    counts = metrics.groupby("query_id", sort=False)["k"].nunique()
    if not counts.eq(len(params["cutoffs"])).all():
        raise RuntimeError("E12 does not contain every cutoff for every query")
    bounded = (
        "strict_adherence",
        "mean_clause_adherence",
        "partial_only_fraction",
        "zero_clause_fraction",
        "useful_component_fraction",
        "strict_component_diversity",
    )
    if not np.isfinite(metrics[list(COMPOSITION_METRICS)].to_numpy()).all():
        raise RuntimeError("E12 contains non-finite metrics")
    if any(not metrics[column].between(0.0, 1.0).all() for column in bounded):
        raise RuntimeError("E12 bounded metric lies outside [0, 1]")
    if not np.allclose(
        metrics["strict_adherence"]
        + metrics["partial_only_fraction"]
        + metrics["zero_clause_fraction"],
        1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("E12 strict, partial, and zero fractions do not partition retrieval")
    if not np.allclose(
        metrics["signed_strict_utility"],
        2.0 * metrics["strict_adherence"] - 1.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("E12 signed utility is inconsistent with strict adherence")
    if metrics["first_strict_rank"].lt(1).any():
        raise RuntimeError("E12 first-strict ranks are invalid")
    summary = _composition_summary(metrics, params)
    if summary != report["summary"]:
        raise RuntimeError("E12 persisted summary differs from recalculation")
    observed_utility = float(summary["primary_k"]["signed_strict_utility"]["mean"])
    expected_utility = float(params["inputs"]["expected_signed_strict_utility_at_10"])
    if not np.isclose(observed_utility, expected_utility, atol=1e-12, rtol=0.0):
        raise RuntimeError("E12 does not reproduce the accepted E11 primary utility")
    if report["tracking"].get("status") != "complete":
        raise RuntimeError("E12 W&B run is incomplete")
    print(
        json.dumps(
            {
                "status": "e12_compositional_outputs_validated",
                "version": version,
                "report_sha256": json_content_sha256(report),
                "metrics_sha256": observed_hash,
                "primary_k": summary["primary_k"],
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
        raise ValueError(f"compute authorization differs from frozen E11 contract: {observed}")
    maximum_cost = float(args.instance_hour_limit) * float(args.price_usd_per_hour)
    if maximum_cost >= 20.0:
        raise ValueError(f"E11 authorization must remain below $20, observed ${maximum_cost:.2f}")
    return observed


def _validated_composition_authorization(
    args: argparse.Namespace, params: dict[str, Any]
) -> dict[str, Any]:
    observed = {
        "approval_reference": args.approval_reference,
        "region": args.region,
        "instance_type": args.instance_type,
        "instance_hour_limit": args.instance_hour_limit,
        "observed_instance_price_usd_per_hour": args.price_usd_per_hour,
    }
    if observed != params["compute_authorization"]:
        raise ValueError(f"compute authorization differs from frozen E12 contract: {observed}")
    maximum_cost = float(args.instance_hour_limit) * float(args.price_usd_per_hour)
    if maximum_cost >= 20.0:
        raise ValueError(f"E12 authorization must remain below $20, observed ${maximum_cost:.2f}")
    return observed


def _is_stable(history: pd.DataFrame, stability: dict[str, Any]) -> bool:
    return bool(
        np.isfinite(history.select_dtypes(include=[np.number]).to_numpy()).all()
        and history.iloc[-1]["loss"]
        <= history.iloc[0]["loss"] * float(stability["maximum_final_loss_ratio"])
        and history["weight_norm_max"].max() <= float(stability["maximum_weight_norm"])
        and history["bias_abs_max"].max() <= float(stability["maximum_bias_abs"])
    )


def _hardware() -> dict[str, Any]:
    import torch

    return {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "precision": "float32",
    }


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


if __name__ == "__main__":
    main()
