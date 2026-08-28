#!/usr/bin/env python3
"""Audit, run, and validate E10 weak-annotation natural-parameter retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import alignment_probe, fixed_representation_alignment, weak_annotations
from vec2vec.lib.serialization import dataframe_content_sha256, json_content_sha256
from vec2vec.lib.text_encoder import FrozenTextEncoder, TextEncoderRecipe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QWEN = "qwen3_embedding_0_6b"
TFIDF = "tfidf_6mer_svd_512"
OUTPUT_DATASETS = (
    "e10_weak_annotation_queries",
    "e10_weak_annotation_checkpoints",
    "e10_weak_annotation_metrics",
    "e10_weak_annotation_report",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "run", "validate"), required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--version")
    parser.add_argument("--approval-reference")
    parser.add_argument("--region")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-hour-limit", type=float)
    parser.add_argument("--price-usd-per-hour", type=float)
    args = parser.parse_args()

    catalog, context_params = _catalog_and_params()
    params = dict(context_params["weak_annotation_experiment"])
    if args.stage == "validate":
        if not args.version:
            parser.error("--version is required for validation")
        _validate_outputs(catalog, params, args.version)
        return
    if args.annotations is None:
        parser.error("--annotations is required for audit and run")

    pairs = catalog.load("e06_pairs", version=str(params["inputs"]["panel_version"]))
    _verify_frame_hash(
        pairs,
        expected=str(params["inputs"]["pairs_sha256"]),
        sort_columns=["panel_role", "sequence_id"],
        name="E06 pairs",
    )
    annotation_rows = _load_annotations(args.annotations, params)
    benchmark = weak_annotations.build_weak_annotation_benchmark(pairs, annotation_rows, params)
    query_hash = dataframe_content_sha256(
        benchmark.queries, sort_columns=["query_kind", "query_id"]
    )
    expected_query_hash = params.get("expected_query_sha256")
    if args.stage == "audit":
        print(
            json.dumps(
                {
                    "status": "e10_weak_annotation_audit_complete",
                    "queries_sha256": query_hash,
                    "query_counts": benchmark.queries["query_kind"].value_counts().to_dict(),
                    "training_positive_pairs": int(benchmark.train_verified.sum()),
                    "sampled_weak_negative_pairs": int(
                        (benchmark.train_known & ~benchmark.train_verified).sum()
                    ),
                    "validation_positive_pairs": int(benchmark.validation_verified.sum()),
                    "annotation_support": _support_summary(benchmark.queries),
                },
                sort_keys=True,
            )
        )
        return
    if not expected_query_hash or query_hash != str(expected_query_hash):
        raise RuntimeError(
            f"E10 query table is not frozen: expected {expected_query_hash}, observed {query_hash}"
        )

    authorization = _validated_authorization(args, params)
    started = time.perf_counter()
    deadline = time.monotonic() + authorization["instance_hour_limit"] * 3600.0
    checkpoints, metrics, run_report = _run(
        catalog,
        context_params,
        params,
        pairs,
        benchmark,
        authorization,
        deadline=deadline,
    )
    elapsed = time.perf_counter() - started
    versions = {name: catalog.get(name).resolve_save_version() for name in OUTPUT_DATASETS}
    if len(set(versions.values())) != 1:
        raise RuntimeError(f"E10 output save versions differ: {versions}")
    output_hashes = {
        "queries_sha256": query_hash,
        "checkpoints_sha256": dataframe_content_sha256(checkpoints, sort_columns=["seed"]),
        "metrics_sha256": dataframe_content_sha256(
            metrics,
            sort_columns=["query_kind", "query_id", "seed", "representation", "k"],
        ),
    }
    report = {
        "protocol_version": str(params["protocol_version"]),
        "weak_label_semantics": (
            "positive means the pinned annotation list contains the normalized feature; "
            "weak negative means the same pipeline did not report it"
        ),
        "resolved_configuration": params,
        "inputs": {
            **params["inputs"],
            "annotation_file_sha256": _file_sha256(args.annotations),
        },
        "population": {
            "training_rows": int(params["inputs"]["training_rows"]),
            "validation_rows": int(params["inputs"]["validation_rows"]),
            "atomic_queries": int(benchmark.queries["query_kind"].eq("atomic").sum()),
            "held_out_conjunction_queries": int(
                benchmark.queries["query_kind"].eq("pair_conjunction").sum()
            ),
            "training_positive_pairs": int(benchmark.train_verified.sum()),
            "sampled_weak_negative_pairs": int(
                (benchmark.train_known & ~benchmark.train_verified).sum()
            ),
            "validation_positive_pairs": int(benchmark.validation_verified.sum()),
            "test_rows_read": False,
        },
        "annotation_support": _support_summary(benchmark.queries),
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
                "weak_annotations_library": _file_sha256(
                    PROJECT_ROOT / "src/vec2vec/lib/weak_annotations.py"
                ),
                "parameters": _file_sha256(PROJECT_ROOT / "conf/base/parameters_modeling_data.yml"),
            },
            "artifact_versions": versions,
        },
        "known_limitations": [
            "Uncalled annotations are noisy weak negatives, not verified biological absences.",
            "Exact normalized names are atoms; biological aliases and feature hierarchies remain.",
            "The reused exploratory validation split is not a final holdout.",
            "Confidence intervals resample queries, not annotation or gallery uncertainty.",
            "Canonical query text does not measure paraphrase robustness.",
        ],
    }
    catalog.save("e10_weak_annotation_queries", benchmark.queries)
    catalog.save("e10_weak_annotation_checkpoints", checkpoints)
    catalog.save("e10_weak_annotation_metrics", metrics)
    catalog.save("e10_weak_annotation_report", report)
    version = next(iter(versions.values()))
    _validate_outputs(catalog, params, version)


def _run(
    catalog: Any,
    context_params: dict[str, Any],
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
        version=str(params["inputs"]["dna_feature_version"]),
    )
    text_features = catalog.load(
        "e06_text_features_qwen3_embedding_0_6b",
        version=str(params["inputs"]["text_feature_version"]),
    )
    _verify_frame_hash(
        dna_features,
        expected=str(params["inputs"]["dna_features_sha256"]),
        sort_columns=["candidate_id", "sequence_sha256"],
        name="E06 DNA features",
    )
    _verify_frame_hash(
        text_features,
        expected=str(params["inputs"]["text_features_sha256"]),
        sort_columns=["candidate_id", "text_role", "text_sha256"],
        name="E06 text features",
    )
    dna_train, dna_validation, text_train, text_validation, query_text, whitening = (
        _prepare_features(
            train,
            validation,
            benchmark.queries,
            dna_features,
            text_features,
            context_params,
            params,
            deadline=deadline,
        )
    )
    # The natural-parameter objective uses DNA and query text only. Document embeddings remain
    # part of train-fitted whitening provenance and preserve the accepted E06 feature geometry.
    del text_train, text_validation
    atomic_positions = np.flatnonzero(benchmark.queries["query_kind"].eq("atomic").to_numpy())
    pair_positions = np.flatnonzero(
        benchmark.queries["query_kind"].eq("pair_conjunction").to_numpy()
    )
    atomic_queries = benchmark.queries.iloc[atomic_positions].reset_index(drop=True)
    pair_queries = benchmark.queries.iloc[pair_positions].reset_index(drop=True)
    log_mass = _component_log_mass(train)
    probe = params["probe"]
    stability = params["stability"]
    checkpoints: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    stability_rows: list[dict[str, Any]] = []
    tracking: list[dict[str, str]] = []
    for seed in map(int, probe["seeds"]):
        run = wandb.init(
            project=str(params["tracking"]["project"]),
            entity=params["tracking"].get("entity"),
            group=str(params["tracking"]["group"]),
            name=f"e10-weak-annotations-seed-{seed}",
            tags=list(params["tracking"]["tags"]),
            config={
                "protocol_version": str(params["protocol_version"]),
                "seed": seed,
                "probe": probe,
                "weak_negatives": params["weak_negatives"],
                "inputs": params["inputs"],
                "authorization": authorization,
            },
            reinit="finish_previous",
        )
        try:
            state, history = alignment_probe.train_maximum_entropy_probe(
                dna_train,
                query_text[atomic_positions],
                benchmark.train_verified,
                benchmark.train_known,
                log_mass,
                base_measure="uniform_v2_component",
                seed=seed,
                projection_dimension=int(probe["projection_dimension"]),
                updates=int(probe["updates"]),
                learning_rate=float(probe["learning_rate"]),
                weight_decay=float(probe["weight_decay"]),
                temperature=float(probe["temperature"]),
                device=str(params["device"]),
                record_initial=True,
                deadline_monotonic=deadline,
            )
            for row in history.itertuples(index=False):
                run.log(
                    {
                        "train/loss": float(row.loss),
                        "train/query_norm_mean": float(row.query_norm_mean),
                        "train/query_norm_max": float(row.query_norm_max),
                        "train/sequence_norm_sample_mean": float(row.sequence_norm_sample_mean),
                        "train/sequence_norm_sample_max": float(row.sequence_norm_sample_max),
                    },
                    step=int(row.update),
                )
            stable = _is_stable(history, stability)
            stability_row = {
                "seed": seed,
                "initial_loss": float(history.iloc[0]["loss"]),
                "final_loss": float(history.iloc[-1]["loss"]),
                "maximum_loss": float(history["loss"].max()),
                "maximum_query_norm": float(history["query_norm_max"].max()),
                "maximum_sequence_norm": float(history["sequence_norm_sample_max"].max()),
                "stable": stable,
            }
            stability_rows.append(stability_row)
            checkpoints.append(_checkpoint(state, history.iloc[-1], stable=stable))

            gallery_vectors = alignment_probe.project_unnormalized(
                dna_validation, state["sequence_head"]
            )
            direct_vectors = alignment_probe.project_unnormalized(query_text, state["text_head"])
            atomic_vectors = direct_vectors[atomic_positions]
            pair_sums = _atomic_sums(atomic_queries, pair_queries, atomic_vectors)
            cutoffs = tuple(map(int, probe["cutoffs"]))
            atomic_scores = alignment_probe.natural_parameter_scores(
                atomic_vectors,
                gallery_vectors,
                _component_log_mass(validation),
                temperature=float(probe["temperature"]),
            )
            pair_direct_scores = alignment_probe.natural_parameter_scores(
                direct_vectors[pair_positions],
                gallery_vectors,
                _component_log_mass(validation),
                temperature=float(probe["temperature"]),
            )
            pair_sum_scores = alignment_probe.natural_parameter_scores(
                pair_sums,
                gallery_vectors,
                _component_log_mass(validation),
                temperature=float(probe["temperature"]),
            )
            seed_metrics = pd.concat(
                [
                    weak_annotations.retrieval_metrics(
                        atomic_scores,
                        atomic_queries,
                        benchmark.validation_verified[atomic_positions],
                        seed=seed,
                        representation="direct_text",
                        cutoffs=cutoffs,
                    ),
                    weak_annotations.retrieval_metrics(
                        pair_direct_scores,
                        pair_queries,
                        benchmark.validation_verified[pair_positions],
                        seed=seed,
                        representation="direct_text",
                        cutoffs=cutoffs,
                    ),
                    weak_annotations.retrieval_metrics(
                        pair_sum_scores,
                        pair_queries,
                        benchmark.validation_verified[pair_positions],
                        seed=seed,
                        representation="atomic_sum",
                        cutoffs=cutoffs,
                    ),
                ],
                ignore_index=True,
            )
            metric_frames.append(seed_metrics)
            primary = seed_metrics.loc[seed_metrics["k"].eq(int(params["primary_k"]))]
            for (query_kind, representation), group in primary.groupby(
                ["query_kind", "representation"], sort=True
            ):
                prefix = f"validation/{query_kind}/{representation}"
                run.summary[f"{prefix}/utility_at_{params['primary_k']}"] = float(
                    group["utility"].mean()
                )
                run.summary[f"{prefix}/positive_fraction_at_{params['primary_k']}"] = float(
                    group["positive_fraction"].mean()
                )
            run.summary["training/stable"] = stable
            run.summary["training/final_loss"] = stability_row["final_loss"]
            run_id, run_url = str(run.id), str(run.url)
            run.finish(exit_code=0)
            tracking.append(
                {"seed": str(seed), "run_id": run_id, "url": run_url, "status": "complete"}
            )
        except BaseException:
            run.finish(exit_code=1)
            raise

    checkpoint_table = pd.concat(checkpoints, ignore_index=True).sort_values(
        "seed", kind="stable", ignore_index=True
    )
    metrics = pd.concat(metric_frames, ignore_index=True).sort_values(
        ["query_kind", "query_id", "seed", "representation", "k"],
        kind="stable",
        ignore_index=True,
    )
    primary = metrics.loc[metrics["k"].eq(int(params["primary_k"]))]
    atomic = primary.loc[
        primary["query_kind"].eq("atomic") & primary["representation"].eq("direct_text")
    ]
    comparison = weak_annotations.paired_query_bootstrap(
        metrics,
        k=int(params["primary_k"]),
        draws=int(probe["bootstrap_draws"]),
        seed=int(probe["bootstrap_seed"]),
    )
    comparison["atomic_direct_text"] = float(atomic["utility"].mean())
    all_stable = all(row["stable"] for row in stability_rows)
    return (
        checkpoint_table,
        metrics,
        {
            "feature_preparation": whitening,
            "stability": {"all_seeds_stable": all_stable, "runs": stability_rows},
            "comparison": comparison,
            "tracking": tracking,
            "decision": {
                "status": "exploratory_weak_label_result" if all_stable else "unstable_rejected",
                "accepted_for_exploratory_iteration": all_stable,
                "confirmatory_claim": False,
                "test_rows_read": False,
            },
        },
    )


def _prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    queries: pd.DataFrame,
    dna_features: pd.DataFrame,
    text_features: pd.DataFrame,
    context_params: dict[str, Any],
    params: dict[str, Any],
    *,
    deadline: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dna_train_raw = fixed_representation_alignment._join_embeddings(
        train,
        dna_features,
        key="sequence_sha256",
    )
    dna_validation_raw = fixed_representation_alignment._join_embeddings(
        validation,
        dna_features,
        key="sequence_sha256",
    )
    document_features = text_features.loc[text_features["text_role"].eq("document")]
    text_train_raw = fixed_representation_alignment._join_embeddings(
        train,
        document_features,
        key="description_sha256",
        feature_key="text_sha256",
    )
    text_validation_raw = fixed_representation_alignment._join_embeddings(
        validation,
        document_features,
        key="description_sha256",
        feature_key="text_sha256",
    )
    epsilon = float(params["probe"]["whitening_epsilon"])
    dna_whitening = alignment_probe.Whitening.fit(dna_train_raw, epsilon=epsilon)
    text_whitening = alignment_probe.Whitening.fit(text_train_raw, epsilon=epsilon)
    recipe = TextEncoderRecipe.model_validate(
        context_params["e06_modeling_features"]["text_candidates"][QWEN]
    )
    encoder = FrozenTextEncoder(
        recipe,
        precision=str(params["text_feature_precision"]),
        device=str(params["device"]),
    )
    try:
        encoded = encoder.encode(
            queries["canonical_query_text"].astype(str).tolist(),
            role="query",
            deadline_monotonic=deadline,
        )
        peak_memory = encoder.peak_device_memory_bytes()
    finally:
        encoder.close()
    query_embeddings = text_whitening.transform(encoded.vectors)
    return (
        dna_whitening.transform(dna_train_raw),
        dna_whitening.transform(dna_validation_raw),
        text_whitening.transform(text_train_raw),
        text_whitening.transform(text_validation_raw),
        query_embeddings,
        {
            "dna_candidate": TFIDF,
            "text_candidate": QWEN,
            "query_embedding_rows": len(queries),
            "query_embedding_elapsed_seconds": encoded.elapsed_seconds,
            "query_embedding_peak_device_memory_bytes": peak_memory,
            "dna_whitening_mean_sha256": _array_sha256(dna_whitening.mean),
            "dna_whitening_matrix_sha256": _array_sha256(dna_whitening.matrix),
            "text_whitening_mean_sha256": _array_sha256(text_whitening.mean),
            "text_whitening_matrix_sha256": _array_sha256(text_whitening.matrix),
        },
    )


def _checkpoint(state: dict[str, Any], final: pd.Series, *, stable: bool) -> pd.DataFrame:
    sequence_head = np.asarray(state["sequence_head"], dtype=np.float32)
    text_head = np.asarray(state["text_head"], dtype=np.float32)
    return pd.DataFrame(
        [
            {
                "seed": int(state["seed"]),
                "base_measure": str(state["base_measure"]),
                "sequence_head": sequence_head.reshape(-1).tolist(),
                "sequence_head_rows": int(sequence_head.shape[0]),
                "sequence_head_columns": int(sequence_head.shape[1]),
                "sequence_head_sha256": _array_sha256(sequence_head),
                "text_head": text_head.reshape(-1).tolist(),
                "text_head_rows": int(text_head.shape[0]),
                "text_head_columns": int(text_head.shape[1]),
                "text_head_sha256": _array_sha256(text_head),
                "temperature": float(state["temperature"]),
                "updates": int(state["updates"]),
                "known_pairs": int(state["known_pairs"]),
                "verified_pairs": int(state["verified_pairs"]),
                "weak_negative_pairs": int(state["contradicted_pairs"]),
                "initial_sequence_head_sha256": str(state["initial_sequence_head_sha256"]),
                "initial_text_head_sha256": str(state["initial_text_head_sha256"]),
                "final_loss": float(final["loss"]),
                "final_query_norm_max": float(final["query_norm_max"]),
                "final_sequence_norm_max": float(final["sequence_norm_sample_max"]),
                "stable": stable,
            }
        ]
    )


def _atomic_sums(
    atomic_queries: pd.DataFrame,
    pair_queries: pd.DataFrame,
    atomic_vectors: np.ndarray,
) -> np.ndarray:
    atomic_lookup = {
        json.loads(str(row.annotation_keys_json))[0]: vector
        for row, vector in zip(atomic_queries.itertuples(index=False), atomic_vectors, strict=True)
    }
    sums = []
    for row in pair_queries.itertuples(index=False):
        left, right = map(str, json.loads(str(row.annotation_keys_json)))
        sums.append(atomic_lookup[left] + atomic_lookup[right])
    result = np.vstack(sums).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("atomic query addition produced non-finite values")
    return result


def _component_log_mass(population: pd.DataFrame) -> np.ndarray:
    components = population["leakage_component_v2"].astype(str)
    sizes = components.value_counts()
    result = -math.log(len(sizes)) - np.log(components.map(sizes).to_numpy(dtype=np.float64))
    if not np.isclose(np.exp(result).sum(), 1.0, rtol=1e-10, atol=1e-10):
        raise RuntimeError("uniform component base measure does not normalize")
    return result


def _is_stable(history: pd.DataFrame, stability: dict[str, Any]) -> bool:
    initial = history.iloc[0]
    final = history.iloc[-1]
    return bool(
        final["loss"] <= initial["loss"] * float(stability["maximum_final_loss_ratio"])
        and history["loss"].max() <= initial["loss"] * float(stability["maximum_loss_ratio"])
        and history["query_norm_max"].max() <= float(stability["maximum_query_norm"])
        and history["sequence_norm_sample_max"].max() <= float(stability["maximum_sequence_norm"])
    )


def _validated_authorization(args: argparse.Namespace, params: dict[str, Any]) -> dict[str, Any]:
    observed = {
        "approval_reference": args.approval_reference,
        "region": args.region,
        "instance_type": args.instance_type,
        "instance_hour_limit": args.instance_hour_limit,
        "observed_instance_price_usd_per_hour": args.price_usd_per_hour,
    }
    expected = params["compute_authorization"]
    if observed != expected:
        raise ValueError(f"compute authorization differs from frozen E10 contract: {observed}")
    maximum_cost = float(args.instance_hour_limit) * float(args.price_usd_per_hour)
    if maximum_cost >= 20.0:
        raise ValueError(f"E10 authorization must remain below $20, observed ${maximum_cost:.2f}")
    return observed


def _load_annotations(path: Path, params: dict[str, Any]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"annotation parquet does not exist: {path}")
    observed_hash = _file_sha256(path)
    expected_hash = str(params["inputs"]["annotation_file_sha256"])
    if observed_hash != expected_hash:
        raise ValueError(
            f"annotation file hash changed: expected {expected_hash}, observed {observed_hash}"
        )
    frame = pd.read_parquet(path, columns=["sequence_id", "sequence_sha256", "annotations"])
    if len(frame) != int(params["inputs"]["annotation_rows"]):
        raise ValueError("annotation row count changed")
    return frame


def _catalog_and_params() -> tuple[Any, dict[str, Any]]:
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    bootstrap_project(PROJECT_ROOT)
    session = KedroSession.create(project_path=PROJECT_ROOT, save_on_close=False)
    context = session.load_context()
    # The catalog keeps no live session dependency after context construction.
    return context.catalog, context.params


def _validate_outputs(catalog: Any, params: dict[str, Any], version: str) -> None:
    queries = catalog.load("e10_weak_annotation_queries", version=version)
    checkpoints = catalog.load("e10_weak_annotation_checkpoints", version=version)
    metrics = catalog.load("e10_weak_annotation_metrics", version=version)
    report = catalog.load("e10_weak_annotation_report", version=version)
    expected = report["output_hashes"]
    observed = {
        "queries_sha256": dataframe_content_sha256(
            queries, sort_columns=["query_kind", "query_id"]
        ),
        "checkpoints_sha256": dataframe_content_sha256(checkpoints, sort_columns=["seed"]),
        "metrics_sha256": dataframe_content_sha256(
            metrics,
            sort_columns=["query_kind", "query_id", "seed", "representation", "k"],
        ),
    }
    if observed != expected:
        raise RuntimeError(f"E10 persisted output hashes differ: {observed}")
    if observed["queries_sha256"] != str(params["expected_query_sha256"]):
        raise RuntimeError("E10 persisted queries differ from the preregistered table")
    expected_seeds = set(map(int, params["probe"]["seeds"]))
    if set(checkpoints["seed"].astype(int)) != expected_seeds:
        raise RuntimeError("E10 checkpoint seed coverage is incomplete")
    for row in checkpoints.itertuples(index=False):
        for prefix in ("sequence_head", "text_head"):
            values = np.asarray(getattr(row, prefix), dtype=np.float32).reshape(
                int(getattr(row, f"{prefix}_rows")),
                int(getattr(row, f"{prefix}_columns")),
            )
            if _array_sha256(values) != str(getattr(row, f"{prefix}_sha256")):
                raise RuntimeError(f"E10 {prefix} hash differs for seed {row.seed}")
    expected_tracking = {str(seed) for seed in expected_seeds}
    complete_tracking = {
        str(row["seed"]) for row in report["tracking"] if row.get("status") == "complete"
    }
    if complete_tracking != expected_tracking:
        raise RuntimeError("E10 W&B run coverage is incomplete")
    print(
        json.dumps(
            {
                "status": "e10_weak_annotation_outputs_validated",
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


def _verify_frame_hash(
    frame: pd.DataFrame,
    *,
    expected: str,
    sort_columns: list[str],
    name: str,
) -> None:
    observed = dataframe_content_sha256(frame, sort_columns=sort_columns)
    if observed != expected:
        raise RuntimeError(f"{name} hash changed: expected {expected}, observed {observed}")


def _support_summary(queries: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for query_kind, group in queries.groupby("query_kind", sort=True):
        result[str(query_kind)] = {
            "queries": int(len(group)),
            "train_positive_rows": {
                "minimum": int(group["train_positive_rows"].min()),
                "median": float(group["train_positive_rows"].median()),
                "maximum": int(group["train_positive_rows"].max()),
            },
            "validation_positive_rows": {
                "minimum": int(group["validation_positive_rows"].min()),
                "median": float(group["validation_positive_rows"].median()),
                "maximum": int(group["validation_positive_rows"].max()),
            },
        }
    return result


def _git_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _hardware() -> dict[str, Any]:
    import torch

    return {
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "probe_precision": "float32",
        "text_feature_precision": "bfloat16",
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


if __name__ == "__main__":
    main()
