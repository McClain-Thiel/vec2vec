#!/usr/bin/env python3
"""Tune description-initialized held-atom retrieval for E14."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from run_text_conditioned_classifier import _folds, _prepare_features, _sample_positions
from run_weak_annotation_classifier import _composition_summary, _hardware, _interval
from run_weak_annotation_experiment import (
    PROJECT_ROOT,
    _array_sha256,
    _catalog_and_params,
    _file_sha256,
    _git_state,
    _load_annotations,
    _verify_frame_hash,
)

from vec2vec.lib import alignment_probe, weak_annotations
from vec2vec.lib.serialization import dataframe_content_sha256, json_content_sha256

OUTPUT_DATASETS = (
    "e14_description_initialized_checkpoint",
    "e14_description_initialized_atomic_metrics",
    "e14_description_initialized_compositional_metrics",
    "e14_description_initialized_report",
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
    params = dict(context_params["description_initialized_tuning_experiment"])
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
        raise RuntimeError("E14 query table differs from the frozen E10-E13 table")

    authorization = _validated_authorization(args, params)
    started = time.perf_counter()
    deadline = time.monotonic() + authorization["instance_hour_limit"] * 3600.0
    checkpoint, atomic, composition, result = _run(
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
        raise RuntimeError(f"E14 output save versions differ: {versions}")
    output_hashes = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["model_id"]),
        "atomic_metrics_sha256": dataframe_content_sha256(
            atomic, sort_columns=["variant", "query_id", "k"]
        ),
        "compositional_metrics_sha256": dataframe_content_sha256(
            composition, sort_columns=["variant", "query_id", "k"]
        ),
    }
    report = {
        "protocol_version": str(params["protocol_version"]),
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
            "E14 tunes on the reused validation split and is not confirmatory.",
            "Uncalled annotations are noisy weak negatives, not verified biological absences.",
            "Only canonical atomic names test held-atom transfer.",
            "The selected all-atom fit is persisted but not evaluated as a deployable model.",
            "Sequence components measure redundancy, not functional diversity.",
        ],
    }
    catalog.save("e14_description_initialized_checkpoint", checkpoint)
    catalog.save("e14_description_initialized_atomic_metrics", atomic)
    catalog.save("e14_description_initialized_compositional_metrics", composition)
    catalog.save("e14_description_initialized_report", report)
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
    features = _prepare_features(
        catalog,
        context_params,
        e10_params,
        params,
        pairs,
        atomic_queries,
        deadline=deadline,
    )
    train = pairs.loc[pairs["panel_role"].eq("alignment_train")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    validation = pairs.loc[pairs["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    probe = params["description_pretraining"]
    run = wandb.init(
        project=str(params["tracking"]["project"]),
        entity=params["tracking"].get("entity"),
        group=str(params["tracking"]["group"]),
        name="e14-description-initialized-held-atom-tuning",
        tags=list(params["tracking"]["tags"]),
        config={
            "protocol_version": str(params["protocol_version"]),
            "description_pretraining": probe,
            "fine_tuning": params["fine_tuning"],
            "authorization": authorization,
        },
        reinit="finish_previous",
    )
    try:
        pretrained, pretrain_history = alignment_probe.train_alignment_probe(
            features["dna_train"].astype(np.float32),
            features["text_train"].astype(np.float32),
            train["sequence_sha256"].astype(str).to_numpy(),
            train["description_sha256"].astype(str).to_numpy(),
            seed=int(probe["seed"]),
            projection_dimension=int(probe["projection_dimension"]),
            epochs=int(probe["epochs"]),
            batch_size=int(probe["batch_size"]),
            learning_rate=float(probe["learning_rate"]),
            weight_decay=float(probe["weight_decay"]),
            initial_temperature=float(probe["initial_temperature"]),
            maximum_logit_scale=float(probe["maximum_logit_scale"]),
            device=str(params["device"]),
            deadline_monotonic=deadline,
        )
        for row in pretrain_history.itertuples(index=False):
            run.log(
                {
                    "pretrain/loss": float(row.mean_loss),
                    "pretrain/logit_scale": float(row.logit_scale),
                },
                step=int(row.epoch),
            )
        variants = _variants(params)
        folds = _folds(atomic_queries["query_id"].astype(str).tolist(), params)
        sample = _sample_positions(
            features["train_ids"],
            count=int(params["calibration"]["training_rows"]),
            salt=str(params["calibration"]["sample_salt"]),
        )
        atomic_tables = []
        composition_tables = []
        variant_reports = {}
        states_by_variant = {}
        for variant, learning_rate in variants.items():
            oof_logits = np.empty(
                (len(features["dna_validation"]), len(atomic_queries)), dtype=np.float32
            )
            fold_reports = []
            for fold in range(int(params["outer_folds"])):
                train_atoms = np.flatnonzero(folds != fold)
                held_atoms = np.flatnonzero(folds == fold)
                state, history = _adapt(
                    pretrained,
                    features,
                    benchmark,
                    train_atoms,
                    params,
                    learning_rate=learning_rate,
                    deadline=deadline,
                )
                raw_train = _scores(
                    features["dna_train"][sample], features["query"][train_atoms], state, params
                )
                slope, intercept, calibration_loss = _fit_shared_calibration(
                    raw_train,
                    benchmark.train_verified[train_atoms][:, sample],
                    params,
                )
                raw_validation = _scores(
                    features["dna_validation"], features["query"][held_atoms], state, params
                )
                oof_logits[:, held_atoms] = (slope * raw_validation + intercept).T
                fold_reports.append(
                    {
                        "fold": fold,
                        "training_atoms": len(train_atoms),
                        "held_atoms": len(held_atoms),
                        "learning_rate": learning_rate,
                        "final_adaptation_loss": (
                            None if history.empty else float(history.iloc[-1]["loss"])
                        ),
                        "calibration_slope": slope,
                        "calibration_intercept": intercept,
                        "calibration_loss": calibration_loss,
                    }
                )
            atomic_metrics, composition_metrics = _evaluate(
                oof_logits,
                atomic_queries,
                pair_queries,
                validation,
                benchmark,
                params,
            )
            atomic_metrics = atomic_metrics.assign(variant=variant)
            composition_metrics = composition_metrics.assign(variant=variant)
            atomic_tables.append(atomic_metrics)
            composition_tables.append(composition_metrics)
            summary = _composition_summary(composition_metrics, params)
            atomic_at_10 = float(
                atomic_metrics.loc[
                    atomic_metrics["k"].eq(int(params["primary_k"])), "utility"
                ].mean()
            )
            variant_reports[variant] = {
                "learning_rate": learning_rate,
                "atomic_utility_at_10": atomic_at_10,
                "compositional_summary": summary,
                "folds": fold_reports,
            }
            run.summary[f"validation/{variant}/atomic_utility_at_10"] = atomic_at_10
            for metric, values in summary["primary_k"].items():
                run.summary[f"validation/{variant}/{metric}_at_10"] = values["mean"]
            states_by_variant[variant] = learning_rate

        atomic = pd.concat(atomic_tables, ignore_index=True).sort_values(
            ["variant", "query_id", "k"], kind="stable", ignore_index=True
        )
        composition = pd.concat(composition_tables, ignore_index=True).sort_values(
            ["variant", "query_id", "k"], kind="stable", ignore_index=True
        )
        selected = _select_variant(variant_reports, params)
        comparison = _compare_to_e13(catalog, composition, selected, params)
        selected_state, selected_history = _adapt(
            pretrained,
            features,
            benchmark,
            np.arange(len(atomic_queries)),
            params,
            learning_rate=states_by_variant[selected],
            deadline=deadline,
        )
        raw_train = _scores(
            features["dna_train"][sample], features["query"], selected_state, params
        )
        slope, intercept, calibration_loss = _fit_shared_calibration(
            raw_train, benchmark.train_verified[:, sample], params
        )
        checkpoint = _checkpoint(
            selected_state,
            features,
            selected,
            slope=slope,
            intercept=intercept,
        )
        run.summary["selection/variant"] = selected
        run.summary["selection/minus_e13"] = comparison["selected_minus_e13"]
        run_id, run_url = str(run.id), str(run.url)
        run.finish(exit_code=0)
    except BaseException:
        run.finish(exit_code=1)
        raise
    return (
        checkpoint,
        atomic,
        composition,
        {
            "feature_preparation": {
                key: value for key, value in features.items() if key.endswith("sha256")
            },
            "description_pretraining": {
                "initial_loss": float(pretrain_history.iloc[0]["mean_loss"]),
                "final_loss": float(pretrain_history.iloc[-1]["mean_loss"]),
                "epochs": len(pretrain_history),
            },
            "variants": variant_reports,
            "selection": {
                "variant": selected,
                "rule": str(params["selection"]["rule"]),
                "all_atom_final_adaptation_loss": (
                    None if selected_history.empty else float(selected_history.iloc[-1]["loss"])
                ),
                "all_atom_calibration_loss": calibration_loss,
                "validation_selected": True,
            },
            "comparison": comparison,
            "tracking": {"run_id": run_id, "url": run_url, "status": "complete"},
            "decision": {
                "status": "exploratory_tuning_result",
                "atom_targets_held_out_during_evaluation": True,
                "confirmatory_claim": False,
                "test_rows_read": False,
            },
        },
    )


def _variants(params: dict[str, Any]) -> dict[str, float | None]:
    values = {"description_only": None}
    for learning_rate in map(float, params["fine_tuning"]["learning_rates"]):
        values[f"atomic_lr_{learning_rate:g}"] = learning_rate
    return values


def _adapt(
    pretrained: dict[str, Any],
    features: dict[str, Any],
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    atom_positions: np.ndarray,
    params: dict[str, Any],
    *,
    learning_rate: float | None,
    deadline: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if learning_rate is None:
        return pretrained, pd.DataFrame()
    fine = params["fine_tuning"]
    return alignment_probe.train_controlled_query_probe(
        features["dna_train"].astype(np.float32),
        features["query"][atom_positions].astype(np.float32),
        benchmark.train_verified[atom_positions],
        objective="verified_set",
        seed=int(fine["seed"]),
        projection_dimension=int(params["description_pretraining"]["projection_dimension"]),
        updates=int(fine["updates"]),
        learning_rate=learning_rate,
        weight_decay=float(fine["weight_decay"]),
        initial_temperature=float(params["description_pretraining"]["initial_temperature"]),
        maximum_logit_scale=float(params["description_pretraining"]["maximum_logit_scale"]),
        device=str(params["device"]),
        deadline_monotonic=deadline,
        initial_state=pretrained,
    )


def _scores(
    dna: np.ndarray,
    query: np.ndarray,
    state: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    sequence_vectors = alignment_probe.project(dna, state["sequence_head"])
    query_vectors = alignment_probe.project(query, state["text_head"])
    scale = min(
        float(np.exp(state["logit_scale"])),
        float(params["description_pretraining"]["maximum_logit_scale"]),
    )
    return scale * query_vectors @ sequence_vectors.T


def _fit_shared_calibration(
    scores: np.ndarray,
    labels: np.ndarray,
    params: dict[str, Any],
) -> tuple[float, float, float]:
    import torch

    values = np.asarray(scores, dtype=np.float32)
    positives = np.asarray(labels)
    if positives.dtype != np.bool_ or positives.shape != values.shape:
        raise ValueError("E14 calibration labels must align with score rows")
    if not positives.any(axis=1).all() or positives.all(axis=1).any():
        raise ValueError("E14 calibration sample needs positives and weak negatives per atom")
    target = torch.device(str(params["device"]))
    score_tensor = torch.as_tensor(values, device=target)
    label_tensor = torch.as_tensor(positives, dtype=torch.float32, device=target)
    raw_slope = torch.nn.Parameter(torch.tensor(0.0, device=target))
    intercept = torch.nn.Parameter(torch.tensor(0.0, device=target))
    calibration = params["calibration"]
    optimizer = torch.optim.Adam([raw_slope, intercept], lr=float(calibration["learning_rate"]))

    def loss_value() -> Any:
        slope = torch.nn.functional.softplus(raw_slope)
        logits = slope * score_tensor + intercept
        positive = (torch.nn.functional.softplus(-logits) * label_tensor).sum(dim=1)
        positive /= label_tensor.sum(dim=1)
        negative_labels = 1.0 - label_tensor
        negative = (torch.nn.functional.softplus(logits) * negative_labels).sum(dim=1)
        negative /= negative_labels.sum(dim=1)
        return (0.5 * (positive + negative)).mean()

    for _ in range(int(calibration["updates"])):
        loss = loss_value()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("E14 shared calibration became non-finite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_loss = loss_value()
    return (
        float(torch.nn.functional.softplus(raw_slope).detach().cpu()),
        float(intercept.detach().cpu()),
        float(final_loss.detach().cpu()),
    )


def _evaluate(
    oof_logits: np.ndarray,
    atomic_queries: pd.DataFrame,
    pair_queries: pd.DataFrame,
    validation: pd.DataFrame,
    benchmark: weak_annotations.WeakAnnotationBenchmark,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoffs = tuple(map(int, params["cutoffs"]))
    atomic_positions = np.flatnonzero(benchmark.queries["query_kind"].eq("atomic").to_numpy())
    atomic = weak_annotations.retrieval_metrics(
        oof_logits.T,
        atomic_queries,
        benchmark.validation_verified[atomic_positions],
        seed=0,
        representation="description_initialized_oof",
        cutoffs=cutoffs,
    ).drop(columns=["query_kind", "seed", "representation"])
    pair_scores = weak_annotations.fuse_atomic_classifier_scores(
        oof_logits, oof_logits, atomic_queries, pair_queries
    )[str(params["score_representation"])]
    composition = weak_annotations.compositional_retrieval_metrics(
        pair_scores,
        pair_queries,
        atomic_queries,
        benchmark.validation_verified[atomic_positions],
        validation[str(params["gallery_component_column"])].to_numpy(),
        cutoffs=cutoffs,
    )
    return atomic, composition


def _select_variant(reports: dict[str, Any], params: dict[str, Any]) -> str:
    strict = {
        name: values["compositional_summary"]["primary_k"]["strict_adherence"]["mean"]
        for name, values in reports.items()
    }
    maximum = max(strict.values())
    tolerance = float(params["selection"]["strict_adherence_tolerance"])
    candidates = [name for name, value in strict.items() if maximum - value <= tolerance]
    return min(
        candidates,
        key=lambda name: (
            -reports[name]["compositional_summary"]["primary_k"]["useful_component_fraction"][
                "mean"
            ],
            name,
        ),
    )


def _compare_to_e13(
    catalog: Any,
    composition: pd.DataFrame,
    selected: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    e13 = catalog.load(
        "e13_text_conditioned_compositional_metrics",
        version=str(params["inputs"]["e13_version"]),
    )
    _verify_frame_hash(
        e13,
        expected=str(params["inputs"]["e13_compositional_metrics_sha256"]),
        sort_columns=["query_id", "k"],
        name="E13 compositional metrics",
    )
    k = int(params["primary_k"])
    current = composition.loc[
        composition["variant"].eq(selected) & composition["k"].eq(k)
    ].set_index("query_id")["signed_strict_utility"]
    reference = e13.loc[e13["k"].eq(k)].set_index("query_id")["signed_strict_utility"]
    joined = pd.concat([current.rename("selected"), reference.rename("e13")], axis=1).dropna()
    if len(joined) != int(params["inputs"]["pair_queries"]):
        raise RuntimeError("E14 selected and E13 query metrics do not align")
    difference = (joined["selected"] - joined["e13"]).to_numpy()
    generator = np.random.default_rng(int(params["bootstrap_seed"]))
    positions = generator.integers(
        0, len(joined), size=(int(params["bootstrap_draws"]), len(joined))
    )
    return {
        "selected_variant": selected,
        "selected_signed_strict_utility_at_10": float(joined["selected"].mean()),
        "e13_signed_strict_utility_at_10": float(joined["e13"].mean()),
        "selected_minus_e13": float(difference.mean()),
        "selected_minus_e13_query_bootstrap_95_interval": _interval(
            difference[positions].mean(axis=1)
        ),
    }


def _checkpoint(
    state: dict[str, Any],
    features: dict[str, Any],
    variant: str,
    *,
    slope: float,
    intercept: float,
) -> pd.DataFrame:
    arrays = {
        "sequence_head": np.asarray(state["sequence_head"], dtype=np.float32),
        "text_head": np.asarray(state["text_head"], dtype=np.float32),
        "dna_whitening_mean": np.asarray(features["dna_whitening_mean"], dtype=np.float32),
        "dna_whitening_matrix": np.asarray(features["dna_whitening_matrix"], dtype=np.float32),
        "text_whitening_mean": np.asarray(features["text_whitening_mean"], dtype=np.float32),
        "text_whitening_matrix": np.asarray(features["text_whitening_matrix"], dtype=np.float32),
    }
    row: dict[str, Any] = {
        "model_id": "description_initialized_selected_all_atom_fit",
        "selected_variant": variant,
        "logit_scale": float(state["logit_scale"]),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "text_encoder_recipe_json": json.dumps(
            features["text_encoder_recipe"], sort_keys=True, separators=(",", ":")
        ),
    }
    for name, values in arrays.items():
        row[name] = values.reshape(-1).tolist()
        row[f"{name}_rows"] = values.shape[0]
        row[f"{name}_columns"] = values.shape[1] if values.ndim == 2 else 1
        row[f"{name}_sha256"] = _array_sha256(values)
    return pd.DataFrame([row])


def _validate_outputs(catalog: Any, params: dict[str, Any], version: str) -> None:
    checkpoint = catalog.load("e14_description_initialized_checkpoint", version=version)
    atomic = catalog.load("e14_description_initialized_atomic_metrics", version=version)
    composition = catalog.load("e14_description_initialized_compositional_metrics", version=version)
    report = catalog.load("e14_description_initialized_report", version=version)
    observed = {
        "checkpoint_sha256": dataframe_content_sha256(checkpoint, sort_columns=["model_id"]),
        "atomic_metrics_sha256": dataframe_content_sha256(
            atomic, sort_columns=["variant", "query_id", "k"]
        ),
        "compositional_metrics_sha256": dataframe_content_sha256(
            composition, sort_columns=["variant", "query_id", "k"]
        ),
    }
    if observed != report["output_hashes"]:
        raise RuntimeError("E14 persisted output hashes differ")
    variants = set(_variants(params))
    expected_atomic = (
        len(variants) * int(params["inputs"]["atomic_queries"]) * len(params["cutoffs"])
    )
    expected_pairs = len(variants) * int(params["inputs"]["pair_queries"]) * len(params["cutoffs"])
    if len(atomic) != expected_atomic or len(composition) != expected_pairs:
        raise RuntimeError("E14 persisted metric row counts differ")
    if (
        set(atomic["variant"].astype(str)) != variants
        or set(composition["variant"].astype(str)) != variants
    ):
        raise RuntimeError("E14 variant coverage is incomplete")
    row = checkpoint.iloc[0]
    for name in (
        "sequence_head",
        "text_head",
        "dna_whitening_mean",
        "dna_whitening_matrix",
        "text_whitening_mean",
        "text_whitening_matrix",
    ):
        values = np.asarray(row[name], dtype=np.float32)
        rows, columns = int(row[f"{name}_rows"]), int(row[f"{name}_columns"])
        values = values.reshape(rows, columns) if columns > 1 else values.reshape(rows)
        if _array_sha256(values) != str(row[f"{name}_sha256"]):
            raise RuntimeError(f"E14 checkpoint {name} hash differs")
    selected = str(report["selection"]["variant"])
    if selected not in variants or selected != str(row["selected_variant"]):
        raise RuntimeError("E14 selected variant and checkpoint differ")
    for variant in variants:
        recalculated = _composition_summary(
            composition.loc[composition["variant"].eq(variant)], params
        )
        if recalculated != report["variants"][variant]["compositional_summary"]:
            raise RuntimeError(f"E14 {variant} summary differs from recalculation")
    if report["tracking"].get("status") != "complete":
        raise RuntimeError("E14 W&B run is incomplete")
    print(
        json.dumps(
            {
                "status": "e14_description_initialized_outputs_validated",
                "version": version,
                "report_sha256": json_content_sha256(report),
                "output_hashes": observed,
                "selection": report["selection"],
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
        raise ValueError(f"compute authorization differs from frozen E14 contract: {observed}")
    maximum_cost = float(args.instance_hour_limit) * float(args.price_usd_per_hour)
    if maximum_cost >= 20.0:
        raise ValueError(f"E14 authorization must remain below $20, observed ${maximum_cost:.2f}")
    return observed


if __name__ == "__main__":
    main()
