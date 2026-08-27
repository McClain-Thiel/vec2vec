"""Gate 2 paired-identity versus verified-set supervision experiment."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import alignment_probe, fixed_representation_alignment
from vec2vec.lib.serialization import json_content_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256

EXPECTED_OBJECTIVES = ("paired_identity", "verified_set")
EXPECTED_SEEDS = (13, 42, 20260818)
EXPECTED_BASE_MEASURES = ("uniform_plasmid", "uniform_v2_component")


def run_set_supervision_comparison(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    validation_states: pd.DataFrame,
    all_query_states: pd.DataFrame,
    query_manifest: dict[str, Any],
    input_manifest: dict[str, Any],
    dna_features: pd.DataFrame,
    dna_manifest: dict[str, Any],
    text_features: pd.DataFrame,
    text_manifest: dict[str, Any],
    params: dict[str, Any],
    *,
    deadline_monotonic: float | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Train both objectives on identical batches and evaluate on validation only."""
    train, gallery, ordered_queries = fixed_representation_alignment._validate_alignment_inputs(
        pairs, queries, validation_states, input_manifest, params
    )
    _validate_query_states(all_query_states, query_manifest, params)
    _validate_feature("dna", dna_features, dna_manifest, input_manifest, params)
    _validate_feature("text", text_features, text_manifest, input_manifest, params)
    train_sequence, gallery_sequence, query_text, whitening_state = _prepare_matrices(
        train,
        gallery,
        ordered_queries,
        dna_features,
        text_features,
        dna_candidate=str(params["accepted_feature_artifacts"]["dna"]["candidate_id"]),
        text_candidate=str(params["accepted_feature_artifacts"]["text"]["candidate_id"]),
        epsilon=float(params["probe"]["whitening_epsilon"]),
    )
    training_queries, evaluation_queries, training_positions, evaluation_positions = (
        _query_partitions(ordered_queries, params)
    )
    training_query_text = query_text[training_positions]
    evaluation_query_text = query_text[evaluation_positions]
    verified = _training_verified_mask(train, training_queries, all_query_states, params)
    objectives, seeds, cutoffs = _validate_axes(params, gallery_rows=len(gallery))

    checkpoints: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    rankings: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    summaries: list[pd.DataFrame] = []
    bootstraps: list[pd.DataFrame] = []
    tracking_records: list[dict[str, Any]] = []
    distribution_gaps: list[dict[str, float | int | str]] = []
    audit_additive = bool(params.get("audit_atomic_sum", False))
    probe = dict(params["probe"])
    for objective in objectives:
        objective_scores: dict[str, list[np.ndarray]] = {"direct_text": []}
        if audit_additive and objective == "verified_set":
            objective_scores["atomic_sum"] = []
        for seed in seeds:
            wandb_run, tracking = _start_wandb(params, objective=objective, seed=seed)
            try:
                state, history = alignment_probe.train_controlled_query_probe(
                    train_sequence,
                    training_query_text,
                    verified,
                    objective=objective,
                    seed=seed,
                    projection_dimension=int(probe["projection_dimension"]),
                    updates=int(probe["updates"]),
                    learning_rate=float(probe["learning_rate"]),
                    weight_decay=float(probe["weight_decay"]),
                    initial_temperature=float(probe["initial_temperature"]),
                    maximum_logit_scale=float(probe["maximum_logit_scale"]),
                    device=str(params["device"]),
                    deadline_monotonic=deadline_monotonic,
                )
            except BaseException:
                _finish_wandb(wandb_run, tracking, exit_code=1)
                raise
            history = history.assign(objective=objective, seed=seed)
            sequence_vectors = alignment_probe.project(gallery_sequence, state["sequence_head"])
            representations = {
                "direct_text": alignment_probe.project(evaluation_query_text, state["text_head"])
            }
            if audit_additive and objective == "verified_set":
                atomic_vectors = alignment_probe.project(training_query_text, state["text_head"])
                representations["atomic_sum"] = _sum_atomic_queries(
                    training_queries, evaluation_queries, atomic_vectors
                )
            seed_summaries = []
            seed_scores = {}
            for representation, query_vectors in representations.items():
                ranking, query_metrics, scores = alignment_probe.query_rankings_and_metrics(
                    query_vectors,
                    sequence_vectors,
                    evaluation_queries,
                    gallery,
                    validation_states,
                    cutoffs=cutoffs,
                )
                ranking = ranking.assign(objective=objective, seed=seed)
                query_metrics = query_metrics.assign(objective=objective, seed=seed)
                summary = _query_summary(query_metrics, objective=objective, seed=seed)
                if audit_additive:
                    ranking = ranking.assign(query_representation=representation)
                    query_metrics = query_metrics.assign(query_representation=representation)
                    summary = summary.assign(query_representation=representation)
                rankings.append(ranking)
                metrics.append(query_metrics)
                summaries.append(summary)
                seed_summaries.append(summary)
                seed_scores[representation] = scores
                objective_scores[representation].append(scores)
            if audit_additive and objective == "verified_set":
                scale = min(
                    float(np.exp(state["logit_scale"])),
                    float(probe["maximum_logit_scale"]),
                )
                divergences = _jensen_shannon_rows(
                    scale * seed_scores["direct_text"], scale * seed_scores["atomic_sum"]
                )
                distribution_gaps.extend(
                    {
                        "seed": seed,
                        "query_id": str(query.query_id),
                        "jensen_shannon_divergence": float(divergence),
                    }
                    for query, divergence in zip(
                        evaluation_queries.itertuples(index=False), divergences, strict=True
                    )
                )
            checkpoints.append(_checkpoint(state))
            histories.append(history)
            wandb_summary = pd.concat(seed_summaries, ignore_index=True)
            if "query_representation" in wandb_summary:
                wandb_summary["query_kind"] = (
                    wandb_summary["query_representation"] + "/" + wandb_summary["query_kind"]
                )
            _log_wandb(wandb_run, tracking, history, wandb_summary)
            _finish_wandb(wandb_run, tracking, exit_code=0)
            tracking_records.append(tracking)
        for representation, scores in objective_scores.items():
            bootstrap = alignment_probe.whole_component_bootstrap_draws(
                scores,
                evaluation_queries,
                gallery,
                validation_states,
                k=int(params["primary_k"]),
                draws=int(probe["bootstrap_draws"]),
                seed=int(probe["bootstrap_seed"]),
                deadline_monotonic=deadline_monotonic,
            ).assign(objective=objective)
            if audit_additive:
                bootstrap = bootstrap.assign(query_representation=representation)
            bootstraps.append(bootstrap)

    representation_columns = ["query_representation"] if audit_additive else []
    checkpoint_table = _concat(checkpoints, ["objective", "seed"])
    history_table = _concat(histories, ["objective", "seed", "update"])
    ranking_table = _concat(
        rankings, ["objective", *representation_columns, "seed", "query_id", "rank"]
    )
    metric_table = _concat(metrics, ["objective", *representation_columns, "seed", "query_id", "k"])
    summary_table = _concat(
        summaries, ["objective", *representation_columns, "seed", "query_kind", "k"]
    )
    bootstrap_table = _concat(
        bootstraps, ["objective", *representation_columns, "query_kind", "draw"]
    )
    output_hashes = {
        "whitening_state_sha256": dataframe_content_sha256(
            whitening_state, sort_columns=["feature_kind", "candidate_id"]
        ),
        "checkpoints_sha256": dataframe_content_sha256(
            checkpoint_table, sort_columns=["objective", "seed"]
        ),
        "training_history_sha256": dataframe_content_sha256(
            history_table, sort_columns=["objective", "seed", "update"]
        ),
        "query_rankings_sha256": dataframe_content_sha256(
            ranking_table,
            sort_columns=["objective", *representation_columns, "seed", "query_id", "rank"],
        ),
        "query_metrics_sha256": dataframe_content_sha256(
            metric_table,
            sort_columns=["objective", *representation_columns, "seed", "query_id", "k"],
        ),
        "query_summaries_sha256": dataframe_content_sha256(
            summary_table,
            sort_columns=["objective", *representation_columns, "seed", "query_kind", "k"],
        ),
        "bootstrap_draws_sha256": dataframe_content_sha256(
            bootstrap_table,
            sort_columns=["objective", *representation_columns, "query_kind", "draw"],
        ),
    }
    gap_table = pd.DataFrame(distribution_gaps)
    if audit_additive:
        gap_table = gap_table.sort_values(["seed", "query_id"], kind="stable", ignore_index=True)
        output_hashes["distribution_gaps_sha256"] = dataframe_content_sha256(
            gap_table, sort_columns=["seed", "query_id"]
        )
    report = {
        "protocol_version": str(params["protocol_version"]),
        "protocol": str(params["protocol_path"]),
        "input_versions": dict(params["input_versions"]),
        "population": {
            "training_rows": int(len(train)),
            "gallery_rows": int(len(gallery)),
            "training_queries": int(len(training_queries)),
            "evaluation_queries": int(len(evaluation_queries)),
            "training_query_kinds": sorted(training_queries["query_kind"].unique().tolist()),
            "evaluation_query_kinds": sorted(evaluation_queries["query_kind"].unique().tolist()),
            "minimum_training_verified_rows": int(verified.sum(axis=1).min()),
            "test_rows_read": False,
        },
        "resolved_configuration": params,
        "comparison": _comparison(summary_table, bootstrap_table, params),
        "tracking": tracking_records,
        "output_hashes": output_hashes,
        "decision": {
            "status": str(params.get("completion_status", "validation_comparison_complete")),
            "validation_only": True,
        },
    }
    if audit_additive:
        report["additive_comparison"] = _additive_comparison(
            summary_table, bootstrap_table, gap_table, params
        )
        report["decision"]["hyperparameter_tuning_performed"] = False
        report["known_limitations"] = [
            "The cosine-normalized probe is not the proposed unnormalized maximum-entropy model.",
            "The historical test split is contaminated and was not read.",
            "Canonical symbolic query text does not test paraphrase robustness.",
        ]
    return (
        whitening_state,
        checkpoint_table,
        history_table,
        ranking_table,
        metric_table,
        summary_table,
        bootstrap_table,
        report,
    )


def run_natural_parameter_comparison(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    validation_states: pd.DataFrame,
    all_query_states: pd.DataFrame,
    query_manifest: dict[str, Any],
    input_manifest: dict[str, Any],
    dna_features: pd.DataFrame,
    dna_manifest: dict[str, Any],
    text_features: pd.DataFrame,
    text_manifest: dict[str, Any],
    params: dict[str, Any],
    *,
    deadline_monotonic: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare two base measures for the frozen unnormalized maximum-entropy model."""
    train, gallery, ordered_queries = fixed_representation_alignment._validate_alignment_inputs(
        pairs, queries, validation_states, input_manifest, params
    )
    _validate_query_states(all_query_states, query_manifest, params)
    _validate_feature("dna", dna_features, dna_manifest, input_manifest, params)
    _validate_feature("text", text_features, text_manifest, input_manifest, params)
    train_sequence, gallery_sequence, query_text, whitening_state = _prepare_matrices(
        train,
        gallery,
        ordered_queries,
        dna_features,
        text_features,
        dna_candidate=str(params["accepted_feature_artifacts"]["dna"]["candidate_id"]),
        text_candidate=str(params["accepted_feature_artifacts"]["text"]["candidate_id"]),
        epsilon=float(params["probe"]["whitening_epsilon"]),
    )
    training_queries, evaluation_queries, training_positions, evaluation_positions = (
        _natural_parameter_partitions(ordered_queries, params)
    )
    verified, known = _training_state_masks(train, training_queries, all_query_states, params)
    base_measures = tuple(map(str, params["base_measures"]))
    seeds = tuple(map(int, params["probe"]["seeds"]))
    if base_measures != EXPECTED_BASE_MEASURES or seeds != EXPECTED_SEEDS:
        raise ValueError("E08 base measures or seeds changed from the frozen protocol")
    cutoffs = tuple(map(int, params["probe"]["cutoffs"]))
    if not cutoffs or max(cutoffs) > len(gallery) or int(params["primary_k"]) not in cutoffs:
        raise ValueError("E08 cutoffs do not cover the frozen primary K")

    training_query_text = query_text[training_positions]
    evaluation_query_text = query_text[evaluation_positions]
    probe = dict(params["probe"])
    checkpoints: list[pd.DataFrame] = []
    histories: list[pd.DataFrame] = []
    seed_summaries: list[pd.DataFrame] = []
    bootstraps: list[pd.DataFrame] = []
    tracking_records: list[dict[str, Any]] = []
    divergence_rows: list[dict[str, float | int | str]] = []
    base_mass_hashes: dict[str, dict[str, str]] = {}

    for base_measure in base_measures:
        train_log_mass = _base_log_mass(train, base_measure)
        gallery_log_mass = _base_log_mass(gallery, base_measure)
        base_mass_hashes[base_measure] = {
            "training": _float64_sha256(train_log_mass),
            "validation": _float64_sha256(gallery_log_mass),
        }
        scores_by_representation: dict[str, list[np.ndarray]] = {
            "direct_text": [],
            "atomic_sum": [],
        }
        for seed in seeds:
            wandb_run, tracking = _start_wandb(
                params, objective=f"maximum_entropy_{base_measure}", seed=seed
            )
            tracking["base_measure"] = base_measure
            try:
                state, history = alignment_probe.train_maximum_entropy_probe(
                    train_sequence,
                    training_query_text,
                    verified,
                    known,
                    train_log_mass,
                    base_measure=base_measure,
                    seed=seed,
                    projection_dimension=int(probe["projection_dimension"]),
                    updates=int(probe["updates"]),
                    learning_rate=float(probe["learning_rate"]),
                    weight_decay=float(probe["weight_decay"]),
                    temperature=float(probe["temperature"]),
                    device=str(params["device"]),
                    deadline_monotonic=deadline_monotonic,
                )
            except BaseException:
                _finish_wandb(wandb_run, tracking, exit_code=1)
                raise
            history = history.assign(base_measure=base_measure, seed=seed)
            histories.append(history)
            checkpoints.append(_natural_parameter_checkpoint(state, history.iloc[-1]))
            gallery_vectors = alignment_probe.project_unnormalized(
                gallery_sequence, state["sequence_head"]
            )
            atomic_vectors = alignment_probe.project_unnormalized(
                training_query_text, state["text_head"]
            )
            representations = {
                "direct_text": alignment_probe.project_unnormalized(
                    evaluation_query_text, state["text_head"]
                ),
                "atomic_sum": _sum_atomic_queries(
                    training_queries, evaluation_queries, atomic_vectors
                ),
            }
            run_summaries: list[pd.DataFrame] = []
            seed_scores: dict[str, np.ndarray] = {}
            for representation, query_vectors in representations.items():
                scores = alignment_probe.natural_parameter_scores(
                    query_vectors,
                    gallery_vectors,
                    gallery_log_mass,
                    temperature=float(probe["temperature"]),
                )
                _, query_metrics, scores = alignment_probe.query_rankings_and_metrics_from_scores(
                    scores,
                    evaluation_queries,
                    gallery,
                    validation_states,
                    cutoffs=cutoffs,
                )
                summary = _query_summary(
                    query_metrics, objective="maximum_entropy", seed=seed
                ).assign(base_measure=base_measure, query_representation=representation)
                seed_summaries.append(summary)
                run_summaries.append(summary)
                scores_by_representation[representation].append(scores)
                seed_scores[representation] = scores
            divergences = _jensen_shannon_rows(
                seed_scores["direct_text"], seed_scores["atomic_sum"]
            )
            divergence_rows.extend(
                {
                    "base_measure": base_measure,
                    "seed": seed,
                    "query_id": str(query.query_id),
                    "jensen_shannon_divergence": float(divergence),
                }
                for query, divergence in zip(
                    evaluation_queries.itertuples(index=False), divergences, strict=True
                )
            )
            _log_natural_parameter_wandb(
                wandb_run, tracking, history, pd.concat(run_summaries, ignore_index=True)
            )
            _finish_wandb(wandb_run, tracking, exit_code=0)
            tracking_records.append(tracking)
        for representation, score_matrices in scores_by_representation.items():
            bootstraps.append(
                alignment_probe.whole_component_bootstrap_draws(
                    score_matrices,
                    evaluation_queries,
                    gallery,
                    validation_states,
                    k=int(params["primary_k"]),
                    draws=int(probe["bootstrap_draws"]),
                    seed=int(probe["bootstrap_seed"]),
                    component_column="leakage_component_v2",
                    deadline_monotonic=deadline_monotonic,
                ).assign(
                    base_measure=base_measure,
                    query_representation=representation,
                )
            )

    checkpoint_table = _concat(checkpoints, ["base_measure", "seed"])
    history_table = _concat(histories, ["base_measure", "seed", "update"])
    per_seed_table = _concat(
        seed_summaries,
        ["base_measure", "query_representation", "seed", "query_kind", "k"],
    )
    bootstrap_table = _concat(
        bootstraps,
        ["base_measure", "query_representation", "query_kind", "draw"],
    )
    divergence_table = pd.DataFrame(divergence_rows).sort_values(
        ["base_measure", "seed", "query_id"], kind="stable", ignore_index=True
    )
    summary_table, comparison = _natural_parameter_summary(
        per_seed_table, bootstrap_table, divergence_table, params
    )
    output_hashes = {
        "whitening_state_sha256": dataframe_content_sha256(
            whitening_state, sort_columns=["feature_kind", "candidate_id"]
        ),
        "checkpoints_sha256": dataframe_content_sha256(
            checkpoint_table, sort_columns=["base_measure", "seed"]
        ),
        "training_history_sha256": dataframe_content_sha256(
            history_table, sort_columns=["base_measure", "seed", "update"]
        ),
        "summary_sha256": dataframe_content_sha256(
            summary_table, sort_columns=["base_measure", "query_representation"]
        ),
        "bootstrap_draws_sha256": dataframe_content_sha256(
            bootstrap_table,
            sort_columns=["base_measure", "query_representation", "query_kind", "draw"],
        ),
        "distribution_gaps_sha256": dataframe_content_sha256(
            divergence_table, sort_columns=["base_measure", "seed", "query_id"]
        ),
    }
    report = {
        "protocol_version": str(params["protocol_version"]),
        "protocol": str(params["protocol_path"]),
        "input_versions": dict(params["input_versions"]),
        "population": {
            "training_rows": int(len(train)),
            "gallery_rows": int(len(gallery)),
            "training_queries": int(len(training_queries)),
            "evaluation_queries": int(len(evaluation_queries)),
            "known_training_pairs": int(known.sum()),
            "verified_training_pairs": int(verified.sum()),
            "contradicted_training_pairs": int((known & ~verified).sum()),
            "test_rows_read": False,
        },
        "resolved_configuration": params,
        "base_mass_hashes": base_mass_hashes,
        "comparison": comparison,
        "tracking": tracking_records,
        "output_hashes": output_hashes,
        "decision": {
            "status": "natural_parameter_validation_complete",
            "validation_only": True,
            "unknown_candidates_excluded_from_training": True,
            "hyperparameter_tuning_performed": False,
        },
        "known_limitations": [
            "Only four frozen atomic constraints have explicit contradicted candidates.",
            "The resulting four-query conjunction evaluation is underpowered.",
            "The historical test split is contaminated and was not read.",
            "Canonical symbolic query text does not test paraphrase robustness.",
        ],
    }
    return checkpoint_table, history_table, summary_table, report


def run_natural_parameter_calibration(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    validation_states: pd.DataFrame,
    all_query_states: pd.DataFrame,
    query_manifest: dict[str, Any],
    input_manifest: dict[str, Any],
    dna_features: pd.DataFrame,
    dna_manifest: dict[str, Any],
    text_features: pd.DataFrame,
    text_manifest: dict[str, Any],
    params: dict[str, Any],
    *,
    deadline_monotonic: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select a stable learning rate without consulting validation retrieval."""
    train, gallery, ordered_queries = fixed_representation_alignment._validate_alignment_inputs(
        pairs, queries, validation_states, input_manifest, params
    )
    _validate_query_states(all_query_states, query_manifest, params)
    _validate_feature("dna", dna_features, dna_manifest, input_manifest, params)
    _validate_feature("text", text_features, text_manifest, input_manifest, params)
    train_sequence, gallery_sequence, query_text, whitening_state = _prepare_matrices(
        train,
        gallery,
        ordered_queries,
        dna_features,
        text_features,
        dna_candidate=str(params["accepted_feature_artifacts"]["dna"]["candidate_id"]),
        text_candidate=str(params["accepted_feature_artifacts"]["text"]["candidate_id"]),
        epsilon=float(params["probe"]["whitening_epsilon"]),
    )
    training_queries, evaluation_queries, training_positions, evaluation_positions = (
        _natural_parameter_partitions(ordered_queries, params)
    )
    verified, known = _training_state_masks(train, training_queries, all_query_states, params)
    base_measures = tuple(map(str, params["base_measures"]))
    seeds = tuple(map(int, params["probe"]["seeds"]))
    learning_rates = tuple(map(float, params["probe"]["learning_rates"]))
    if base_measures != EXPECTED_BASE_MEASURES or seeds != EXPECTED_SEEDS:
        raise ValueError("E09 base measures or seeds changed from the frozen protocol")
    if not learning_rates or tuple(sorted(set(learning_rates))) != learning_rates:
        raise ValueError("E09 learning rates must be positive, unique, and increasing")
    if learning_rates[0] <= 0.0:
        raise ValueError("E09 learning rates must be positive")
    cutoffs = tuple(map(int, params["probe"]["cutoffs"]))
    if not cutoffs or max(cutoffs) > len(gallery) or int(params["primary_k"]) not in cutoffs:
        raise ValueError("E09 cutoffs do not cover the frozen primary K")

    training_query_text = query_text[training_positions]
    evaluation_query_text = query_text[evaluation_positions]
    probe = dict(params["probe"])
    stability = dict(params["stability"])
    maximum_loss_ratio = float(stability["maximum_loss_ratio"])
    maximum_final_loss_ratio = float(stability["maximum_final_loss_ratio"])
    maximum_query_norm = float(stability["maximum_query_norm"])
    maximum_sequence_norm = float(stability["maximum_sequence_norm"])
    if (
        min(
            maximum_loss_ratio,
            maximum_final_loss_ratio,
            maximum_query_norm,
            maximum_sequence_norm,
        )
        <= 0.0
    ):
        raise ValueError("E09 stability thresholds must be positive")

    histories: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    stable_states: dict[float, dict[tuple[str, int], dict[str, Any]]] = {}
    tracking_records: list[dict[str, Any]] = []
    base_mass_hashes: dict[str, dict[str, str]] = {}
    train_masses = {}
    gallery_masses = {}
    for base_measure in base_measures:
        train_masses[base_measure] = _base_log_mass(train, base_measure)
        gallery_masses[base_measure] = _base_log_mass(gallery, base_measure)
        base_mass_hashes[base_measure] = {
            "training": _float64_sha256(train_masses[base_measure]),
            "validation": _float64_sha256(gallery_masses[base_measure]),
        }

    for learning_rate in learning_rates:
        candidate_states: dict[tuple[str, int], dict[str, Any]] = {}
        candidate_stable = True
        candidate_params = {
            **params,
            "probe": {**probe, "learning_rate": learning_rate},
        }
        for base_measure in base_measures:
            for seed in seeds:
                objective = f"calibration_{base_measure}_lr_{learning_rate:g}"
                wandb_run, tracking = _start_wandb(candidate_params, objective=objective, seed=seed)
                tracking.update(
                    phase="calibration",
                    base_measure=base_measure,
                    learning_rate=learning_rate,
                )
                try:
                    state, history = alignment_probe.train_maximum_entropy_probe(
                        train_sequence,
                        training_query_text,
                        verified,
                        known,
                        train_masses[base_measure],
                        base_measure=base_measure,
                        seed=seed,
                        projection_dimension=int(probe["projection_dimension"]),
                        updates=int(probe["updates"]),
                        learning_rate=learning_rate,
                        weight_decay=float(probe["weight_decay"]),
                        temperature=float(probe["temperature"]),
                        device=str(params["device"]),
                        record_initial=True,
                        deadline_monotonic=deadline_monotonic,
                    )
                except BaseException:
                    _finish_wandb(wandb_run, tracking, exit_code=1)
                    raise
                initial = history.iloc[0]
                final = history.iloc[-1]
                run_stable = bool(
                    final["loss"] <= initial["loss"] * maximum_final_loss_ratio
                    and history["loss"].max() <= initial["loss"] * maximum_loss_ratio
                    and history["query_norm_max"].max() <= maximum_query_norm
                    and history["sequence_norm_sample_max"].max() <= maximum_sequence_norm
                )
                candidate_stable = candidate_stable and run_stable
                candidate_states[(base_measure, seed)] = state
                calibration_rows.append(
                    {
                        "learning_rate": learning_rate,
                        "base_measure": base_measure,
                        "seed": seed,
                        "initial_loss": float(initial["loss"]),
                        "final_loss": float(final["loss"]),
                        "minimum_loss": float(history["loss"].min()),
                        "maximum_loss": float(history["loss"].max()),
                        "maximum_query_norm": float(history["query_norm_max"].max()),
                        "maximum_sequence_norm": float(history["sequence_norm_sample_max"].max()),
                        "stable": run_stable,
                    }
                )
                history = history.assign(
                    learning_rate=learning_rate,
                    base_measure=base_measure,
                    seed=seed,
                )
                histories.append(history)
                _log_natural_parameter_wandb(
                    wandb_run,
                    tracking,
                    history,
                    pd.DataFrame(),
                )
                _finish_wandb(wandb_run, tracking, exit_code=0)
                tracking_records.append(tracking)
        if candidate_stable:
            stable_states[learning_rate] = candidate_states

    calibration_table = pd.DataFrame(calibration_rows).sort_values(
        ["learning_rate", "base_measure", "seed"], kind="stable", ignore_index=True
    )
    history_table = _concat(histories, ["learning_rate", "base_measure", "seed", "update"])
    candidate_summary = (
        calibration_table.groupby("learning_rate", sort=True)
        .agg(
            stable_runs=("stable", "sum"),
            total_runs=("stable", "size"),
            mean_initial_loss=("initial_loss", "mean"),
            mean_final_loss=("final_loss", "mean"),
            worst_loss=("maximum_loss", "max"),
            worst_query_norm=("maximum_query_norm", "max"),
            worst_sequence_norm=("maximum_sequence_norm", "max"),
        )
        .reset_index()
    )
    eligible = candidate_summary.loc[
        candidate_summary["stable_runs"].eq(candidate_summary["total_runs"])
    ]
    selected_learning_rate = (
        None
        if eligible.empty
        else float(
            eligible.sort_values(["mean_final_loss", "learning_rate"], kind="stable").iloc[0][
                "learning_rate"
            ]
        )
    )
    candidate_summary["selected"] = candidate_summary["learning_rate"].eq(selected_learning_rate)

    checkpoints: list[pd.DataFrame] = []
    seed_summaries: list[pd.DataFrame] = []
    bootstraps: list[pd.DataFrame] = []
    divergence_rows: list[dict[str, float | int | str]] = []
    if selected_learning_rate is not None:
        selected_states = stable_states[selected_learning_rate]
        selected_params = {
            **params,
            "probe": {**probe, "learning_rate": selected_learning_rate},
        }
        for base_measure in base_measures:
            scores_by_representation: dict[str, list[np.ndarray]] = {
                "direct_text": [],
                "atomic_sum": [],
            }
            for seed in seeds:
                state = selected_states[(base_measure, seed)]
                selected_history = history_table.loc[
                    history_table["learning_rate"].eq(selected_learning_rate)
                    & history_table["base_measure"].eq(base_measure)
                    & history_table["seed"].eq(seed)
                ]
                checkpoint = _natural_parameter_checkpoint(state, selected_history.iloc[-1])
                checkpoints.append(checkpoint.assign(learning_rate=selected_learning_rate))
                gallery_vectors = alignment_probe.project_unnormalized(
                    gallery_sequence, state["sequence_head"]
                )
                atomic_vectors = alignment_probe.project_unnormalized(
                    training_query_text, state["text_head"]
                )
                representations = {
                    "direct_text": alignment_probe.project_unnormalized(
                        evaluation_query_text, state["text_head"]
                    ),
                    "atomic_sum": _sum_atomic_queries(
                        training_queries, evaluation_queries, atomic_vectors
                    ),
                }
                run_summaries: list[pd.DataFrame] = []
                seed_scores = {}
                for representation, query_vectors in representations.items():
                    scores = alignment_probe.natural_parameter_scores(
                        query_vectors,
                        gallery_vectors,
                        gallery_masses[base_measure],
                        temperature=float(probe["temperature"]),
                    )
                    _, query_metrics, scores = (
                        alignment_probe.query_rankings_and_metrics_from_scores(
                            scores,
                            evaluation_queries,
                            gallery,
                            validation_states,
                            cutoffs=cutoffs,
                        )
                    )
                    summary = _query_summary(
                        query_metrics, objective="maximum_entropy", seed=seed
                    ).assign(
                        base_measure=base_measure,
                        query_representation=representation,
                    )
                    seed_summaries.append(summary)
                    run_summaries.append(summary)
                    scores_by_representation[representation].append(scores)
                    seed_scores[representation] = scores
                divergences = _jensen_shannon_rows(
                    seed_scores["direct_text"], seed_scores["atomic_sum"]
                )
                divergence_rows.extend(
                    {
                        "base_measure": base_measure,
                        "seed": seed,
                        "query_id": str(query.query_id),
                        "jensen_shannon_divergence": float(divergence),
                    }
                    for query, divergence in zip(
                        evaluation_queries.itertuples(index=False), divergences, strict=True
                    )
                )
                wandb_run, tracking = _start_wandb(
                    selected_params, objective=f"evaluation_{base_measure}", seed=seed
                )
                tracking.update(
                    phase="selected_validation_evaluation",
                    base_measure=base_measure,
                    learning_rate=selected_learning_rate,
                )
                _log_natural_parameter_wandb(
                    wandb_run,
                    tracking,
                    pd.DataFrame(),
                    pd.concat(run_summaries, ignore_index=True),
                )
                _finish_wandb(wandb_run, tracking, exit_code=0)
                tracking_records.append(tracking)
            for representation, score_matrices in scores_by_representation.items():
                bootstraps.append(
                    alignment_probe.whole_component_bootstrap_draws(
                        score_matrices,
                        evaluation_queries,
                        gallery,
                        validation_states,
                        k=int(params["primary_k"]),
                        draws=int(probe["bootstrap_draws"]),
                        seed=int(probe["bootstrap_seed"]),
                        component_column="leakage_component_v2",
                        deadline_monotonic=deadline_monotonic,
                    ).assign(
                        base_measure=base_measure,
                        query_representation=representation,
                    )
                )

        checkpoint_table = _concat(checkpoints, ["base_measure", "seed"])
        per_seed_table = _concat(
            seed_summaries,
            ["base_measure", "query_representation", "seed", "query_kind", "k"],
        )
        bootstrap_table = _concat(
            bootstraps,
            ["base_measure", "query_representation", "query_kind", "draw"],
        )
        divergence_table = pd.DataFrame(divergence_rows).sort_values(
            ["base_measure", "seed", "query_id"], kind="stable", ignore_index=True
        )
        summary_table, comparison = _natural_parameter_summary(
            per_seed_table, bootstrap_table, divergence_table, params
        )
    else:
        checkpoint_table = pd.DataFrame()
        summary_table = pd.DataFrame()
        bootstrap_table = pd.DataFrame()
        divergence_table = pd.DataFrame()
        comparison = None

    output_hashes = {
        "whitening_state_sha256": dataframe_content_sha256(
            whitening_state, sort_columns=["feature_kind", "candidate_id"]
        ),
        "calibration_history_sha256": dataframe_content_sha256(
            history_table,
            sort_columns=["learning_rate", "base_measure", "seed", "update"],
        ),
    }
    if selected_learning_rate is not None:
        output_hashes.update(
            selected_checkpoints_sha256=dataframe_content_sha256(
                checkpoint_table, sort_columns=["base_measure", "seed"]
            ),
            selected_summary_sha256=dataframe_content_sha256(
                summary_table, sort_columns=["base_measure", "query_representation"]
            ),
            bootstrap_draws_sha256=dataframe_content_sha256(
                bootstrap_table,
                sort_columns=[
                    "base_measure",
                    "query_representation",
                    "query_kind",
                    "draw",
                ],
            ),
            distribution_gaps_sha256=dataframe_content_sha256(
                divergence_table, sort_columns=["base_measure", "seed", "query_id"]
            ),
        )
    report = {
        "protocol_version": str(params["protocol_version"]),
        "protocol": str(params["protocol_path"]),
        "input_versions": dict(params["input_versions"]),
        "population": {
            "training_rows": int(len(train)),
            "gallery_rows": int(len(gallery)),
            "training_queries": int(len(training_queries)),
            "evaluation_queries": int(len(evaluation_queries)),
            "known_training_pairs": int(known.sum()),
            "verified_training_pairs": int(verified.sum()),
            "contradicted_training_pairs": int((known & ~verified).sum()),
            "test_rows_read": False,
        },
        "resolved_configuration": params,
        "base_mass_hashes": base_mass_hashes,
        "calibration": {
            "selection_uses_validation_retrieval": False,
            "selected_learning_rate": selected_learning_rate,
            "candidates": candidate_summary.to_dict("records"),
        },
        "comparison": comparison,
        "tracking": tracking_records,
        "output_hashes": output_hashes,
        "decision": {
            "status": (
                "stable_learning_rate_selected"
                if selected_learning_rate is not None
                else "no_stable_learning_rate"
            ),
            "validation_only": True,
            "unknown_candidates_excluded_from_training": True,
            "hyperparameter_selected_from_training_diagnostics_only": True,
        },
        "known_limitations": [
            "Only four frozen atomic constraints have explicit contradicted candidates.",
            "The resulting four-query conjunction evaluation is underpowered.",
            "The historical test split is contaminated and was not read.",
            "The annotation list contains positive calls but no explicit absence evidence.",
        ],
    }
    return checkpoint_table, history_table, summary_table, report


def _natural_parameter_partitions(
    queries: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    atomic_ids = tuple(map(str, params["training_semantic_query_ids"]))
    conjunction_ids = tuple(map(str, params["evaluation_semantic_query_ids"]))
    if len(atomic_ids) != int(params["expected_training_queries"]) or len(conjunction_ids) != int(
        params["expected_evaluation_queries"]
    ):
        raise ValueError("E08 query identifiers do not match the frozen counts")
    if len(set(atomic_ids)) != len(atomic_ids) or len(set(conjunction_ids)) != len(conjunction_ids):
        raise ValueError("E08 query identifiers must be unique")
    positions = {
        value: index for index, value in enumerate(queries["semantic_query_id"].astype(str))
    }
    missing = set((*atomic_ids, *conjunction_ids)).difference(positions)
    if missing:
        raise ValueError(f"E08 frozen queries are missing: {sorted(missing)}")
    training_positions = np.asarray([positions[value] for value in atomic_ids], dtype=np.int64)
    evaluation_positions = np.asarray(
        [positions[value] for value in conjunction_ids], dtype=np.int64
    )
    training = queries.iloc[training_positions].reset_index(drop=True)
    evaluation = queries.iloc[evaluation_positions].reset_index(drop=True)
    if set(training["query_kind"].astype(str)) != {"atomic"}:
        raise ValueError("E08 training queries must all be atomic")
    if set(evaluation["query_kind"].astype(str)) != {"pair_conjunction"}:
        raise ValueError("E08 evaluation queries must all be pair conjunctions")
    if set(evaluation["controlled_split"].astype(str)) != {"atoms_seen_conjunction_unseen"}:
        raise ValueError("E08 evaluation queries changed controlled split")
    atomic_constraints = {
        str(values[0])
        for raw in training["constraint_ids_json"]
        if len(values := json.loads(str(raw))) == 1
    }
    pair_constraints = [json.loads(str(raw)) for raw in evaluation["constraint_ids_json"]]
    if len(atomic_constraints) != len(training) or any(
        len(values) != 2 or not set(map(str, values)) <= atomic_constraints
        for values in pair_constraints
    ):
        raise ValueError("E08 conjunctions must contain exactly two frozen training atoms")
    return training, evaluation, training_positions, evaluation_positions


def _training_state_masks(
    train: pd.DataFrame,
    queries: pd.DataFrame,
    states: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    query_index = {
        value: index for index, value in enumerate(queries["semantic_query_id"].astype(str))
    }
    sequence_index = {value: index for index, value in enumerate(train["sequence_id"].astype(str))}
    selected = states.loc[
        states["semantic_query_id"].astype(str).isin(query_index)
        & states["sequence_id"].astype(str).isin(sequence_index)
    ]
    verified = np.zeros((len(queries), len(train)), dtype=bool)
    known = np.zeros_like(verified)
    for row in selected.itertuples(index=False):
        query_position = query_index[str(row.semantic_query_id)]
        sequence_position = sequence_index[str(row.sequence_id)]
        known[query_position, sequence_position] = True
        verified[query_position, sequence_position] = str(row.state) == "verified"
    minimum = int(params["minimum_training_state_rows"])
    if np.any(verified.sum(axis=1) < minimum) or np.any((known & ~verified).sum(axis=1) < minimum):
        raise ValueError(
            "every E08 atom must have the frozen minimum verified and contradicted support"
        )
    return verified, known


def _base_log_mass(population: pd.DataFrame, base_measure: str) -> np.ndarray:
    if population.empty:
        raise ValueError("base measure needs a non-empty population")
    if base_measure == "uniform_plasmid":
        result = np.full(len(population), -math.log(len(population)), dtype=np.float64)
    elif base_measure == "uniform_v2_component":
        if (
            "leakage_component_v2" not in population
            or population["leakage_component_v2"].isna().any()
        ):
            raise ValueError("uniform component mass needs complete v2 leakage components")
        components = population["leakage_component_v2"].astype(str)
        sizes = components.value_counts()
        result = -math.log(len(sizes)) - np.log(components.map(sizes).to_numpy(dtype=np.float64))
    else:
        raise ValueError(f"unsupported E08 base measure: {base_measure}")
    if not np.isclose(np.exp(result).sum(), 1.0, rtol=1e-10, atol=1e-10):
        raise RuntimeError(f"E08 base measure does not normalize: {base_measure}")
    return result


def _natural_parameter_checkpoint(state: dict[str, Any], final_history: pd.Series) -> pd.DataFrame:
    sequence_head = np.asarray(state["sequence_head"], dtype=np.float32)
    text_head = np.asarray(state["text_head"], dtype=np.float32)
    return pd.DataFrame(
        [
            {
                "objective": "maximum_entropy",
                "base_measure": str(state["base_measure"]),
                "seed": int(state["seed"]),
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
                "contradicted_pairs": int(state["contradicted_pairs"]),
                "initial_sequence_head_sha256": str(state["initial_sequence_head_sha256"]),
                "initial_text_head_sha256": str(state["initial_text_head_sha256"]),
                "final_loss": float(final_history["loss"]),
                "final_query_norm_mean": float(final_history["query_norm_mean"]),
                "final_query_norm_max": float(final_history["query_norm_max"]),
                "final_sequence_norm_sample_mean": float(
                    final_history["sequence_norm_sample_mean"]
                ),
                "final_sequence_norm_sample_max": float(final_history["sequence_norm_sample_max"]),
            }
        ]
    )


def _natural_parameter_summary(
    per_seed: pd.DataFrame,
    bootstrap: pd.DataFrame,
    divergences: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary_k = int(params["primary_k"])
    selected = per_seed.loc[
        per_seed["query_kind"].eq("pair_conjunction") & per_seed["k"].eq(primary_k)
    ]
    values = ["verified_fraction", "contradicted_fraction", "unknown_fraction", "utility"]
    point = (
        selected.groupby(["base_measure", "query_representation"], sort=True)[values]
        .mean()
        .reset_index()
    )
    primary_draws = bootstrap.loc[bootstrap["query_kind"].eq("pair_conjunction")]
    intervals = (
        primary_draws.groupby(["base_measure", "query_representation"], sort=True)["utility"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "interval_lower", 0.975: "interval_upper"})
        .reset_index()
    )
    summary = point.merge(
        intervals, on=["base_measure", "query_representation"], validate="one_to_one"
    )
    summary.insert(2, "query_count", int(params["expected_evaluation_queries"]))
    expected_rows = {
        (measure, representation)
        for measure in EXPECTED_BASE_MEASURES
        for representation in ("direct_text", "atomic_sum")
    }
    observed_rows = set(zip(summary["base_measure"], summary["query_representation"], strict=True))
    if observed_rows != expected_rows:
        raise ValueError("E08 summary lacks a base-measure or query-representation result")

    draws = primary_draws.pivot(
        index="draw", columns=["base_measure", "query_representation"], values="utility"
    )
    component_difference = (
        draws[("uniform_v2_component", "atomic_sum")] - draws[("uniform_plasmid", "atomic_sum")]
    )
    component_lower, component_upper = component_difference.quantile([0.025, 0.975])
    atomic_points = summary.loc[summary["query_representation"].eq("atomic_sum")].set_index(
        "base_measure"
    )["utility"]
    observed = float(atomic_points["uniform_v2_component"] - atomic_points["uniform_plasmid"])
    threshold = float(params["minimum_practical_improvement"])
    if observed >= threshold and component_lower > 0.0:
        selection = "uniform_v2_component"
    elif observed <= -threshold and component_upper < 0.0:
        selection = "uniform_plasmid"
    else:
        selection = "indistinguishable"
    direct_vs_sum = {}
    for base_measure in EXPECTED_BASE_MEASURES:
        difference = draws[(base_measure, "atomic_sum")] - draws[(base_measure, "direct_text")]
        lower, upper = difference.quantile([0.025, 0.975])
        base_points = summary.loc[summary["base_measure"].eq(base_measure)].set_index(
            "query_representation"
        )["utility"]
        direct_vs_sum[base_measure] = {
            "atomic_sum_minus_direct_text": float(
                base_points["atomic_sum"] - base_points["direct_text"]
            ),
            "paired_component_bootstrap_95_interval": [float(lower), float(upper)],
            "mean_jensen_shannon_divergence": float(
                divergences.loc[
                    divergences["base_measure"].eq(base_measure),
                    "jensen_shannon_divergence",
                ].mean()
            ),
        }
    comparison = {
        "primary_metric": f"validation_pair_query_macro_utility_at_{primary_k}",
        "primary_query_representation": "atomic_sum",
        "uniform_plasmid": float(atomic_points["uniform_plasmid"]),
        "uniform_v2_component": float(atomic_points["uniform_v2_component"]),
        "uniform_v2_component_minus_uniform_plasmid": observed,
        "paired_component_bootstrap_95_interval": [
            float(component_lower),
            float(component_upper),
        ],
        "minimum_practical_improvement": threshold,
        "selection": selection,
        "direct_vs_atomic_sum": direct_vs_sum,
    }
    return summary.sort_values(
        ["base_measure", "query_representation"], kind="stable", ignore_index=True
    ), comparison


def _float64_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def _sum_atomic_queries(
    atomic_queries: pd.DataFrame,
    conjunction_queries: pd.DataFrame,
    atomic_vectors: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(atomic_vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(atomic_queries):
        raise ValueError("atomic query vectors do not align with atomic query metadata")
    positions: dict[str, int] = {}
    for index, raw in enumerate(atomic_queries["constraint_ids_json"]):
        identifiers = json.loads(str(raw))
        if not isinstance(identifiers, list) or len(identifiers) != 1:
            raise ValueError("atomic query must contain exactly one constraint")
        constraint_id = str(identifiers[0])
        if constraint_id in positions:
            raise ValueError("atomic query constraints are not unique")
        positions[constraint_id] = index

    rows = []
    for raw in conjunction_queries["constraint_ids_json"]:
        identifiers = json.loads(str(raw))
        if not isinstance(identifiers, list) or len(identifiers) != 2:
            raise ValueError("additive audit requires two-constraint conjunctions")
        try:
            left, right = (positions[str(value)] for value in identifiers)
        except KeyError as error:
            missing = error.args[0]
            raise ValueError(f"conjunction contains an unavailable atom: {missing}") from error
        rows.append(vectors[left] + vectors[right])
    result = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(result).all() or np.any(np.linalg.norm(result, axis=1) == 0.0):
        raise ValueError("summed atomic queries produced a zero or non-finite vector")
    return result


def _jensen_shannon_rows(left_scores: np.ndarray, right_scores: np.ndarray) -> np.ndarray:
    left = np.asarray(left_scores, dtype=np.float64)
    right = np.asarray(right_scores, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or not np.isfinite(left).all():
        raise ValueError("distribution score matrices must be finite and have equal shapes")
    if not np.isfinite(right).all():
        raise ValueError("distribution score matrices must be finite and have equal shapes")

    def log_probabilities(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    left_log_probability = log_probabilities(left)
    right_log_probability = log_probabilities(right)
    midpoint_log_probability = np.logaddexp(left_log_probability, right_log_probability) - math.log(
        2.0
    )
    return 0.5 * np.sum(
        np.exp(left_log_probability) * (left_log_probability - midpoint_log_probability)
        + np.exp(right_log_probability) * (right_log_probability - midpoint_log_probability),
        axis=1,
    )


def _additive_comparison(
    summaries: pd.DataFrame,
    bootstrap: pd.DataFrame,
    gaps: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    primary_k = int(params["primary_k"])
    selected = summaries.loc[
        summaries["objective"].eq("verified_set")
        & summaries["query_kind"].eq("pair_conjunction")
        & summaries["k"].eq(primary_k)
    ]
    point = selected.groupby("query_representation", sort=True)["utility"].mean()
    expected = {"atomic_sum", "direct_text"}
    if set(point.index) != expected:
        raise ValueError("additive audit lacks one query representation")
    draws = bootstrap.loc[
        bootstrap["objective"].eq("verified_set") & bootstrap["query_kind"].eq("pair_conjunction")
    ].pivot(index="draw", columns="query_representation", values="utility")
    if set(draws.columns) != expected or draws.isna().any(axis=None):
        raise ValueError("additive audit bootstrap lacks one query representation")
    differences = draws["atomic_sum"] - draws["direct_text"]
    difference_lower, difference_upper = differences.quantile([0.025, 0.975])
    atomic_lower, atomic_upper = draws["atomic_sum"].quantile([0.025, 0.975])
    return {
        "primary_metric": f"validation_pair_query_macro_utility_at_{primary_k}",
        "direct_text": float(point["direct_text"]),
        "atomic_sum": float(point["atomic_sum"]),
        "atomic_sum_minus_direct_text": float(point["atomic_sum"] - point["direct_text"]),
        "paired_component_bootstrap_95_interval": [
            float(difference_lower),
            float(difference_upper),
        ],
        "atomic_sum_component_bootstrap_95_interval": [
            float(atomic_lower),
            float(atomic_upper),
        ],
        "mean_jensen_shannon_divergence": float(gaps["jensen_shannon_divergence"].mean()),
        "interpretation": "exploratory_validation_only",
    }


def _query_partitions(
    queries: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    training_kind = params.get("training_query_kind")
    evaluation_kind = params.get("evaluation_query_kind")
    if training_kind is None and evaluation_kind is None:
        positions = np.arange(len(queries), dtype=np.int64)
        return queries, queries, positions, positions
    if (training_kind, evaluation_kind) != ("atomic", "pair_conjunction"):
        raise ValueError(
            "unseen-composition partitions must train on atomic and evaluate on pair queries"
        )

    training_positions = np.flatnonzero(queries["query_kind"].eq(training_kind).to_numpy())
    evaluation_positions = np.flatnonzero(queries["query_kind"].eq(evaluation_kind).to_numpy())
    training = queries.iloc[training_positions].reset_index(drop=True)
    evaluation = queries.iloc[evaluation_positions].reset_index(drop=True)
    expected_training = int(params["expected_training_queries"])
    expected_evaluation = int(params["expected_evaluation_queries"])
    if len(training) != expected_training or len(evaluation) != expected_evaluation:
        raise ValueError(
            "unseen-composition query counts changed: "
            f"expected {expected_training}/{expected_evaluation}, "
            f"observed {len(training)}/{len(evaluation)}"
        )
    if set(training["semantic_query_id"]) & set(evaluation["semantic_query_id"]):
        raise ValueError("training and evaluation semantic queries overlap")
    expected_split = str(params["expected_evaluation_controlled_split"])
    if set(evaluation["controlled_split"].astype(str)) != {expected_split}:
        raise ValueError("evaluation queries are not the frozen unseen conjunctions")

    atomic_constraints = {
        str(values[0])
        for raw in training["constraint_ids_json"]
        if len(values := json.loads(str(raw))) == 1
    }
    pair_constraints = [json.loads(str(raw)) for raw in evaluation["constraint_ids_json"]]
    if len(atomic_constraints) != len(training):
        raise ValueError("atomic training queries do not map one-to-one to constraints")
    if any(
        len(values) != 2 or not set(map(str, values)) <= atomic_constraints
        for values in pair_constraints
    ):
        raise ValueError("an evaluation conjunction contains an unseen atomic constraint")
    return training, evaluation, training_positions, evaluation_positions


def _prepare_matrices(
    train: pd.DataFrame,
    gallery: pd.DataFrame,
    queries: pd.DataFrame,
    dna_features: pd.DataFrame,
    text_features: pd.DataFrame,
    *,
    dna_candidate: str,
    text_candidate: str,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    dna, dna_state = fixed_representation_alignment._whiten_dna_candidates(
        train, gallery, {dna_candidate: dna_features}, epsilon=epsilon
    )
    text, text_state = fixed_representation_alignment._whiten_text_candidates(
        train,
        gallery,
        queries,
        {text_candidate: text_features},
        epsilon=epsilon,
    )
    query_matrix = text[text_candidate].queries
    if query_matrix is None:
        raise RuntimeError("selected Qwen feature artifact has no controlled-query matrix")
    states = pd.DataFrame([*dna_state, *text_state]).sort_values(
        ["feature_kind", "candidate_id"], kind="stable", ignore_index=True
    )
    return (
        dna[dna_candidate].train,
        dna[dna_candidate].gallery,
        query_matrix,
        states,
    )


def _training_verified_mask(
    train: pd.DataFrame,
    queries: pd.DataFrame,
    states: pd.DataFrame,
    params: dict[str, Any],
) -> np.ndarray:
    query_index = {
        value: index for index, value in enumerate(queries["semantic_query_id"].astype(str))
    }
    sequence_index = {value: index for index, value in enumerate(train["sequence_id"].astype(str))}
    selected = states.loc[
        states["state"].eq("verified")
        & states["semantic_query_id"].astype(str).isin(query_index)
        & states["sequence_id"].astype(str).isin(sequence_index)
    ]
    mask = np.zeros((len(queries), len(train)), dtype=bool)
    for row in selected.itertuples(index=False):
        mask[query_index[str(row.semantic_query_id)], sequence_index[str(row.sequence_id)]] = True
    support = mask.sum(axis=1)
    minimum = int(params["minimum_training_verified_rows"])
    if np.any(support < minimum):
        failed = queries.loc[support < minimum, "semantic_query_id"].astype(str).head(5).tolist()
        raise ValueError(f"controlled queries below {minimum} training positives: {failed}")
    return mask


def _validate_query_states(
    states: pd.DataFrame, manifest: dict[str, Any], params: dict[str, Any]
) -> None:
    required = {"semantic_query_id", "sequence_id", "state"}
    if states.empty or required.difference(states.columns):
        raise ValueError("training query-state artifact is empty or incomplete")
    if states.duplicated(["semantic_query_id", "sequence_id"]).any():
        raise ValueError("training query-state artifact repeats a query and sequence")
    if not set(states["state"].astype(str)) <= {"verified", "contradicted"}:
        raise ValueError("training query states contain an unsupported state")
    observed = dataframe_content_sha256(
        states, sort_columns=["semantic_query_id", "state", "sequence_id"]
    )
    expected = str(params["expected_training_query_states_sha256"])
    if observed != expected:
        raise ValueError(
            f"training query-state hash changed: expected {expected}, observed {observed}"
        )
    if manifest.get("output_content_hashes", {}).get("query_candidate_state_sha256") != expected:
        raise ValueError("query-benchmark manifest does not describe the training query states")


def _validate_feature(
    kind: str,
    features: pd.DataFrame,
    manifest: dict[str, Any],
    input_manifest: dict[str, Any],
    params: dict[str, Any],
) -> None:
    accepted = dict(params["accepted_feature_artifacts"])[kind]
    expected_candidate = str(accepted["candidate_id"])
    sort_columns = (
        ["candidate_id", "sequence_sha256"]
        if kind == "dna"
        else ["candidate_id", "text_role", "text_sha256"]
    )
    feature_hash = dataframe_content_sha256(features, sort_columns=sort_columns)
    if set(features["candidate_id"].astype(str)) != {expected_candidate}:
        raise ValueError(f"{kind} artifact contains an unexpected candidate")
    if feature_hash != accepted["features_sha256"]:
        raise ValueError(f"accepted {kind} feature hash changed")
    if json_content_sha256(manifest) != accepted["manifest_sha256"]:
        raise ValueError(f"accepted {kind} feature manifest changed")
    if manifest.get("input_manifest_sha256") != json_content_sha256(input_manifest):
        raise ValueError(f"accepted {kind} feature input changed")


def _validate_axes(
    params: dict[str, Any], *, gallery_rows: int
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    objectives = tuple(str(value) for value in params["objectives"])
    seeds = tuple(int(value) for value in params["probe"]["seeds"])
    cutoffs = tuple(int(value) for value in params["probe"]["cutoffs"])
    if objectives != EXPECTED_OBJECTIVES or seeds != EXPECTED_SEEDS:
        raise ValueError("Gate 2 objectives or seeds changed from the frozen protocol")
    if not cutoffs or len(cutoffs) != len(set(cutoffs)) or min(cutoffs) < 1:
        raise ValueError("Gate 2 cutoffs must be positive and unique")
    if max(cutoffs) > gallery_rows or int(params["primary_k"]) not in cutoffs:
        raise ValueError("Gate 2 cutoffs do not cover the frozen primary K")
    return objectives, seeds, cutoffs


def _checkpoint(state: dict[str, Any]) -> pd.DataFrame:
    sequence_head = np.asarray(state["sequence_head"], dtype=np.float32)
    text_head = np.asarray(state["text_head"], dtype=np.float32)
    return pd.DataFrame(
        [
            {
                "objective": str(state["objective"]),
                "seed": int(state["seed"]),
                "sequence_head": sequence_head.reshape(-1).tolist(),
                "sequence_head_rows": int(sequence_head.shape[0]),
                "sequence_head_columns": int(sequence_head.shape[1]),
                "sequence_head_sha256": _array_sha256(sequence_head),
                "text_head": text_head.reshape(-1).tolist(),
                "text_head_rows": int(text_head.shape[0]),
                "text_head_columns": int(text_head.shape[1]),
                "text_head_sha256": _array_sha256(text_head),
                "logit_scale": float(state["logit_scale"]),
                "updates": int(state["updates"]),
                "batch_rows": int(state["batch_rows"]),
                "sampler_sha256": str(state["sampler_sha256"]),
                "initial_sequence_head_sha256": str(state["initial_sequence_head_sha256"]),
                "initial_text_head_sha256": str(state["initial_text_head_sha256"]),
            }
        ]
    )


def _query_summary(metrics: pd.DataFrame, *, objective: str, seed: int) -> pd.DataFrame:
    values = [
        "verified_fraction",
        "contradicted_fraction",
        "unknown_fraction",
        "known_fraction",
        "utility",
    ]
    by_kind = (
        metrics.groupby(["query_kind", "k"], sort=True, observed=True)
        .agg(queries=("query_id", "nunique"), **{value: (value, "mean") for value in values})
        .reset_index()
    )
    combined = (
        metrics.groupby("k", sort=True)
        .agg(queries=("query_id", "nunique"), **{value: (value, "mean") for value in values})
        .reset_index()
    )
    combined.insert(0, "query_kind", "combined")
    return pd.concat([by_kind, combined], ignore_index=True).assign(objective=objective, seed=seed)


def _comparison(
    summaries: pd.DataFrame, bootstrap: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    primary_k = int(params["primary_k"])
    if "query_representation" in summaries:
        summaries = summaries.loc[summaries["query_representation"].eq("direct_text")]
        bootstrap = bootstrap.loc[bootstrap["query_representation"].eq("direct_text")]
    selected = summaries.loc[
        summaries["query_kind"].eq("pair_conjunction") & summaries["k"].eq(primary_k)
    ]
    point = selected.groupby("objective", sort=True)["utility"].mean()
    if set(point.index) != set(EXPECTED_OBJECTIVES):
        raise ValueError("primary comparison lacks one Gate 2 objective")
    draws = bootstrap.loc[bootstrap["query_kind"].eq("pair_conjunction")].pivot(
        index="draw", columns="objective", values="utility"
    )
    if set(draws.columns) != set(EXPECTED_OBJECTIVES) or draws.isna().any(axis=None):
        raise ValueError("paired bootstrap lacks one Gate 2 objective")
    differences = draws["verified_set"] - draws["paired_identity"]
    lower, upper = differences.quantile([0.025, 0.975])
    observed_difference = float(point["verified_set"] - point["paired_identity"])
    threshold = float(params["minimum_practical_improvement"])
    return {
        "primary_metric": f"validation_pair_query_macro_utility_at_{primary_k}",
        "paired_identity": float(point["paired_identity"]),
        "verified_set": float(point["verified_set"]),
        "verified_set_minus_paired_identity": observed_difference,
        "paired_component_bootstrap_95_interval": [float(lower), float(upper)],
        "minimum_practical_improvement": threshold,
        "supports_set_supervision": bool(observed_difference >= threshold and lower > 0.0),
    }


def _concat(frames: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True).sort_values(
        columns, kind="stable", ignore_index=True
    )


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float32).tobytes()).hexdigest()


def _start_wandb(
    params: dict[str, Any], *, objective: str, seed: int
) -> tuple[Any | None, dict[str, Any]]:
    tracking = dict(params["tracking"])
    record: dict[str, Any] = {"objective": objective, "seed": seed, "enabled": False}
    if not bool(tracking["enabled"]):
        record["status"] = "disabled_by_configuration"
        return None, record
    try:
        import wandb

        run_prefix = str(params.get("run_name_prefix", "gate2"))
        run = wandb.init(
            project=str(tracking["project"]),
            entity=tracking.get("entity"),
            name=f"{run_prefix}-{objective}-seed-{seed}",
            group=str(tracking["group"]),
            tags=list(tracking["tags"]),
            config={
                "protocol_version": str(params["protocol_version"]),
                "objective": objective,
                "seed": seed,
                "training_query_kind": params.get("training_query_kind", "all"),
                "evaluation_query_kind": params.get("evaluation_query_kind", "all"),
                "query_representations": params.get("query_representations", ["direct_text"]),
                "precision": str(params.get("precision", "float32")),
                "probe": dict(params["probe"]),
                "input_versions": dict(params["input_versions"]),
            },
            reinit="finish_previous",
        )
    except Exception as error:
        record.update(
            status="initialization_failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
        return None, record
    record.update(enabled=True, status="running", url=str(run.url), run_id=str(run.id))
    return run, record


def _log_wandb(
    run: Any | None,
    record: dict[str, Any],
    history: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    if run is None:
        return
    try:
        for row in history.itertuples(index=False):
            run.log(
                {
                    "train/loss": float(row.loss),
                    "train/logit_scale": float(row.logit_scale),
                    "train/true_positive_pairs": int(row.true_positive_pairs),
                    "train/unique_candidate_rows": int(row.unique_candidate_rows),
                },
                step=int(row.update),
            )
        for row in summary.itertuples(index=False):
            prefix = f"validation/{row.query_kind}/k{row.k}"
            run.summary[f"{prefix}/verified_fraction"] = float(row.verified_fraction)
            run.summary[f"{prefix}/contradicted_fraction"] = float(row.contradicted_fraction)
            run.summary[f"{prefix}/unknown_fraction"] = float(row.unknown_fraction)
            run.summary[f"{prefix}/utility"] = float(row.utility)
        record["status"] = "logged"
    except Exception as error:
        record.update(
            status="logging_failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )


def _log_natural_parameter_wandb(
    run: Any | None,
    record: dict[str, Any],
    history: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    if run is None:
        return
    try:
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
        for row in summary.itertuples(index=False):
            prefix = f"validation/{row.query_representation}/{row.query_kind}/k{row.k}"
            run.summary[f"{prefix}/verified_fraction"] = float(row.verified_fraction)
            run.summary[f"{prefix}/contradicted_fraction"] = float(row.contradicted_fraction)
            run.summary[f"{prefix}/unknown_fraction"] = float(row.unknown_fraction)
            run.summary[f"{prefix}/utility"] = float(row.utility)
        record["status"] = "logged"
    except Exception as error:
        record.update(
            status="logging_failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )


def _finish_wandb(run: Any | None, record: dict[str, Any], *, exit_code: int) -> None:
    if run is None:
        return
    try:
        run.finish(exit_code=exit_code)
        if record["status"] == "running":
            record["status"] = "finished_without_metrics"
        elif record["status"] == "logged":
            record["status"] = "complete"
    except Exception as error:
        record.update(
            status="finish_failed",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
