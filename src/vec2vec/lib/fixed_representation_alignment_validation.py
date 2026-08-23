"""Independent read-back validation for persisted E02b alignment evidence."""

from __future__ import annotations

import hashlib
import math
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import alignment_probe, fixed_representation_alignment
from vec2vec.lib.fixed_representation_invariance import json_content_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.lib.text import sha256_text

FLOAT_RECOMPUTATION_ATOL = 1e-6
FLOAT_RECOMPUTATION_RTOL = 1e-6
WHITENING_MEAN_ATOL = 1e-6
WHITENING_DIRECTION_COSINE_MINIMUM = 0.9999
WHITENING_SCALE_RTOL = 1e-3

OUTPUT_SORT_COLUMNS = {
    "whitening_state_sha256": ["feature_kind", "candidate_id"],
    "probe_checkpoints_sha256": ["dna_candidate_id", "text_candidate_id", "seed"],
    "training_history_sha256": [
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "epoch",
    ],
    "paired_metrics_sha256": ["dna_candidate_id", "text_candidate_id", "seed"],
    "query_rankings_sha256": [
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "query_id",
        "rank",
    ],
    "query_metrics_sha256": [
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "query_id",
        "k",
    ],
    "query_summaries_sha256": [
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "query_kind",
        "k",
    ],
    "bootstrap_draws_sha256": [
        "dna_candidate_id",
        "text_candidate_id",
        "query_kind",
        "draw",
    ],
}


def validate_alignment_outputs(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    query_states: pd.DataFrame,
    input_manifest: dict[str, Any],
    dna_features: dict[str, pd.DataFrame],
    text_features: dict[str, pd.DataFrame],
    whitening_state: pd.DataFrame,
    probe_checkpoints: pd.DataFrame,
    training_history: pd.DataFrame,
    paired_metrics: pd.DataFrame,
    query_rankings: pd.DataFrame,
    query_metrics: pd.DataFrame,
    query_summaries: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    selection_report: dict[str, Any],
    *,
    params: dict[str, Any],
    expected_compute_authorization: dict[str, Any],
) -> dict[str, Any]:
    """Validate one exact persisted validation-only E02b alignment version."""
    train, gallery = _validate_frozen_inputs(
        pairs,
        queries,
        query_states,
        input_manifest,
        params=params,
    )
    dna_candidates, text_candidates, seeds, cutoffs = _frozen_axes(params)
    configurations = set(product(dna_candidates, text_candidates, seeds))
    _validate_feature_tables(
        dna_features,
        text_features,
        dna_candidates=dna_candidates,
        text_candidates=text_candidates,
        accepted=params["accepted_feature_artifacts"],
    )
    _validate_report_contract(
        selection_report,
        input_manifest=input_manifest,
        training_rows=len(train),
        gallery_rows=len(gallery),
        query_rows=len(queries),
        dna_candidates=dna_candidates,
        text_candidates=text_candidates,
        seeds=seeds,
        params=params,
        expected_compute_authorization=expected_compute_authorization,
    )

    output_tables = {
        "whitening_state_sha256": whitening_state,
        "probe_checkpoints_sha256": probe_checkpoints,
        "training_history_sha256": training_history,
        "paired_metrics_sha256": paired_metrics,
        "query_rankings_sha256": query_rankings,
        "query_metrics_sha256": query_metrics,
        "query_summaries_sha256": query_summaries,
        "bootstrap_draws_sha256": bootstrap_draws,
    }
    output_hashes = {
        name: dataframe_content_sha256(table, sort_columns=OUTPUT_SORT_COLUMNS[name])
        for name, table in output_tables.items()
    }
    if output_hashes != selection_report.get("output_hashes"):
        raise ValueError("persisted E02b alignment output hashes changed")

    dimensions = _validate_whitening_state(
        whitening_state,
        dna_candidates=dna_candidates,
        text_candidates=text_candidates,
    )
    _validate_probe_checkpoints(
        probe_checkpoints,
        configurations=configurations,
        dimensions=dimensions,
        training_rows=len(train),
        params=params,
    )
    _validate_training_history(
        training_history,
        probe_checkpoints,
        configurations=configurations,
        params=params,
    )
    _validate_paired_metrics(
        paired_metrics,
        configurations=configurations,
        gallery_rows=len(gallery),
    )
    recomputed_metrics = _validate_query_evidence(
        query_rankings,
        query_metrics,
        queries,
        gallery,
        query_states,
        configurations=configurations,
        cutoffs=cutoffs,
    )
    _validate_query_summaries(
        query_summaries,
        recomputed_metrics,
        configurations=configurations,
        cutoffs=cutoffs,
        query_kinds=tuple(sorted(set(queries["query_kind"].astype(str)))),
    )
    _validate_bootstrap_draws(
        bootstrap_draws,
        queries,
        dna_candidates=dna_candidates,
        text_candidates=text_candidates,
        seeds=seeds,
        draws=int(params["probe"]["bootstrap_draws"]),
    )
    _validate_recomputed_model_evidence(
        pairs,
        queries,
        query_states,
        dna_features,
        text_features,
        whitening_state,
        probe_checkpoints,
        paired_metrics,
        query_rankings,
        query_metrics,
        bootstrap_draws,
        dna_candidates=dna_candidates,
        text_candidates=text_candidates,
        seeds=seeds,
        cutoffs=cutoffs,
        params=params,
    )
    expected_selection = fixed_representation_alignment._selection_report(
        query_summaries,
        bootstrap_draws,
        params["accepted_feature_artifacts"],
        params["probe"],
    )
    if json_content_sha256(expected_selection) != json_content_sha256(
        selection_report.get("selection", {})
    ):
        raise ValueError("persisted E02b alignment selection changed from its evidence")

    return {
        "status": "passed_e02b_alignment_readback",
        "selection_report_sha256": json_content_sha256(selection_report),
        "output_hashes": output_hashes,
        "planned_configurations": len(configurations),
        "completed_configurations": len(probe_checkpoints),
        "training_history_rows": len(training_history),
        "ranking_rows": len(query_rankings),
        "query_metric_rows": len(query_metrics),
        "bootstrap_rows": len(bootstrap_draws),
        "selected_pair": expected_selection["selected_pair"],
        "floating_recomputation_tolerance": {
            "absolute": FLOAT_RECOMPUTATION_ATOL,
            "relative": FLOAT_RECOMPUTATION_RTOL,
        },
        "whitening_refit_tolerance": {
            "mean_absolute": WHITENING_MEAN_ATOL,
            "direction_cosine_minimum": WHITENING_DIRECTION_COSINE_MINIMUM,
            "scale_relative": WHITENING_SCALE_RTOL,
        },
        "validation_only": True,
        "test_rows_read_by_alignment": False,
        "current_test_split_contaminated_before_e02b": True,
    }


def _validate_frozen_inputs(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    query_states: pd.DataFrame,
    input_manifest: dict[str, Any],
    *,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted = params.get("accepted_input_artifact")
    if not isinstance(accepted, dict):
        raise ValueError("accepted E02b input artifact is missing")
    observed_hashes = {
        "manifest_sha256": json_content_sha256(input_manifest),
        "pairs_sha256": dataframe_content_sha256(pairs, sort_columns=["panel_role", "sequence_id"]),
        "queries_sha256": dataframe_content_sha256(queries, sort_columns=["query_id"]),
        "query_states_sha256": dataframe_content_sha256(
            query_states, sort_columns=["semantic_query_id", "sequence_id"]
        ),
    }
    if any(observed_hashes[name] != accepted.get(name) for name in observed_hashes):
        raise ValueError("persisted alignment inputs differ from the accepted E02b input")
    manifest_hashes = input_manifest.get("output_hashes", {})
    if any(
        observed_hashes[name] != manifest_hashes.get(name)
        for name in ("pairs_sha256", "queries_sha256", "query_states_sha256")
    ):
        raise ValueError("persisted alignment inputs differ from their E02b input manifest")

    required_pairs = {
        "sequence_id",
        "sequence_sha256",
        "description_sha256",
        "panel_role",
        "similarity_component_primary",
        "length_bp",
        "component_size",
    }
    required_queries = {
        "query_id",
        "semantic_query_id",
        "query_kind",
        "canonical_query_text",
    }
    required_states = {"semantic_query_id", "sequence_id", "state"}
    _require_complete_frame(pairs, required_pairs, name="alignment pairs")
    _require_complete_frame(queries, required_queries, name="alignment queries")
    _require_complete_frame(query_states, required_states, name="alignment query states")
    if pairs["sequence_id"].duplicated().any():
        raise ValueError("persisted alignment pairs repeat sequence identifiers")
    if queries["query_id"].duplicated().any():
        raise ValueError("persisted alignment queries repeat query identifiers")
    if query_states.duplicated(["semantic_query_id", "sequence_id"]).any():
        raise ValueError("persisted alignment query states repeat query and sequence pairs")
    if not set(query_states["state"].astype(str)) <= {"verified", "contradicted"}:
        raise ValueError("persisted alignment query states contain an invalid label")
    if set(pairs["panel_role"].astype(str)) != {"alignment_train", "validation_gallery"}:
        raise ValueError("persisted alignment pairs contain an unexpected panel role")

    train = pairs.loc[pairs["panel_role"].eq("alignment_train")]
    gallery = pairs.loc[pairs["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    if len(train) != int(params["training_rows"]):
        raise ValueError("persisted alignment training row count changed")
    if set(query_states["sequence_id"].astype(str)).difference(gallery["sequence_id"].astype(str)):
        raise ValueError("persisted alignment labels refer to a sequence outside the gallery")
    if set(query_states["semantic_query_id"].astype(str)).difference(
        queries["semantic_query_id"].astype(str)
    ):
        raise ValueError("persisted alignment labels refer to an unknown semantic query")
    return train, gallery


def _frozen_axes(
    params: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    dna_candidates = tuple(
        sorted({str(params["tfidf"]["candidate_id"]), *map(str, params["dna_candidates"])})
    )
    text_candidates = tuple(sorted(map(str, params["text_candidates"])))
    seeds = tuple(int(seed) for seed in params["probe"]["seeds"])
    cutoffs = tuple(int(cutoff) for cutoff in params["probe"]["cutoffs"])
    if set(dna_candidates) != fixed_representation_alignment.EXPECTED_DNA_CANDIDATES:
        raise ValueError("configured DNA factorial differs from frozen E02b")
    if set(text_candidates) != fixed_representation_alignment.EXPECTED_TEXT_CANDIDATES:
        raise ValueError("configured text factorial differs from frozen E02b")
    if seeds != fixed_representation_alignment.EXPECTED_PROBE_SEEDS:
        raise ValueError("configured seeds differ from frozen E02b")
    if cutoffs != (1, 5, 10, 50):
        raise ValueError("configured retrieval cutoffs differ from frozen E02b")
    return dna_candidates, text_candidates, seeds, cutoffs


def _validate_feature_tables(
    dna_features: dict[str, pd.DataFrame],
    text_features: dict[str, pd.DataFrame],
    *,
    dna_candidates: tuple[str, ...],
    text_candidates: tuple[str, ...],
    accepted: dict[str, Any],
) -> None:
    """Bind recomputation to every exact feature table accepted before alignment."""
    expected_by_kind = {"dna": set(dna_candidates), "text": set(text_candidates)}
    observed_by_kind = {"dna": set(dna_features), "text": set(text_features)}
    if observed_by_kind != expected_by_kind:
        raise ValueError("persisted alignment validation loaded an incomplete feature factorial")
    if not isinstance(accepted, dict) or set(accepted) != {"dna", "text"}:
        raise ValueError("accepted E02b feature artifact records are incomplete")
    for feature_kind, tables in (("dna", dna_features), ("text", text_features)):
        accepted_kind = accepted.get(feature_kind)
        if (
            not isinstance(accepted_kind, dict)
            or set(accepted_kind) != expected_by_kind[feature_kind]
        ):
            raise ValueError(f"accepted {feature_kind} feature artifact records are incomplete")
        for candidate_id, table in tables.items():
            sort_columns = (
                ["candidate_id", "sequence_sha256"]
                if feature_kind == "dna"
                else ["candidate_id", "text_role", "text_sha256"]
            )
            observed_hash = dataframe_content_sha256(table, sort_columns=sort_columns)
            if observed_hash != accepted_kind[candidate_id].get("features_sha256"):
                raise ValueError(f"accepted feature table changed for {candidate_id}")
            if set(table["candidate_id"].astype(str)) != {candidate_id}:
                raise ValueError(f"feature table contains another candidate for {candidate_id}")


def _validate_report_contract(
    report: dict[str, Any],
    *,
    input_manifest: dict[str, Any],
    training_rows: int,
    gallery_rows: int,
    query_rows: int,
    dna_candidates: tuple[str, ...],
    text_candidates: tuple[str, ...],
    seeds: tuple[int, ...],
    params: dict[str, Any],
    expected_compute_authorization: dict[str, Any],
) -> None:
    if report.get("protocol_version") != params["protocol_version"]:
        raise ValueError("persisted alignment protocol version changed")
    if report.get("protocol") != params["protocol_path"]:
        raise ValueError("persisted alignment protocol path changed")
    if report.get("input_manifest_sha256") != json_content_sha256(input_manifest):
        raise ValueError("persisted alignment report changed its input manifest")
    expected_configurations = len(dna_candidates) * len(text_candidates) * len(seeds)
    expected_factorial = {
        "dna_candidates": list(dna_candidates),
        "text_candidates": list(text_candidates),
        "seeds": list(seeds),
        "planned_configurations": expected_configurations,
        "completed_configurations": expected_configurations,
    }
    if report.get("factorial") != expected_factorial:
        raise ValueError("persisted alignment report records an incomplete factorial")
    expected_population = {
        "training_rows": training_rows,
        "gallery_rows": gallery_rows,
        "queries": query_rows,
        "test_rows_read_by_alignment": False,
        "current_test_split_contaminated_before_e02b": True,
    }
    if report.get("validation_population") != expected_population:
        raise ValueError("persisted alignment report changed its validation population")
    if report.get("accepted_feature_artifacts") != params["accepted_feature_artifacts"]:
        raise ValueError("persisted alignment report changed its accepted feature artifacts")
    expected_configuration = {
        "device": str(params["device"]),
        "probe": dict(params["probe"]),
    }
    if report.get("resolved_alignment_configuration") != expected_configuration:
        raise ValueError("persisted alignment report changed its resolved configuration")
    expected_decision = {
        "status": "validation_pair_selected",
        "validation_only": True,
        "gate2_started": False,
    }
    if report.get("decision") != expected_decision:
        raise ValueError("persisted alignment report changed its validation-only decision")
    if report.get("compute_authorization") != expected_compute_authorization:
        raise ValueError("persisted alignment compute authorization changed")
    elapsed_seconds = float(report.get("alignment_elapsed_seconds", np.nan))
    if (
        not np.isfinite(elapsed_seconds)
        or elapsed_seconds <= 0.0
        or elapsed_seconds > float(expected_compute_authorization["instance_hour_limit"]) * 3600.0
    ):
        raise ValueError("persisted alignment elapsed time is outside its authorization")
    expected_cost = (
        elapsed_seconds
        / 3600.0
        * float(expected_compute_authorization["observed_instance_price_usd_per_hour"])
    )
    observed_cost = float(report.get("alignment_estimated_compute_cost_usd", np.nan))
    if not np.isfinite(observed_cost) or not np.isclose(
        observed_cost, expected_cost, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("persisted alignment estimated compute cost changed")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict) or not isinstance(runtime.get("packages"), dict):
        raise ValueError("persisted alignment report lacks runtime provenance")
    if not all(str(runtime.get(field, "")).strip() for field in ("python", "hostname", "machine")):
        raise ValueError("persisted alignment runtime provenance is incomplete")
    required_packages = {"numpy", "pandas", "scikit-learn", "torch"}
    packages = runtime["packages"]
    if any(not str(packages.get(name, "")).strip() for name in required_packages):
        raise ValueError("persisted alignment package provenance is incomplete")
    _validate_clean_git_provenance(report.get("git"))


def _validate_clean_git_provenance(git: Any) -> None:
    if not isinstance(git, dict):
        raise ValueError("persisted alignment report lacks Git provenance")
    commit = str(git.get("commit", ""))
    valid_commit = len(commit) == 40 and not set(commit).difference("0123456789abcdef")
    if (
        not valid_commit
        or git.get("worktree_dirty") is not False
        or git.get("changed_paths") != []
        or git.get("worktree_status_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError("persisted alignment run did not use a clean Git commit")


def _validate_whitening_state(
    state: pd.DataFrame,
    *,
    dna_candidates: tuple[str, ...],
    text_candidates: tuple[str, ...],
) -> dict[tuple[str, str], int]:
    required = {
        "feature_kind",
        "candidate_id",
        "input_dimension",
        "output_dimension",
        "epsilon_rule",
        "mean",
        "matrix",
        "matrix_rows",
        "matrix_columns",
        "mean_sha256",
        "matrix_sha256",
    }
    _require_complete_frame(state, required, name="alignment whitening state")
    expected = {
        *(("dna", candidate) for candidate in dna_candidates),
        *(("text", candidate) for candidate in text_candidates),
    }
    observed = set(
        zip(
            state["feature_kind"].astype(str),
            state["candidate_id"].astype(str),
            strict=True,
        )
    )
    if observed != expected or state.duplicated(["feature_kind", "candidate_id"]).any():
        raise ValueError("persisted whitening state does not cover the exact feature factorial")
    dimensions: dict[tuple[str, str], int] = {}
    for row in state.itertuples(index=False):
        input_dimension = int(row.input_dimension)
        output_dimension = int(row.output_dimension)
        matrix_rows = int(row.matrix_rows)
        matrix_columns = int(row.matrix_columns)
        if (
            input_dimension < 1
            or output_dimension != input_dimension
            or matrix_rows != input_dimension
            or matrix_columns != output_dimension
            or str(row.epsilon_rule) != "standard_deviation_plus_epsilon"
        ):
            raise ValueError("persisted whitening dimensions or rule changed")
        mean = np.asarray(row.mean, dtype=np.float32)
        matrix = np.asarray(row.matrix, dtype=np.float32)
        if mean.shape != (input_dimension,) or matrix.shape != (matrix_rows * matrix_columns,):
            raise ValueError("persisted whitening array shape changed")
        if not np.isfinite(mean).all() or not np.isfinite(matrix).all():
            raise ValueError("persisted whitening state contains a non-finite value")
        if _array_sha256(mean) != row.mean_sha256:
            raise ValueError("persisted whitening mean hash changed")
        if _array_sha256(matrix.reshape(matrix_rows, matrix_columns)) != row.matrix_sha256:
            raise ValueError("persisted whitening matrix hash changed")
        dimensions[(str(row.feature_kind), str(row.candidate_id))] = output_dimension
    return dimensions


def _validate_probe_checkpoints(
    checkpoints: pd.DataFrame,
    *,
    configurations: set[tuple[str, str, int]],
    dimensions: dict[tuple[str, str], int],
    training_rows: int,
    params: dict[str, Any],
) -> None:
    required = {
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "sequence_head",
        "sequence_head_rows",
        "sequence_head_columns",
        "sequence_head_sha256",
        "text_head",
        "text_head_rows",
        "text_head_columns",
        "text_head_sha256",
        "logit_scale",
        "batches_per_epoch",
        "last_batch_rows",
        "dropped_rows_per_epoch",
    }
    _require_complete_frame(checkpoints, required, name="alignment probe checkpoints")
    _require_exact_configurations(checkpoints, configurations, name="probe checkpoints")
    projection_dimension = int(params["probe"]["projection_dimension"])
    batch_size = int(params["probe"]["batch_size"])
    expected_batches = math.ceil(training_rows / batch_size)
    expected_last_batch = training_rows % batch_size or batch_size
    for row in checkpoints.itertuples(index=False):
        dna_dimension = dimensions[("dna", str(row.dna_candidate_id))]
        text_dimension = dimensions[("text", str(row.text_candidate_id))]
        if (int(row.sequence_head_rows), int(row.sequence_head_columns)) != (
            projection_dimension,
            dna_dimension,
        ):
            raise ValueError("persisted sequence-head shape changed")
        if (int(row.text_head_rows), int(row.text_head_columns)) != (
            projection_dimension,
            text_dimension,
        ):
            raise ValueError("persisted text-head shape changed")
        sequence_head = np.asarray(row.sequence_head, dtype=np.float32)
        text_head = np.asarray(row.text_head, dtype=np.float32)
        if sequence_head.shape != (projection_dimension * dna_dimension,):
            raise ValueError("persisted sequence-head values changed shape")
        if text_head.shape != (projection_dimension * text_dimension,):
            raise ValueError("persisted text-head values changed shape")
        if not np.isfinite(sequence_head).all() or not np.isfinite(text_head).all():
            raise ValueError("persisted probe checkpoint contains a non-finite head")
        if _array_sha256(sequence_head.reshape(projection_dimension, dna_dimension)) != str(
            row.sequence_head_sha256
        ):
            raise ValueError("persisted sequence-head hash changed")
        if _array_sha256(text_head.reshape(projection_dimension, text_dimension)) != str(
            row.text_head_sha256
        ):
            raise ValueError("persisted text-head hash changed")
        if not np.isfinite(float(row.logit_scale)):
            raise ValueError("persisted probe checkpoint has a non-finite logit scale")
        if (
            int(row.batches_per_epoch) != expected_batches
            or int(row.last_batch_rows) != expected_last_batch
            or int(row.dropped_rows_per_epoch) != 0
        ):
            raise ValueError("persisted probe checkpoint changed batch accounting")


def _validate_training_history(
    history: pd.DataFrame,
    checkpoints: pd.DataFrame,
    *,
    configurations: set[tuple[str, str, int]],
    params: dict[str, Any],
) -> None:
    required = {
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "epoch",
        "mean_loss",
        "logit_scale",
    }
    _require_complete_frame(history, required, name="alignment training history")
    _require_exact_configurations(history, configurations, name="training history")
    epochs = int(params["probe"]["epochs"])
    maximum_logit_scale = float(params["probe"]["maximum_logit_scale"])
    expected_rows = len(configurations) * epochs
    if len(history) != expected_rows:
        raise ValueError("persisted training history row count changed")
    for _, group in history.groupby(["dna_candidate_id", "text_candidate_id", "seed"], sort=False):
        observed_epochs = group["epoch"].astype(int).sort_values().to_numpy()
        if not np.array_equal(observed_epochs, np.arange(1, epochs + 1)):
            raise ValueError("persisted training history has incomplete epochs")
    numeric = history[["mean_loss", "logit_scale"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any(numeric[:, 0] < 0.0):
        raise ValueError("persisted training history contains an invalid loss or scale")
    if np.any(numeric[:, 1] <= 0.0) or np.any(numeric[:, 1] > maximum_logit_scale):
        raise ValueError("persisted training history logit scale is outside its contract")
    last = history.loc[history["epoch"].astype(int).eq(epochs)].merge(
        checkpoints[["dna_candidate_id", "text_candidate_id", "seed", "logit_scale"]].rename(
            columns={"logit_scale": "checkpoint_logit_scale"}
        ),
        on=["dna_candidate_id", "text_candidate_id", "seed"],
        validate="one_to_one",
    )
    expected_final = np.exp(
        np.minimum(
            last["checkpoint_logit_scale"].to_numpy(dtype=np.float64),
            np.log(maximum_logit_scale),
        )
    )
    if not np.allclose(
        last["logit_scale"].to_numpy(dtype=np.float64),
        expected_final,
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("persisted final training scale differs from its checkpoint")


def _validate_paired_metrics(
    metrics: pd.DataFrame,
    *,
    configurations: set[tuple[str, str, int]],
    gallery_rows: int,
) -> None:
    fraction_columns = [
        "sequence_to_description_r1",
        "sequence_to_description_r10",
        "description_to_sequence_r1",
        "description_to_sequence_r10",
    ]
    median_columns = ["sequence_to_description_median_rank", "description_to_sequence_median_rank"]
    required = {
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "rows",
        *fraction_columns,
        *median_columns,
    }
    _require_complete_frame(metrics, required, name="alignment paired metrics")
    _require_exact_configurations(metrics, configurations, name="paired metrics")
    if (
        len(metrics) != len(configurations)
        or not metrics["rows"].astype(int).eq(gallery_rows).all()
    ):
        raise ValueError("persisted paired metrics changed the validation population")
    fractions = metrics[fraction_columns].to_numpy(dtype=np.float64)
    medians = metrics[median_columns].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(fractions).all()
        or np.any(fractions < 0.0)
        or np.any(fractions > 1.0)
        or not np.isfinite(medians).all()
        or np.any(medians < 1.0)
        or np.any(medians > gallery_rows)
    ):
        raise ValueError("persisted paired metrics contain an invalid value")


def _validate_query_evidence(
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    queries: pd.DataFrame,
    gallery: pd.DataFrame,
    query_states: pd.DataFrame,
    *,
    configurations: set[tuple[str, str, int]],
    cutoffs: tuple[int, ...],
) -> pd.DataFrame:
    ranking_required = {
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "query_id",
        "semantic_query_id",
        "query_kind",
        "rank",
        "sequence_id",
        "similarity_component_primary",
        "length_bp",
        "component_size",
        "score",
        "state",
    }
    _require_complete_frame(rankings, ranking_required, name="alignment query rankings")
    _require_exact_configurations(rankings, configurations, name="query rankings")
    maximum_rank = max(cutoffs)
    expected_rows = len(configurations) * len(queries) * maximum_rank
    if len(rankings) != expected_rows:
        raise ValueError("persisted query ranking row count changed")
    if not np.isfinite(rankings["score"].to_numpy(dtype=np.float64)).all():
        raise ValueError("persisted query rankings contain a non-finite score")

    metric_required = {
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "query_id",
        "semantic_query_id",
        "query_kind",
        "k",
        "verified_fraction",
        "contradicted_fraction",
        "unknown_fraction",
        "known_fraction",
        "utility",
    }
    _require_complete_frame(metrics, metric_required, name="alignment query metrics")
    _require_exact_configurations(metrics, configurations, name="query metrics")

    query_lookup = queries.set_index("query_id")[["semantic_query_id", "query_kind"]]
    gallery_lookup = gallery.set_index("sequence_id")[
        ["similarity_component_primary", "length_bp", "component_size"]
    ]
    if set(rankings["query_id"].astype(str)) != set(query_lookup.index.astype(str)):
        raise ValueError("persisted rankings changed the query set")
    if set(rankings["sequence_id"].astype(str)).difference(gallery_lookup.index.astype(str)):
        raise ValueError("persisted rankings contain a sequence outside the gallery")
    for row in rankings.itertuples(index=False):
        query = query_lookup.loc[str(row.query_id)]
        candidate = gallery_lookup.loc[str(row.sequence_id)]
        if str(row.semantic_query_id) != str(query["semantic_query_id"]):
            raise ValueError("persisted ranking semantic-query identity changed")
        if str(row.query_kind) != str(query["query_kind"]):
            raise ValueError("persisted ranking query kind changed")
        if (
            str(row.similarity_component_primary) != str(candidate["similarity_component_primary"])
            or int(row.length_bp) != int(candidate["length_bp"])
            or int(row.component_size) != int(candidate["component_size"])
        ):
            raise ValueError("persisted ranking gallery metadata changed")
    state_lookup = {
        (str(row.semantic_query_id), str(row.sequence_id)): str(row.state)
        for row in query_states.itertuples(index=False)
    }
    expected_states = [
        state_lookup.get((str(row.semantic_query_id), str(row.sequence_id)), "unknown")
        for row in rankings.itertuples(index=False)
    ]
    if expected_states != rankings["state"].astype(str).tolist():
        raise ValueError("persisted ranking state differs from the frozen query labels")

    recomputed_rows: list[dict[str, Any]] = []
    gallery_positions = {
        sequence_id: position
        for position, sequence_id in enumerate(gallery["sequence_id"].astype(str))
    }
    grouping = ["dna_candidate_id", "text_candidate_id", "seed", "query_id"]
    for keys, group in rankings.groupby(grouping, sort=False):
        ordered = group.sort_values("rank", kind="stable")
        if not np.array_equal(ordered["rank"].astype(int), np.arange(1, maximum_rank + 1)):
            raise ValueError("persisted query rankings have incomplete ranks")
        if ordered["sequence_id"].duplicated().any():
            raise ValueError("persisted query ranking repeats a gallery sequence")
        scores = ordered["score"].to_numpy(dtype=np.float64)
        if np.any(scores[1:] > scores[:-1]):
            raise ValueError("persisted query ranking scores are not non-increasing")
        positions = ordered["sequence_id"].astype(str).map(gallery_positions).to_numpy()
        tied = scores[1:] == scores[:-1]
        if np.any(tied & (positions[1:] < positions[:-1])):
            raise ValueError("persisted query ranking changed stable gallery tie order")
        first = ordered.iloc[0]
        for cutoff in cutoffs:
            states = ordered["state"].astype(str).iloc[:cutoff]
            verified = float(states.eq("verified").mean())
            contradicted = float(states.eq("contradicted").mean())
            unknown = float(states.eq("unknown").mean())
            recomputed_rows.append(
                {
                    "query_id": str(keys[3]),
                    "semantic_query_id": str(first["semantic_query_id"]),
                    "query_kind": str(first["query_kind"]),
                    "k": cutoff,
                    "verified_fraction": verified,
                    "contradicted_fraction": contradicted,
                    "unknown_fraction": unknown,
                    "known_fraction": verified + contradicted,
                    "utility": verified - contradicted,
                    "dna_candidate_id": str(keys[0]),
                    "text_candidate_id": str(keys[1]),
                    "seed": int(keys[2]),
                }
            )
    recomputed = pd.DataFrame(recomputed_rows)
    _compare_metric_frames(recomputed, metrics, name="query metrics")
    return recomputed


def _validate_query_summaries(
    summaries: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    configurations: set[tuple[str, str, int]],
    cutoffs: tuple[int, ...],
    query_kinds: tuple[str, ...],
) -> None:
    required = {
        "dna_candidate_id",
        "text_candidate_id",
        "seed",
        "query_kind",
        "k",
        "queries",
        "verified_fraction",
        "contradicted_fraction",
        "unknown_fraction",
        "known_fraction",
        "utility",
    }
    _require_complete_frame(summaries, required, name="alignment query summaries")
    _require_exact_configurations(summaries, configurations, name="query summaries")
    expected_rows = len(configurations) * len(cutoffs) * (len(query_kinds) + 1)
    if len(summaries) != expected_rows:
        raise ValueError("persisted query summary row count changed")
    value_columns = [
        "verified_fraction",
        "contradicted_fraction",
        "unknown_fraction",
        "known_fraction",
        "utility",
    ]
    grouped = (
        metrics.groupby(
            ["dna_candidate_id", "text_candidate_id", "seed", "query_kind", "k"],
            sort=True,
            observed=True,
        )
        .agg(
            queries=("query_id", "nunique"),
            **{column: (column, "mean") for column in value_columns},
        )
        .reset_index()
    )
    combined = (
        metrics.groupby(["dna_candidate_id", "text_candidate_id", "seed", "k"], sort=True)
        .agg(
            queries=("query_id", "nunique"),
            **{column: (column, "mean") for column in value_columns},
        )
        .reset_index()
    )
    combined.insert(3, "query_kind", "combined")
    recomputed = pd.concat([grouped, combined], ignore_index=True)
    _compare_metric_frames(recomputed, summaries, name="query summaries")


def _validate_bootstrap_draws(
    bootstrap: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    dna_candidates: tuple[str, ...],
    text_candidates: tuple[str, ...],
    seeds: tuple[int, ...],
    draws: int,
) -> None:
    required = {
        "dna_candidate_id",
        "text_candidate_id",
        "query_kind",
        "draw",
        "k",
        "utility",
        "probe_seeds",
        "query_seed_observations",
    }
    _require_complete_frame(bootstrap, required, name="alignment bootstrap draws")
    query_kind_counts = queries["query_kind"].astype(str).value_counts().to_dict()
    query_kind_counts["combined"] = len(queries)
    expected_pairs = set(product(dna_candidates, text_candidates))
    observed_pairs = set(
        zip(
            bootstrap["dna_candidate_id"].astype(str),
            bootstrap["text_candidate_id"].astype(str),
            strict=True,
        )
    )
    if observed_pairs != expected_pairs:
        raise ValueError("persisted bootstrap changed the DNA-by-text candidate pairs")
    expected_rows = len(expected_pairs) * len(query_kind_counts) * draws
    if len(bootstrap) != expected_rows:
        raise ValueError("persisted bootstrap row count changed")
    if set(bootstrap["query_kind"].astype(str)) != set(query_kind_counts):
        raise ValueError("persisted bootstrap changed the query-kind populations")
    utility = bootstrap["utility"].to_numpy(dtype=np.float64)
    if not np.isfinite(utility).all() or np.any(utility < -1.0) or np.any(utility > 1.0):
        raise ValueError("persisted bootstrap contains an invalid utility")
    if not bootstrap["k"].astype(int).eq(10).all():
        raise ValueError("persisted bootstrap changed the primary retrieval cutoff")
    if not bootstrap["probe_seeds"].astype(int).eq(len(seeds)).all():
        raise ValueError("persisted bootstrap changed its probe-seed count")
    for keys, group in bootstrap.groupby(
        ["dna_candidate_id", "text_candidate_id", "query_kind"], sort=False
    ):
        if not np.array_equal(group["draw"].astype(int).sort_values(), np.arange(draws)):
            raise ValueError("persisted bootstrap has incomplete draw indices")
        expected_observations = int(query_kind_counts[str(keys[2])]) * len(seeds)
        if not group["query_seed_observations"].astype(int).eq(expected_observations).all():
            raise ValueError("persisted bootstrap changed its query-seed observations")


def _validate_recomputed_model_evidence(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    query_states: pd.DataFrame,
    dna_features: dict[str, pd.DataFrame],
    text_features: dict[str, pd.DataFrame],
    whitening_state: pd.DataFrame,
    checkpoints: pd.DataFrame,
    paired_metrics: pd.DataFrame,
    query_rankings: pd.DataFrame,
    query_metrics: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    *,
    dna_candidates: tuple[str, ...],
    text_candidates: tuple[str, ...],
    seeds: tuple[int, ...],
    cutoffs: tuple[int, ...],
    params: dict[str, Any],
) -> None:
    """Reconstruct all score-derived evidence from accepted features and saved heads."""
    train = pairs.loc[pairs["panel_role"].eq("alignment_train")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    gallery = pairs.loc[pairs["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    ordered_queries = queries.sort_values("query_id", kind="stable", ignore_index=True)
    dna_gallery, text_gallery, text_queries = _reconstruct_whitened_matrices(
        train,
        gallery,
        ordered_queries,
        dna_features,
        text_features,
        whitening_state,
        dna_candidates=dna_candidates,
        text_candidates=text_candidates,
        epsilon=float(params["probe"]["whitening_epsilon"]),
    )
    if checkpoints.duplicated(["dna_candidate_id", "text_candidate_id", "seed"]).any():
        raise ValueError("persisted probe checkpoints repeat a factorial configuration")
    checkpoint_lookup = checkpoints.set_index(["dna_candidate_id", "text_candidate_id", "seed"])

    paired_rows: list[dict[str, Any]] = []
    ranking_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []
    sequence_groups = gallery["sequence_sha256"].astype(str).to_numpy()
    description_groups = gallery["description_sha256"].astype(str).to_numpy()
    bootstrap_draw_count = int(params["probe"]["bootstrap_draws"])
    bootstrap_seed = int(params["probe"]["bootstrap_seed"])

    for dna_candidate_id in dna_candidates:
        for text_candidate_id in text_candidates:
            scores_by_seed: list[np.ndarray] = []
            for seed in seeds:
                checkpoint = checkpoint_lookup.loc[(dna_candidate_id, text_candidate_id, seed)]
                sequence_head = np.asarray(checkpoint["sequence_head"], dtype=np.float32).reshape(
                    int(checkpoint["sequence_head_rows"]),
                    int(checkpoint["sequence_head_columns"]),
                )
                text_head = np.asarray(checkpoint["text_head"], dtype=np.float32).reshape(
                    int(checkpoint["text_head_rows"]),
                    int(checkpoint["text_head_columns"]),
                )
                projected_sequence = alignment_probe.project(
                    dna_gallery[dna_candidate_id], sequence_head
                )
                projected_text = alignment_probe.project(text_gallery[text_candidate_id], text_head)
                projected_queries = alignment_probe.project(
                    text_queries[text_candidate_id], text_head
                )
                paired_rows.append(
                    {
                        "dna_candidate_id": dna_candidate_id,
                        "text_candidate_id": text_candidate_id,
                        "seed": seed,
                        **alignment_probe.paired_retrieval_metrics(
                            projected_sequence,
                            projected_text,
                            sequence_groups,
                            description_groups,
                        ),
                    }
                )
                rankings, metrics, scores = alignment_probe.query_rankings_and_metrics(
                    projected_queries,
                    projected_sequence,
                    ordered_queries,
                    gallery,
                    query_states,
                    cutoffs=cutoffs,
                )
                scores_by_seed.append(scores)
                ranking_frames.append(
                    rankings.assign(
                        dna_candidate_id=dna_candidate_id,
                        text_candidate_id=text_candidate_id,
                        seed=seed,
                    )
                )
                metric_frames.append(
                    metrics.assign(
                        dna_candidate_id=dna_candidate_id,
                        text_candidate_id=text_candidate_id,
                        seed=seed,
                    )
                )
            draws = alignment_probe.whole_component_bootstrap_draws(
                scores_by_seed,
                ordered_queries,
                gallery,
                query_states,
                k=10,
                draws=bootstrap_draw_count,
                seed=bootstrap_seed,
            )
            bootstrap_frames.append(
                draws.assign(
                    dna_candidate_id=dna_candidate_id,
                    text_candidate_id=text_candidate_id,
                )
            )

    expected_paired = pd.DataFrame(paired_rows)
    expected_rankings = pd.concat(ranking_frames, ignore_index=True)
    expected_metrics = pd.concat(metric_frames, ignore_index=True)
    expected_bootstrap = pd.concat(bootstrap_frames, ignore_index=True)
    _require_recomputed_table(
        expected_paired,
        paired_metrics,
        sort_columns=OUTPUT_SORT_COLUMNS["paired_metrics_sha256"],
        name="paired metrics",
    )
    _require_recomputed_table(
        expected_rankings,
        query_rankings,
        sort_columns=OUTPUT_SORT_COLUMNS["query_rankings_sha256"],
        name="query rankings",
    )
    _require_recomputed_table(
        expected_metrics,
        query_metrics,
        sort_columns=OUTPUT_SORT_COLUMNS["query_metrics_sha256"],
        name="query metrics",
    )
    _require_recomputed_table(
        expected_bootstrap,
        bootstrap_draws,
        sort_columns=OUTPUT_SORT_COLUMNS["bootstrap_draws_sha256"],
        name="bootstrap draws",
    )


def _reconstruct_whitened_matrices(
    train: pd.DataFrame,
    gallery: pd.DataFrame,
    queries: pd.DataFrame,
    dna_features: dict[str, pd.DataFrame],
    text_features: dict[str, pd.DataFrame],
    whitening_state: pd.DataFrame,
    *,
    dna_candidates: tuple[str, ...],
    text_candidates: tuple[str, ...],
    epsilon: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Refit train-only whitening and return matrices from the persisted accepted features."""
    state_lookup = whitening_state.set_index(["feature_kind", "candidate_id"])
    dna_gallery: dict[str, np.ndarray] = {}
    text_gallery: dict[str, np.ndarray] = {}
    text_queries: dict[str, np.ndarray] = {}
    for candidate_id in dna_candidates:
        features = dna_features[candidate_id]
        train_matrix = _join_feature_embeddings(train, features, key="sequence_sha256")
        gallery_matrix = _join_feature_embeddings(gallery, features, key="sequence_sha256")
        whitening = _validated_persisted_whitening(
            train_matrix,
            state_lookup.loc[("dna", candidate_id)],
            candidate_id=candidate_id,
            epsilon=epsilon,
        )
        dna_gallery[candidate_id] = whitening.transform(gallery_matrix)

    query_keys = queries.loc[:, ["query_id"]].copy()
    query_keys["text_sha256"] = queries["canonical_query_text"].astype(str).map(sha256_text)
    for candidate_id in text_candidates:
        features = text_features[candidate_id]
        documents = features.loc[features["text_role"].eq("document")]
        query_features = features.loc[features["text_role"].eq("query")]
        train_matrix = _join_feature_embeddings(
            train,
            documents,
            key="description_sha256",
            feature_key="text_sha256",
        )
        gallery_matrix = _join_feature_embeddings(
            gallery,
            documents,
            key="description_sha256",
            feature_key="text_sha256",
        )
        query_matrix = _join_feature_embeddings(
            query_keys,
            query_features,
            key="text_sha256",
        )
        whitening = _validated_persisted_whitening(
            train_matrix,
            state_lookup.loc[("text", candidate_id)],
            candidate_id=candidate_id,
            epsilon=epsilon,
        )
        text_gallery[candidate_id] = whitening.transform(gallery_matrix)
        text_queries[candidate_id] = whitening.transform(query_matrix)
    return dna_gallery, text_gallery, text_queries


def _join_feature_embeddings(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    *,
    key: str,
    feature_key: str | None = None,
) -> np.ndarray:
    feature_key = feature_key or key
    required = {feature_key, "embedding", "embedding_dimension"}
    _require_complete_frame(features, required, name="accepted feature table")
    if features[feature_key].duplicated().any():
        raise ValueError(f"accepted feature table repeats {feature_key}")
    joined = (
        rows.loc[:, [key]]
        .assign(_row_order=np.arange(len(rows)))
        .merge(
            features.loc[:, [feature_key, "embedding", "embedding_dimension"]].rename(
                columns={feature_key: key}
            ),
            on=key,
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        .sort_values("_row_order", kind="stable")
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"accepted feature join left {unmatched} {key} values unmatched")
    dimensions = set(joined["embedding_dimension"].astype(int))
    if len(dimensions) != 1:
        raise ValueError("accepted feature embedding dimension changed within a candidate")
    dimension = dimensions.pop()
    vectors = [np.asarray(value, dtype=np.float32) for value in joined["embedding"]]
    if any(vector.shape != (dimension,) for vector in vectors):
        raise ValueError("accepted feature embedding shape changed")
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("accepted feature table contains a non-finite embedding")
    return matrix


def _validated_persisted_whitening(
    train_matrix: np.ndarray,
    persisted: pd.Series,
    *,
    candidate_id: str,
    epsilon: float,
) -> alignment_probe.Whitening:
    """Check a persisted PCA transform against a tolerant, sign-invariant refit."""
    persisted_mean = np.asarray(persisted["mean"], dtype=np.float32)
    persisted_matrix = np.asarray(persisted["matrix"], dtype=np.float32).reshape(
        int(persisted["matrix_rows"]), int(persisted["matrix_columns"])
    )
    fitted = alignment_probe.Whitening.fit(train_matrix, epsilon=epsilon)
    if not np.allclose(
        fitted.mean,
        persisted_mean,
        atol=WHITENING_MEAN_ATOL,
        rtol=FLOAT_RECOMPUTATION_RTOL,
    ):
        raise ValueError(f"persisted whitening state was not train-fitted for {candidate_id}")
    fitted_norms = np.linalg.norm(fitted.matrix.astype(np.float64), axis=0)
    persisted_norms = np.linalg.norm(persisted_matrix.astype(np.float64), axis=0)
    if np.any(fitted_norms == 0.0) or np.any(persisted_norms == 0.0):
        raise ValueError(f"persisted whitening state has a zero direction for {candidate_id}")
    direction_cosines = np.abs(
        np.sum(fitted.matrix.astype(np.float64) * persisted_matrix.astype(np.float64), axis=0)
        / (fitted_norms * persisted_norms)
    )
    # LAPACK can flip singular-vector signs and change low bits across hosts. Direction cosine
    # and scale tolerances validate the same fitted transform without requiring bit identity.
    if np.any(direction_cosines < WHITENING_DIRECTION_COSINE_MINIMUM) or not np.allclose(
        fitted_norms,
        persisted_norms,
        atol=1e-5,
        rtol=WHITENING_SCALE_RTOL,
    ):
        raise ValueError(f"persisted whitening state was not train-fitted for {candidate_id}")
    return alignment_probe.Whitening(mean=persisted_mean, matrix=persisted_matrix)


def _require_recomputed_table(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    sort_columns: list[str],
    name: str,
) -> None:
    float_columns_by_name = {
        "paired metrics": {
            "sequence_to_description_r1",
            "sequence_to_description_r10",
            "sequence_to_description_median_rank",
            "description_to_sequence_r1",
            "description_to_sequence_r10",
            "description_to_sequence_median_rank",
        },
        "query rankings": {"score"},
        "query metrics": {
            "verified_fraction",
            "contradicted_fraction",
            "unknown_fraction",
            "known_fraction",
            "utility",
        },
        "bootstrap draws": {"utility"},
    }
    float_columns = float_columns_by_name[name]
    if set(expected.columns) != set(observed.columns):
        raise ValueError(f"persisted {name} differ from accepted features and model state")
    columns = sorted(expected.columns)
    exact_columns = [column for column in columns if column not in float_columns]
    expected_sorted = expected.sort_values(sort_columns, kind="stable", ignore_index=True)
    observed_sorted = observed.sort_values(sort_columns, kind="stable", ignore_index=True)
    if len(expected_sorted) != len(observed_sorted) or not expected_sorted[exact_columns].astype(
        str
    ).equals(observed_sorted[exact_columns].astype(str)):
        raise ValueError(f"persisted {name} differ from accepted features and model state")
    expected_values = expected_sorted[sorted(float_columns)].to_numpy(dtype=np.float64)
    observed_values = observed_sorted[sorted(float_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(observed_values).all() or not np.allclose(
        expected_values,
        observed_values,
        atol=FLOAT_RECOMPUTATION_ATOL,
        rtol=FLOAT_RECOMPUTATION_RTOL,
    ):
        raise ValueError(f"persisted {name} differ from accepted features and model state")


def _compare_metric_frames(expected: pd.DataFrame, observed: pd.DataFrame, *, name: str) -> None:
    keys = [
        column
        for column in (
            "dna_candidate_id",
            "text_candidate_id",
            "seed",
            "query_id",
            "semantic_query_id",
            "query_kind",
            "k",
        )
        if column in expected.columns
    ]
    values = [
        column
        for column in (
            "queries",
            "verified_fraction",
            "contradicted_fraction",
            "unknown_fraction",
            "known_fraction",
            "utility",
        )
        if column in expected.columns
    ]
    expected_sorted = expected.sort_values(keys, kind="stable", ignore_index=True)
    observed_sorted = observed.sort_values(keys, kind="stable", ignore_index=True)
    if len(expected_sorted) != len(observed_sorted):
        raise ValueError(f"persisted {name} row count differs from recomputation")
    if not expected_sorted[keys].astype(str).equals(observed_sorted[keys].astype(str)):
        raise ValueError(f"persisted {name} identities differ from recomputation")
    expected_values = expected_sorted[values].to_numpy(dtype=np.float64)
    observed_values = observed_sorted[values].to_numpy(dtype=np.float64)
    if not np.isfinite(observed_values).all() or not np.allclose(
        expected_values, observed_values, atol=1e-12, rtol=0.0
    ):
        raise ValueError(f"persisted {name} values differ from recomputation")


def _require_exact_configurations(
    frame: pd.DataFrame,
    expected: set[tuple[str, str, int]],
    *,
    name: str,
) -> None:
    observed = set(
        zip(
            frame["dna_candidate_id"].astype(str),
            frame["text_candidate_id"].astype(str),
            frame["seed"].astype(int),
            strict=True,
        )
    )
    if observed != expected:
        raise ValueError(f"persisted {name} changed the frozen factorial configurations")


def _require_complete_frame(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"persisted {name} are missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any(axis=None):
        raise ValueError(f"persisted {name} must be non-empty and complete")


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()
