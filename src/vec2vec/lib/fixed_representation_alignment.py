"""Frozen E02b factorial alignment and validation-only model selection."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import alignment_probe
from vec2vec.lib.fixed_representation_invariance import json_content_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.lib.text import sha256_text

EXPECTED_DNA_CANDIDATES = {
    "tfidf_6mer_svd_512",
    "carbon_500m",
    "generanno_prokaryote_500m",
    "generator_v2_prokaryote_1_2b",
}
EXPECTED_TEXT_CANDIDATES = {
    "bge_base_en_v1_5",
    "gte_modernbert_base",
    "qwen3_embedding_0_6b",
}
EXPECTED_PROBE_SEEDS = (13, 42, 20260818)


@dataclass(frozen=True)
class CandidateMatrices:
    """Whitened matrices in the frozen train, gallery, and optional query order."""

    train: np.ndarray
    gallery: np.ndarray
    queries: np.ndarray | None


def run_factorial_alignment(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    query_states: pd.DataFrame,
    input_manifest: dict[str, Any],
    dna_features: dict[str, pd.DataFrame],
    dna_manifests: dict[str, dict[str, Any]],
    text_features: dict[str, pd.DataFrame],
    text_manifests: dict[str, dict[str, Any]],
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
    pd.DataFrame,
    dict[str, Any],
]:
    """Fit all frozen DNA-by-text probes and select from validation utility only."""
    _ensure_before_deadline(deadline_monotonic, operation="factorial alignment setup")
    train, gallery, ordered_queries = _validate_alignment_inputs(
        pairs, queries, query_states, input_manifest, params
    )
    accepted = _validate_factorial_feature_artifacts(
        dna_features,
        dna_manifests,
        text_features,
        text_manifests,
        input_manifest,
        params,
    )
    probe = dict(params["probe"])
    seeds, cutoffs = validated_probe_axes(probe, gallery_rows=len(gallery))

    dna_matrices, dna_whitening = _whiten_dna_candidates(
        train, gallery, dna_features, epsilon=float(probe["whitening_epsilon"])
    )
    text_matrices, text_whitening = _whiten_text_candidates(
        train,
        gallery,
        ordered_queries,
        text_features,
        epsilon=float(probe["whitening_epsilon"]),
    )
    whitening_state = pd.DataFrame([*dna_whitening, *text_whitening]).sort_values(
        ["feature_kind", "candidate_id"], kind="stable", ignore_index=True
    )

    checkpoint_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    paired_rows: list[dict[str, Any]] = []
    ranking_frames: list[pd.DataFrame] = []
    query_metric_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    bootstrap_frames: list[pd.DataFrame] = []

    for dna_candidate_id in sorted(dna_matrices):
        for text_candidate_id in sorted(text_matrices):
            _ensure_before_deadline(
                deadline_monotonic,
                operation=f"alignment pair {dna_candidate_id} x {text_candidate_id}",
            )
            dna = dna_matrices[dna_candidate_id]
            text = text_matrices[text_candidate_id]
            if text.queries is None:
                raise RuntimeError("text candidate has no frozen query matrix")
            scores_by_seed: list[np.ndarray] = []
            for seed in seeds:
                _ensure_before_deadline(
                    deadline_monotonic,
                    operation=(
                        f"alignment pair {dna_candidate_id} x {text_candidate_id} seed {seed}"
                    ),
                )
                state, history = alignment_probe.train_alignment_probe(
                    dna.train,
                    text.train,
                    train["sequence_sha256"].astype(str).to_numpy(),
                    train["description_sha256"].astype(str).to_numpy(),
                    seed=seed,
                    projection_dimension=int(probe["projection_dimension"]),
                    epochs=int(probe["epochs"]),
                    batch_size=int(probe["batch_size"]),
                    learning_rate=float(probe["learning_rate"]),
                    weight_decay=float(probe["weight_decay"]),
                    initial_temperature=float(probe["initial_temperature"]),
                    maximum_logit_scale=float(probe["maximum_logit_scale"]),
                    device=str(params["device"]),
                    deadline_monotonic=deadline_monotonic,
                )
                sequence_gallery = alignment_probe.project(dna.gallery, state["sequence_head"])
                text_gallery = alignment_probe.project(text.gallery, state["text_head"])
                query_vectors = alignment_probe.project(text.queries, state["text_head"])
                scores_by_seed.append(query_vectors @ sequence_gallery.T)

                checkpoint_frames.append(
                    _checkpoint_frame(dna_candidate_id, text_candidate_id, seed, state)
                )
                history_frames.append(
                    history.assign(
                        dna_candidate_id=dna_candidate_id,
                        text_candidate_id=text_candidate_id,
                        seed=seed,
                    )
                )
                paired_rows.append(
                    {
                        "dna_candidate_id": dna_candidate_id,
                        "text_candidate_id": text_candidate_id,
                        "seed": seed,
                        **alignment_probe.paired_retrieval_metrics(
                            sequence_gallery,
                            text_gallery,
                            gallery["sequence_sha256"].astype(str).to_numpy(),
                            gallery["description_sha256"].astype(str).to_numpy(),
                        ),
                    }
                )
                rankings, metrics, observed_scores = alignment_probe.query_rankings_and_metrics(
                    query_vectors,
                    sequence_gallery,
                    ordered_queries,
                    gallery,
                    query_states,
                    cutoffs=cutoffs,
                )
                if not np.array_equal(observed_scores, scores_by_seed[-1]):
                    raise RuntimeError("query score matrix changed between ranking and bootstrap")
                ranking_frames.append(
                    rankings.assign(
                        dna_candidate_id=dna_candidate_id,
                        text_candidate_id=text_candidate_id,
                        seed=seed,
                    )
                )
                query_metric_frames.append(
                    metrics.assign(
                        dna_candidate_id=dna_candidate_id,
                        text_candidate_id=text_candidate_id,
                        seed=seed,
                    )
                )
                summary_frames.append(
                    _query_macro_summary(
                        metrics,
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
                draws=int(probe["bootstrap_draws"]),
                seed=int(probe["bootstrap_seed"]),
                deadline_monotonic=deadline_monotonic,
            )
            bootstrap_frames.append(
                draws.assign(
                    dna_candidate_id=dna_candidate_id,
                    text_candidate_id=text_candidate_id,
                )
            )

    checkpoints = pd.concat(checkpoint_frames, ignore_index=True).sort_values(
        ["dna_candidate_id", "text_candidate_id", "seed"], kind="stable", ignore_index=True
    )
    histories = pd.concat(history_frames, ignore_index=True).sort_values(
        ["dna_candidate_id", "text_candidate_id", "seed", "epoch"],
        kind="stable",
        ignore_index=True,
    )
    paired_metrics = pd.DataFrame(paired_rows).sort_values(
        ["dna_candidate_id", "text_candidate_id", "seed"], kind="stable", ignore_index=True
    )
    rankings = pd.concat(ranking_frames, ignore_index=True).sort_values(
        ["dna_candidate_id", "text_candidate_id", "seed", "query_id", "rank"],
        kind="stable",
        ignore_index=True,
    )
    query_metrics = pd.concat(query_metric_frames, ignore_index=True).sort_values(
        ["dna_candidate_id", "text_candidate_id", "seed", "query_id", "k"],
        kind="stable",
        ignore_index=True,
    )
    query_summaries = pd.concat(summary_frames, ignore_index=True).sort_values(
        ["dna_candidate_id", "text_candidate_id", "seed", "query_kind", "k"],
        kind="stable",
        ignore_index=True,
    )
    bootstrap_draws = pd.concat(bootstrap_frames, ignore_index=True).sort_values(
        ["dna_candidate_id", "text_candidate_id", "query_kind", "draw"],
        kind="stable",
        ignore_index=True,
    )
    expected_configurations = (
        len(EXPECTED_DNA_CANDIDATES) * len(EXPECTED_TEXT_CANDIDATES) * len(EXPECTED_PROBE_SEEDS)
    )
    if len(checkpoints) != expected_configurations:
        raise RuntimeError(
            "E02b factorial is incomplete: "
            f"expected {expected_configurations} checkpoints, observed {len(checkpoints)}"
        )
    selection = _selection_report(query_summaries, bootstrap_draws, accepted, probe)
    outputs = {
        "whitening_state_sha256": dataframe_content_sha256(
            whitening_state, sort_columns=["feature_kind", "candidate_id"]
        ),
        "probe_checkpoints_sha256": dataframe_content_sha256(
            checkpoints, sort_columns=["dna_candidate_id", "text_candidate_id", "seed"]
        ),
        "training_history_sha256": dataframe_content_sha256(
            histories,
            sort_columns=["dna_candidate_id", "text_candidate_id", "seed", "epoch"],
        ),
        "paired_metrics_sha256": dataframe_content_sha256(
            paired_metrics, sort_columns=["dna_candidate_id", "text_candidate_id", "seed"]
        ),
        "query_rankings_sha256": dataframe_content_sha256(
            rankings,
            sort_columns=["dna_candidate_id", "text_candidate_id", "seed", "query_id", "rank"],
        ),
        "query_metrics_sha256": dataframe_content_sha256(
            query_metrics,
            sort_columns=["dna_candidate_id", "text_candidate_id", "seed", "query_id", "k"],
        ),
        "query_summaries_sha256": dataframe_content_sha256(
            query_summaries,
            sort_columns=[
                "dna_candidate_id",
                "text_candidate_id",
                "seed",
                "query_kind",
                "k",
            ],
        ),
        "bootstrap_draws_sha256": dataframe_content_sha256(
            bootstrap_draws,
            sort_columns=["dna_candidate_id", "text_candidate_id", "query_kind", "draw"],
        ),
    }
    report = {
        "protocol_version": str(params["protocol_version"]),
        "input_manifest_sha256": json_content_sha256(input_manifest),
        "factorial": {
            "dna_candidates": sorted(dna_matrices),
            "text_candidates": sorted(text_matrices),
            "seeds": list(seeds),
            "planned_configurations": expected_configurations,
            "completed_configurations": int(len(checkpoints)),
        },
        "validation_population": {
            "training_rows": int(len(train)),
            "gallery_rows": int(len(gallery)),
            "queries": int(len(ordered_queries)),
            "test_rows_read_by_alignment": False,
            "current_test_split_contaminated_before_e02b": True,
        },
        "selection": selection,
        "output_hashes": outputs,
        "decision": {
            "status": "validation_pair_selected",
            "validation_only": True,
            "gate2_started": False,
        },
    }
    return (
        whitening_state,
        checkpoints,
        histories,
        paired_metrics,
        rankings,
        query_metrics,
        query_summaries,
        bootstrap_draws,
        report,
    )


def validated_probe_axes(
    probe: dict[str, Any],
    *,
    gallery_rows: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate the frozen E02b seed and retrieval-cutoff axes."""
    seeds = tuple(int(seed) for seed in probe["seeds"])
    cutoffs = tuple(int(cutoff) for cutoff in probe["cutoffs"])
    if seeds != EXPECTED_PROBE_SEEDS:
        raise ValueError(f"probe seeds must remain {EXPECTED_PROBE_SEEDS}, observed {seeds}")
    if len(cutoffs) != len(set(cutoffs)) or min(cutoffs) < 1:
        raise ValueError("probe cutoffs must be positive and unique")
    if max(cutoffs) > gallery_rows:
        raise ValueError("probe cutoffs exceed the frozen validation gallery")
    return seeds, cutoffs


def _validate_alignment_inputs(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    query_states: pd.DataFrame,
    input_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accepted_input = params.get("accepted_input_artifact")
    if not isinstance(accepted_input, dict):
        raise ValueError("accepted_input_artifact must be frozen before alignment")
    if json_content_sha256(input_manifest) != accepted_input.get("manifest_sha256"):
        raise ValueError("E02b input manifest hash changed before alignment")
    if dataframe_content_sha256(
        pairs, sort_columns=["panel_role", "sequence_id"]
    ) != accepted_input.get("pairs_sha256"):
        raise ValueError("E02b pair table hash changed before alignment")
    observed_query_hash = dataframe_content_sha256(queries, sort_columns=["query_id"])
    observed_state_hash = dataframe_content_sha256(
        query_states, sort_columns=["semantic_query_id", "sequence_id"]
    )
    expected_hashes = input_manifest.get("output_hashes", {})
    if observed_query_hash != expected_hashes.get("queries_sha256"):
        raise ValueError("E02b query table hash changed before alignment")
    if observed_state_hash != expected_hashes.get("query_states_sha256"):
        raise ValueError("E02b query-state table hash changed before alignment")
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
    for name, frame, required in (
        ("pairs", pairs, required_pairs),
        ("queries", queries, required_queries),
        ("query states", query_states, required_states),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"E02b {name} are missing columns: {sorted(missing)}")
        if frame.empty or frame[list(required)].isna().any(axis=None):
            raise ValueError(f"E02b {name} must be non-empty and complete")
    if pairs["sequence_id"].duplicated().any():
        raise ValueError("E02b pairs repeat sequence identifiers")
    if queries["query_id"].duplicated().any():
        raise ValueError("E02b queries repeat query identifiers")
    if query_states.duplicated(["semantic_query_id", "sequence_id"]).any():
        raise ValueError("E02b query states repeat semantic-query and sequence pairs")
    train = pairs.loc[pairs["panel_role"].eq("alignment_train")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    gallery = pairs.loc[pairs["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    ordered_queries = queries.sort_values("query_id", kind="stable", ignore_index=True)
    if len(train) != int(params["training_rows"]):
        raise ValueError("E02b alignment training row count changed")
    if set(pairs["panel_role"].astype(str)) != {"alignment_train", "validation_gallery"}:
        raise ValueError("E02b pairs contain an unexpected panel role")
    unknown_state_ids = set(query_states["sequence_id"].astype(str)).difference(
        gallery["sequence_id"].astype(str)
    )
    if unknown_state_ids:
        raise ValueError(f"query states contain {len(unknown_state_ids)} rows outside the gallery")
    return train, gallery, ordered_queries


def _validate_factorial_feature_artifacts(
    dna_features: dict[str, pd.DataFrame],
    dna_manifests: dict[str, dict[str, Any]],
    text_features: dict[str, pd.DataFrame],
    text_manifests: dict[str, dict[str, Any]],
    input_manifest: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    configured_dna = {str(params["tfidf"]["candidate_id"]), *map(str, params["dna_candidates"])}
    configured_text = set(map(str, params["text_candidates"]))
    if configured_dna != EXPECTED_DNA_CANDIDATES:
        raise ValueError("configured DNA candidates changed from the frozen E02b factorial")
    if configured_text != EXPECTED_TEXT_CANDIDATES:
        raise ValueError("configured text candidates changed from the frozen E02b factorial")
    expected_dna = EXPECTED_DNA_CANDIDATES
    expected_text = EXPECTED_TEXT_CANDIDATES
    for name, observed, expected in (
        ("DNA features", set(dna_features), expected_dna),
        ("DNA manifests", set(dna_manifests), expected_dna),
        ("text features", set(text_features), expected_text),
        ("text manifests", set(text_manifests), expected_text),
    ):
        if observed != expected:
            raise ValueError(
                f"E02b {name} candidate set changed: expected {sorted(expected)}, "
                f"observed {sorted(observed)}"
            )
    accepted = params.get("accepted_feature_artifacts")
    if not isinstance(accepted, dict) or set(accepted) != {"dna", "text"}:
        raise ValueError("accepted_feature_artifacts must freeze DNA and text artifacts")
    input_hash = json_content_sha256(input_manifest)
    for feature_kind, features_by_id, manifests_by_id, expected in (
        ("dna", dna_features, dna_manifests, expected_dna),
        ("text", text_features, text_manifests, expected_text),
    ):
        accepted_kind = accepted.get(feature_kind)
        if not isinstance(accepted_kind, dict) or set(accepted_kind) != expected:
            raise ValueError(f"accepted {feature_kind} feature artifacts are incomplete")
        for candidate_id in sorted(expected):
            feature = features_by_id[candidate_id]
            manifest = manifests_by_id[candidate_id]
            record = accepted_kind[candidate_id]
            if not isinstance(record, dict):
                raise ValueError(f"accepted feature record is invalid for {candidate_id}")
            if manifest.get("candidate_id") != candidate_id:
                raise ValueError(f"feature manifest candidate changed for {candidate_id}")
            if manifest.get("input_manifest_sha256") != input_hash:
                raise ValueError(f"feature input manifest changed for {candidate_id}")
            manifest_hash = json_content_sha256(manifest)
            feature_hash = dataframe_content_sha256(
                feature,
                sort_columns=(
                    ["candidate_id", "sequence_sha256"]
                    if feature_kind == "dna"
                    else ["candidate_id", "text_role", "text_sha256"]
                ),
            )
            if manifest_hash != record.get("manifest_sha256"):
                raise ValueError(f"accepted manifest hash changed for {candidate_id}")
            if feature_hash != record.get("features_sha256"):
                raise ValueError(f"accepted feature hash changed for {candidate_id}")
            if manifest.get("output_hashes", {}).get("features_sha256") != feature_hash:
                raise ValueError(f"feature manifest does not describe {candidate_id} features")
            if set(feature["candidate_id"].astype(str)) != {candidate_id}:
                raise ValueError(f"feature rows contain another candidate for {candidate_id}")
            for field in ("version", "extraction_gpu_hours", "persisted_bytes"):
                if field not in record:
                    raise ValueError(f"accepted feature record for {candidate_id} lacks {field}")
            if float(record["extraction_gpu_hours"]) < 0.0 or int(record["persisted_bytes"]) < 1:
                raise ValueError(f"accepted feature cost evidence is invalid for {candidate_id}")
    return accepted


def _whiten_dna_candidates(
    train: pd.DataFrame,
    gallery: pd.DataFrame,
    features_by_id: dict[str, pd.DataFrame],
    *,
    epsilon: float,
) -> tuple[dict[str, CandidateMatrices], list[dict[str, Any]]]:
    matrices: dict[str, CandidateMatrices] = {}
    states: list[dict[str, Any]] = []
    for candidate_id, features in sorted(features_by_id.items()):
        train_matrix = _join_embeddings(train, features, key="sequence_sha256")
        gallery_matrix = _join_embeddings(gallery, features, key="sequence_sha256")
        whitening = alignment_probe.Whitening.fit(train_matrix, epsilon=epsilon)
        matrices[candidate_id] = CandidateMatrices(
            train=whitening.transform(train_matrix),
            gallery=whitening.transform(gallery_matrix),
            queries=None,
        )
        states.append(_whitening_row("dna", candidate_id, whitening))
    return matrices, states


def _whiten_text_candidates(
    train: pd.DataFrame,
    gallery: pd.DataFrame,
    queries: pd.DataFrame,
    features_by_id: dict[str, pd.DataFrame],
    *,
    epsilon: float,
) -> tuple[dict[str, CandidateMatrices], list[dict[str, Any]]]:
    matrices: dict[str, CandidateMatrices] = {}
    states: list[dict[str, Any]] = []
    query_keys = queries.loc[:, ["query_id"]].copy()
    query_keys["text_sha256"] = queries["canonical_query_text"].astype(str).map(sha256_text)
    for candidate_id, features in sorted(features_by_id.items()):
        documents = features.loc[features["text_role"].eq("document")]
        query_features = features.loc[features["text_role"].eq("query")]
        train_matrix = _join_embeddings(
            train, documents, key="description_sha256", feature_key="text_sha256"
        )
        gallery_matrix = _join_embeddings(
            gallery, documents, key="description_sha256", feature_key="text_sha256"
        )
        query_matrix = _join_embeddings(query_keys, query_features, key="text_sha256")
        whitening = alignment_probe.Whitening.fit(train_matrix, epsilon=epsilon)
        matrices[candidate_id] = CandidateMatrices(
            train=whitening.transform(train_matrix),
            gallery=whitening.transform(gallery_matrix),
            queries=whitening.transform(query_matrix),
        )
        states.append(_whitening_row("text", candidate_id, whitening))
    return matrices, states


def _join_embeddings(
    rows: pd.DataFrame,
    features: pd.DataFrame,
    *,
    key: str,
    feature_key: str | None = None,
) -> np.ndarray:
    feature_key = feature_key or key
    required = {feature_key, "embedding", "embedding_dimension", "embedding_sha256"}
    missing = required.difference(features.columns)
    if missing:
        raise ValueError(f"feature table is missing columns: {sorted(missing)}")
    if features[feature_key].duplicated().any():
        raise ValueError(f"feature table repeats {feature_key}")
    lookup = features.loc[:, list(required)].rename(columns={feature_key: key})
    joined = (
        rows.loc[:, [key]]
        .assign(_row_order=np.arange(len(rows)))
        .merge(
            lookup,
            on=key,
            how="left",
            validate="many_to_one",
            indicator=True,
        )
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"feature join left {unmatched} {key} values unmatched")
    joined = joined.sort_values("_row_order", kind="stable")
    dimensions = set(joined["embedding_dimension"].astype(int))
    if len(dimensions) != 1:
        raise ValueError("feature embedding dimension changed within a candidate")
    vectors = [np.asarray(value, dtype=np.float32) for value in joined["embedding"]]
    expected_dimension = dimensions.pop()
    if any(vector.ndim != 1 or len(vector) != expected_dimension for vector in vectors):
        raise ValueError("feature vector shape does not match its recorded dimension")
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("feature table contains a non-finite embedding")
    return matrix


def _whitening_row(
    feature_kind: str,
    candidate_id: str,
    whitening: alignment_probe.Whitening,
) -> dict[str, Any]:
    return {
        "feature_kind": feature_kind,
        "candidate_id": candidate_id,
        "input_dimension": int(len(whitening.mean)),
        "output_dimension": int(whitening.matrix.shape[1]),
        "epsilon_rule": "standard_deviation_plus_epsilon",
        "mean": whitening.mean.tolist(),
        "matrix": whitening.matrix.reshape(-1).tolist(),
        "matrix_rows": int(whitening.matrix.shape[0]),
        "matrix_columns": int(whitening.matrix.shape[1]),
        "mean_sha256": _array_sha256(whitening.mean),
        "matrix_sha256": _array_sha256(whitening.matrix),
    }


def _checkpoint_frame(
    dna_candidate_id: str,
    text_candidate_id: str,
    seed: int,
    state: dict[str, Any],
) -> pd.DataFrame:
    sequence_head = np.asarray(state["sequence_head"], dtype=np.float32)
    text_head = np.asarray(state["text_head"], dtype=np.float32)
    return pd.DataFrame(
        [
            {
                "dna_candidate_id": dna_candidate_id,
                "text_candidate_id": text_candidate_id,
                "seed": seed,
                "sequence_head": sequence_head.reshape(-1).tolist(),
                "sequence_head_rows": int(sequence_head.shape[0]),
                "sequence_head_columns": int(sequence_head.shape[1]),
                "sequence_head_sha256": _array_sha256(sequence_head),
                "text_head": text_head.reshape(-1).tolist(),
                "text_head_rows": int(text_head.shape[0]),
                "text_head_columns": int(text_head.shape[1]),
                "text_head_sha256": _array_sha256(text_head),
                "logit_scale": float(state["logit_scale"]),
                "batches_per_epoch": int(state["batches_per_epoch"]),
                "last_batch_rows": int(state["last_batch_rows"]),
                "dropped_rows_per_epoch": int(state["dropped_rows_per_epoch"]),
            }
        ]
    )


def _query_macro_summary(
    metrics: pd.DataFrame,
    *,
    dna_candidate_id: str,
    text_candidate_id: str,
    seed: int,
) -> pd.DataFrame:
    value_columns = [
        "verified_fraction",
        "contradicted_fraction",
        "unknown_fraction",
        "known_fraction",
        "utility",
    ]
    grouped = (
        metrics.groupby(["query_kind", "k"], sort=True, observed=True)
        .agg(
            queries=("query_id", "nunique"),
            **{column: (column, "mean") for column in value_columns},
        )
        .reset_index()
    )
    combined = (
        metrics.groupby("k", sort=True)
        .agg(
            queries=("query_id", "nunique"),
            **{column: (column, "mean") for column in value_columns},
        )
        .reset_index()
    )
    combined.insert(0, "query_kind", "combined")
    return pd.concat([grouped, combined], ignore_index=True).assign(
        dna_candidate_id=dna_candidate_id,
        text_candidate_id=text_candidate_id,
        seed=seed,
    )


def _selection_report(
    summaries: pd.DataFrame,
    bootstrap_draws: pd.DataFrame,
    accepted: dict[str, dict[str, dict[str, Any]]],
    probe: dict[str, Any],
) -> dict[str, Any]:
    primary = summaries.loc[summaries["query_kind"].eq("combined") & summaries["k"].eq(10)]
    leaderboard = (
        primary.groupby(["dna_candidate_id", "text_candidate_id"], sort=True)
        .agg(mean_utility_at_10=("utility", "mean"), seeds=("seed", "nunique"))
        .reset_index()
    )
    intervals = (
        bootstrap_draws.loc[bootstrap_draws["query_kind"].eq("combined")]
        .groupby(["dna_candidate_id", "text_candidate_id"], sort=True)["utility"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={0.025: "bootstrap_lower", 0.975: "bootstrap_upper"})
        .reset_index()
    )
    leaderboard = leaderboard.merge(
        intervals, on=["dna_candidate_id", "text_candidate_id"], validate="one_to_one"
    )
    expected_seeds = len(probe["seeds"])
    if not leaderboard["seeds"].eq(expected_seeds).all():
        raise ValueError("candidate selection received incomplete seed coverage")
    costs = []
    for row in leaderboard.itertuples(index=False):
        dna = accepted["dna"][str(row.dna_candidate_id)]
        text = accepted["text"][str(row.text_candidate_id)]
        costs.append(
            {
                "dna_candidate_id": str(row.dna_candidate_id),
                "text_candidate_id": str(row.text_candidate_id),
                "extraction_gpu_hours": float(dna["extraction_gpu_hours"])
                + float(text["extraction_gpu_hours"]),
                "persisted_feature_bytes": int(dna["persisted_bytes"])
                + int(text["persisted_bytes"]),
            }
        )
    leaderboard = leaderboard.merge(
        pd.DataFrame(costs),
        on=["dna_candidate_id", "text_candidate_id"],
        validate="one_to_one",
    ).sort_values(
        ["mean_utility_at_10", "dna_candidate_id", "text_candidate_id"],
        ascending=[False, True, True],
        kind="stable",
        ignore_index=True,
    )
    top = leaderboard.iloc[0]
    practical_tie = float(probe["practical_tie_utility"])
    tied = leaderboard.loc[
        leaderboard["mean_utility_at_10"]
        .rsub(float(top["mean_utility_at_10"]))
        .abs()
        .lt(practical_tie)
        & leaderboard.apply(
            lambda row: not (
                float(row["bootstrap_upper"]) < float(top["bootstrap_lower"])
                or float(top["bootstrap_upper"]) < float(row["bootstrap_lower"])
            ),
            axis=1,
        )
    ].copy()
    minimum_cost = float(tied["extraction_gpu_hours"].min())
    cost_tie_fraction = float(probe["cost_tie_fraction"])
    if minimum_cost == 0.0:
        cost_tied = tied.loc[tied["extraction_gpu_hours"].eq(0.0)]
    else:
        cost_tied = tied.loc[
            tied["extraction_gpu_hours"].le(minimum_cost * (1.0 + cost_tie_fraction))
        ]
    preferred = cost_tied.sort_values(
        [
            "persisted_feature_bytes",
            "extraction_gpu_hours",
            "dna_candidate_id",
            "text_candidate_id",
        ],
        kind="stable",
    ).iloc[0]

    baseline = leaderboard.loc[
        leaderboard["dna_candidate_id"].eq("carbon_500m")
        & leaderboard["text_candidate_id"].eq("bge_base_en_v1_5")
    ]
    if len(baseline) != 1:
        raise ValueError("Carbon-500M plus BGE-base incumbent is missing from the factorial")
    baseline_row = baseline.iloc[0]
    best_improvement = float(top["mean_utility_at_10"] - baseline_row["mean_utility_at_10"])
    cost_preferred_improvement = float(
        preferred["mean_utility_at_10"] - baseline_row["mean_utility_at_10"]
    )
    retained_incumbent = best_improvement < float(probe["minimum_incumbent_improvement"])
    selected = baseline_row if retained_incumbent else preferred
    return {
        "primary_metric": "validation_query_macro_utility_at_10",
        "higher_is_better": True,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "highest_utility_pair": _pair_identity(top),
        "practical_tie_pairs": [_pair_identity(row) for _, row in tied.iterrows()],
        "cost_preferred_pair": _pair_identity(preferred),
        "incumbent_pair": _pair_identity(baseline_row),
        "highest_utility_improvement_over_incumbent": best_improvement,
        "cost_preferred_improvement_over_incumbent": cost_preferred_improvement,
        "minimum_incumbent_improvement": float(probe["minimum_incumbent_improvement"]),
        "incumbent_retained": retained_incumbent,
        "selected_pair": _pair_identity(selected),
    }


def _pair_identity(row: pd.Series) -> dict[str, Any]:
    return {
        "dna_candidate_id": str(row["dna_candidate_id"]),
        "text_candidate_id": str(row["text_candidate_id"]),
        "mean_utility_at_10": float(row["mean_utility_at_10"]),
        "bootstrap_lower": float(row["bootstrap_lower"]),
        "bootstrap_upper": float(row["bootstrap_upper"]),
        "extraction_gpu_hours": float(row["extraction_gpu_hours"]),
        "persisted_feature_bytes": int(row["persisted_feature_bytes"]),
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.float32)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _ensure_before_deadline(deadline_monotonic: float | None, *, operation: str) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError(f"authorized compute deadline reached before {operation}")
