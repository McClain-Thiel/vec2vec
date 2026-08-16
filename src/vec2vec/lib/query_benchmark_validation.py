"""Independent read-back validation for the frozen query benchmark."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from vec2vec.lib import query_benchmark, similarity_graph


def validate_query_benchmark_outputs(
    query_catalog: pd.DataFrame,
    galleries: pd.DataFrame,
    query_states: pd.DataFrame,
    base_masses: pd.DataFrame,
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: dict[str, Any],
    source_states: pd.DataFrame,
    *,
    expected_rows: int,
    expected_retrieval_version: str,
    expected_graph_artifact_version: str,
    expected_split_artifact_version: str,
    expected_constraint_state_artifact_version: str,
    require_gate0_data_ready: bool = True,
) -> dict[str, Any]:
    """Validate reloaded benchmark tables, hashes, and answer-set identities."""
    top_k = tuple(int(value) for value in manifest["resolved_configuration"]["top_k"])
    table_checks = query_benchmark.validate_query_benchmark_tables(
        query_catalog,
        galleries,
        query_states,
        base_masses,
        rankings,
        metrics,
        expected_rows=expected_rows,
        top_k=top_k,
    )
    expected_versions = {
        "retrieval": expected_retrieval_version,
        "graph": expected_graph_artifact_version,
        "split": expected_split_artifact_version,
        "constraint_state": expected_constraint_state_artifact_version,
    }
    if manifest.get("input_versions") != expected_versions:
        raise RuntimeError("query-benchmark input versions differ from the pinned versions")
    decision = manifest.get("decision", {})
    if decision.get("artifact_invariants_passed") is not True:
        raise RuntimeError("query-benchmark manifest does not accept its artifact invariants")
    if decision.get("model_outcomes_inspected") is not False:
        raise RuntimeError("query-benchmark manifest reports model-outcome inspection")
    if require_gate0_data_ready and decision.get("gate0_data_ready") is not True:
        raise RuntimeError(
            "query benchmark is structurally valid but fails the preregistered Gate 0 support rule"
        )

    observed_hashes = {
        "query_catalog_sha256": similarity_graph.dataframe_content_sha256(
            query_catalog, sort_columns=["query_id"]
        ),
        "candidate_galleries_sha256": similarity_graph.dataframe_content_sha256(
            galleries, sort_columns=["gallery_id", "sequence_id"]
        ),
        "query_candidate_state_sha256": similarity_graph.dataframe_content_sha256(
            query_states,
            sort_columns=["semantic_query_id", "state", "sequence_id"],
        ),
        "candidate_base_mass_sha256": similarity_graph.dataframe_content_sha256(
            base_masses, sort_columns=["gallery_id", "base_measure", "sequence_id"]
        ),
        "control_rankings_sha256": similarity_graph.dataframe_content_sha256(
            rankings, sort_columns=["query_id", "control", "rank"]
        ),
        "control_metrics_sha256": similarity_graph.dataframe_content_sha256(
            metrics, sort_columns=["query_id", "control", "k"]
        ),
    }
    expected_hashes = manifest.get("output_content_hashes", {})
    failures = {
        key: {"expected": expected_hashes.get(key), "observed": observed}
        for key, observed in observed_hashes.items()
        if expected_hashes.get(key) != observed
    }
    if failures:
        raise RuntimeError(f"reloaded query-benchmark content hashes differ: {failures}")

    _validate_answer_set_identities(query_catalog, galleries, query_states, source_states)
    return {
        "status": "accepted_independent_s3_readback",
        "benchmark_version": manifest.get("benchmark_version"),
        "input_versions": expected_versions,
        "semantic_queries": int(query_catalog["semantic_query_id"].nunique()),
        "catalog_rows": int(len(query_catalog)),
        "sparse_query_state_rows": int(len(query_states)),
        "gate0_support": manifest.get("gate0_support"),
        "gate0_data_ready": bool(decision.get("gate0_data_ready")),
        "table_checks": table_checks,
        "answer_set_identities_recomputed": True,
        "content_hashes": observed_hashes,
    }


def _validate_answer_set_identities(
    query_catalog: pd.DataFrame,
    galleries: pd.DataFrame,
    query_states: pd.DataFrame,
    source_states: pd.DataFrame,
) -> None:
    required_source = {"sequence_id", "constraint_id", "state"}
    missing_source = required_source.difference(source_states.columns)
    if missing_source:
        raise ValueError(f"source states are missing columns: {sorted(missing_source)}")
    source = source_states.loc[:, sorted(required_source)].copy()
    source["sequence_id"] = source["sequence_id"].astype(str)
    source["constraint_id"] = source["constraint_id"].astype(str)
    if source.duplicated(["sequence_id", "constraint_id"]).any():
        raise RuntimeError("source states contain duplicate sequence-constraint pairs")
    by_constraint: dict[str, dict[str, set[str]]] = {
        query_benchmark.VERIFIED: {},
        query_benchmark.CONTRADICTED: {},
    }
    for (constraint_id, state), group in source.groupby(["constraint_id", "state"], sort=True):
        by_constraint[str(state)][str(constraint_id)] = set(group["sequence_id"].astype(str))

    explicit: dict[tuple[str, str], set[str]] = {}
    for (semantic_query_id, state), group in query_states.groupby(
        ["semantic_query_id", "state"], sort=True
    ):
        explicit[str(semantic_query_id), str(state)] = set(group["sequence_id"].astype(str))
    definitions = query_catalog.sort_values("query_id", kind="stable").drop_duplicates(
        "semantic_query_id"
    )
    expected_by_query: dict[str, tuple[set[str], set[str]]] = {}
    for row in definitions.itertuples(index=False):
        constraint_ids = tuple(str(value) for value in json.loads(row.constraint_ids_json))
        if not constraint_ids:
            raise RuntimeError("a persisted query has no constraints")
        verified_sets = [
            by_constraint[query_benchmark.VERIFIED].get(value, set()) for value in constraint_ids
        ]
        expected_verified = set.intersection(*verified_sets)
        expected_contradicted = set().union(
            *(
                by_constraint[query_benchmark.CONTRADICTED].get(value, set())
                for value in constraint_ids
            )
        )
        expected_contradicted.difference_update(expected_verified)
        semantic_query_id = str(row.semantic_query_id)
        observed_verified = explicit.get((semantic_query_id, query_benchmark.VERIFIED), set())
        observed_contradicted = explicit.get(
            (semantic_query_id, query_benchmark.CONTRADICTED), set()
        )
        if observed_verified != expected_verified:
            raise RuntimeError(
                f"verified answer set differs from source constraints: {semantic_query_id}"
            )
        if observed_contradicted != expected_contradicted:
            raise RuntimeError(
                f"contradicted set differs from source constraints: {semantic_query_id}"
            )
        expected_by_query[semantic_query_id] = (expected_verified, expected_contradicted)

    gallery_ids = {
        str(gallery_id): set(group["sequence_id"].astype(str))
        for gallery_id, group in galleries.groupby("gallery_id", sort=True)
    }
    for row in query_catalog.itertuples(index=False):
        members = gallery_ids[str(row.gallery_id)]
        verified, contradicted = expected_by_query[str(row.semantic_query_id)]
        verified_count = len(verified.intersection(members))
        contradicted_count = len(contradicted.intersection(members))
        unknown_count = len(members) - verified_count - contradicted_count
        observed = (
            int(row.answer_set_size),
            int(row.contradiction_set_size),
            int(row.unknown_set_size),
        )
        expected = (verified_count, contradicted_count, unknown_count)
        if observed != expected:
            raise RuntimeError(f"query catalog set sizes differ after read-back: {row.query_id}")
