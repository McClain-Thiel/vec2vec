"""Frozen symbolic queries, galleries, answer sets, base measures, and controls."""

from __future__ import annotations

import hashlib
import heapq
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import similarity_graph, similarity_split, splits
from vec2vec.lib.serialization import stable_json
from vec2vec.lib.text import sha256_text

VERIFIED = "verified"
CONTRADICTED = "contradicted"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class QueryDefinition:
    """One semantic query selected only from v2 training evidence."""

    semantic_query_id: str
    query_kind: str
    canonical_query_text: str
    constraint_ids: tuple[str, ...]
    facets: tuple[str, ...]
    train_verified_rows: int
    train_verified_components: int
    train_contradicted_rows: int
    train_contradicted_components: int
    contradiction_control_eligible: bool


def build_query_benchmark(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    graph_edges: pd.DataFrame,
    graph_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    vocabulary: pd.DataFrame,
    states: pd.DataFrame,
    state_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Build and validate the complete versioned query-benchmark data product."""
    input_content_hashes = _validate_inputs(
        retrieval,
        split_mapping,
        graph_edges,
        graph_manifest,
        split_manifest,
        vocabulary,
        states,
        state_manifest,
        params,
    )
    population = _population(retrieval, split_mapping)
    state_table = _state_table(states, population, vocabulary)
    support = _training_support(state_table, vocabulary)
    definitions = _select_queries(support, state_table, population, params)
    verified, contradicted = _answer_sets(definitions, state_table)
    query_states = _query_state_table(definitions, verified, contradicted)
    galleries, gallery_members = _build_galleries(population, params)
    query_catalog = _build_query_catalog(
        definitions,
        verified,
        contradicted,
        gallery_members,
        params,
    )
    base_masses = _build_base_masses(population, galleries, graph_edges)
    rankings, metrics = _run_controls(
        definitions,
        query_catalog,
        gallery_members,
        verified,
        contradicted,
        state_table,
        support,
        params,
    )
    checks = validate_query_benchmark_tables(
        query_catalog,
        galleries,
        query_states,
        base_masses,
        rankings,
        metrics,
        expected_rows=len(population),
        top_k=tuple(int(value) for value in params["top_k"]),
    )
    output_hashes = {
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
    expected_output_hashes = params.get("expected_output_content_hashes")
    if expected_output_hashes is not None and output_hashes != expected_output_hashes:
        raise RuntimeError("query benchmark changed from the accepted content hashes")
    pair_count = sum(definition.query_kind == "pair_conjunction" for definition in definitions)
    support_gate = _gate0_support_summary(query_catalog, params)
    support_passed = bool(support_gate["passed"])
    manifest = {
        "benchmark_version": str(params["benchmark_version"]),
        "protocol": "modeling_data_v1",
        "input_content_hashes": input_content_hashes,
        "resolved_configuration": params,
        "population_rows": int(len(population)),
        "semantic_queries": int(len(definitions)),
        "atomic_queries": int(sum(definition.query_kind == "atomic" for definition in definitions)),
        "pair_conjunction_queries": int(pair_count),
        "catalog_rows": int(len(query_catalog)),
        "sparse_query_state_rows": int(len(query_states)),
        "gallery_rows": int(len(galleries)),
        "base_mass_rows": int(len(base_masses)),
        "control_ranking_rows": int(len(rankings)),
        "control_metric_rows": int(len(metrics)),
        "checks": checks,
        "gate0_support": support_gate,
        "output_content_hashes": output_hashes,
        "selection_policy": {
            "training_split_only": True,
            "test_support_used_for_query_selection": False,
            "different_facet_pairs_only": True,
            "paraphrases_included": False,
            "triples_included": False,
            "source_queries_included": False,
        },
        "exclusion_policy": {
            "status": "not_applicable_no_source_queries",
            "source_exclusion_applied": False,
            "identical_sequence_exclusion_applied": False,
            "same_backbone_exclusion_applied": False,
        },
        "decision": {
            "status": (
                "accepted_gate0_data"
                if pair_count and support_passed
                else (
                    "narrow_insufficient_evaluation_support"
                    if pair_count
                    else "stop_no_eligible_conjunctions"
                )
            ),
            "artifact_invariants_passed": True,
            "paired_origin_control": "not_applicable_no_source_queries",
            "model_outcomes_inspected": False,
            "gate0_data_ready": bool(pair_count and support_passed),
        },
        "known_limitations": [
            "States are narrow Addgene metadata claims, not biological function.",
            "Positive-only facets cannot support an atomic contradiction control.",
            "Canonical symbolic text does not test language understanding.",
            "The open gallery contains training rows.",
        ],
    }
    return (
        query_catalog,
        galleries,
        query_states,
        base_masses,
        rankings,
        metrics,
        manifest,
    )


def _validate_inputs(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    graph_edges: pd.DataFrame,
    graph_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    vocabulary: pd.DataFrame,
    states: pd.DataFrame,
    state_manifest: dict[str, Any],
    params: dict[str, Any],
) -> dict[str, str]:
    expected_population = str(params["expected_input_population_sha256"])
    if state_manifest.get("input_population_sha256") != expected_population:
        raise RuntimeError("constraint-state population hash differs from the benchmark pin")
    if state_manifest.get("pair_state_conflicts") != 0:
        raise RuntimeError("constraint-state manifest reports a pair-state conflict")
    split_decision = split_manifest.get("decision", {})
    if split_decision.get("status") != "accepted_strict_similarity_closed_split":
        raise RuntimeError("split manifest is not accepted for benchmark construction")
    graph_decision = graph_manifest.get("decision", {})
    if graph_decision.get("edge_enumeration_complete_under_configured_caps") is not True:
        raise RuntimeError("graph manifest does not accept complete configured-cap enumeration")

    _require_columns(
        retrieval,
        {"sequence_id", "sequence_sha256", "family_key"},
        name="retrieval",
    )
    _require_columns(split_mapping, set(similarity_split.MAPPING_COLUMNS), name="v2 mapping")
    _require_columns(
        graph_edges,
        {"sequence_a", "sequence_b", "primary_near_duplicate"},
        name="graph edges",
    )
    _require_columns(
        vocabulary,
        {
            "constraint_id",
            "facet",
            "relation",
            "canonical_value",
            "rule_id",
            "rule_version",
            "has_reviewed_conflict_rule",
        },
        name="constraint vocabulary",
    )
    _require_columns(states, {"sequence_id", "constraint_id", "state"}, name="states")
    if retrieval["sequence_id"].duplicated().any():
        raise ValueError("retrieval sequence IDs are not unique")
    if split_mapping["sequence_id"].duplicated().any():
        raise ValueError("v2 mapping sequence IDs are not unique")
    if vocabulary["constraint_id"].duplicated().any():
        raise ValueError("constraint identifiers are not unique")
    if states.duplicated(["sequence_id", "constraint_id"]).any():
        raise ValueError("state table contains duplicate sequence-constraint pairs")
    if set(states["state"].astype(str)) - {VERIFIED, CONTRADICTED}:
        raise ValueError("state table contains a value other than verified or contradicted")
    retrieval_ids = set(retrieval["sequence_id"].astype(str))
    mapping_ids = set(split_mapping["sequence_id"].astype(str))
    if retrieval_ids != mapping_ids:
        raise ValueError("v2 mapping identifiers differ from the retrieval population")
    if set(states["sequence_id"].astype(str)) - retrieval_ids:
        raise ValueError("state table contains an unknown sequence identifier")
    if set(states["constraint_id"].astype(str)) - set(vocabulary["constraint_id"].astype(str)):
        raise ValueError("state table contains an unknown constraint identifier")
    invalid_splits = set(split_mapping["split_grouped_v2"].astype(str)) - set(splits.SPLIT_LABELS)
    if invalid_splits:
        raise ValueError(f"v2 mapping contains invalid splits: {sorted(invalid_splits)}")
    if split_mapping.groupby("leakage_component_v2")["split_grouped_v2"].nunique().gt(1).any():
        raise RuntimeError("a v2 leakage component crosses splits")
    mapping_hash = similarity_graph.dataframe_content_sha256(
        split_mapping.loc[:, list(similarity_split.MAPPING_COLUMNS)],
        sort_columns=["sequence_id"],
    )
    if split_manifest.get("build", {}).get("mapping_sha256") != mapping_hash:
        raise RuntimeError("v2 mapping content differs from its accepted manifest")
    edge_hash = similarity_graph.dataframe_content_sha256(
        graph_edges, sort_columns=["sequence_a", "sequence_b"]
    )
    if graph_manifest.get("output_content_hashes", {}).get("edges_sha256") != edge_hash:
        raise RuntimeError("graph edges differ from their accepted manifest")
    vocabulary_hash = similarity_graph.dataframe_content_sha256(
        vocabulary, sort_columns=["constraint_id"]
    )
    states_hash = similarity_graph.dataframe_content_sha256(
        states, sort_columns=["sequence_id", "constraint_id", "state"]
    )
    observed_state_hashes = {
        "vocabulary_sha256": vocabulary_hash,
        "states_sha256": states_hash,
    }
    expected_state_hashes = dict(params["expected_constraint_artifact_hashes"])
    if observed_state_hashes != expected_state_hashes:
        raise RuntimeError("constraint tables differ from the accepted content hashes")
    manifest_state_hashes = state_manifest.get("output_content_hashes")
    if manifest_state_hashes is not None and manifest_state_hashes != observed_state_hashes:
        raise RuntimeError("constraint tables differ from their manifest")
    retrieval_hash = similarity_graph.dataframe_content_sha256(
        retrieval, sort_columns=["sequence_id"]
    )
    return {
        "retrieval_sha256": retrieval_hash,
        "graph_edges_sha256": edge_hash,
        "split_mapping_sha256": mapping_hash,
        "constraint_vocabulary_sha256": vocabulary_hash,
        "constraint_states_sha256": states_hash,
    }


def _population(retrieval: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    left = retrieval.loc[:, ["sequence_id", "sequence_sha256", "family_key"]].copy()
    left["sequence_id"] = left["sequence_id"].astype(str)
    right = mapping.loc[:, ["sequence_id", "leakage_component_v2", "split_grouped_v2"]].copy()
    right["sequence_id"] = right["sequence_id"].astype(str)
    population = left.merge(right, on="sequence_id", how="inner", validate="one_to_one")
    if population.isna().any().any():
        raise ValueError("benchmark population contains missing identity or grouping values")
    return population.sort_values("sequence_id", kind="stable").reset_index(drop=True)


def _state_table(
    states: pd.DataFrame,
    population: pd.DataFrame,
    vocabulary: pd.DataFrame,
) -> pd.DataFrame:
    table = states.loc[:, ["sequence_id", "constraint_id", "state"]].copy()
    table["sequence_id"] = table["sequence_id"].astype(str)
    table["constraint_id"] = table["constraint_id"].astype(str)
    table = table.merge(
        population.loc[:, ["sequence_id", "leakage_component_v2", "split_grouped_v2"]],
        on="sequence_id",
        how="inner",
        validate="many_to_one",
    )
    table = table.merge(
        vocabulary.loc[:, ["constraint_id", "facet"]],
        on="constraint_id",
        how="inner",
        validate="many_to_one",
    )
    return table.sort_values(["constraint_id", "state", "sequence_id"], kind="stable").reset_index(
        drop=True
    )


def _training_support(state_table: pd.DataFrame, vocabulary: pd.DataFrame) -> pd.DataFrame:
    train = state_table.loc[state_table["split_grouped_v2"].eq(splits.TRAIN)]
    result = vocabulary.loc[
        :,
        [
            "constraint_id",
            "facet",
            "relation",
            "canonical_value",
            "rule_id",
            "rule_version",
            "has_reviewed_conflict_rule",
        ],
    ].copy()
    for state in (VERIFIED, CONTRADICTED):
        selected = train.loc[train["state"].eq(state)]
        rows = selected.groupby("constraint_id")["sequence_id"].nunique()
        components = selected.groupby("constraint_id")["leakage_component_v2"].nunique()
        result[f"train_{state}_rows"] = result["constraint_id"].map(rows).fillna(0).astype(int)
        result[f"train_{state}_components"] = (
            result["constraint_id"].map(components).fillna(0).astype(int)
        )
    return result.sort_values("constraint_id", kind="stable").reset_index(drop=True)


def _select_queries(
    support: pd.DataFrame,
    state_table: pd.DataFrame,
    population: pd.DataFrame,
    params: dict[str, Any],
) -> list[QueryDefinition]:
    min_atomic_rows = int(params["minimum_atomic_train_verified_rows"])
    min_atomic_components = int(params["minimum_atomic_train_verified_components"])
    min_pair_rows = int(params["minimum_pair_train_verified_rows"])
    min_pair_components = int(params["minimum_pair_train_verified_components"])
    min_negative_rows = int(params["minimum_train_contradicted_rows"])
    min_negative_components = int(params["minimum_train_contradicted_components"])
    if min(min_atomic_rows, min_atomic_components, min_pair_rows, min_pair_components) < 1:
        raise ValueError("verified support thresholds must be positive")
    if min(min_negative_rows, min_negative_components) < 1:
        raise ValueError("contradiction support thresholds must be positive")

    eligible = support.loc[
        support["train_verified_rows"].ge(min_atomic_rows)
        & support["train_verified_components"].ge(min_atomic_components)
    ].copy()
    if eligible.empty:
        raise RuntimeError("no atomic constraint passes the fixed training-support rules")
    by_constraint = support.set_index("constraint_id").to_dict("index")
    train_ids = set(
        population.loc[population["split_grouped_v2"].eq(splits.TRAIN), "sequence_id"].astype(str)
    )
    component_by_id = population.set_index("sequence_id")["leakage_component_v2"].astype(str)
    verified, contradicted = _constraint_state_sets(state_table)
    definitions: list[QueryDefinition] = []
    version = str(params["benchmark_version"])
    text_revision = str(params["canonical_text_revision"])

    for row in eligible.sort_values("constraint_id", kind="stable").itertuples(index=False):
        constraint_id = str(row.constraint_id)
        details = by_constraint[constraint_id]
        text = _constraint_phrase(details)
        definitions.append(
            _query_definition(
                version=version,
                text_revision=text_revision,
                query_kind="atomic",
                text=text,
                constraint_ids=(constraint_id,),
                facets=(str(details["facet"]),),
                verified_rows=int(details["train_verified_rows"]),
                verified_components=int(details["train_verified_components"]),
                contradicted_rows=int(details["train_contradicted_rows"]),
                contradicted_components=int(details["train_contradicted_components"]),
                minimum_negative_rows=min_negative_rows,
                minimum_negative_components=min_negative_components,
            )
        )

    eligible_ids = tuple(sorted(eligible["constraint_id"].astype(str)))
    for left, right in combinations(eligible_ids, 2):
        left_details = by_constraint[left]
        right_details = by_constraint[right]
        if str(left_details["facet"]) == str(right_details["facet"]):
            continue
        pair_verified = verified.get(left, set()).intersection(
            verified.get(right, set()), train_ids
        )
        pair_contradicted = (
            contradicted.get(left, set())
            .union(contradicted.get(right, set()))
            .intersection(train_ids)
            - pair_verified
        )
        verified_components = int(component_by_id.loc[list(pair_verified)].nunique())
        contradicted_components = int(component_by_id.loc[list(pair_contradicted)].nunique())
        if len(pair_verified) < min_pair_rows or verified_components < min_pair_components:
            continue
        if (
            len(pair_contradicted) < min_negative_rows
            or contradicted_components < min_negative_components
        ):
            continue
        ordered = tuple(sorted((left, right)))
        texts = sorted((_constraint_phrase(left_details), _constraint_phrase(right_details)))
        facets = tuple(sorted((str(left_details["facet"]), str(right_details["facet"]))))
        definitions.append(
            _query_definition(
                version=version,
                text_revision=text_revision,
                query_kind="pair_conjunction",
                text=" AND ".join(texts),
                constraint_ids=ordered,
                facets=facets,
                verified_rows=len(pair_verified),
                verified_components=verified_components,
                contradicted_rows=len(pair_contradicted),
                contradicted_components=contradicted_components,
                minimum_negative_rows=min_negative_rows,
                minimum_negative_components=min_negative_components,
            )
        )
    return sorted(definitions, key=lambda definition: definition.semantic_query_id)


def _query_definition(
    *,
    version: str,
    text_revision: str,
    query_kind: str,
    text: str,
    constraint_ids: tuple[str, ...],
    facets: tuple[str, ...],
    verified_rows: int,
    verified_components: int,
    contradicted_rows: int,
    contradicted_components: int,
    minimum_negative_rows: int,
    minimum_negative_components: int,
) -> QueryDefinition:
    semantic_query_id = sha256_text(
        stable_json(
            {
                "benchmark_version": version,
                "canonical_text_revision": text_revision,
                "constraint_ids": constraint_ids,
                "query_kind": query_kind,
                "query_text": text,
            }
        )
    )
    return QueryDefinition(
        semantic_query_id=semantic_query_id,
        query_kind=query_kind,
        canonical_query_text=text,
        constraint_ids=constraint_ids,
        facets=facets,
        train_verified_rows=verified_rows,
        train_verified_components=verified_components,
        train_contradicted_rows=contradicted_rows,
        train_contradicted_components=contradicted_components,
        contradiction_control_eligible=(
            contradicted_rows >= minimum_negative_rows
            and contradicted_components >= minimum_negative_components
        ),
    )


def _constraint_phrase(details: dict[str, Any]) -> str:
    facet = str(details["facet"]).replace("_", " ")
    relation = str(details["relation"]).replace("_", " ")
    value = str(details["canonical_value"]).replace("_", " ")
    return f"{facet}: {relation} {value}"


def _constraint_state_sets(
    state_table: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {VERIFIED: {}, CONTRADICTED: {}}
    for (constraint_id, state), group in state_table.groupby(["constraint_id", "state"], sort=True):
        result[str(state)][str(constraint_id)] = set(group["sequence_id"].astype(str))
    return result[VERIFIED], result[CONTRADICTED]


def _answer_sets(
    definitions: list[QueryDefinition],
    state_table: pd.DataFrame,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    by_constraint_verified, by_constraint_contradicted = _constraint_state_sets(state_table)
    verified: dict[str, set[str]] = {}
    contradicted: dict[str, set[str]] = {}
    for definition in definitions:
        constraint_sets = [
            by_constraint_verified.get(constraint_id, set())
            for constraint_id in definition.constraint_ids
        ]
        valid = set.intersection(*constraint_sets) if constraint_sets else set()
        negative = set().union(
            *(by_constraint_contradicted.get(value, set()) for value in definition.constraint_ids)
        )
        negative.difference_update(valid)
        if valid.intersection(negative):
            raise RuntimeError("a query has overlapping verified and contradicted sets")
        verified[definition.semantic_query_id] = valid
        contradicted[definition.semantic_query_id] = negative
    return verified, contradicted


def _query_state_table(
    definitions: list[QueryDefinition],
    verified: dict[str, set[str]],
    contradicted: dict[str, set[str]],
) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for definition in definitions:
        for state, values in (
            (VERIFIED, verified[definition.semantic_query_id]),
            (CONTRADICTED, contradicted[definition.semantic_query_id]),
        ):
            if values:
                tables.append(
                    pd.DataFrame(
                        {
                            "semantic_query_id": definition.semantic_query_id,
                            "sequence_id": sorted(values),
                            "state": state,
                        }
                    )
                )
    if not tables:
        return pd.DataFrame(columns=["semantic_query_id", "sequence_id", "state"])
    return (
        pd.concat(tables, ignore_index=True)
        .sort_values(["semantic_query_id", "state", "sequence_id"], kind="stable")
        .reset_index(drop=True)
    )


def _build_galleries(
    population: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    evaluation_splits = tuple(str(value) for value in params["evaluation_splits"])
    if set(evaluation_splits) - {splits.VAL, splits.TEST}:
        raise ValueError("query benchmark evaluation_splits can contain only val and test")
    frames: list[pd.DataFrame] = []
    members: dict[str, pd.DataFrame] = {}
    columns = [
        "sequence_id",
        "sequence_sha256",
        "family_key",
        "leakage_component_v2",
        "split_grouped_v2",
    ]
    for evaluation_split in evaluation_splits:
        for gallery_kind in ("closed_grouped_v2", "open_all"):
            gallery_id = f"{gallery_kind}:{evaluation_split}"
            selected = (
                population.loc[population["split_grouped_v2"].eq(evaluation_split), columns]
                if gallery_kind == "closed_grouped_v2"
                else population.loc[:, columns]
            ).copy()
            selected.insert(0, "evaluation_split", evaluation_split)
            selected.insert(0, "gallery_kind", gallery_kind)
            selected.insert(0, "gallery_id", gallery_id)
            selected = selected.sort_values("sequence_id", kind="stable").reset_index(drop=True)
            frames.append(selected)
            members[gallery_id] = selected
    galleries = pd.concat(frames, ignore_index=True)
    if galleries.duplicated(["gallery_id", "sequence_id"]).any():
        raise RuntimeError("candidate gallery contains duplicate membership")
    return galleries, members


def _build_query_catalog(
    definitions: list[QueryDefinition],
    verified: dict[str, set[str]],
    contradicted: dict[str, set[str]],
    galleries: dict[str, pd.DataFrame],
    params: dict[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    version = str(params["benchmark_version"])
    text_revision = str(params["canonical_text_revision"])
    for definition in definitions:
        valid = verified[definition.semantic_query_id]
        negative = contradicted[definition.semantic_query_id]
        for gallery_id, members in sorted(galleries.items()):
            member_ids = set(members["sequence_id"].astype(str))
            valid_count = len(valid.intersection(member_ids))
            negative_count = len(negative.intersection(member_ids))
            valid_components = int(
                members.loc[members["sequence_id"].isin(valid), "leakage_component_v2"].nunique()
            )
            negative_components = int(
                members.loc[members["sequence_id"].isin(negative), "leakage_component_v2"].nunique()
            )
            candidate_count = len(member_ids)
            unknown_count = candidate_count - valid_count - negative_count
            gallery_kind, evaluation_split = gallery_id.split(":", maxsplit=1)
            query_id = sha256_text(
                stable_json(
                    {
                        "benchmark_version": version,
                        "evaluation_split": evaluation_split,
                        "exclusion_policy": "not_applicable_no_source_query",
                        "gallery_id": gallery_id,
                        "semantic_query_id": definition.semantic_query_id,
                        "text_revision": text_revision,
                    }
                )
            )
            records.append(
                {
                    "query_id": query_id,
                    "semantic_query_id": definition.semantic_query_id,
                    "query_kind": definition.query_kind,
                    "canonical_query_text": definition.canonical_query_text,
                    "canonical_text_revision": text_revision,
                    "constraint_ids_json": stable_json(definition.constraint_ids),
                    "facets_json": stable_json(definition.facets),
                    "controlled_split": (
                        "atomic_seen"
                        if definition.query_kind == "atomic"
                        else "atoms_seen_conjunction_unseen"
                    ),
                    "gallery_id": gallery_id,
                    "gallery_kind": gallery_kind,
                    "evaluation_split": evaluation_split,
                    "exclusion_policy": "not_applicable_no_source_query",
                    "source_sequence_id": None,
                    "candidate_count": candidate_count,
                    "answer_set_size": valid_count,
                    "answer_set_component_count": valid_components,
                    "contradiction_set_size": negative_count,
                    "contradiction_set_component_count": negative_components,
                    "unknown_set_size": unknown_count,
                    "unknown_fraction": unknown_count / candidate_count,
                    "answer_set_bucket": _answer_bucket(valid_count),
                    "specificity_bits_uniform_plasmid": (
                        math.inf if valid_count == 0 else -math.log2(valid_count / candidate_count)
                    ),
                    "train_verified_rows": definition.train_verified_rows,
                    "train_verified_components": definition.train_verified_components,
                    "train_contradicted_rows": definition.train_contradicted_rows,
                    "train_contradicted_components": definition.train_contradicted_components,
                    "contradiction_control_eligible": (definition.contradiction_control_eligible),
                    "measurement_eligible": (
                        valid_count >= int(params["minimum_gallery_verified_rows_for_measurement"])
                        and valid_components
                        >= int(params["minimum_gallery_verified_components_for_measurement"])
                    ),
                    "contradiction_measurement_eligible": (
                        negative_count
                        >= int(params["minimum_gallery_contradicted_rows_for_control"])
                        and negative_components
                        >= int(params["minimum_gallery_contradicted_components_for_control"])
                    ),
                }
            )
    catalog = pd.DataFrame.from_records(records)
    if catalog.empty or catalog["query_id"].duplicated().any():
        raise RuntimeError("query catalog is empty or has duplicate query identifiers")
    return catalog.sort_values("query_id", kind="stable").reset_index(drop=True)


def _gate0_support_summary(
    query_catalog: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "usable_atomic_queries": int(params["minimum_usable_atomic_queries_each_closed_eval"]),
        "usable_pair_queries": int(params["minimum_usable_pair_queries_each_closed_eval"]),
        "usable_pair_contradiction_controls": int(
            params["minimum_usable_pair_contradiction_controls_each_closed_eval"]
        ),
    }
    by_split: dict[str, dict[str, int | bool]] = {}
    for evaluation_split in tuple(str(value) for value in params["evaluation_splits"]):
        selected = query_catalog.loc[
            query_catalog["gallery_kind"].eq("closed_grouped_v2")
            & query_catalog["evaluation_split"].eq(evaluation_split)
        ]
        atoms = selected.loc[selected["query_kind"].eq("atomic")]
        pairs = selected.loc[selected["query_kind"].eq("pair_conjunction")]
        observed = {
            "usable_atomic_queries": int(atoms["measurement_eligible"].sum()),
            "usable_pair_queries": int(pairs["measurement_eligible"].sum()),
            "usable_pair_contradiction_controls": int(
                (pairs["measurement_eligible"] & pairs["contradiction_measurement_eligible"]).sum()
            ),
        }
        observed["passed"] = all(observed[name] >= minimum for name, minimum in required.items())
        by_split[evaluation_split] = observed
    return {
        "required_each_closed_evaluation": required,
        "by_evaluation_split": by_split,
        "passed": all(bool(values["passed"]) for values in by_split.values()),
    }


def _answer_bucket(size: int) -> str:
    if size == 0:
        return "zero"
    if size == 1:
        return "singleton"
    if size <= 10:
        return "2_to_10"
    if size <= 100:
        return "11_to_100"
    if size <= 1_000:
        return "101_to_1000"
    return "over_1000"


def _build_base_masses(
    population: pd.DataFrame,
    galleries: pd.DataFrame,
    graph_edges: pd.DataFrame,
) -> pd.DataFrame:
    primary_edges = graph_edges.loc[graph_edges["primary_near_duplicate"].astype(bool)]
    degree = (
        pd.concat([primary_edges["sequence_a"], primary_edges["sequence_b"]], ignore_index=True)
        .astype(str)
        .value_counts()
    )
    population_scores = population.loc[:, ["sequence_id"]].copy()
    population_scores["primary_degree"] = (
        population_scores["sequence_id"].map(degree).fillna(0).astype(int)
    )
    frames: list[pd.DataFrame] = []
    for gallery_id, group in galleries.groupby("gallery_id", sort=True):
        frame = group.merge(population_scores, on="sequence_id", validate="one_to_one")
        count = len(frame)
        specifications = {
            "uniform_plasmid": np.full(count, -math.log(count)),
            "uniform_v2_component": _group_log_mass(frame["leakage_component_v2"]),
            "uniform_declared_family": _group_log_mass(frame["family_key"]),
        }
        density_weight = 1.0 / (1.0 + frame["primary_degree"].to_numpy(dtype=float))
        specifications["primary_degree_corrected"] = np.log(density_weight) - math.log(
            float(density_weight.sum())
        )
        for measure, log_mass in specifications.items():
            frames.append(
                pd.DataFrame(
                    {
                        "gallery_id": str(gallery_id),
                        "base_measure": measure,
                        "sequence_id": frame["sequence_id"].astype(str).to_numpy(),
                        "log_base_mass": log_mass,
                    }
                )
            )
    result = pd.concat(frames, ignore_index=True)
    sums = (
        result.assign(mass=np.exp(result["log_base_mass"]))
        .groupby(["gallery_id", "base_measure"])["mass"]
        .sum()
    )
    if not np.allclose(sums.to_numpy(), 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError("a candidate base measure does not normalize to one")
    return result.sort_values(
        ["gallery_id", "base_measure", "sequence_id"], kind="stable"
    ).reset_index(drop=True)


def _group_log_mass(groups: pd.Series) -> np.ndarray:
    values = groups.astype(str)
    sizes = values.value_counts()
    group_count = len(sizes)
    return -math.log(group_count) - np.log(values.map(sizes).to_numpy(dtype=float))


def _run_controls(
    definitions: list[QueryDefinition],
    query_catalog: pd.DataFrame,
    galleries: dict[str, pd.DataFrame],
    verified: dict[str, set[str]],
    contradicted: dict[str, set[str]],
    state_table: pd.DataFrame,
    support: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_k = tuple(sorted(int(value) for value in params["top_k"]))
    if not top_k or top_k[0] < 1 or len(top_k) != len(set(top_k)):
        raise ValueError("top_k must contain unique positive integers")
    maximum_k = top_k[-1]
    random_seed = int(params["random_seed"])
    definition_by_id = {value.semantic_query_id: value for value in definitions}
    verified_by_constraint, contradicted_by_constraint = _constraint_state_sets(state_table)

    train_support = support.set_index("constraint_id")["train_verified_rows"].astype(float)
    verified_states = state_table.loc[
        state_table["state"].eq(VERIFIED), ["sequence_id", "constraint_id"]
    ].copy()
    verified_states["weight"] = np.log1p(
        verified_states["constraint_id"].map(train_support).fillna(0.0)
    )
    prevalence_score = verified_states.groupby("sequence_id")["weight"].sum()
    gallery_cache: dict[str, dict[str, Any]] = {}
    for gallery_id, members in galleries.items():
        candidate_ids = tuple(members["sequence_id"].astype(str))
        prevalence = np.asarray(
            [float(prevalence_score.get(sequence_id, 0.0)) for sequence_id in candidate_ids]
        )
        order = np.lexsort((np.asarray(candidate_ids, dtype=object), -prevalence))
        gallery_cache[gallery_id] = {
            "ids": candidate_ids,
            "set": set(candidate_ids),
            "prevalence_top": tuple(candidate_ids[index] for index in order[:maximum_k]),
            "stable_hash": np.asarray(
                [_stable_u64(value) for value in candidate_ids], dtype=np.uint64
            ),
        }

    ranking_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    for row in query_catalog.itertuples(index=False):
        definition = definition_by_id[str(row.semantic_query_id)]
        cache = gallery_cache[str(row.gallery_id)]
        candidate_ids = cache["ids"]
        member_ids = cache["set"]
        valid = verified[definition.semantic_query_id].intersection(member_ids)
        negative = contradicted[definition.semantic_query_id].intersection(member_ids)
        unknown_count = len(candidate_ids) - len(valid) - len(negative)
        controls = {
            "verified_first_oracle": _priority_top(
                candidate_ids, valid, negative, first=VERIFIED, maximum_k=maximum_k
            ),
            "contradiction_first": _priority_top(
                candidate_ids, valid, negative, first=CONTRADICTED, maximum_k=maximum_k
            ),
            "metadata_prevalence_prior": cache["prevalence_top"],
            "deterministic_random": _random_top(
                candidate_ids,
                cache["stable_hash"],
                query_id=str(row.query_id),
                seed=random_seed,
                maximum_k=maximum_k,
            ),
        }
        for control, ranked_ids in controls.items():
            top_records: list[dict[str, Any]] = []
            for rank, sequence_id in enumerate(ranked_ids, start=1):
                state = (
                    VERIFIED
                    if sequence_id in valid
                    else (CONTRADICTED if sequence_id in negative else UNKNOWN)
                )
                verified_fraction, contradicted_fraction = _constraint_fractions(
                    sequence_id,
                    definition.constraint_ids,
                    verified_by_constraint,
                    contradicted_by_constraint,
                )
                record = {
                    "query_id": str(row.query_id),
                    "semantic_query_id": definition.semantic_query_id,
                    "gallery_id": str(row.gallery_id),
                    "control": control,
                    "rank": rank,
                    "sequence_id": sequence_id,
                    "state": state,
                    "verified_constraint_fraction": verified_fraction,
                    "contradicted_constraint_fraction": contradicted_fraction,
                    "unknown_constraint_fraction": (
                        1.0 - verified_fraction - contradicted_fraction
                    ),
                }
                ranking_records.append(record)
                top_records.append(record)
            for k in top_k:
                selected = top_records[: min(k, len(top_records))]
                denominator = len(selected)
                state_counts = pd.Series(
                    [value["state"] for value in selected], dtype="object"
                ).value_counts()
                metric_records.append(
                    {
                        "query_id": str(row.query_id),
                        "semantic_query_id": definition.semantic_query_id,
                        "gallery_id": str(row.gallery_id),
                        "control": control,
                        "k": k,
                        "evaluated_candidates": denominator,
                        "verified_at_k": float(state_counts.get(VERIFIED, 0) / denominator),
                        "contradicted_at_k": float(state_counts.get(CONTRADICTED, 0) / denominator),
                        "unknown_at_k": float(state_counts.get(UNKNOWN, 0) / denominator),
                        "known_at_k": float(
                            (state_counts.get(VERIFIED, 0) + state_counts.get(CONTRADICTED, 0))
                            / denominator
                        ),
                        "verified_constraint_fraction_at_k": float(
                            np.mean([value["verified_constraint_fraction"] for value in selected])
                        ),
                        "contradicted_constraint_fraction_at_k": float(
                            np.mean(
                                [value["contradicted_constraint_fraction"] for value in selected]
                            )
                        ),
                        "unknown_constraint_fraction_at_k": float(
                            np.mean([value["unknown_constraint_fraction"] for value in selected])
                        ),
                        "analytic_random_verified_fraction": len(valid) / len(candidate_ids),
                        "analytic_random_contradicted_fraction": len(negative) / len(candidate_ids),
                        "analytic_random_unknown_fraction": unknown_count / len(candidate_ids),
                    }
                )
    rankings = (
        pd.DataFrame.from_records(ranking_records)
        .sort_values(["query_id", "control", "rank"], kind="stable")
        .reset_index(drop=True)
    )
    metrics = (
        pd.DataFrame.from_records(metric_records)
        .sort_values(["query_id", "control", "k"], kind="stable")
        .reset_index(drop=True)
    )
    return rankings, metrics


def _priority_top(
    candidate_ids: tuple[str, ...],
    valid: set[str],
    negative: set[str],
    *,
    first: str,
    maximum_k: int,
) -> tuple[str, ...]:
    first_set = valid if first == VERIFIED else negative
    last_set = negative if first == VERIFIED else valid
    result = heapq.nsmallest(maximum_k, first_set)
    if len(result) < maximum_k:
        for sequence_id in candidate_ids:
            if sequence_id not in valid and sequence_id not in negative:
                result.append(sequence_id)
                if len(result) == maximum_k:
                    break
    if len(result) < maximum_k:
        result.extend(heapq.nsmallest(maximum_k - len(result), last_set))
    return tuple(result)


def _random_top(
    candidate_ids: tuple[str, ...],
    stable_hashes: np.ndarray,
    *,
    query_id: str,
    seed: int,
    maximum_k: int,
) -> tuple[str, ...]:
    key = np.uint64(_stable_u64(f"{seed}|{query_id}"))
    scores = _splitmix64(stable_hashes ^ key)
    count = min(maximum_k, len(candidate_ids))
    indices = np.argpartition(scores, count - 1)[:count]
    candidate_array = np.asarray(candidate_ids, dtype=object)
    order = np.lexsort((candidate_array[indices], scores[indices]))
    return tuple(candidate_array[indices[order]].tolist())


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big", signed=False)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        result = values + np.uint64(0x9E3779B97F4A7C15)
        result = (result ^ (result >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        result = (result ^ (result >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return result ^ (result >> np.uint64(31))


def _constraint_fractions(
    sequence_id: str,
    constraint_ids: tuple[str, ...],
    verified: dict[str, set[str]],
    contradicted: dict[str, set[str]],
) -> tuple[float, float]:
    count = len(constraint_ids)
    verified_count = sum(sequence_id in verified.get(value, set()) for value in constraint_ids)
    contradicted_count = sum(
        sequence_id in contradicted.get(value, set()) for value in constraint_ids
    )
    return verified_count / count, contradicted_count / count


def validate_query_benchmark_tables(
    query_catalog: pd.DataFrame,
    galleries: pd.DataFrame,
    query_states: pd.DataFrame,
    base_masses: pd.DataFrame,
    rankings: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    expected_rows: int,
    top_k: tuple[int, ...],
) -> dict[str, Any]:
    """Validate persisted-table invariants without rebuilding query definitions."""
    _require_columns(
        query_catalog,
        {
            "query_id",
            "semantic_query_id",
            "query_kind",
            "gallery_id",
            "candidate_count",
            "answer_set_size",
            "contradiction_set_size",
            "unknown_set_size",
        },
        name="query catalog",
    )
    _require_columns(
        galleries,
        {"gallery_id", "gallery_kind", "sequence_id", "leakage_component_v2"},
        name="candidate galleries",
    )
    _require_columns(
        query_states,
        {"semantic_query_id", "sequence_id", "state"},
        name="query candidate states",
    )
    _require_columns(
        base_masses,
        {"gallery_id", "base_measure", "sequence_id", "log_base_mass"},
        name="candidate base masses",
    )
    _require_columns(
        rankings,
        {"query_id", "control", "rank", "sequence_id", "state"},
        name="control rankings",
    )
    _require_columns(
        metrics,
        {"query_id", "control", "k", "verified_at_k", "contradicted_at_k"},
        name="control metrics",
    )
    if query_catalog.empty or query_catalog["query_id"].duplicated().any():
        raise RuntimeError("query catalog is empty or has duplicate query identifiers")
    if galleries.duplicated(["gallery_id", "sequence_id"]).any():
        raise RuntimeError("candidate galleries contain duplicate memberships")
    if query_states.duplicated(["semantic_query_id", "sequence_id"]).any():
        raise RuntimeError("a query-candidate pair has more than one explicit state")
    if set(query_states["state"].astype(str)) - {VERIFIED, CONTRADICTED}:
        raise RuntimeError("query-candidate states contain an invalid explicit state")
    if rankings.duplicated(["query_id", "control", "rank"]).any():
        raise RuntimeError("control rankings contain duplicate ranks")
    if rankings.duplicated(["query_id", "control", "sequence_id"]).any():
        raise RuntimeError("control rankings contain a duplicate candidate")
    if metrics.duplicated(["query_id", "control", "k"]).any():
        raise RuntimeError("control metrics contain duplicate keys")
    open_sizes = (
        galleries.loc[galleries["gallery_kind"].eq("open_all")]
        .groupby("gallery_id")["sequence_id"]
        .nunique()
    )
    if open_sizes.empty or not open_sizes.eq(expected_rows).all():
        raise RuntimeError("an open gallery does not contain the complete population")
    if galleries.groupby("leakage_component_v2")["split_grouped_v2"].nunique().gt(1).any():
        raise RuntimeError("a v2 component crosses gallery split labels")
    sizes = query_catalog[["answer_set_size", "contradiction_set_size", "unknown_set_size"]].sum(
        axis=1
    )
    if not sizes.equals(query_catalog["candidate_count"]):
        raise RuntimeError("query state sizes do not sum to candidate count")
    mass_sums = (
        base_masses.assign(mass=np.exp(base_masses["log_base_mass"]))
        .groupby(["gallery_id", "base_measure"])["mass"]
        .sum()
    )
    if not np.allclose(mass_sums.to_numpy(), 1.0, rtol=1e-12, atol=1e-12):
        raise RuntimeError("a reloaded candidate base measure does not normalize")
    if set(metrics["k"].astype(int)) != set(top_k):
        raise RuntimeError("control metrics do not contain the configured K values")
    joined = metrics.merge(
        query_catalog[["query_id", "answer_set_size", "contradiction_set_size"]],
        on="query_id",
        how="inner",
        validate="many_to_one",
    )
    oracle_failures = joined.loc[
        joined["control"].eq("verified_first_oracle")
        & joined["answer_set_size"].ge(joined["k"])
        & ~np.isclose(joined["verified_at_k"], 1.0)
    ]
    if not oracle_failures.empty:
        raise RuntimeError("verified-first oracle is not perfect where enough answers exist")
    contradiction_failures = joined.loc[
        joined["control"].eq("contradiction_first")
        & joined["contradiction_set_size"].ge(joined["k"])
        & ~np.isclose(joined["contradicted_at_k"], 1.0)
    ]
    if not contradiction_failures.empty:
        raise RuntimeError(
            "contradiction-first control is not perfect where enough contradictions exist"
        )
    return {
        "status": "accepted_table_invariants",
        "query_rows": int(len(query_catalog)),
        "semantic_queries": int(query_catalog["semantic_query_id"].nunique()),
        "gallery_count": int(galleries["gallery_id"].nunique()),
        "open_gallery_rows": {str(key): int(value) for key, value in open_sizes.items()},
        "query_state_pairs_disjoint": True,
        "base_measures_normalized": True,
        "verified_first_oracle_checks_passed": True,
        "contradiction_first_checks_passed": True,
    }


def _require_columns(frame: pd.DataFrame, columns: set[str], *, name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
