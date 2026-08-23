"""Candidate-neutral inputs for the reduced-population Gate 1 bake-off."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any

import pandas as pd

from vec2vec.lib.constraint_state import retrieval_population_sha256
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.lib.text import sha256_text

PAIR_HASH_COLUMNS = [
    "sequence_id",
    "sequence_sha256",
    "description_sha256",
    "split_grouped_v2",
    "similarity_component_primary",
    "length_bp",
    "component_size",
    "selection_pass",
    "selection_sha256",
]


def validated_compute_authorization(
    params: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Return an explicit paid-compute authorization for one E02b stage."""
    authorization = params.get("compute_authorization")
    if not isinstance(authorization, dict):
        raise ValueError(f"{stage} requires an explicit compute_authorization")
    required_text = ("approval_reference", "region", "instance_type")
    for name in required_text:
        if not str(authorization.get(name, "")).strip():
            raise ValueError(f"{stage} compute_authorization requires {name}")
    for name in ("instance_hour_limit", "observed_instance_price_usd_per_hour"):
        try:
            value = float(authorization[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{stage} compute_authorization requires numeric {name}") from error
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{stage} compute_authorization {name} must be finite and positive")
    authorized_stage = str(authorization.get("stage", ""))
    if authorized_stage != stage:
        raise ValueError(
            f"compute_authorization is for {authorized_stage!r}, not requested stage {stage!r}"
        )
    return {
        "stage": stage,
        "approval_reference": str(authorization["approval_reference"]),
        "region": str(authorization["region"]),
        "instance_type": str(authorization["instance_type"]),
        "instance_hour_limit": float(authorization["instance_hour_limit"]),
        "observed_instance_price_usd_per_hour": float(
            authorization["observed_instance_price_usd_per_hour"]
        ),
    }


def build_bakeoff_inputs(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    split_manifest: dict[str, Any],
    query_catalog: pd.DataFrame,
    query_states: pd.DataFrame,
    query_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Freeze the E02b train pairs, validation gallery, queries, and exclusions."""
    _validate_source_artifacts(
        retrieval,
        split_mapping,
        split_manifest,
        query_catalog,
        query_states,
        query_manifest,
        params,
    )
    rows = retrieval.merge(
        split_mapping,
        on="sequence_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(rows["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"split mapping has {unmatched} unmatched retrieval rows")
    rows = rows.drop(columns="_merge")
    rows["description_sha256"] = rows["description"].astype(str).map(sha256_text)
    rows["component_size"] = rows.groupby("similarity_component_primary")["sequence_id"].transform(
        "size"
    )
    if params.get("eligible_sequence_alphabet") != "ACGT":
        raise ValueError("E02b eligible_sequence_alphabet must be ACGT")
    rows["eligible"] = rows["sequence"].map(lambda value: set(str(value)) <= set("ACGT"))
    _validate_sequences(rows)

    eligible = rows.loc[rows["eligible"] & rows["split_grouped_v2"].isin(["train", "val"])].copy()
    train = eligible.loc[eligible["split_grouped_v2"].eq("train")].copy()
    validation = eligible.loc[eligible["split_grouped_v2"].eq("val")].copy()
    selected_train = select_component_balanced_train_panel(
        train,
        rows=int(params["training_rows"]),
        maximum_rows_per_component=int(params["maximum_rows_per_component"]),
        salt=str(params["selection_salt"]),
    )
    selected_train["panel_role"] = "alignment_train"
    validation["selection_pass"] = pd.NA
    validation["selection_sha256"] = validation["sequence_id"].map(
        lambda sequence_id: sha256_text(f"{params['selection_salt']}|validation|{sequence_id}")
    )
    validation["panel_role"] = "validation_gallery"
    pairs = pd.concat([selected_train, validation], ignore_index=True)
    pairs = pairs.loc[
        :,
        [
            "sequence_id",
            "sequence",
            "sequence_sha256",
            "description",
            "description_sha256",
            "length_bp",
            "similarity_component_primary",
            "leakage_component_v2",
            "split_grouped_v2",
            "component_size",
            "selection_pass",
            "selection_sha256",
            "panel_role",
        ],
    ].sort_values(["panel_role", "sequence_id"], kind="stable", ignore_index=True)

    exclusions = _build_exclusions(rows)
    queries, filtered_states = _validation_queries(
        query_catalog,
        query_states,
        validation_ids=set(validation["sequence_id"].astype(str)),
        minimum_verified_rows=int(params["minimum_verified_rows"]),
    )
    hashes = {
        "pairs_sha256": dataframe_content_sha256(
            pairs,
            sort_columns=["panel_role", "sequence_id"],
        ),
        "exclusions_sha256": dataframe_content_sha256(
            exclusions,
            sort_columns=["split_grouped_v2", "sequence_id"],
        ),
        "queries_sha256": dataframe_content_sha256(queries, sort_columns=["query_id"]),
        "query_states_sha256": dataframe_content_sha256(
            filtered_states,
            sort_columns=["semantic_query_id", "sequence_id"],
        ),
    }
    report = {
        "protocol_version": str(params["protocol_version"]),
        "input_versions": dict(params["input_versions"]),
        "input_population_sha256": str(params["expected_input_population_sha256"]),
        "eligibility_rule": "uppercase_acgt_only_before_model_processing",
        "training_selection": {
            "rows": int(len(selected_train)),
            "components": int(selected_train["similarity_component_primary"].nunique()),
            "maximum_rows_per_component": int(
                selected_train.groupby("similarity_component_primary").size().max()
            ),
            "selection_salt": str(params["selection_salt"]),
        },
        "population_flow": _population_flow(rows, pairs),
        "queries": {
            "rows": int(len(queries)),
            "atomic": int(queries["query_kind"].eq("atomic").sum()),
            "pair_conjunction": int(queries["query_kind"].eq("pair_conjunction").sum()),
            "minimum_verified_rows": int(queries["eligible_verified_rows"].min()),
        },
        "output_hashes": hashes,
        "candidate_disposition": {
            "carbon_3b": {
                "status": "not_evaluable_technical_ineligibility",
                "reason": "exceeded_22_and_44_gib_and_80_gib_capacity_unavailable",
            }
        },
        "decision": {
            "validation_only": True,
            "current_test_split_contaminated_before_e02b": True,
            "model_outcomes_read": False,
            "candidate_selected": False,
        },
    }
    return pairs, exclusions, queries, filtered_states, report


def select_component_balanced_train_panel(
    train: pd.DataFrame,
    *,
    rows: int,
    maximum_rows_per_component: int,
    salt: str,
) -> pd.DataFrame:
    """Select hash-ordered rows through inverse-size weighted component passes."""
    if rows < 1 or maximum_rows_per_component < 1:
        raise ValueError("training rows and component cap must be positive")
    if len(train) < rows:
        raise ValueError(f"eligible training pool has {len(train)} rows, below requested {rows}")
    required = {
        "sequence_id",
        "sequence_sha256",
        "similarity_component_primary",
        "component_size",
    }
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"training pool is missing columns: {sorted(missing)}")

    ordered = train.copy()
    ordered["selection_sha256"] = ordered.apply(
        lambda row: sha256_text(
            "|".join(
                [
                    salt,
                    str(row["similarity_component_primary"]),
                    str(row["sequence_id"]),
                    str(row["sequence_sha256"]),
                ]
            )
        ),
        axis=1,
    )
    ordered = ordered.sort_values(
        ["similarity_component_primary", "selection_sha256"], kind="stable"
    )
    ordered["selection_pass"] = ordered.groupby("similarity_component_primary").cumcount()
    ordered = ordered.loc[ordered["selection_pass"].lt(maximum_rows_per_component)].copy()

    selected: list[int] = []
    for selection_pass, candidates in ordered.groupby("selection_pass", sort=True):
        current_pass = int(selection_pass)
        candidates = candidates.copy()
        candidates["component_key"] = candidates.apply(
            lambda row, selection_pass=current_pass: _weighted_component_key(
                salt,
                selection_pass,
                str(row["similarity_component_primary"]),
                int(row["component_size"]),
            ),
            axis=1,
        )
        candidates = candidates.sort_values(
            ["component_key", "similarity_component_primary"], kind="stable"
        )
        remaining = rows - len(selected)
        selected.extend(candidates.index[:remaining].tolist())
        if len(selected) == rows:
            break
    if len(selected) != rows:
        raise ValueError(
            f"component cap permits only {len(selected)} training rows, below requested {rows}"
        )
    result = ordered.loc[selected].copy()
    counts = result.groupby("similarity_component_primary").size()
    if int(counts.max()) > maximum_rows_per_component:
        raise RuntimeError("training panel exceeded the component row cap")
    return result.sort_values(["selection_pass", "selection_sha256"], kind="stable")


def _weighted_component_key(salt: str, selection_pass: int, component: str, size: int) -> float:
    digest = hashlib.sha256(f"{salt}|{selection_pass}|{component}".encode()).digest()
    uniform = (int.from_bytes(digest[:8], "big") + 1) / (2**64 + 1)
    return -math.log(uniform) * size


def _validation_queries(
    query_catalog: pd.DataFrame,
    query_states: pd.DataFrame,
    *,
    validation_ids: set[str],
    minimum_verified_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = query_catalog.loc[
        query_catalog["evaluation_split"].eq("val")
        & query_catalog["gallery_kind"].eq("closed_grouped_v2")
        & query_catalog["measurement_eligible"].astype(bool)
    ].copy()
    if queries.empty or queries["query_id"].duplicated().any():
        raise ValueError("validation query selection is empty or has duplicate query_id values")
    semantic_ids = set(queries["semantic_query_id"].astype(str))
    states = query_states.loc[
        query_states["semantic_query_id"].astype(str).isin(semantic_ids)
        & query_states["sequence_id"].astype(str).isin(validation_ids)
    ].copy()
    if not set(states["state"].astype(str)) <= {"verified", "contradicted"}:
        raise ValueError("query states contain values outside verified and contradicted")
    if states.duplicated(["semantic_query_id", "sequence_id"]).any():
        raise ValueError("validation query states repeat a semantic-query and sequence pair")
    support = (
        states.groupby(["semantic_query_id", "state"], observed=True)
        .size()
        .unstack(fill_value=0)
        .rename(
            columns={
                "verified": "eligible_verified_rows",
                "contradicted": "eligible_contradicted_rows",
            }
        )
    )
    for column in ("eligible_verified_rows", "eligible_contradicted_rows"):
        if column not in support:
            support[column] = 0
    queries = queries.merge(
        support.reset_index(),
        on="semantic_query_id",
        how="left",
        validate="many_to_one",
    )
    queries[["eligible_verified_rows", "eligible_contradicted_rows"]] = (
        queries[["eligible_verified_rows", "eligible_contradicted_rows"]].fillna(0).astype("int64")
    )
    unusable = queries["eligible_verified_rows"].lt(minimum_verified_rows)
    if unusable.any():
        identifiers = queries.loc[unusable, "query_id"].astype(str).head(5).tolist()
        raise ValueError(
            f"{int(unusable.sum())} validation queries lost measurement support: {identifiers}"
        )
    return (
        queries.sort_values("query_id", kind="stable", ignore_index=True),
        states.sort_values(["semantic_query_id", "sequence_id"], kind="stable", ignore_index=True),
    )


def _build_exclusions(rows: pd.DataFrame) -> pd.DataFrame:
    excluded = rows.loc[rows["split_grouped_v2"].isin(["train", "val"]) & ~rows["eligible"]].copy()
    excluded["unsupported_symbol_counts_json"] = excluded["sequence"].map(
        lambda value: json.dumps(
            dict(sorted(Counter(symbol for symbol in str(value) if symbol not in "ACGT").items())),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return excluded.loc[
        :,
        [
            "sequence_id",
            "sequence_sha256",
            "split_grouped_v2",
            "similarity_component_primary",
            "length_bp",
            "component_size",
            "unsupported_symbol_counts_json",
        ],
    ].sort_values(["split_grouped_v2", "sequence_id"], kind="stable", ignore_index=True)


def _population_flow(rows: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("train", "val"):
        source = rows.loc[rows["split_grouped_v2"].eq(split)]
        eligible = source.loc[source["eligible"]]
        selected = pairs.loc[pairs["split_grouped_v2"].eq(split)]
        result[split] = {
            "original_rows": int(len(source)),
            "eligible_rows": int(len(eligible)),
            "excluded_rows": int(len(source) - len(eligible)),
            "excluded_fraction": float(1.0 - len(eligible) / len(source)),
            "selected_rows": int(len(selected)),
            "original_components": int(source["similarity_component_primary"].nunique()),
            "eligible_components": int(eligible["similarity_component_primary"].nunique()),
            "selected_components": int(selected["similarity_component_primary"].nunique()),
        }
    return result


def _validate_source_artifacts(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    split_manifest: dict[str, Any],
    query_catalog: pd.DataFrame,
    query_states: pd.DataFrame,
    query_manifest: dict[str, Any],
    params: dict[str, Any],
) -> None:
    required_retrieval = {
        "sequence_id",
        "sequence",
        "sequence_sha256",
        "description",
        "length_bp",
    }
    required_split = {
        "sequence_id",
        "similarity_component_primary",
        "leakage_component_v2",
        "split_grouped_v2",
    }
    required_queries = {
        "query_id",
        "semantic_query_id",
        "query_kind",
        "canonical_query_text",
        "evaluation_split",
        "gallery_kind",
        "measurement_eligible",
    }
    required_states = {"semantic_query_id", "sequence_id", "state"}
    for name, frame, required in (
        ("retrieval", retrieval, required_retrieval),
        ("split mapping", split_mapping, required_split),
        ("query catalog", query_catalog, required_queries),
        ("query states", query_states, required_states),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} must not be empty")
    duplicate_retrieval = retrieval["sequence_id"].duplicated().any()
    duplicate_split = split_mapping["sequence_id"].duplicated().any()
    if duplicate_retrieval or duplicate_split:
        raise ValueError("retrieval and split sequence identifiers must be unique")
    observed_population = retrieval_population_sha256(retrieval)
    expected_population = str(params["expected_input_population_sha256"])
    if observed_population != expected_population:
        raise ValueError(
            "retrieval population changed: "
            f"expected {expected_population}, observed {observed_population}"
        )
    expected_versions = {
        key: value
        for key, value in dict(params["input_versions"]).items()
        if key != "query_benchmark"
    }
    observed_versions = dict(query_manifest.get("input_versions", {}))
    if observed_versions != expected_versions:
        raise ValueError(
            "query benchmark input versions changed: "
            f"expected {expected_versions}, observed {observed_versions}"
        )
    expected_hashes = dict(params["expected_query_artifact_hashes"])
    observed_hashes = dict(query_manifest.get("output_content_hashes", {}))
    for name in ("query_catalog_sha256", "query_candidate_state_sha256"):
        if observed_hashes.get(name) != expected_hashes.get(name):
            raise ValueError(f"query benchmark manifest hash changed for {name}")
    observed_query_hash = dataframe_content_sha256(query_catalog, sort_columns=["query_id"])
    observed_state_hash = dataframe_content_sha256(
        query_states,
        sort_columns=["semantic_query_id", "state", "sequence_id"],
    )
    if observed_query_hash != expected_hashes["query_catalog_sha256"]:
        raise ValueError("loaded query catalog hash changed")
    if observed_state_hash != expected_hashes["query_candidate_state_sha256"]:
        raise ValueError("loaded query state hash changed")
    if (
        split_manifest.get("decision", {}).get("status")
        != "accepted_strict_similarity_closed_split"
    ):
        raise ValueError("split_grouped_v2 manifest is not accepted")
    observed_mapping_hash = dataframe_content_sha256(split_mapping, sort_columns=["sequence_id"])
    if observed_mapping_hash != split_manifest.get("build", {}).get("mapping_sha256"):
        raise ValueError("loaded split_grouped_v2 mapping hash changed")


def _validate_sequences(rows: pd.DataFrame) -> None:
    missing = rows[["sequence", "sequence_sha256", "description", "length_bp"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"retrieval rows have {int(missing.sum())} missing model inputs")
    for row in rows.itertuples(index=False):
        sequence = str(row.sequence)
        if len(sequence) != int(row.length_bp):
            raise ValueError(f"sequence {row.sequence_id} length mismatch")
        if sequence_sha256(sequence) != str(row.sequence_sha256):
            raise ValueError(f"sequence {row.sequence_id} SHA-256 mismatch")
