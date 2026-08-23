"""Independent read-back validation for E02b benchmark input artifacts."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vec2vec.lib.similarity_graph import dataframe_content_sha256


def validate_bakeoff_inputs(
    pairs: pd.DataFrame,
    exclusions: pd.DataFrame,
    queries: pd.DataFrame,
    query_states: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    expected_protocol_version: str,
    expected_training_rows: int,
) -> dict[str, Any]:
    """Recompute the frozen input contract from persisted E02b products."""
    if manifest.get("protocol_version") != expected_protocol_version:
        raise ValueError("persisted E02b protocol version changed")
    expected_hashes = manifest.get("output_hashes")
    if not isinstance(expected_hashes, dict):
        raise ValueError("persisted E02b manifest lacks output hashes")
    observed_hashes = {
        "pairs_sha256": dataframe_content_sha256(pairs, sort_columns=["panel_role", "sequence_id"]),
        "exclusions_sha256": dataframe_content_sha256(
            exclusions, sort_columns=["split_grouped_v2", "sequence_id"]
        ),
        "queries_sha256": dataframe_content_sha256(queries, sort_columns=["query_id"]),
        "query_states_sha256": dataframe_content_sha256(
            query_states, sort_columns=["semantic_query_id", "sequence_id"]
        ),
    }
    if observed_hashes != expected_hashes:
        raise ValueError(
            "persisted E02b output hashes changed: "
            f"expected {expected_hashes}, observed {observed_hashes}"
        )
    required_pairs = {
        "sequence_id",
        "sequence",
        "sequence_sha256",
        "panel_role",
        "split_grouped_v2",
        "similarity_component_primary",
    }
    missing_pairs = required_pairs.difference(pairs.columns)
    if missing_pairs:
        raise ValueError(f"persisted E02b pairs are missing columns: {sorted(missing_pairs)}")
    if pairs.empty or pairs[list(required_pairs)].isna().any(axis=None):
        raise ValueError("persisted E02b pairs must be non-empty and complete")
    if pairs["sequence_id"].duplicated().any():
        raise ValueError("persisted E02b pairs repeat sequence identifiers")
    if set(pairs["panel_role"].astype(str)) != {"alignment_train", "validation_gallery"}:
        raise ValueError("persisted E02b pairs contain an unexpected panel role")
    expected_split_by_role = {
        "alignment_train": "train",
        "validation_gallery": "val",
    }
    observed_split = pairs["panel_role"].map(expected_split_by_role)
    if not observed_split.eq(pairs["split_grouped_v2"].astype(str)).all():
        raise ValueError("persisted E02b panel roles do not match split_grouped_v2")
    invalid_sequences = pairs["sequence"].map(
        lambda value: bool(set(str(value)).difference("ACGT"))
    )
    if invalid_sequences.any():
        raise ValueError(
            f"persisted E02b pairs contain {int(invalid_sequences.sum())} non-ACGT sequences"
        )
    training_rows = int(pairs["panel_role"].eq("alignment_train").sum())
    if training_rows != expected_training_rows:
        raise ValueError(
            f"persisted E02b training rows changed: expected {expected_training_rows}, "
            f"observed {training_rows}"
        )
    gallery = pairs.loc[pairs["panel_role"].eq("validation_gallery")]
    population_flow = manifest.get("population_flow", {})
    if len(gallery) != population_flow.get("val", {}).get("selected_rows"):
        raise ValueError("persisted E02b validation row count differs from its manifest")
    expected_exclusions = sum(
        int(population_flow.get(split, {}).get("excluded_rows", -1)) for split in ("train", "val")
    )
    if len(exclusions) != expected_exclusions:
        raise ValueError("persisted E02b exclusion count differs from its manifest")
    if queries.empty or queries["query_id"].duplicated().any():
        raise ValueError("persisted E02b queries are empty or repeat query identifiers")
    if query_states.duplicated(["semantic_query_id", "sequence_id"]).any():
        raise ValueError("persisted E02b query states repeat a query and sequence pair")
    if not set(query_states["state"].astype(str)) <= {"verified", "contradicted"}:
        raise ValueError("persisted E02b query states contain an invalid state")
    unknown_semantic_ids = set(query_states["semantic_query_id"].astype(str)).difference(
        queries["semantic_query_id"].astype(str)
    )
    unknown_sequence_ids = set(query_states["sequence_id"].astype(str)).difference(
        gallery["sequence_id"].astype(str)
    )
    if unknown_semantic_ids or unknown_sequence_ids:
        raise ValueError(
            "persisted E02b query states refer to rows outside the frozen query/gallery inputs"
        )
    decision = manifest.get("decision", {})
    if (
        decision.get("model_outcomes_read") is not False
        or decision.get("candidate_selected") is not False
    ):
        raise ValueError("persisted E02b input manifest records a premature model decision")
    return {
        "status": "passed_e02b_input_readback",
        "protocol_version": expected_protocol_version,
        "training_rows": training_rows,
        "validation_rows": int(len(gallery)),
        "excluded_rows": int(len(exclusions)),
        "queries": int(len(queries)),
        "query_states": int(len(query_states)),
        "output_hashes": observed_hashes,
        "validation_only": True,
        "current_test_split_contaminated_before_e02b": True,
    }
