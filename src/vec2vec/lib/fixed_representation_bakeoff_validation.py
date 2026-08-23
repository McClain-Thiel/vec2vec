"""Independent read-back validation for E02b benchmark input artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import fixed_representation
from vec2vec.lib.fixed_representation_invariance import json_content_sha256
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
        "manifest_sha256": json_content_sha256(manifest),
        "output_hashes": observed_hashes,
        "validation_only": True,
        "current_test_split_contaminated_before_e02b": True,
    }


def validate_tfidf_features(
    pairs: pd.DataFrame,
    input_manifest: dict[str, Any],
    features: pd.DataFrame,
    vocabulary: pd.DataFrame,
    svd_state: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    expected_candidate_id: str,
    expected_dimension: int,
    expected_training_rows: int,
) -> dict[str, Any]:
    """Validate one persisted train-fitted TF-IDF/SVD feature product."""
    if manifest.get("feature_kind") != "tfidf_dna":
        raise ValueError("persisted TF-IDF manifest has the wrong feature kind")
    if manifest.get("candidate_id") != expected_candidate_id:
        raise ValueError("persisted TF-IDF candidate identifier changed")
    if manifest.get("input_manifest_sha256") != json_content_sha256(input_manifest):
        raise ValueError("persisted TF-IDF input manifest hash changed")
    if manifest.get("training_rows") != expected_training_rows:
        raise ValueError("persisted TF-IDF training row count changed")
    if manifest.get("decision", {}).get("status") != "frozen_features_complete":
        raise ValueError("persisted TF-IDF feature product is not frozen as complete")
    if manifest.get("decision", {}).get("validation_rankings_computed") is not False:
        raise ValueError("persisted TF-IDF extraction read validation rankings")

    feature_hash = dataframe_content_sha256(
        features, sort_columns=["candidate_id", "sequence_sha256"]
    )
    vocabulary_hash = dataframe_content_sha256(vocabulary, sort_columns=["term_index"])
    state_hash = dataframe_content_sha256(svd_state, sort_columns=["component"])
    observed_hashes = {
        "features_sha256": feature_hash,
        "vocabulary_sha256": vocabulary_hash,
        "svd_state_sha256": state_hash,
    }
    if observed_hashes != manifest.get("output_hashes"):
        raise ValueError("persisted TF-IDF output hashes changed")

    required_features = {
        "candidate_id",
        "sequence_sha256",
        "embedding_dimension",
        "embedding",
        "embedding_sha256",
    }
    missing_features = required_features.difference(features.columns)
    if missing_features:
        raise ValueError(
            f"persisted TF-IDF features are missing columns: {sorted(missing_features)}"
        )
    if features.empty or features[list(required_features)].isna().any(axis=None):
        raise ValueError("persisted TF-IDF features must be non-empty and complete")
    if features["sequence_sha256"].duplicated().any():
        raise ValueError("persisted TF-IDF features repeat sequence hashes")
    expected_sequence_hashes = set(pairs["sequence_sha256"].astype(str))
    observed_sequence_hashes = set(features["sequence_sha256"].astype(str))
    if observed_sequence_hashes != expected_sequence_hashes:
        raise ValueError("persisted TF-IDF features do not cover the exact E02b sequence set")
    if set(features["candidate_id"].astype(str)) != {expected_candidate_id}:
        raise ValueError("persisted TF-IDF features contain another candidate")
    if set(features["embedding_dimension"].astype(int)) != {expected_dimension}:
        raise ValueError("persisted TF-IDF embedding dimension changed")
    vectors = [np.asarray(value, dtype=np.float32) for value in features["embedding"]]
    if any(vector.shape != (expected_dimension,) for vector in vectors):
        raise ValueError("persisted TF-IDF embedding shape changed")
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("persisted TF-IDF feature contains a non-finite value")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("persisted TF-IDF features are not L2-normalized")
    observed_embedding_hashes = [fixed_representation.embedding_sha256(row) for row in matrix]
    if observed_embedding_hashes != features["embedding_sha256"].astype(str).tolist():
        raise ValueError("persisted TF-IDF per-row embedding hashes changed")

    required_vocabulary = {"term", "term_index", "idf"}
    if required_vocabulary.difference(vocabulary.columns) or vocabulary.empty:
        raise ValueError("persisted TF-IDF vocabulary schema is incomplete")
    ordered_indices = vocabulary["term_index"].astype(int).sort_values().to_numpy()
    if not np.array_equal(ordered_indices, np.arange(len(vocabulary))):
        raise ValueError("persisted TF-IDF vocabulary indices are not contiguous")
    invalid_terms = (
        vocabulary["term"]
        .astype(str)
        .map(lambda term: len(term) != 6 or bool(set(term).difference("ACGT")))
    )
    if invalid_terms.any():
        raise ValueError("persisted TF-IDF vocabulary contains a non-ACGT 6-mer")
    idf = vocabulary["idf"].to_numpy(dtype=np.float64)
    if not np.isfinite(idf).all() or np.any(idf <= 0.0):
        raise ValueError("persisted TF-IDF vocabulary contains an invalid IDF")

    required_state = {"component", "singular_value", "vector"}
    if required_state.difference(svd_state.columns) or len(svd_state) != expected_dimension:
        raise ValueError("persisted TF-IDF SVD state has the wrong schema or dimension")
    components = svd_state["component"].astype(int).to_numpy()
    if not np.array_equal(components, np.arange(expected_dimension)):
        raise ValueError("persisted TF-IDF SVD component indices changed")
    state_vectors = [np.asarray(value, dtype=np.float32) for value in svd_state["vector"]]
    if any(vector.shape != (len(vocabulary),) for vector in state_vectors):
        raise ValueError("persisted TF-IDF SVD vector shape changed")
    if not np.isfinite(np.vstack(state_vectors)).all():
        raise ValueError("persisted TF-IDF SVD state contains a non-finite value")
    return {
        "status": "passed_e02b_tfidf_readback",
        "candidate_id": expected_candidate_id,
        "manifest_sha256": json_content_sha256(manifest),
        "output_hashes": observed_hashes,
        "source_rows": int(len(pairs)),
        "unique_sequences": int(len(features)),
        "embedding_dimension": expected_dimension,
        "vocabulary_terms": int(len(vocabulary)),
        "extraction_gpu_hours": 0.0,
        "elapsed_seconds": float(manifest["elapsed_seconds"]),
        "validation_rankings_computed": False,
    }
