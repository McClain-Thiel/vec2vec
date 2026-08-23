"""Independent read-back validation for E02b benchmark input artifacts."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import fixed_representation
from vec2vec.lib.fixed_representation_invariance import json_content_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.lib.text import sha256_text


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


def validate_neural_dna_features(
    pairs: pd.DataFrame,
    input_manifest: dict[str, Any],
    invariance_manifest: dict[str, Any],
    smoke_manifest: dict[str, Any],
    features: pd.DataFrame,
    coverage: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    expected_candidate_id: str,
    expected_candidate: dict[str, Any],
    expected_invariance_manifest_sha256: str,
    expected_configuration: dict[str, Any],
    accepted_input_artifact: dict[str, Any],
    expected_compute_authorization: dict[str, Any],
) -> dict[str, Any]:
    """Validate one persisted neural-DNA feature product and its full-base coverage."""
    _validate_accepted_input_identity(
        pairs,
        input_manifest,
        accepted_input_artifact=accepted_input_artifact,
    )
    _validate_neural_feature_manifest(
        input_manifest,
        manifest,
        feature_kind="neural_dna",
        expected_candidate_id=expected_candidate_id,
        expected_candidate=expected_candidate,
        expected_configuration=expected_configuration,
        expected_compute_authorization=expected_compute_authorization,
    )
    observed_invariance_hash = json_content_sha256(invariance_manifest)
    if observed_invariance_hash != expected_invariance_manifest_sha256:
        raise ValueError("persisted neural-DNA invariance artifact changed")
    if manifest.get("accepted_invariance_manifest_sha256") != observed_invariance_hash:
        raise ValueError("persisted neural-DNA manifest is not bound to accepted invariance")
    if invariance_manifest.get("candidate_id") != expected_candidate_id:
        raise ValueError("persisted neural-DNA invariance candidate changed")
    if invariance_manifest.get("candidate") != expected_candidate:
        raise ValueError("persisted neural-DNA invariance recipe changed")
    if invariance_manifest.get("decision", {}).get("status") != "passed_invariance_check":
        raise ValueError("persisted neural-DNA candidate did not pass invariance")
    accepted_smoke = invariance_manifest.get("accepted_numerical_smoke_artifact", {})
    if json_content_sha256(smoke_manifest) != accepted_smoke.get("observed_smoke_manifest_sha256"):
        raise ValueError("persisted neural-DNA numerical-smoke artifact changed")
    if smoke_manifest.get("candidate_id") != expected_candidate_id:
        raise ValueError("persisted neural-DNA numerical-smoke candidate changed")
    if smoke_manifest.get("candidate") != expected_candidate:
        raise ValueError("persisted neural-DNA numerical-smoke recipe changed")
    if smoke_manifest.get("decision", {}).get("status") != "passed_numerical_smoke":
        raise ValueError("persisted neural-DNA candidate did not pass numerical smoke")
    expected_maximum_content_bp = int(
        smoke_manifest["precision_runs"]["bfloat16"]["maximum_content_bp"]
    )
    if int(manifest.get("maximum_content_bp", 0)) != expected_maximum_content_bp:
        raise ValueError("persisted neural-DNA maximum content length differs from numerical smoke")
    expected_dimension = int(
        invariance_manifest["diagnostic_summary"]["geometry"]["embedding_dimension"]
    )

    observed_hashes = {
        "features_sha256": dataframe_content_sha256(
            features, sort_columns=["candidate_id", "sequence_sha256"]
        ),
        "coverage_sha256": dataframe_content_sha256(
            coverage,
            sort_columns=["candidate_id", "sequence_sha256", "window_index"],
        ),
    }
    if observed_hashes != manifest.get("output_hashes"):
        raise ValueError("persisted neural-DNA output hashes changed")

    required_features = {
        "candidate_id",
        "sequence_sha256",
        "representative_sequence_id",
        "length_bp",
        "embedding_dimension",
        "embedding",
        "embedding_sha256",
        "elapsed_seconds",
    }
    _require_complete_frame(features, required_features, name="neural-DNA features")
    if features["sequence_sha256"].duplicated().any():
        raise ValueError("persisted neural-DNA features repeat sequence hashes")
    expected_sequence_hashes = set(pairs["sequence_sha256"].astype(str))
    if set(features["sequence_sha256"].astype(str)) != expected_sequence_hashes:
        raise ValueError("persisted neural-DNA features do not cover the exact E02b sequence set")
    if set(features["candidate_id"].astype(str)) != {expected_candidate_id}:
        raise ValueError("persisted neural-DNA features contain another candidate")
    dimensions = set(features["embedding_dimension"].astype(int))
    if dimensions != {expected_dimension}:
        raise ValueError(
            "persisted neural-DNA embedding dimension differs from accepted invariance"
        )
    _validate_normalized_vectors(features, expected_dimension=expected_dimension)
    elapsed_by_row = features["elapsed_seconds"].to_numpy(dtype=np.float64)
    if not np.isfinite(elapsed_by_row).all() or np.any(elapsed_by_row <= 0.0):
        raise ValueError("persisted neural-DNA feature times must be finite and positive")

    expected_lengths = (
        pairs.loc[:, ["sequence_sha256", "sequence"]]
        .drop_duplicates("sequence_sha256")
        .assign(expected_length=lambda frame: frame["sequence"].astype(str).str.len())
        .set_index("sequence_sha256")["expected_length"]
    )
    observed_lengths = features.set_index("sequence_sha256")["length_bp"].astype(int)
    if not observed_lengths.sort_index().equals(expected_lengths.astype(int).sort_index()):
        raise ValueError("persisted neural-DNA feature lengths changed from the E02b input")
    _validate_dna_coverage(
        coverage,
        features,
        pairs,
        expected_candidate_id=expected_candidate_id,
        tokenizer_unit_bp=int(expected_candidate["tokenizer_unit_bp"]),
        model_max_tokens=int(expected_candidate["model_max_tokens"]),
        maximum_content_bp=expected_maximum_content_bp,
        overlap_fraction=float(expected_configuration["window_overlap_fraction"]),
    )
    return {
        "status": "passed_e02b_neural_dna_readback",
        "candidate_id": expected_candidate_id,
        "manifest_sha256": json_content_sha256(manifest),
        "output_hashes": observed_hashes,
        "source_rows": int(len(pairs)),
        "unique_sequences": int(len(features)),
        "coverage_window_rows": int(len(coverage)),
        "embedding_dimension": expected_dimension,
        "elapsed_seconds": float(manifest["elapsed_seconds"]),
        "extraction_gpu_hours": float(manifest["elapsed_seconds"]) / 3600.0,
        "validation_rankings_computed": False,
    }


def validate_text_features(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    input_manifest: dict[str, Any],
    features: pd.DataFrame,
    manifest: dict[str, Any],
    *,
    expected_candidate_id: str,
    expected_candidate: dict[str, Any],
    expected_configuration: dict[str, Any],
    accepted_input_artifact: dict[str, Any],
    expected_compute_authorization: dict[str, Any],
) -> dict[str, Any]:
    """Validate one persisted document-and-query text feature product."""
    _validate_accepted_input_identity(
        pairs,
        input_manifest,
        accepted_input_artifact=accepted_input_artifact,
        queries=queries,
    )
    _validate_neural_feature_manifest(
        input_manifest,
        manifest,
        feature_kind="neural_text",
        expected_candidate_id=expected_candidate_id,
        expected_candidate=expected_candidate,
        expected_configuration=expected_configuration,
        expected_compute_authorization=expected_compute_authorization,
    )
    observed_hashes = {
        "features_sha256": dataframe_content_sha256(
            features, sort_columns=["candidate_id", "text_role", "text_sha256"]
        )
    }
    if observed_hashes != manifest.get("output_hashes"):
        raise ValueError("persisted text-feature output hash changed")
    required = {
        "candidate_id",
        "text_role",
        "text_sha256",
        "token_count",
        "embedding_dimension",
        "embedding",
        "embedding_sha256",
    }
    _require_complete_frame(features, required, name="text features")
    if features.duplicated(["text_role", "text_sha256"]).any():
        raise ValueError("persisted text features repeat a role and text hash")
    if set(features["candidate_id"].astype(str)) != {expected_candidate_id}:
        raise ValueError("persisted text features contain another candidate")
    if set(features["text_role"].astype(str)) != {"document", "query"}:
        raise ValueError("persisted text features have an unexpected text role")

    expected_documents = set(pairs["description_sha256"].astype(str))
    expected_queries = set(queries["canonical_query_text"].astype(str).map(sha256_text))
    documents = features.loc[features["text_role"].eq("document"), "text_sha256"]
    query_rows = features.loc[features["text_role"].eq("query"), "text_sha256"]
    if set(documents.astype(str)) != expected_documents:
        raise ValueError("persisted document features do not cover the exact E02b text set")
    if set(query_rows.astype(str)) != expected_queries:
        raise ValueError("persisted query features do not cover the exact E02b query-text set")
    if manifest.get("unique_documents") != len(expected_documents):
        raise ValueError("persisted text manifest document count changed")
    if manifest.get("unique_queries") != len(expected_queries):
        raise ValueError("persisted text manifest query count changed")

    dimensions = set(features["embedding_dimension"].astype(int))
    if len(dimensions) != 1:
        raise ValueError("persisted text embedding dimension is not constant")
    expected_dimension = dimensions.pop()
    _validate_normalized_vectors(features, expected_dimension=expected_dimension)
    token_counts = features["token_count"].to_numpy(dtype=np.int64)
    maximum_tokens = int(expected_candidate["max_tokens"])
    if np.any(token_counts < 1) or np.any(token_counts > maximum_tokens):
        raise ValueError("persisted text token count is outside the frozen model limit")
    if int(manifest.get("maximum_token_count", -1)) != int(token_counts.max()):
        raise ValueError("persisted text manifest maximum token count changed")
    return {
        "status": "passed_e02b_text_readback",
        "candidate_id": expected_candidate_id,
        "manifest_sha256": json_content_sha256(manifest),
        "output_hashes": observed_hashes,
        "unique_documents": int(len(expected_documents)),
        "unique_queries": int(len(expected_queries)),
        "embedding_dimension": expected_dimension,
        "maximum_token_count": int(token_counts.max()),
        "elapsed_seconds": float(manifest["elapsed_seconds"]),
        "extraction_gpu_hours": float(manifest["elapsed_seconds"]) / 3600.0,
        "validation_rankings_computed": False,
    }


def _validate_neural_feature_manifest(
    input_manifest: dict[str, Any],
    manifest: dict[str, Any],
    *,
    feature_kind: str,
    expected_candidate_id: str,
    expected_candidate: dict[str, Any],
    expected_configuration: dict[str, Any],
    expected_compute_authorization: dict[str, Any],
) -> None:
    if manifest.get("feature_kind") != feature_kind:
        raise ValueError(f"persisted {feature_kind} manifest has the wrong feature kind")
    if manifest.get("candidate_id") != expected_candidate_id:
        raise ValueError(f"persisted {feature_kind} candidate identifier changed")
    if manifest.get("candidate") != expected_candidate:
        raise ValueError(f"persisted {feature_kind} candidate recipe changed")
    if manifest.get("input_manifest_sha256") != json_content_sha256(input_manifest):
        raise ValueError(f"persisted {feature_kind} input manifest hash changed")
    if manifest.get("resolved_feature_configuration") != expected_configuration:
        raise ValueError(f"persisted {feature_kind} resolved configuration changed")
    decision = manifest.get("decision", {})
    if decision.get("status") != "frozen_features_complete":
        raise ValueError(f"persisted {feature_kind} feature product is not frozen as complete")
    if decision.get("validation_rankings_computed") is not False:
        raise ValueError(f"persisted {feature_kind} extraction read validation rankings")
    if decision.get("candidate_selected") is not False:
        raise ValueError(f"persisted {feature_kind} extraction selected a candidate")
    if decision.get("current_test_split_contaminated_before_e02b") is not True:
        raise ValueError(f"persisted {feature_kind} manifest lost the test-contamination record")
    git = manifest.get("git", {})
    commit = str(git.get("commit", ""))
    valid_commit = len(commit) == 40 and not set(commit).difference("0123456789abcdef")
    if (
        git.get("worktree_dirty") is not False
        or not valid_commit
        or git.get("changed_paths") != []
        or git.get("worktree_status_sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise ValueError(f"persisted {feature_kind} run did not use a clean Git commit")
    runtime_transformers = manifest.get("runtime", {}).get("packages", {}).get("transformers")
    if runtime_transformers != expected_candidate["transformers_version"]:
        raise ValueError(f"persisted {feature_kind} Transformers version changed")
    compute = manifest.get("compute_authorization")
    if compute != expected_compute_authorization:
        raise ValueError(f"persisted {feature_kind} compute authorization changed")
    elapsed = float(manifest.get("elapsed_seconds", np.nan))
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError(f"persisted {feature_kind} elapsed time must be finite and positive")
    if elapsed > float(compute["instance_hour_limit"]) * 3600.0:
        raise ValueError(f"persisted {feature_kind} elapsed time exceeds its authorized limit")


def _require_complete_frame(frame: pd.DataFrame, required: set[str], *, name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"persisted {name} are missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any(axis=None):
        raise ValueError(f"persisted {name} must be non-empty and complete")


def _validate_normalized_vectors(features: pd.DataFrame, *, expected_dimension: int) -> None:
    vectors = [np.asarray(value, dtype=np.float32) for value in features["embedding"]]
    if expected_dimension < 1 or any(vector.shape != (expected_dimension,) for vector in vectors):
        raise ValueError("persisted feature embedding shape changed")
    matrix = np.vstack(vectors)
    if not np.isfinite(matrix).all():
        raise ValueError("persisted feature contains a non-finite value")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("persisted features are not L2-normalized")
    observed_hashes = [fixed_representation.embedding_sha256(row) for row in matrix]
    if observed_hashes != features["embedding_sha256"].astype(str).tolist():
        raise ValueError("persisted per-row embedding hashes changed")


def _validate_dna_coverage(
    coverage: pd.DataFrame,
    features: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    expected_candidate_id: str,
    tokenizer_unit_bp: int,
    model_max_tokens: int,
    maximum_content_bp: int,
    overlap_fraction: float,
) -> None:
    required = {
        "candidate_id",
        "sequence_sha256",
        "sequence_id",
        "sequence_length_bp",
        "window_index",
        "start_bp",
        "input_base_count",
        "newly_covered_base_count",
        "wrapped_input_base_count",
        "input_token_count",
        "content_token_count",
        "special_token_count",
        "out_of_vocabulary_token_count",
    }
    _require_complete_frame(coverage, required, name="neural-DNA coverage")
    if set(coverage["candidate_id"].astype(str)) != {expected_candidate_id}:
        raise ValueError("persisted neural-DNA coverage contains another candidate")
    if set(coverage["sequence_sha256"].astype(str)) != set(features["sequence_sha256"].astype(str)):
        raise ValueError("persisted neural-DNA coverage changed the sequence set")
    if coverage.duplicated(["sequence_sha256", "window_index"]).any():
        raise ValueError("persisted neural-DNA coverage repeats a sequence window")
    numeric_columns = [
        "sequence_length_bp",
        "window_index",
        "start_bp",
        "input_base_count",
        "newly_covered_base_count",
        "wrapped_input_base_count",
        "input_token_count",
        "content_token_count",
        "special_token_count",
        "out_of_vocabulary_token_count",
    ]
    values = coverage[numeric_columns].to_numpy(dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("persisted neural-DNA coverage contains a negative count")
    if np.any(coverage["newly_covered_base_count"].to_numpy(dtype=np.int64) < 1):
        raise ValueError("persisted neural-DNA coverage contains an empty coverage weight")
    if not coverage["start_bp"].astype(int).lt(coverage["sequence_length_bp"].astype(int)).all():
        raise ValueError("persisted neural-DNA window starts outside its sequence")
    if (
        not coverage["newly_covered_base_count"]
        .astype(int)
        .le(coverage["input_base_count"].astype(int))
        .all()
    ):
        raise ValueError("persisted neural-DNA coverage weight exceeds its input window")
    if (
        not coverage["wrapped_input_base_count"]
        .astype(int)
        .le(coverage["input_base_count"].astype(int))
        .all()
    ):
        raise ValueError("persisted neural-DNA wrapped-base count exceeds its input window")
    if (
        not coverage["input_token_count"]
        .astype(int)
        .eq(
            coverage["content_token_count"].astype(int)
            + coverage["special_token_count"].astype(int)
        )
        .all()
    ):
        raise ValueError("persisted neural-DNA input-token accounting changed")
    if (
        not (coverage["content_token_count"].astype(int) * tokenizer_unit_bp)
        .eq(coverage["input_base_count"].astype(int))
        .all()
    ):
        raise ValueError("persisted neural-DNA content-token base accounting changed")
    if not coverage["out_of_vocabulary_token_count"].astype(int).eq(0).all():
        raise ValueError("persisted neural-DNA coverage contains an out-of-vocabulary token")
    if not coverage["input_token_count"].astype(int).le(model_max_tokens).all():
        raise ValueError("persisted neural-DNA input token count exceeds the model limit")

    grouped = coverage.groupby("sequence_sha256", sort=True)
    summary = grouped.agg(
        newly_covered_base_count=("newly_covered_base_count", "sum"),
        sequence_length_bp=("sequence_length_bp", "first"),
        sequence_length_values=("sequence_length_bp", "nunique"),
        sequence_id=("sequence_id", "first"),
        sequence_id_values=("sequence_id", "nunique"),
    )
    if not summary["newly_covered_base_count"].eq(summary["sequence_length_bp"]).all():
        raise ValueError("persisted neural-DNA coverage does not cover each base exactly once")
    if (
        not summary["sequence_length_values"].eq(1).all()
        or not summary["sequence_id_values"].eq(1).all()
    ):
        raise ValueError("persisted neural-DNA coverage changes identity within a sequence")
    feature_identity = features.set_index("sequence_sha256")[
        ["representative_sequence_id", "length_bp"]
    ].sort_index()
    if (
        not summary["sequence_id"]
        .astype(str)
        .equals(feature_identity["representative_sequence_id"].astype(str))
    ):
        raise ValueError("persisted neural-DNA coverage sequence identifier changed")
    if (
        not summary["sequence_length_bp"]
        .astype(int)
        .equals(feature_identity["length_bp"].astype(int))
    ):
        raise ValueError("persisted neural-DNA coverage sequence length changed")
    source_sequences = (
        pairs.loc[:, ["sequence_sha256", "sequence"]]
        .drop_duplicates("sequence_sha256")
        .set_index("sequence_sha256")["sequence"]
    )
    window_columns = [
        "window_index",
        "start_bp",
        "input_base_count",
        "newly_covered_base_count",
        "wrapped_input_base_count",
    ]
    for sequence_hash, group in grouped:
        sequence = str(source_sequences.at[sequence_hash])
        plan = fixed_representation.circular_window_plan(
            len(sequence),
            maximum_content_bp=maximum_content_bp,
            tokenizer_unit_bp=tokenizer_unit_bp,
            overlap_fraction=overlap_fraction,
        )
        expected_windows = np.asarray(
            [
                [
                    window.index,
                    window.start_bp,
                    window.input_base_count,
                    window.newly_covered_base_count,
                    window.wrapped_input_base_count,
                ]
                for window in plan
            ],
            dtype=np.int64,
        )
        observed_windows = group.sort_values("window_index", kind="stable")[
            window_columns
        ].to_numpy(dtype=np.int64)
        if not np.array_equal(observed_windows, expected_windows):
            raise ValueError(
                f"persisted neural-DNA circular-window plan changed for {sequence_hash}"
            )


def _validate_accepted_input_identity(
    pairs: pd.DataFrame,
    input_manifest: dict[str, Any],
    *,
    accepted_input_artifact: dict[str, Any],
    queries: pd.DataFrame | None = None,
) -> None:
    observed_manifest_hash = json_content_sha256(input_manifest)
    if observed_manifest_hash != accepted_input_artifact.get("manifest_sha256"):
        raise ValueError("persisted feature input manifest differs from the accepted E02b input")
    observed_pairs_hash = dataframe_content_sha256(
        pairs, sort_columns=["panel_role", "sequence_id"]
    )
    if observed_pairs_hash != accepted_input_artifact.get("pairs_sha256"):
        raise ValueError("persisted feature pairs differ from the accepted E02b input")
    output_hashes = input_manifest.get("output_hashes", {})
    if observed_pairs_hash != output_hashes.get("pairs_sha256"):
        raise ValueError("persisted feature pairs differ from their E02b input manifest")
    if queries is not None:
        observed_queries_hash = dataframe_content_sha256(queries, sort_columns=["query_id"])
        if observed_queries_hash != accepted_input_artifact.get("queries_sha256"):
            raise ValueError("persisted feature queries differ from the accepted E02b input")
        if observed_queries_hash != output_hashes.get("queries_sha256"):
            raise ValueError("persisted feature queries differ from their E02b input manifest")
