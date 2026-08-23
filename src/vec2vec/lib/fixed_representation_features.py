"""Frozen feature extraction for the reduced-population Gate 1 bake-off."""

from __future__ import annotations

import gc
import time
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import fixed_representation
from vec2vec.lib.dna_encoder import EncoderRecipe, FrozenDnaEncoder
from vec2vec.lib.fixed_representation_invariance import json_content_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.lib.text import sha256_text
from vec2vec.lib.text_encoder import FrozenTextEncoder, TextEncoderRecipe


def extract_neural_dna_features(
    pairs: pd.DataFrame,
    input_manifest: dict[str, Any],
    invariance_manifest: dict[str, Any],
    params: dict[str, Any],
    candidate_id: str,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Encode each unique eligible sequence once with one accepted DNA candidate."""
    _validate_input_artifact(pairs, input_manifest, params)
    recipes = dict(params["dna_candidates"])
    if candidate_id not in recipes:
        raise ValueError(f"unknown DNA candidate: {candidate_id}")
    recipe = EncoderRecipe.model_validate(recipes[candidate_id])
    _validate_invariance_manifest(invariance_manifest, params, candidate_id, recipe)
    unique = _unique_content(
        pairs,
        hash_column="sequence_sha256",
        value_column="sequence",
        id_column="sequence_id",
    )
    invalid = unique["sequence"].map(lambda value: sorted(set(str(value)).difference("ACGT")))
    if invalid.map(bool).any():
        raise ValueError("DNA feature input contains a sequence outside the E02b A/C/G/T rule")

    encoder = FrozenDnaEncoder(
        recipe,
        precision=str(params["precision"]),
        device=str(params["device"]),
        overlap_fraction=float(params["window_overlap_fraction"]),
    )
    rows: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        _ensure_before_deadline(deadline_monotonic, operation="DNA model loading")
        encoder.load()
        _ensure_before_deadline(deadline_monotonic, operation="DNA feature extraction")
        encoder.reset_peak_device_memory()
        for row in unique.itertuples(index=False):
            _ensure_before_deadline(deadline_monotonic, operation=f"DNA feature {row.sequence_id}")
            result = encoder.encode_sequence(str(row.sequence_id), str(row.sequence))
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "sequence_sha256": str(row.sequence_sha256),
                    "representative_sequence_id": str(row.sequence_id),
                    "length_bp": len(str(row.sequence)),
                    "embedding_dimension": int(len(result.vector)),
                    "embedding": result.vector.tolist(),
                    "embedding_sha256": fixed_representation.embedding_sha256(result.vector),
                    "elapsed_seconds": float(result.elapsed_seconds),
                }
            )
            coverage.extend(
                {
                    **record,
                    "candidate_id": candidate_id,
                    "sequence_sha256": str(row.sequence_sha256),
                    "sequence_length_bp": len(str(row.sequence)),
                }
                for record in result.coverage
            )
        peak_memory = encoder.peak_device_memory_bytes()
        maximum_content_bp = encoder.maximum_content_bp
    finally:
        encoder.close()
        del encoder
        gc.collect()
    features = pd.DataFrame(rows).sort_values("sequence_sha256", kind="stable", ignore_index=True)
    coverage_frame = pd.DataFrame(coverage).sort_values(
        ["sequence_sha256", "window_index"], kind="stable", ignore_index=True
    )
    elapsed = time.perf_counter() - started
    hashes = {
        "features_sha256": dataframe_content_sha256(
            features, sort_columns=["candidate_id", "sequence_sha256"]
        ),
        "coverage_sha256": dataframe_content_sha256(
            coverage_frame,
            sort_columns=["candidate_id", "sequence_sha256", "window_index"],
        ),
    }
    summary = {
        "feature_kind": "neural_dna",
        "candidate_id": candidate_id,
        "candidate": recipe.model_dump(mode="json"),
        "input_manifest_sha256": json_content_sha256(input_manifest),
        "accepted_invariance_manifest_sha256": json_content_sha256(invariance_manifest),
        "unique_sequences": int(len(features)),
        "source_rows": int(len(pairs)),
        "maximum_content_bp": int(maximum_content_bp or 0),
        "elapsed_seconds": elapsed,
        "base_pairs_per_second": float(unique["sequence"].str.len().sum() / elapsed),
        "peak_device_memory_bytes": peak_memory,
        "output_hashes": hashes,
    }
    return features, coverage_frame, summary


def extract_text_features(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    input_manifest: dict[str, Any],
    params: dict[str, Any],
    candidate_id: str,
    *,
    deadline_monotonic: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Encode each unique paired description and query-role text once."""
    _validate_input_artifact(pairs, input_manifest, params)
    _validate_query_artifact(queries, input_manifest)
    recipes = dict(params["text_candidates"])
    if candidate_id not in recipes:
        raise ValueError(f"unknown text candidate: {candidate_id}")
    recipe = TextEncoderRecipe.model_validate(recipes[candidate_id])
    documents = _unique_content(
        pairs,
        hash_column="description_sha256",
        value_column="description",
        id_column="sequence_id",
    ).rename(columns={"description_sha256": "text_sha256", "description": "text"})
    query_texts = queries.loc[:, ["query_id", "canonical_query_text"]].copy()
    query_texts["text_sha256"] = query_texts["canonical_query_text"].astype(str).map(sha256_text)
    query_texts = _unique_content(
        query_texts,
        hash_column="text_sha256",
        value_column="canonical_query_text",
        id_column="query_id",
    ).rename(columns={"canonical_query_text": "text"})

    encoder = FrozenTextEncoder(
        recipe, precision=str(params["precision"]), device=str(params["device"])
    )
    started = time.perf_counter()
    try:
        document_result = encoder.encode(
            documents["text"].astype(str).tolist(),
            role="document",
            deadline_monotonic=deadline_monotonic,
        )
        query_result = encoder.encode(
            query_texts["text"].astype(str).tolist(),
            role="query",
            deadline_monotonic=deadline_monotonic,
        )
        peak_memory = encoder.peak_device_memory_bytes()
    finally:
        encoder.close()
        del encoder
        gc.collect()
    frames = []
    for role, metadata, result in (
        ("document", documents, document_result),
        ("query", query_texts, query_result),
    ):
        frame = metadata.loc[:, ["text_sha256"]].copy()
        frame.insert(0, "candidate_id", candidate_id)
        frame.insert(1, "text_role", role)
        frame["token_count"] = result.token_counts
        frame["embedding_dimension"] = result.vectors.shape[1]
        frame["embedding"] = [row.tolist() for row in result.vectors]
        frame["embedding_sha256"] = [
            fixed_representation.embedding_sha256(row) for row in result.vectors
        ]
        frames.append(frame)
    features = pd.concat(frames, ignore_index=True).sort_values(
        ["text_role", "text_sha256"], kind="stable", ignore_index=True
    )
    elapsed = time.perf_counter() - started
    summary = {
        "feature_kind": "neural_text",
        "candidate_id": candidate_id,
        "candidate": recipe.model_dump(mode="json"),
        "input_manifest_sha256": json_content_sha256(input_manifest),
        "unique_documents": int(len(documents)),
        "unique_queries": int(len(query_texts)),
        "maximum_token_count": int(features["token_count"].max()),
        "elapsed_seconds": elapsed,
        "texts_per_second": float(len(features) / elapsed),
        "peak_device_memory_bytes": peak_memory,
        "output_hashes": {
            "features_sha256": dataframe_content_sha256(
                features, sort_columns=["candidate_id", "text_role", "text_sha256"]
            )
        },
    }
    return features, summary


def fit_tfidf_dna_features(
    pairs: pd.DataFrame,
    input_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit the frozen train-only 6-mer TF-IDF/SVD representation and transform all rows."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    _validate_input_artifact(pairs, input_manifest, params)
    recipe = dict(params["tfidf"])
    if recipe["analyzer"] != "char" or int(recipe["ngram_size"]) != 6:
        raise ValueError("E02b TF-IDF requires exact character 6-mers")
    train = pairs.loc[pairs["panel_role"].eq("alignment_train")]
    if len(train) != int(params["training_rows"]):
        raise ValueError("TF-IDF training panel row count changed")
    unique = _unique_content(
        pairs,
        hash_column="sequence_sha256",
        value_column="sequence",
        id_column="sequence_id",
    )
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(6, 6),
        lowercase=bool(recipe["lowercase"]),
        norm=str(recipe["norm"]),
        use_idf=bool(recipe["use_idf"]),
        smooth_idf=bool(recipe["smooth_idf"]),
        sublinear_tf=bool(recipe["sublinear_tf"]),
        dtype=np.float32,
    )
    started = time.perf_counter()
    training_matrix = vectorizer.fit_transform(train["sequence"].astype(str))
    components = int(recipe["svd_components"])
    if components >= min(training_matrix.shape):
        raise ValueError(
            f"TF-IDF matrix shape {training_matrix.shape} cannot fit {components} SVD components"
        )
    svd = TruncatedSVD(
        n_components=components,
        algorithm=str(recipe["svd_algorithm"]),
        n_iter=int(recipe["svd_iterations"]),
        random_state=int(recipe["seed"]),
    )
    svd.fit(training_matrix)
    matrix = normalize(svd.transform(vectorizer.transform(unique["sequence"].astype(str))))
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("TF-IDF/SVD produced a non-finite feature")
    candidate_id = str(recipe["candidate_id"])
    features = unique.loc[:, ["sequence_sha256", "sequence_id"]].rename(
        columns={"sequence_id": "representative_sequence_id"}
    )
    features.insert(0, "candidate_id", candidate_id)
    features["length_bp"] = unique["sequence"].str.len().to_numpy()
    features["embedding_dimension"] = matrix.shape[1]
    features["embedding"] = [row.tolist() for row in matrix]
    features["embedding_sha256"] = [fixed_representation.embedding_sha256(row) for row in matrix]
    vocabulary = pd.DataFrame(
        {
            "term": list(vectorizer.get_feature_names_out()),
            "term_index": np.arange(len(vectorizer.get_feature_names_out()), dtype=np.int64),
            "idf": vectorizer.idf_.astype(np.float64),
        }
    )
    svd_state = pd.DataFrame(
        {
            "component": np.arange(components, dtype=np.int64),
            "singular_value": svd.singular_values_.astype(np.float64),
            "explained_variance": svd.explained_variance_.astype(np.float64),
            "explained_variance_ratio": svd.explained_variance_ratio_.astype(np.float64),
            "vector": [row.astype(np.float32).tolist() for row in svd.components_],
        }
    )
    hashes = {
        "features_sha256": dataframe_content_sha256(
            features, sort_columns=["candidate_id", "sequence_sha256"]
        ),
        "vocabulary_sha256": dataframe_content_sha256(vocabulary, sort_columns=["term_index"]),
        "svd_state_sha256": dataframe_content_sha256(svd_state, sort_columns=["component"]),
    }
    summary = {
        "feature_kind": "tfidf_dna",
        "candidate_id": candidate_id,
        "recipe": recipe,
        "input_manifest_sha256": json_content_sha256(input_manifest),
        "training_rows": int(len(train)),
        "unique_sequences": int(len(unique)),
        "vocabulary_terms": int(len(vocabulary)),
        "elapsed_seconds": time.perf_counter() - started,
        "output_hashes": hashes,
    }
    return features, vocabulary, svd_state, summary


def _validate_input_artifact(
    pairs: pd.DataFrame,
    manifest: dict[str, Any],
    params: dict[str, Any],
) -> None:
    accepted = params.get("accepted_input_artifact")
    if not isinstance(accepted, dict):
        raise ValueError("accepted_input_artifact must be frozen before feature extraction")
    observed_manifest_hash = json_content_sha256(manifest)
    if observed_manifest_hash != accepted.get("manifest_sha256"):
        raise ValueError("E02b input manifest hash changed")
    observed_pairs_hash = dataframe_content_sha256(
        pairs, sort_columns=["panel_role", "sequence_id"]
    )
    if observed_pairs_hash != accepted.get("pairs_sha256"):
        raise ValueError("E02b pairs hash changed")
    if manifest.get("output_hashes", {}).get("pairs_sha256") != observed_pairs_hash:
        raise ValueError("E02b input manifest does not describe the loaded pairs")


def _validate_query_artifact(
    queries: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    """Bind text-query extraction to the exact accepted E02b query table."""
    if "query_id" not in queries.columns or queries.empty:
        raise ValueError("E02b query artifact must be non-empty and contain query_id")
    observed_hash = dataframe_content_sha256(queries, sort_columns=["query_id"])
    expected_hash = manifest.get("output_hashes", {}).get("queries_sha256")
    if observed_hash != expected_hash:
        raise ValueError("E02b query table hash changed before text extraction")


def _validate_invariance_manifest(
    manifest: dict[str, Any],
    params: dict[str, Any],
    candidate_id: str,
    recipe: EncoderRecipe,
) -> None:
    accepted = dict(params["accepted_invariance_artifacts"]).get(candidate_id)
    if not isinstance(accepted, dict):
        raise ValueError(f"candidate {candidate_id} has no accepted invariance artifact")
    if json_content_sha256(manifest) != accepted.get("manifest_sha256"):
        raise ValueError(f"candidate {candidate_id} invariance manifest hash changed")
    if manifest.get("candidate_id") != candidate_id:
        raise ValueError("invariance manifest candidate changed")
    if manifest.get("candidate") != recipe.model_dump(mode="json"):
        raise ValueError("invariance manifest recipe changed")
    if manifest.get("decision", {}).get("status") != "passed_invariance_check":
        raise ValueError("DNA candidate did not pass the frozen invariance gate")


def _unique_content(
    frame: pd.DataFrame,
    *,
    hash_column: str,
    value_column: str,
    id_column: str,
) -> pd.DataFrame:
    required = {hash_column, value_column, id_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feature input is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any(axis=None):
        raise ValueError("feature input contains missing content identities")
    inconsistent = frame.groupby(hash_column)[value_column].nunique(dropna=False).gt(1)
    if inconsistent.any():
        raise ValueError(f"{int(inconsistent.sum())} hashes map to multiple source values")
    return (
        frame.sort_values([hash_column, id_column], kind="stable")
        .drop_duplicates(hash_column, keep="first")
        .loc[:, [hash_column, value_column, id_column]]
        .reset_index(drop=True)
    )


def _ensure_before_deadline(deadline_monotonic: float | None, *, operation: str) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError(f"authorized compute deadline reached before {operation}")
