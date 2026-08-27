"""Final frozen vec2vec fit and deployable retrieval bundle."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import alignment_probe
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.lib.text_encoder import FrozenTextEncoder, TextEncoderRecipe

EXPECTED_SOURCE_ROWS = 115_120
EXPECTED_ELIGIBLE_ROWS = 110_267
EXPECTED_ATOMIC_QUERIES = 28
MINIMUM_VERIFIED_ROWS = 20
EXPECTED_SOURCE_FILE_SHA256 = "eaf4ef6885aded6e984f974c71f1c32ffb08b74cf1cf96aa69af8d6f3993f855"
EXPECTED_STATES_SHA256 = "571b45e807e21f74699a5400faf7b20678df7d9595de3ef591a981e2a34d3208"
EXPECTED_QUERIES_SHA256 = "a440e26a32468a9e613aaa2034b476b453a8bae988c08128df3f14f251c4552c"

FINAL_RECIPE: dict[str, Any] = {
    "protocol_version": "final-model-v1",
    "eligible_sequence_alphabet": "ACGT",
    "objective": "verified_set",
    "seed": 20260818,
    "projection_dimension": 512,
    "whitening_epsilon": 1e-6,
    "updates": 300,
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "initial_temperature": 0.07,
    "maximum_logit_scale": 100.0,
    "tfidf": {
        "analyzer": "char",
        "ngram_size": 6,
        "lowercase": False,
        "norm": "l2",
        "use_idf": True,
        "smooth_idf": True,
        "sublinear_tf": False,
        "svd_components": 512,
        "svd_algorithm": "randomized",
        "svd_iterations": 7,
        "seed": 20260818,
    },
    "text": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        "transformers_version": "5.12.1",
        "trust_remote_code": False,
        "max_tokens": 32768,
        "pooling": "last_token",
        "document_prefix": "",
        "query_prefix": (
            "Instruct: Given a plasmid constraint query, retrieve plasmid descriptions that "
            "satisfy the recorded constraint\nQuery:"
        ),
        "normalize": True,
        "attention_implementation": "sdpa",
        "batch_size": 32,
    },
}


def file_sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_final_inputs(
    source_path: Path,
    states_path: Path,
    queries_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Load and validate the full population and accepted atomic supervision."""
    observed_source_hash = file_sha256(source_path)
    if observed_source_hash != EXPECTED_SOURCE_FILE_SHA256:
        raise ValueError(
            "structured source file changed: "
            f"expected {EXPECTED_SOURCE_FILE_SHA256}, observed {observed_source_hash}"
        )
    columns = [
        "uuid",
        "sequence_id",
        "sequence",
        "sequence_sha256",
        "name",
        "description",
        "length_bp",
        "split_grouped",
    ]
    plasmids = pd.read_parquet(source_path, columns=columns)
    if len(plasmids) != EXPECTED_SOURCE_ROWS:
        raise ValueError(f"expected {EXPECTED_SOURCE_ROWS} source rows, observed {len(plasmids)}")
    if plasmids[columns].isna().any(axis=None):
        raise ValueError("structured source contains missing final-model fields")
    if plasmids["sequence_id"].duplicated().any():
        raise ValueError("structured source repeats sequence_id")
    observed_sequence_hashes = plasmids["sequence"].astype(str).map(sequence_sha256)
    if not observed_sequence_hashes.equals(plasmids["sequence_sha256"].astype(str)):
        raise ValueError("structured source sequence hashes do not match sequence content")

    queries = pd.read_parquet(queries_path)
    observed_query_hash = dataframe_content_sha256(queries, sort_columns=["query_id"])
    if observed_query_hash != EXPECTED_QUERIES_SHA256:
        raise ValueError(
            f"accepted query table changed: expected {EXPECTED_QUERIES_SHA256}, "
            f"observed {observed_query_hash}"
        )
    atomic = queries.loc[queries["query_kind"].eq("atomic")].copy()
    if len(atomic) != EXPECTED_ATOMIC_QUERIES:
        raise ValueError(
            f"expected {EXPECTED_ATOMIC_QUERIES} atomic queries, observed {len(atomic)}"
        )
    atomic["constraint_id"] = atomic["constraint_ids_json"].map(_single_constraint_id)
    if atomic["constraint_id"].duplicated().any():
        raise ValueError("atomic queries repeat a constraint")
    atomic = atomic.sort_values("query_id", kind="stable", ignore_index=True)

    states = pd.read_parquet(states_path)
    observed_states_hash = dataframe_content_sha256(
        states, sort_columns=["sequence_id", "constraint_id", "state"]
    )
    if observed_states_hash != EXPECTED_STATES_SHA256:
        raise ValueError(
            f"accepted constraint states changed: expected {EXPECTED_STATES_SHA256}, "
            f"observed {observed_states_hash}"
        )
    state_identities = states.loc[:, ["sequence_id", "sequence_sha256"]].drop_duplicates()
    joined = plasmids.loc[:, ["sequence_id", "sequence_sha256"]].merge(
        state_identities,
        on="sequence_id",
        how="outer",
        suffixes=("_source", "_state"),
        indicator=True,
        validate="one_to_one",
    )
    if set(joined["_merge"].astype(str)) != {"both"}:
        raise ValueError("constraint states and HF source do not contain the same sequence IDs")
    if joined["sequence_sha256_source"].ne(joined["sequence_sha256_state"]).any():
        raise ValueError("constraint states and HF source disagree on sequence identity")

    eligible = plasmids["sequence"].str.fullmatch("[ACGT]+", na=False)
    training = plasmids.loc[eligible].sort_values("sequence_id", kind="stable", ignore_index=True)
    if len(training) != EXPECTED_ELIGIBLE_ROWS:
        raise ValueError(
            f"expected {EXPECTED_ELIGIBLE_ROWS} A/C/G/T rows, observed {len(training)}"
        )
    verified_mask = _verified_mask(training, atomic, states)
    support = verified_mask.sum(axis=1)
    if np.any(support < MINIMUM_VERIFIED_ROWS):
        failed = (
            atomic.loc[support < MINIMUM_VERIFIED_ROWS, "semantic_query_id"].astype(str).tolist()
        )
        raise ValueError(
            f"atomic queries below {MINIMUM_VERIFIED_ROWS} final-fit positives: {failed}"
        )
    audit = {
        "source_rows": int(len(plasmids)),
        "eligible_rows": int(len(training)),
        "excluded_non_acgt_rows": int((~eligible).sum()),
        "source_rows_by_historical_split": {
            str(key): int(value)
            for key, value in plasmids["split_grouped"].value_counts().sort_index().items()
        },
        "eligible_rows_by_historical_split": {
            str(key): int(value)
            for key, value in training["split_grouped"].value_counts().sort_index().items()
        },
        "unique_sequence_hashes": int(training["sequence_sha256"].nunique()),
        "atomic_queries": int(len(atomic)),
        "minimum_verified_rows": int(support.min()),
        "maximum_verified_rows": int(support.max()),
        "source_file_sha256": observed_source_hash,
        "constraint_states_sha256": observed_states_hash,
        "queries_sha256": observed_query_hash,
    }
    return training, atomic, verified_mask, audit


def fit_final_model(
    training: pd.DataFrame,
    atomic_queries: pd.DataFrame,
    verified_mask: np.ndarray,
    *,
    device: str,
    deadline_monotonic: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, pd.DataFrame, dict[str, float]]:
    """Fit the selected frozen encoders and verified-set projection heads."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    recipe = FINAL_RECIPE
    tfidf = dict(recipe["tfidf"])
    timings: dict[str, float] = {}
    started = time.perf_counter()
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(int(tfidf["ngram_size"]), int(tfidf["ngram_size"])),
        lowercase=bool(tfidf["lowercase"]),
        norm=str(tfidf["norm"]),
        use_idf=bool(tfidf["use_idf"]),
        smooth_idf=bool(tfidf["smooth_idf"]),
        sublinear_tf=bool(tfidf["sublinear_tf"]),
        dtype=np.float32,
    )
    sparse = vectorizer.fit_transform(training["sequence"].astype(str))
    if sparse.shape[1] != 4096:
        raise ValueError(f"final A/C/G/T 6-mer vocabulary has {sparse.shape[1]} terms, not 4096")
    svd = TruncatedSVD(
        n_components=int(tfidf["svd_components"]),
        algorithm=str(tfidf["svd_algorithm"]),
        n_iter=int(tfidf["svd_iterations"]),
        random_state=int(tfidf["seed"]),
    )
    sequence_features = np.asarray(normalize(svd.fit_transform(sparse)), dtype=np.float32)
    del sparse
    _ensure_before_deadline(deadline_monotonic, "DNA feature extraction")
    timings["dna_features_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    text_recipe = TextEncoderRecipe.model_validate(recipe["text"])
    encoder = FrozenTextEncoder(text_recipe, precision="bfloat16", device=device)
    try:
        documents = encoder.encode(
            training["description"].astype(str).tolist(),
            role="document",
            deadline_monotonic=deadline_monotonic,
        ).vectors
        query_features = encoder.encode(
            atomic_queries["canonical_query_text"].astype(str).tolist(),
            role="query",
            deadline_monotonic=deadline_monotonic,
        ).vectors
    finally:
        encoder.close()
    timings["text_features_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    epsilon = float(recipe["whitening_epsilon"])
    dna_whitening = alignment_probe.Whitening.fit(sequence_features, epsilon=epsilon)
    text_whitening = alignment_probe.Whitening.fit(documents, epsilon=epsilon)
    whitened_sequences = dna_whitening.transform(sequence_features)
    whitened_queries = text_whitening.transform(query_features)
    del documents, sequence_features, query_features
    timings["whitening_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    state, history = alignment_probe.train_controlled_query_probe(
        whitened_sequences,
        whitened_queries,
        verified_mask,
        objective=str(recipe["objective"]),
        seed=int(recipe["seed"]),
        projection_dimension=int(recipe["projection_dimension"]),
        updates=int(recipe["updates"]),
        learning_rate=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
        initial_temperature=float(recipe["initial_temperature"]),
        maximum_logit_scale=float(recipe["maximum_logit_scale"]),
        device=device,
        deadline_monotonic=deadline_monotonic,
    )
    index = alignment_probe.project(whitened_sequences, state["sequence_head"])
    timings["probe_and_index_seconds"] = time.perf_counter() - started
    model = {
        "terms": np.asarray(vectorizer.get_feature_names_out(), dtype=np.str_),
        "idf": np.asarray(vectorizer.idf_, dtype=np.float64),
        "svd_components": np.asarray(svd.components_, dtype=np.float32),
        "dna_whitening_mean": dna_whitening.mean,
        "dna_whitening_matrix": dna_whitening.matrix,
        "text_whitening_mean": text_whitening.mean,
        "text_whitening_matrix": text_whitening.matrix,
        "sequence_head": np.asarray(state["sequence_head"], dtype=np.float32),
        "text_head": np.asarray(state["text_head"], dtype=np.float32),
        "logit_scale": np.asarray([state["logit_scale"]], dtype=np.float32),
    }
    return model, index, history, timings


def save_bundle(
    output_dir: Path,
    model: dict[str, np.ndarray],
    index: np.ndarray,
    training: pd.DataFrame,
    atomic_queries: pd.DataFrame,
    history: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Write the portable model state, memory-mappable index, and identities."""
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "model.npz", **model)
    np.save(output_dir / "sequence_index.npy", np.asarray(index, dtype=np.float32))
    training.loc[:, ["uuid", "sequence_id", "sequence_sha256", "name", "length_bp"]].reset_index(
        names="index_row"
    ).to_parquet(output_dir / "sequence_ids.parquet", index=False)
    atomic_queries.loc[
        :, ["query_id", "semantic_query_id", "canonical_query_text", "constraint_id"]
    ].to_parquet(output_dir / "training_queries.parquet", index=False)
    history.to_parquet(output_dir / "training_history.parquet", index=False)
    files = {}
    for path in sorted(output_dir.iterdir()):
        files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    complete = {**manifest, "files": files}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(complete, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    validate_bundle(output_dir)
    return complete


def validate_bundle(output_dir: Path) -> dict[str, Any]:
    """Reload and validate every persisted bundle product."""
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = output_dir / name
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"bundle file is missing or changed: {name}")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"bundle file hash changed: {name}")
    with np.load(output_dir / "model.npz", allow_pickle=False) as model:
        required = {
            "terms",
            "idf",
            "svd_components",
            "dna_whitening_mean",
            "dna_whitening_matrix",
            "text_whitening_mean",
            "text_whitening_matrix",
            "sequence_head",
            "text_head",
            "logit_scale",
        }
        if set(model.files) != required:
            raise ValueError("model.npz fields changed")
        if model["svd_components"].shape != (512, 4096):
            raise ValueError("SVD state shape changed")
        if model["sequence_head"].shape != (512, 512):
            raise ValueError("sequence head shape changed")
        if model["text_head"].shape[0] != 512:
            raise ValueError("text head shape changed")
        if any(not np.isfinite(model[name]).all() for name in required - {"terms"}):
            raise ValueError("model state contains non-finite values")
    index = np.load(output_dir / "sequence_index.npy", mmap_mode="r", allow_pickle=False)
    identities = pd.read_parquet(output_dir / "sequence_ids.parquet")
    if index.shape != (EXPECTED_ELIGIBLE_ROWS, 512) or len(identities) != len(index):
        raise ValueError("sequence index shape or identity count changed")
    norms = np.linalg.norm(np.asarray(index[::1000]), axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("sequence index is not L2 normalized")
    return manifest


def project_sequences(bundle_path: Path, sequences: list[str]) -> np.ndarray:
    """Transform raw A/C/G/T sequences with a persisted final model."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    if not sequences or any(
        len(sequence) < 6 or set(sequence) - set("ACGT") for sequence in sequences
    ):
        raise ValueError("final model accepts uppercase A/C/G/T sequences at least 6 bp long")
    with np.load(bundle_path, allow_pickle=False) as model:
        terms = model["terms"].astype(str).tolist()
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(6, 6),
            lowercase=False,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            vocabulary={term: index for index, term in enumerate(terms)},
            dtype=np.float32,
        )
        vectorizer.idf_ = model["idf"]
        tfidf = vectorizer.transform(sequences)
        features = np.asarray(normalize(tfidf @ model["svd_components"].T), dtype=np.float32)
        whitened = (features - model["dna_whitening_mean"]) @ model["dna_whitening_matrix"]
        return alignment_probe.project(whitened, model["sequence_head"])


def project_query_embeddings(bundle_path: Path, embeddings: np.ndarray) -> np.ndarray:
    """Project raw frozen-Qwen query embeddings with the persisted final state."""
    with np.load(bundle_path, allow_pickle=False) as model:
        values = np.asarray(embeddings, dtype=np.float32)
        expected_dimension = int(model["text_whitening_mean"].shape[0])
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != expected_dimension
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                "query embeddings must be a non-empty finite matrix with "
                f"{expected_dimension} columns"
            )
        whitened = (values - model["text_whitening_mean"]) @ model["text_whitening_matrix"]
        return alignment_probe.project(whitened, model["text_head"])


def encode_queries(
    bundle_dir: Path,
    queries: list[str],
    *,
    device: str,
) -> tuple[np.ndarray, list[int]]:
    """Encode text queries with the model recipe stored in a validated bundle."""
    manifest = validate_bundle(bundle_dir)
    if manifest.get("protocol_version") != FINAL_RECIPE["protocol_version"]:
        raise ValueError("bundle is not an accepted final-model-v1 artifact")
    if manifest.get("recipe") != FINAL_RECIPE:
        raise ValueError("bundle recipe does not match the accepted final recipe")
    text_recipe = TextEncoderRecipe.model_validate(FINAL_RECIPE["text"])
    precision = "bfloat16" if str(device).startswith("cuda") else "float32"
    encoder = FrozenTextEncoder(text_recipe, precision=precision, device=device)
    try:
        encoded = encoder.encode(queries, role="query")
    finally:
        encoder.close()
    vectors = project_query_embeddings(bundle_dir / "model.npz", encoded.vectors)
    return vectors, encoded.token_counts


def retrieve(bundle_dir: Path, query_vectors: np.ndarray, *, top_k: int) -> pd.DataFrame:
    """Return deterministic cosine-ranked plasmid identities for projected queries."""
    vectors = np.asarray(query_vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] != 512:
        raise ValueError("query vectors must be a non-empty matrix with 512 columns")
    if not np.isfinite(vectors).all():
        raise ValueError("query vectors contain non-finite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("query vectors must be L2 normalized")

    index = np.load(bundle_dir / "sequence_index.npy", mmap_mode="r", allow_pickle=False)
    identities = pd.read_parquet(bundle_dir / "sequence_ids.parquet")
    if not 1 <= top_k <= len(index):
        raise ValueError(f"top_k must be between 1 and {len(index)}")
    if len(identities) != len(index) or not np.array_equal(
        identities["index_row"].to_numpy(), np.arange(len(index))
    ):
        raise ValueError("sequence identities do not match index rows")

    rows = []
    index_rows = np.arange(len(index))
    for query_index, vector in enumerate(vectors):
        scores = np.asarray(index @ vector, dtype=np.float32)
        order = np.lexsort((index_rows, -scores))[:top_k]
        selected = identities.iloc[order].copy()
        selected.insert(0, "score", scores[order])
        selected.insert(0, "rank", np.arange(1, top_k + 1))
        selected.insert(0, "query_index", query_index)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True)


def _single_constraint_id(value: str) -> str:
    identifiers = json.loads(str(value))
    if not isinstance(identifiers, list) or len(identifiers) != 1:
        raise ValueError("atomic query must contain exactly one constraint ID")
    return str(identifiers[0])


def _verified_mask(
    training: pd.DataFrame,
    atomic_queries: pd.DataFrame,
    states: pd.DataFrame,
) -> np.ndarray:
    sequence_index = {
        value: index for index, value in enumerate(training["sequence_id"].astype(str))
    }
    query_index = {
        value: index for index, value in enumerate(atomic_queries["constraint_id"].astype(str))
    }
    selected = states.loc[
        states["state"].eq("verified")
        & states["sequence_id"].astype(str).isin(sequence_index)
        & states["constraint_id"].astype(str).isin(query_index),
        ["sequence_id", "constraint_id"],
    ]
    mask = np.zeros((len(atomic_queries), len(training)), dtype=bool)
    for row in selected.itertuples(index=False):
        mask[query_index[str(row.constraint_id)], sequence_index[str(row.sequence_id)]] = True
    return mask


def _ensure_before_deadline(deadline_monotonic: float | None, operation: str) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError(f"authorized compute deadline reached after {operation}")
