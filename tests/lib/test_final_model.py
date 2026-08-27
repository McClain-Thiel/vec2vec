import itertools
import json

import numpy as np
import pandas as pd

from vec2vec.lib import final_model
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256


def test_load_final_inputs_validates_and_collapses_splits(tmp_path, monkeypatch):
    source = pd.DataFrame(
        [
            _source_row("p1", "AAAAAA", "train"),
            _source_row("p2", "CCCCCC", "val"),
            _source_row("p3", "GGGGGN", "test"),
        ]
    )
    queries = pd.DataFrame(
        [
            _query_row("q1", "s1", "c1", "first query"),
            _query_row("q2", "s2", "c2", "second query"),
        ]
    )
    states = pd.DataFrame(
        [
            _state_row("p1", "AAAAAA", "c1"),
            _state_row("p2", "CCCCCC", "c1"),
            _state_row("p2", "CCCCCC", "c2"),
            _state_row("p3", "GGGGGN", "c2"),
        ]
    )
    source_path = tmp_path / "source.parquet"
    query_path = tmp_path / "queries.parquet"
    states_path = tmp_path / "states.parquet"
    source.to_parquet(source_path, index=False)
    queries.to_parquet(query_path, index=False)
    states.to_parquet(states_path, index=False)
    monkeypatch.setattr(final_model, "EXPECTED_SOURCE_ROWS", 3)
    monkeypatch.setattr(final_model, "EXPECTED_ELIGIBLE_ROWS", 2)
    monkeypatch.setattr(final_model, "EXPECTED_ATOMIC_QUERIES", 2)
    monkeypatch.setattr(final_model, "MINIMUM_VERIFIED_ROWS", 1)
    monkeypatch.setattr(
        final_model, "EXPECTED_SOURCE_FILE_SHA256", final_model.file_sha256(source_path)
    )
    monkeypatch.setattr(
        final_model,
        "EXPECTED_QUERIES_SHA256",
        dataframe_content_sha256(queries, sort_columns=["query_id"]),
    )
    monkeypatch.setattr(
        final_model,
        "EXPECTED_STATES_SHA256",
        dataframe_content_sha256(states, sort_columns=["sequence_id", "constraint_id", "state"]),
    )

    training, atomic, mask, audit = final_model.load_final_inputs(
        source_path, states_path, query_path
    )

    assert training["sequence_id"].tolist() == ["p1", "p2"]
    assert atomic["query_id"].tolist() == ["q1", "q2"]
    assert mask.tolist() == [[True, True], [False, True]]
    assert audit["source_rows"] == 3
    assert audit["eligible_rows"] == 2
    assert audit["excluded_non_acgt_rows"] == 1


def test_bundle_round_trip_and_inference(tmp_path, monkeypatch):
    rows = 3
    monkeypatch.setattr(final_model, "EXPECTED_ELIGIBLE_ROWS", rows)
    terms = np.asarray(["".join(value) for value in itertools.product("ACGT", repeat=6)])
    generator = np.random.default_rng(7)
    model = {
        "terms": terms,
        "idf": np.ones(4096, dtype=np.float64),
        "svd_components": generator.normal(size=(512, 4096)).astype(np.float32),
        "dna_whitening_mean": np.zeros(512, dtype=np.float32),
        "dna_whitening_matrix": np.eye(512, dtype=np.float32),
        "text_whitening_mean": np.zeros(3, dtype=np.float32),
        "text_whitening_matrix": np.eye(3, dtype=np.float32),
        "sequence_head": np.eye(512, dtype=np.float32),
        "text_head": generator.normal(size=(512, 3)).astype(np.float32),
        "logit_scale": np.asarray([1.0], dtype=np.float32),
    }
    index = generator.normal(size=(rows, 512)).astype(np.float32)
    index /= np.linalg.norm(index, axis=1, keepdims=True)
    training = pd.DataFrame(
        [
            {
                "uuid": f"u{row}",
                "sequence_id": f"p{row}",
                "sequence_sha256": str(row) * 64,
                "name": f"plasmid {row}",
                "length_bp": 6,
            }
            for row in range(rows)
        ]
    )
    queries = pd.DataFrame(
        [
            {
                "query_id": "q1",
                "semantic_query_id": "s1",
                "canonical_query_text": "query",
                "constraint_id": "c1",
            }
        ]
    )
    history = pd.DataFrame([{"update": 1, "loss": 0.5}])
    output = tmp_path / "bundle"

    manifest = final_model.save_bundle(
        output, model, index, training, queries, history, {"protocol_version": "test"}
    )

    assert final_model.validate_bundle(output) == manifest
    sequence_vectors = final_model.project_sequences(output / "model.npz", ["AAAAAA"])
    query_vectors = final_model.project_query_embeddings(
        output / "model.npz", np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    )
    assert sequence_vectors.shape == (1, 512)
    assert query_vectors.shape == (1, 512)
    assert np.allclose(np.linalg.norm(sequence_vectors, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0)


def _source_row(sequence_id, sequence, split):
    return {
        "uuid": f"uuid-{sequence_id}",
        "sequence_id": sequence_id,
        "sequence": sequence,
        "sequence_sha256": sequence_sha256(sequence),
        "name": sequence_id,
        "description": f"description {sequence_id}",
        "length_bp": len(sequence),
        "split_grouped": split,
    }


def _query_row(query_id, semantic_id, constraint_id, text):
    return {
        "query_id": query_id,
        "semantic_query_id": semantic_id,
        "query_kind": "atomic",
        "canonical_query_text": text,
        "constraint_ids_json": json.dumps([constraint_id]),
    }


def _state_row(sequence_id, sequence, constraint_id):
    return {
        "sequence_id": sequence_id,
        "sequence_sha256": sequence_sha256(sequence),
        "constraint_id": constraint_id,
        "state": "verified",
    }
