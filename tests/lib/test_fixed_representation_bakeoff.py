from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vec2vec.lib import (
    fixed_representation,
    fixed_representation_bakeoff,
    fixed_representation_bakeoff_validation,
    fixed_representation_features,
)
from vec2vec.lib.constraint_state import retrieval_population_sha256
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retrieval_rows = []
    split_rows = []
    for index in range(30):
        sequence = "ACGT" * (index + 2)
        split = "train" if index < 20 else "val"
        if index in {3, 25}:
            sequence = f"W{sequence[1:]}"
        sequence_id = f"sequence-{index:02d}"
        retrieval_rows.append(
            {
                "sequence_id": sequence_id,
                "sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
                "description": f"description {index // 2}",
                "length_bp": len(sequence),
                "leakage_component": index // 3,
                "split_grouped": split,
            }
        )
        split_rows.append(
            {
                "sequence_id": sequence_id,
                "similarity_component_primary": f"component-{index // 3:02d}",
                "leakage_component_v2": f"component-{index // 3:02d}",
                "split_grouped_v2": split,
            }
        )
    queries = pd.DataFrame(
        [
            {
                "query_id": f"query-{index}",
                "semantic_query_id": f"semantic-{index}",
                "query_kind": "atomic" if index == 0 else "pair_conjunction",
                "canonical_query_text": f"query text {index}",
                "evaluation_split": "val",
                "gallery_kind": "closed_grouped_v2",
                "measurement_eligible": True,
            }
            for index in range(2)
        ]
    )
    states = pd.DataFrame(
        [
            {
                "semantic_query_id": f"semantic-{query}",
                "sequence_id": f"sequence-{sequence:02d}",
                "state": "verified" if sequence < 23 else "contradicted",
            }
            for query in range(2)
            for sequence in (20, 21, 22, 23)
        ]
    )
    return pd.DataFrame(retrieval_rows), pd.DataFrame(split_rows), queries, states


def _params(
    retrieval: pd.DataFrame,
    queries: pd.DataFrame,
    states: pd.DataFrame,
) -> dict[str, object]:
    return {
        "protocol_version": "e02b-test",
        "input_versions": {
            "retrieval": "retrieval-v1",
            "split": "split-v1",
            "constraint_state": "state-v1",
            "graph": "graph-v1",
        },
        "expected_input_population_sha256": retrieval_population_sha256(retrieval),
        "expected_query_artifact_hashes": {
            "query_catalog_sha256": dataframe_content_sha256(queries, sort_columns=["query_id"]),
            "query_candidate_state_sha256": dataframe_content_sha256(
                states, sort_columns=["semantic_query_id", "state", "sequence_id"]
            ),
        },
        "eligible_sequence_alphabet": "ACGT",
        "training_rows": 12,
        "maximum_rows_per_component": 2,
        "selection_salt": "selection-v1",
        "minimum_verified_rows": 2,
    }


def _manifest(params: dict[str, object]) -> dict[str, object]:
    return {
        "input_versions": params["input_versions"],
        "output_content_hashes": params["expected_query_artifact_hashes"],
    }


def _split_manifest(split: pd.DataFrame) -> dict[str, object]:
    return {
        "build": {"mapping_sha256": dataframe_content_sha256(split, sort_columns=["sequence_id"])},
        "decision": {"status": "accepted_strict_similarity_closed_split"},
    }


def test_bakeoff_inputs_are_deterministic_and_filter_before_sampling(tmp_path) -> None:
    retrieval, split, queries, states = _inputs()
    params = _params(retrieval, queries, states)

    first = fixed_representation_bakeoff.build_bakeoff_inputs(
        retrieval,
        split,
        _split_manifest(split),
        queries,
        states,
        _manifest(params),
        params,
    )
    second = fixed_representation_bakeoff.build_bakeoff_inputs(
        retrieval.sample(frac=1.0, random_state=4),
        split.sample(frac=1.0, random_state=7),
        _split_manifest(split),
        queries.sample(frac=1.0, random_state=8),
        states.sample(frac=1.0, random_state=9),
        _manifest(params),
        params,
    )
    first_pairs, first_exclusions, first_queries, first_states, first_report = first
    second_pairs, _, _, _, second_report = second

    assert first_report["output_hashes"] == second_report["output_hashes"]
    assert first_pairs["sequence_id"].tolist() == second_pairs["sequence_id"].tolist()
    assert len(first_pairs.query("panel_role == 'alignment_train'")) == 12
    assert len(first_pairs.query("panel_role == 'validation_gallery'")) == 9
    assert set(first_exclusions["sequence_id"]) == {"sequence-03", "sequence-25"}
    assert len(first_queries) == 2
    assert len(first_states) == 8
    assert first_report["population_flow"]["train"]["excluded_rows"] == 1
    assert first_report["population_flow"]["val"]["excluded_rows"] == 1
    readback = fixed_representation_bakeoff_validation.validate_bakeoff_inputs(
        first_pairs,
        first_exclusions,
        first_queries,
        first_states,
        first_report,
        expected_protocol_version="e02b-test",
        expected_training_rows=12,
    )
    assert readback["status"] == "passed_e02b_input_readback"

    persisted_tables = []
    for name, table in (
        ("pairs", first_pairs),
        ("exclusions", first_exclusions),
        ("queries", first_queries),
        ("states", first_states),
    ):
        path = tmp_path / f"{name}.parquet"
        table.to_parquet(path)
        persisted_tables.append(pd.read_parquet(path))
    persisted_readback = fixed_representation_bakeoff_validation.validate_bakeoff_inputs(
        *persisted_tables,
        first_report,
        expected_protocol_version="e02b-test",
        expected_training_rows=12,
    )
    assert persisted_readback["output_hashes"] == first_report["output_hashes"]


def test_component_panel_respects_cap_and_inverse_size_passes() -> None:
    retrieval, split, _, _ = _inputs()
    rows = retrieval.merge(split, on="sequence_id", validate="one_to_one")
    rows["component_size"] = rows.groupby("similarity_component_primary")["sequence_id"].transform(
        "size"
    )

    panel = fixed_representation_bakeoff.select_component_balanced_train_panel(
        rows.query("split_grouped_v2 == 'train' and sequence_id != 'sequence-03'"),
        rows=12,
        maximum_rows_per_component=2,
        salt="selection-v1",
    )

    assert len(panel) == 12
    assert panel.groupby("similarity_component_primary").size().max() <= 2
    assert set(panel["selection_pass"]) <= {0, 1}


def test_bakeoff_inputs_fail_when_a_query_loses_support() -> None:
    retrieval, split, queries, states = _inputs()
    params = _params(retrieval, queries, states)
    params["minimum_verified_rows"] = 4

    with pytest.raises(ValueError, match="lost measurement support"):
        fixed_representation_bakeoff.build_bakeoff_inputs(
            retrieval,
            split,
            _split_manifest(split),
            queries,
            states,
            _manifest(params),
            params,
        )


def test_bakeoff_inputs_reject_changed_query_manifest() -> None:
    retrieval, split, queries, states = _inputs()
    params = _params(retrieval, queries, states)
    manifest = _manifest(params)
    manifest["output_content_hashes"] = {
        "query_catalog_sha256": "changed",
        "query_candidate_state_sha256": "state-hash",
    }

    with pytest.raises(ValueError, match="manifest hash changed"):
        fixed_representation_bakeoff.build_bakeoff_inputs(
            retrieval,
            split,
            _split_manifest(split),
            queries,
            states,
            manifest,
            params,
        )


def test_paid_stage_requires_exact_compute_authorization() -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        fixed_representation_bakeoff.validated_compute_authorization(
            {"compute_authorization": None}, stage="alignment_probe"
        )

    with pytest.raises(ValueError, match="not requested stage"):
        fixed_representation_bakeoff.validated_compute_authorization(
            {
                "compute_authorization": {
                    "stage": "dna_features:carbon_500m",
                    "approval_reference": "approval-1",
                    "region": "us-east-1",
                    "instance_type": "g6.2xlarge",
                    "instance_hour_limit": 1.0,
                    "observed_instance_price_usd_per_hour": 0.98,
                }
            },
            stage="alignment_probe",
        )


def test_text_feature_extraction_rejects_a_changed_query_table() -> None:
    queries = pd.DataFrame([{"query_id": "query-1", "canonical_query_text": "original"}])
    manifest = {
        "output_hashes": {
            "queries_sha256": dataframe_content_sha256(queries, sort_columns=["query_id"])
        }
    }
    changed = queries.assign(canonical_query_text="changed")

    with pytest.raises(ValueError, match="query table hash changed"):
        fixed_representation_features._validate_query_artifact(changed, manifest)


def test_tfidf_readback_validates_vectors_vocabulary_and_state() -> None:
    pairs = pd.DataFrame(
        {
            "sequence_sha256": ["sequence-a", "sequence-b", "sequence-c"],
        }
    )
    matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]], dtype=np.float32)
    features = pd.DataFrame(
        {
            "candidate_id": "tfidf",
            "sequence_sha256": pairs["sequence_sha256"],
            "embedding_dimension": 2,
            "embedding": [row.tolist() for row in matrix],
            "embedding_sha256": [fixed_representation.embedding_sha256(row) for row in matrix],
        }
    )
    vocabulary = pd.DataFrame(
        {
            "term": ["AAAAAA", "CCCCCC"],
            "term_index": [0, 1],
            "idf": [1.0, 2.0],
        }
    )
    svd_state = pd.DataFrame(
        {
            "component": [0, 1],
            "singular_value": [2.0, 1.0],
            "vector": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    input_manifest = {"input": "accepted"}
    output_hashes = {
        "features_sha256": dataframe_content_sha256(
            features, sort_columns=["candidate_id", "sequence_sha256"]
        ),
        "vocabulary_sha256": dataframe_content_sha256(vocabulary, sort_columns=["term_index"]),
        "svd_state_sha256": dataframe_content_sha256(svd_state, sort_columns=["component"]),
    }
    manifest = {
        "feature_kind": "tfidf_dna",
        "candidate_id": "tfidf",
        "input_manifest_sha256": (
            fixed_representation_bakeoff_validation.json_content_sha256(input_manifest)
        ),
        "training_rows": 2,
        "elapsed_seconds": 1.0,
        "output_hashes": output_hashes,
        "decision": {
            "status": "frozen_features_complete",
            "validation_rankings_computed": False,
        },
    }

    report = fixed_representation_bakeoff_validation.validate_tfidf_features(
        pairs,
        input_manifest,
        features,
        vocabulary,
        svd_state,
        manifest,
        expected_candidate_id="tfidf",
        expected_dimension=2,
        expected_training_rows=2,
    )

    assert report["status"] == "passed_e02b_tfidf_readback"
    assert report["extraction_gpu_hours"] == 0.0
