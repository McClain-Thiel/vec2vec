from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.lib import fixed_representation_bakeoff, fixed_representation_features
from vec2vec.lib.constraint_state import retrieval_population_sha256
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.serialization import dataframe_content_sha256


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
        "protocol_version": "modeling-data-test",
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


def _query_manifest(params: dict[str, object]) -> dict[str, object]:
    return {"output_content_hashes": params["expected_query_artifact_hashes"]}


def _split_manifest(split: pd.DataFrame) -> dict[str, object]:
    return {
        "build": {"mapping_sha256": dataframe_content_sha256(split, sort_columns=["sequence_id"])},
        "decision": {"status": "accepted_strict_similarity_closed_split"},
    }


def test_model_inputs_are_deterministic_and_filter_before_sampling() -> None:
    retrieval, split, queries, states = _inputs()
    params = _params(retrieval, queries, states)
    first = fixed_representation_bakeoff.build_bakeoff_inputs(
        retrieval,
        split,
        _split_manifest(split),
        queries,
        states,
        _query_manifest(params),
        params,
    )
    second = fixed_representation_bakeoff.build_bakeoff_inputs(
        retrieval.sample(frac=1.0, random_state=4),
        split.sample(frac=1.0, random_state=7),
        _split_manifest(split),
        queries.sample(frac=1.0, random_state=8),
        states.sample(frac=1.0, random_state=9),
        _query_manifest(params),
        params,
    )
    pairs, exclusions, selected_queries, selected_states, report = first

    assert report["output_hashes"] == second[-1]["output_hashes"]
    assert pairs["sequence_id"].tolist() == second[0]["sequence_id"].tolist()
    assert len(pairs.query("panel_role == 'alignment_train'")) == 12
    assert len(pairs.query("panel_role == 'validation_gallery'")) == 9
    assert set(exclusions["sequence_id"]) == {"sequence-03", "sequence-25"}
    assert len(selected_queries) == 2
    assert len(selected_states) == 8


def test_model_inputs_reject_changed_query_content() -> None:
    retrieval, split, queries, states = _inputs()
    params = _params(retrieval, queries, states)
    manifest = _query_manifest(params)
    manifest["output_content_hashes"] = {
        "query_catalog_sha256": "changed",
        "query_candidate_state_sha256": "changed",
    }
    with pytest.raises(ValueError, match="manifest hash changed"):
        fixed_representation_bakeoff.build_bakeoff_inputs(
            retrieval, split, _split_manifest(split), queries, states, manifest, params
        )


def test_paid_stage_requires_explicit_bounded_authorization() -> None:
    with pytest.raises(ValueError, match="requires an explicit"):
        fixed_representation_bakeoff.validated_compute_authorization(
            {"compute_authorization": None}, stage="text_features:qwen3_embedding_0_6b"
        )
    authorization = {
        "stage": "text_features:qwen3_embedding_0_6b",
        "approval_reference": "approval-1",
        "region": "us-east-1",
        "instance_type": "g6.4xlarge",
        "instance_hour_limit": 1.0,
        "observed_instance_price_usd_per_hour": 1.3232,
    }
    assert (
        fixed_representation_bakeoff.validated_compute_authorization(
            {"compute_authorization": authorization}, stage=authorization["stage"]
        )
        == authorization
    )
    with pytest.raises(ValueError, match="not requested stage"):
        fixed_representation_bakeoff.validated_compute_authorization(
            {"compute_authorization": authorization}, stage="other-stage"
        )


def test_text_features_reject_a_changed_query_table() -> None:
    queries = pd.DataFrame([{"query_id": "query-1", "canonical_query_text": "original"}])
    manifest = {
        "output_hashes": {
            "queries_sha256": dataframe_content_sha256(queries, sort_columns=["query_id"])
        }
    }
    with pytest.raises(ValueError, match="query table hash changed"):
        fixed_representation_features._validate_query_artifact(
            queries.assign(canonical_query_text="changed"), manifest
        )
