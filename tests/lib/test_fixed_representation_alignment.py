from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from vec2vec.lib import fixed_representation_alignment
from vec2vec.lib.fixed_representation_invariance import json_content_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.lib.text import sha256_text


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for index in range(24):
        role = "alignment_train" if index < 12 else "validation_gallery"
        rows.append(
            {
                "sequence_id": f"sequence-{index:02d}",
                "sequence_sha256": f"sequence-hash-{index:02d}",
                "description_sha256": f"description-hash-{index:02d}",
                "panel_role": role,
                "similarity_component_primary": f"component-{index:02d}",
                "length_bp": 100 + index,
                "component_size": 1,
            }
        )
    pairs = pd.DataFrame(rows)
    queries = pd.DataFrame(
        [
            {
                "query_id": "query-atomic",
                "semantic_query_id": "semantic-atomic",
                "query_kind": "atomic",
                "canonical_query_text": "atomic query",
            },
            {
                "query_id": "query-pair",
                "semantic_query_id": "semantic-pair",
                "query_kind": "pair_conjunction",
                "canonical_query_text": "pair query",
            },
        ]
    )
    states = pd.DataFrame(
        [
            {
                "semantic_query_id": "semantic-atomic",
                "sequence_id": "sequence-12",
                "state": "verified",
            },
            {
                "semantic_query_id": "semantic-atomic",
                "sequence_id": "sequence-13",
                "state": "contradicted",
            },
            {
                "semantic_query_id": "semantic-pair",
                "sequence_id": "sequence-14",
                "state": "verified",
            },
            {
                "semantic_query_id": "semantic-pair",
                "sequence_id": "sequence-15",
                "state": "contradicted",
            },
        ]
    )
    manifest = {
        "protocol_version": "e02b-test",
        "output_hashes": {
            "queries_sha256": dataframe_content_sha256(queries, sort_columns=["query_id"]),
            "query_states_sha256": dataframe_content_sha256(
                states, sort_columns=["semantic_query_id", "sequence_id"]
            ),
        },
    }
    return pairs, queries, states, manifest


def _dna_features(candidate_id: str, pairs: pd.DataFrame, seed: int) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    matrix = generator.normal(size=(len(pairs), 3)).astype(np.float32)
    return pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "sequence_sha256": pairs["sequence_sha256"],
            "representative_sequence_id": pairs["sequence_id"],
            "length_bp": pairs["length_bp"],
            "embedding_dimension": 3,
            "embedding": [row.tolist() for row in matrix],
            "embedding_sha256": [f"embedding-{index}" for index in range(len(pairs))],
        }
    )


def _text_features(
    candidate_id: str,
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
) -> pd.DataFrame:
    generator = np.random.default_rng(19)
    documents = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "text_role": "document",
            "text_sha256": pairs["description_sha256"],
            "token_count": 4,
            "embedding_dimension": 3,
            "embedding": [row.tolist() for row in generator.normal(size=(len(pairs), 3))],
            "embedding_sha256": [f"document-{index}" for index in range(len(pairs))],
        }
    )
    query_matrix = generator.normal(size=(len(queries), 3))
    query_rows = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "text_role": "query",
            "text_sha256": queries["canonical_query_text"].map(sha256_text),
            "token_count": 4,
            "embedding_dimension": 3,
            "embedding": [row.tolist() for row in query_matrix],
            "embedding_sha256": [f"query-{index}" for index in range(len(queries))],
        }
    )
    return pd.concat([documents, query_rows], ignore_index=True)


def _manifest_and_acceptance(
    candidate_id: str,
    features: pd.DataFrame,
    input_manifest: dict[str, Any],
    *,
    feature_kind: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sort_columns = (
        ["candidate_id", "sequence_sha256"]
        if feature_kind == "dna"
        else ["candidate_id", "text_role", "text_sha256"]
    )
    feature_hash = dataframe_content_sha256(features, sort_columns=sort_columns)
    manifest = {
        "candidate_id": candidate_id,
        "input_manifest_sha256": json_content_sha256(input_manifest),
        "output_hashes": {"features_sha256": feature_hash},
    }
    accepted = {
        "version": f"{candidate_id}-v1",
        "manifest_sha256": json_content_sha256(manifest),
        "features_sha256": feature_hash,
        "extraction_gpu_hours": 0.001,
        "persisted_bytes": 100,
    }
    return manifest, accepted


def test_factorial_alignment_persists_all_seeds_draws_and_selection(monkeypatch) -> None:
    pairs, queries, states, input_manifest = _inputs()
    dna_features = {
        "carbon_500m": _dna_features("carbon_500m", pairs, 11),
        "generanno_prokaryote_500m": _dna_features("generanno_prokaryote_500m", pairs, 13),
        "generator_v2_prokaryote_1_2b": _dna_features("generator_v2_prokaryote_1_2b", pairs, 15),
        "tfidf_6mer_svd_512": _dna_features("tfidf_6mer_svd_512", pairs, 17),
    }
    text_features = {
        "bge_base_en_v1_5": _text_features("bge_base_en_v1_5", pairs, queries),
        "gte_modernbert_base": _text_features("gte_modernbert_base", pairs, queries),
        "qwen3_embedding_0_6b": _text_features("qwen3_embedding_0_6b", pairs, queries),
    }
    dna_manifests = {}
    text_manifests = {}
    accepted_dna = {}
    accepted_text = {}
    for candidate_id, features in dna_features.items():
        manifest, accepted = _manifest_and_acceptance(
            candidate_id, features, input_manifest, feature_kind="dna"
        )
        dna_manifests[candidate_id] = manifest
        accepted_dna[candidate_id] = accepted
    for candidate_id, features in text_features.items():
        manifest, accepted = _manifest_and_acceptance(
            candidate_id, features, input_manifest, feature_kind="text"
        )
        text_manifests[candidate_id] = manifest
        accepted_text[candidate_id] = accepted

    def fake_train(
        sequence_train: np.ndarray,
        text_train: np.ndarray,
        sequence_groups: np.ndarray,
        description_groups: np.ndarray,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        dimension = int(kwargs["projection_dimension"])
        sequence_head = np.eye(dimension, sequence_train.shape[1], dtype=np.float32)
        text_head = np.eye(dimension, text_train.shape[1], dtype=np.float32)
        state = {
            "sequence_head": sequence_head,
            "text_head": text_head,
            "logit_scale": 1.0,
            "batches_per_epoch": 3,
            "last_batch_rows": 4,
            "dropped_rows_per_epoch": 0,
        }
        history = pd.DataFrame(
            [
                {"epoch": 1, "mean_loss": 1.0, "logit_scale": 2.0},
                {"epoch": 2, "mean_loss": 0.5, "logit_scale": 2.1},
            ]
        )
        return state, history

    monkeypatch.setattr(
        fixed_representation_alignment.alignment_probe,
        "train_alignment_probe",
        fake_train,
    )
    params = {
        "protocol_version": "e02b-test",
        "training_rows": 12,
        "device": "cpu",
        "tfidf": {"candidate_id": "tfidf_6mer_svd_512"},
        "dna_candidates": {
            "carbon_500m": {},
            "generanno_prokaryote_500m": {},
            "generator_v2_prokaryote_1_2b": {},
        },
        "text_candidates": {
            "bge_base_en_v1_5": {},
            "gte_modernbert_base": {},
            "qwen3_embedding_0_6b": {},
        },
        "accepted_input_artifact": {
            "manifest_sha256": json_content_sha256(input_manifest),
            "pairs_sha256": dataframe_content_sha256(
                pairs, sort_columns=["panel_role", "sequence_id"]
            ),
        },
        "accepted_feature_artifacts": {"dna": accepted_dna, "text": accepted_text},
        "probe": {
            "seeds": [13, 42, 20260818],
            "projection_dimension": 2,
            "whitening_epsilon": 1e-6,
            "epochs": 2,
            "batch_size": 4,
            "learning_rate": 1e-3,
            "weight_decay": 0.01,
            "initial_temperature": 0.07,
            "maximum_logit_scale": 100.0,
            "cutoffs": [1, 5, 10],
            "bootstrap_draws": 20,
            "bootstrap_seed": 7,
            "practical_tie_utility": 0.01,
            "minimum_incumbent_improvement": 0.01,
            "cost_tie_fraction": 0.10,
        },
    }

    outputs = fixed_representation_alignment.run_factorial_alignment(
        pairs,
        queries,
        states,
        input_manifest,
        dna_features,
        dna_manifests,
        text_features,
        text_manifests,
        params,
    )
    (
        whitening,
        checkpoints,
        histories,
        paired,
        rankings,
        query_metrics,
        summaries,
        draws,
        report,
    ) = outputs

    assert len(whitening) == 7
    assert len(checkpoints) == 36
    assert len(histories) == 72
    assert len(paired) == 36
    assert len(rankings) == 720
    assert len(query_metrics) == 216
    assert len(summaries) == 324
    assert len(draws) == 720
    assert report["factorial"]["completed_configurations"] == 36
    assert report["decision"]["status"] == "validation_pair_selected"
    assert report["selection"]["selected_pair"]["dna_candidate_id"] in dna_features


def test_alignment_rejects_changed_query_labels_and_text() -> None:
    pairs, queries, states, input_manifest = _inputs()
    params = {
        "training_rows": 12,
        "accepted_input_artifact": {
            "manifest_sha256": json_content_sha256(input_manifest),
            "pairs_sha256": dataframe_content_sha256(
                pairs, sort_columns=["panel_role", "sequence_id"]
            ),
        },
    }
    changed_states = states.copy()
    changed_states.loc[0, "state"] = "contradicted"
    with pytest.raises(ValueError, match="query-state table hash changed"):
        fixed_representation_alignment._validate_alignment_inputs(
            pairs, queries, changed_states, input_manifest, params
        )

    changed_queries = queries.copy()
    changed_queries.loc[0, "canonical_query_text"] = "changed after freezing"
    with pytest.raises(ValueError, match="query table hash changed"):
        fixed_representation_alignment._validate_alignment_inputs(
            pairs, changed_queries, states, input_manifest, params
        )


def test_alignment_requires_the_three_frozen_probe_seeds() -> None:
    with pytest.raises(ValueError, match="probe seeds must remain"):
        fixed_representation_alignment.validated_probe_axes(
            {"seeds": [13], "cutoffs": [1, 5, 10]}, gallery_rows=12
        )


def test_incumbent_threshold_uses_highest_utility_before_cost_tie_break() -> None:
    utilities = {
        "carbon_500m": 0.100,
        "highest": 0.114,
        "lower_cost_tie": 0.109,
    }
    summaries = pd.DataFrame(
        [
            {
                "dna_candidate_id": dna_candidate_id,
                "text_candidate_id": "bge_base_en_v1_5",
                "seed": seed,
                "query_kind": "combined",
                "k": 10,
                "utility": utility,
            }
            for dna_candidate_id, utility in utilities.items()
            for seed in (13, 42, 20260818)
        ]
    )
    draws = pd.DataFrame(
        [
            {
                "dna_candidate_id": dna_candidate_id,
                "text_candidate_id": "bge_base_en_v1_5",
                "query_kind": "combined",
                "draw": draw,
                "utility": utility + offset,
            }
            for dna_candidate_id, utility in utilities.items()
            for draw, offset in enumerate((-0.02, 0.02))
        ]
    )
    accepted = {
        "dna": {
            "carbon_500m": {"extraction_gpu_hours": 0.020, "persisted_bytes": 300},
            "highest": {"extraction_gpu_hours": 0.030, "persisted_bytes": 200},
            "lower_cost_tie": {"extraction_gpu_hours": 0.010, "persisted_bytes": 100},
        },
        "text": {
            "bge_base_en_v1_5": {
                "extraction_gpu_hours": 0.010,
                "persisted_bytes": 100,
            }
        },
    }

    report = fixed_representation_alignment._selection_report(
        summaries,
        draws,
        accepted,
        {
            "seeds": [13, 42, 20260818],
            "practical_tie_utility": 0.01,
            "cost_tie_fraction": 0.10,
            "minimum_incumbent_improvement": 0.01,
        },
    )

    assert report["highest_utility_improvement_over_incumbent"] == pytest.approx(0.014)
    assert report["cost_preferred_improvement_over_incumbent"] == pytest.approx(0.009)
    assert not report["incumbent_retained"]
    assert report["selected_pair"]["dna_candidate_id"] == "lower_cost_tie"
