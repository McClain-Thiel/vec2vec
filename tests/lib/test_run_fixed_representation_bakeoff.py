from __future__ import annotations

import argparse

import pytest
from scripts import run_fixed_representation_bakeoff


def _arguments(stage: str, candidate: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        stage=stage,
        candidate=candidate,
        approval_reference="approval-1",
        region="us-east-1",
        instance_type="g6.2xlarge",
        instance_hour_limit=1.0,
        observed_instance_price_usd_per_hour=0.98,
        shutdown_reserve_seconds=30.0,
        internal_child=False,
    )


def _configuration() -> dict[str, object]:
    dna_candidates = {
        "carbon_500m": {},
        "generanno_prokaryote_500m": {},
        "generator_v2_prokaryote_1_2b": {},
    }
    text_candidates = {
        "bge_base_en_v1_5": {},
        "gte_modernbert_base": {},
        "qwen3_embedding_0_6b": {},
    }
    dna_feature_ids = {"tfidf_6mer_svd_512", *dna_candidates}
    paid_stages = {
        "dna_features:carbon_500m",
        "dna_features:generanno_prokaryote_500m",
        "dna_features:generator_v2_prokaryote_1_2b",
        "text_features:bge_base_en_v1_5",
        "text_features:gte_modernbert_base",
        "text_features:qwen3_embedding_0_6b",
        "alignment_probe",
    }
    return {
        "accepted_input_artifact": {"version": "input-v1"},
        "dna_candidates": dna_candidates,
        "text_candidates": text_candidates,
        "accepted_invariance_artifacts": {
            candidate_id: {"version": f"invariance-{candidate_id}"}
            for candidate_id in dna_candidates
        },
        "accepted_feature_artifacts": {
            "dna": {
                candidate_id: {"version": f"dna-{candidate_id}"} for candidate_id in dna_feature_ids
            },
            "text": {
                candidate_id: {"version": f"text-{candidate_id}"}
                for candidate_id in text_candidates
            },
        },
        "approved_compute_authorization": {
            "approval_reference": "approval-1",
            "region": "us-east-1",
            "instance_type": "g6.2xlarge",
            "observed_instance_price_usd_per_hour": 0.98,
            "total_instance_hour_limit": 7.0,
            "stage_instance_hour_limits": {stage: 1.0 for stage in paid_stages},
        },
    }


def test_paid_timeout_reserves_time_for_forced_shutdown() -> None:
    assert (
        run_fixed_representation_bakeoff._paid_timeout_seconds(1.0, shutdown_reserve_seconds=30.0)
        == 3570.0
    )

    with pytest.raises(ValueError, match="leaves no paid stage execution time"):
        run_fixed_representation_bakeoff._paid_timeout_seconds(0.01, shutdown_reserve_seconds=36.0)


def test_dna_stage_binds_input_and_accepted_invariance_versions() -> None:
    pipeline, versions = run_fixed_representation_bakeoff._stage_contract(
        _arguments("dna_features", "carbon_500m"), _configuration()
    )

    assert pipeline == "fixed_representation_bakeoff_dna_features"
    assert versions == {
        "e02b_pairs": "input-v1",
        "e02b_input_manifest": "input-v1",
        "e02_fixed_representation_invariance_manifest": "invariance-carbon_500m",
    }


def test_alignment_stage_requires_and_binds_every_feature_version() -> None:
    pipeline, versions = run_fixed_representation_bakeoff._stage_contract(
        _arguments("alignment_probe"), _configuration()
    )

    assert pipeline == "fixed_representation_bakeoff_alignment"
    assert len(versions) == 18
    assert versions["e02b_dna_features_carbon_500m"] == "dna-carbon_500m"
    assert versions["e02b_dna_manifest_tfidf_6mer_svd_512"] == ("dna-tfidf_6mer_svd_512")
    assert versions["e02b_text_features_qwen3_embedding_0_6b"] == ("text-qwen3_embedding_0_6b")
    assert versions["e02b_query_states"] == "input-v1"
