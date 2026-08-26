from argparse import Namespace

import pandas as pd
import pytest
from scripts import summarize_results


class RecordingCatalog:
    def __init__(self):
        self.loads = []

    def load(self, name, version=None):
        self.loads.append((name, version))
        return {"name": name, "version": version}


def _config():
    return {
        "inputs": {
            "e02b_version": "inputs-v1",
            "query_benchmark_version": "queries-v1",
            "manifest_sha256": "manifest-hash",
            "pairs_sha256": "pairs-hash",
            "training_query_states_sha256": "states-hash",
        },
        "features": {
            "dna": {
                "tfidf_6mer_svd_512": {"version": "tfidf-v1"},
                "carbon_500m": {"version": "carbon-v1"},
            },
            "text": {"qwen3_embedding_0_6b": {"version": "qwen-v1"}},
        },
        "alignment": {
            "protocol_version": "alignment-v1",
            "training_rows": 20,
            "device": "cuda",
            "probe": {"seeds": [13, 42, 20260818]},
        },
        "supervision": {
            "protocol_version": "supervision-v1",
            "training_rows": 20,
            "minimum_training_verified_rows": 2,
            "objectives": ["paired_identity", "verified_set"],
            "device": "cuda",
            "primary_k": 10,
            "minimum_practical_improvement": 0.01,
            "probe": {"seeds": [13, 42, 20260818]},
            "tracking": {"enabled": True},
        },
        "composition": {
            "protocol_version": "composition-v1",
            "training_rows": 20,
            "training_query_kind": "atomic",
            "evaluation_query_kind": "pair_conjunction",
            "expected_training_queries": 2,
            "expected_evaluation_queries": 3,
            "expected_evaluation_controlled_split": "atoms_seen_conjunction_unseen",
            "minimum_training_verified_rows": 2,
            "objectives": ["paired_identity", "verified_set"],
            "device": "cuda",
            "primary_k": 10,
            "minimum_practical_improvement": 0.01,
            "probe": {"seeds": [13, 42, 20260818]},
            "tracking": {"enabled": True},
        },
        "scale": {
            "protocol_version": "scale-v1",
            "inputs": {
                "panel_version": "e06-inputs-v1",
                "query_benchmark_version": "queries-v1",
                "manifest_sha256": "e06-manifest-hash",
                "pairs_sha256": "e06-pairs-hash",
                "training_query_states_sha256": "states-hash",
            },
            "features": {
                "dna": {"version": "e06-dna-v1"},
                "text": {"version": "e06-text-v1"},
            },
            "training_rows": 88,
            "training_query_kind": "atomic",
            "evaluation_query_kind": "pair_conjunction",
            "expected_training_queries": 2,
            "expected_evaluation_queries": 3,
            "expected_evaluation_controlled_split": "atoms_seen_conjunction_unseen",
            "minimum_training_verified_rows": 2,
            "objectives": ["paired_identity", "verified_set"],
            "device": "cuda",
            "primary_k": 10,
            "minimum_practical_improvement": 0.01,
            "probe": {"seeds": [13, 42, 20260818]},
            "tracking": {"enabled": True},
        },
    }


def test_feature_loader_uses_each_frozen_version() -> None:
    catalog = RecordingCatalog()

    features, manifests = summarize_results._load_features(catalog, _config(), kind="dna")

    assert set(features) == {"tfidf_6mer_svd_512", "carbon_500m"}
    assert set(manifests) == set(features)
    assert catalog.loads == [
        ("e02b_dna_features_tfidf_6mer_svd_512", "carbon-v1"),
        ("e02b_dna_manifest_tfidf_6mer_svd_512", "carbon-v1"),
        ("e02b_dna_features_tfidf_6mer_svd_512", "tfidf-v1"),
        ("e02b_dna_manifest_tfidf_6mer_svd_512", "tfidf-v1"),
    ]


def test_reproduction_parameters_bind_frozen_inputs_and_features() -> None:
    config = _config()

    alignment = summarize_results._alignment_params(config)
    supervision = summarize_results._supervision_params(config)

    assert alignment["accepted_input_artifact"] == {
        "manifest_sha256": "manifest-hash",
        "pairs_sha256": "pairs-hash",
    }
    assert alignment["dna_candidates"] == {"carbon_500m": {}}
    assert supervision["expected_training_query_states_sha256"] == "states-hash"
    assert supervision["input_versions"]["text_features"] == "qwen-v1"

    composition = summarize_results._supervision_params(config, section_name="composition")
    assert composition["training_query_kind"] == "atomic"
    assert composition["evaluation_query_kind"] == "pair_conjunction"
    assert composition["run_name_prefix"] == "e05"

    scale = summarize_results._supervision_params(config, section_name="scale")
    assert scale["training_rows"] == 88
    assert scale["accepted_input_artifact"]["pairs_sha256"] == "e06-pairs-hash"
    assert scale["input_versions"]["e06_inputs"] == "e06-inputs-v1"
    assert scale["run_name_prefix"] == "e06"


def test_scale_feature_loader_uses_e06_datasets() -> None:
    catalog = RecordingCatalog()

    features, manifests = summarize_results._load_scale_features(
        catalog, _config()["scale"], kind="text"
    )

    assert set(features) == {"qwen3_embedding_0_6b"}
    assert set(manifests) == set(features)
    assert catalog.loads == [
        ("e06_text_features_qwen3_embedding_0_6b", "e06-text-v1"),
        ("e06_text_manifest_qwen3_embedding_0_6b", "e06-text-v1"),
    ]


def test_output_hash_verification_reports_changed_artifact() -> None:
    with pytest.raises(ValueError, match="recomputed artifact hashes differ"):
        summarize_results._verify_output_hashes({"metrics": "changed"}, {"metrics": "accepted"})


def test_result_table_can_ignore_new_tracking_urls(tmp_path) -> None:
    path = tmp_path / "supervision.csv"
    pd.DataFrame([{"metric": 0.5, "wandb_runs": "accepted-run"}]).to_csv(path, index=False)

    summarize_results._verify_result_table(
        path,
        [{"metric": 0.5, "wandb_runs": "reproduction-run"}],
        ignored_columns=("wandb_runs",),
    )


def test_paid_reproduction_requires_explicit_authorization() -> None:
    arguments = Namespace(
        approval_reference=None,
        region=None,
        instance_type=None,
        instance_hour_limit=None,
        observed_instance_price_usd_per_hour=None,
    )

    with pytest.raises(ValueError, match="approval-reference"):
        summarize_results._authorization(arguments)


def test_e05_authorization_must_match_frozen_contract() -> None:
    expected = {
        "approval_reference": "approval",
        "region": "us-east-1",
        "instance_type": "g6.4xlarge",
        "instance_hour_limit": 0.5,
        "observed_instance_price_usd_per_hour": 1.3232,
    }

    summarize_results._validate_frozen_authorization(expected, expected)
    summarize_results._validate_frozen_authorization(
        {**expected, "instance_hour_limit": 0.44}, expected
    )
    with pytest.raises(ValueError, match="differs from frozen"):
        summarize_results._validate_frozen_authorization(
            {**expected, "instance_hour_limit": 1.0}, expected
        )
