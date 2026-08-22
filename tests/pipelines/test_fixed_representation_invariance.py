from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from scripts import validate_fixed_representation_invariance_artifacts

from vec2vec.lib import fixed_representation, fixed_representation_invariance
from vec2vec.lib.dna_encoder import EncodedSequence
from vec2vec.lib.fixed_representation_validation import validate_invariance_outputs
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.pipelines.fixed_representation_invariance import nodes

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _panel() -> pd.DataFrame:
    sequences = ["ACGT" * 25, "GCGT" * 50, "ATAT" * 100, "GCCC" * 200]
    rows = []
    for index, sequence in enumerate(sequences):
        rows.append(
            {
                "sequence_id": f"sequence-{index}",
                "sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
                "description": f"description {index}",
                "description_sha256": f"description-hash-{index}",
                "length_bp": len(sequence),
                "leakage_component": index,
                "split_grouped": "train",
                "similarity_component_primary": f"primary-{index}",
                "leakage_component_v2": f"v2-{index}",
                "split_grouped_v2": "train",
                "length_decile": index,
                "selection_sha256": f"selection-hash-{index}",
                "in_numerical_smoke_panel": index == 0,
            }
        )
    return pd.DataFrame(rows)


def _smoke_params() -> dict[str, Any]:
    return {
        "candidates": {
            "candidate": {
                "model_id": "organization/model",
                "revision": "a" * 40,
                "transformers_version": "5.12.1",
                "model_class": "causal_lm",
                "trust_remote_code": True,
                "model_max_tokens": 128,
                "tokenizer_unit_bp": 1,
                "sequence_prefix": "<dna>",
                "sequence_suffix": "</dna>",
                "excluded_content_tokens": ["<dna>", "</dna>"],
                "out_of_vocabulary_token": "<oov>",
                "pooling_layers": 4,
                "attention_implementation": "sdpa",
            }
        }
    }


def _smoke_manifest(status: str = "passed_numerical_smoke") -> dict[str, Any]:
    return {
        "candidate_id": "candidate",
        "candidate": _smoke_params()["candidates"]["candidate"],
        "decision": {"status": status},
    }


def _panel_manifest(panel_hash: str) -> dict[str, Any]:
    return {
        "protocol": "studies/example.md",
        "summary": {"panel_sha256": panel_hash},
    }


def _params(panel_hash: str) -> dict[str, Any]:
    panel_manifest = _panel_manifest(panel_hash)
    smoke_manifest = _smoke_manifest()
    return {
        "protocol_version": "fixed_representation_bakeoff_v0.1",
        "expected_panel_sha256": panel_hash,
        "expected_rows": 4,
        "accepted_smoke_artifacts": {
            "candidate": {
                "version": "2026-08-18T00.00.00.000Z",
                "panel_manifest_sha256": (
                    fixed_representation_invariance.json_content_sha256(panel_manifest)
                ),
                "smoke_manifest_sha256": (
                    fixed_representation_invariance.json_content_sha256(smoke_manifest)
                ),
                "transformers_version_source": "candidate.transformers_version",
            }
        },
        "rotation_fractions": {"rotate_25": 0.25, "rotate_50": 0.5, "rotate_75": 0.75},
        "reverse_complement_variant": "reverse_complement",
        "minimum_median_transform_cosine": 0.90,
        "minimum_effective_rank_fraction": 0.01,
        "window_overlap_fraction": 0.25,
        "seed": 13,
        "device": "cpu",
        "compute_authorization": {
            "approval_reference": "test approval",
            "region": "test-region-1",
            "instance_type": "test.instance",
            "instance_hour_limit": 1.0,
            "batch_instance_hour_limit": 1.0,
            "observed_instance_price_usd_per_hour": 1.0,
        },
    }


class _FakeEncoder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.maximum_content_bp = 1_000

    def load(self) -> None:
        return None

    def reset_peak_device_memory(self) -> None:
        return None

    def peak_device_memory_bytes(self) -> int:
        return 123

    def close(self) -> None:
        return None

    def encode_sequence(self, sequence_id: str, sequence: str) -> EncodedSequence:
        index = int(sequence_id.rsplit("-", 1)[1])
        vectors = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return EncodedSequence(
            vector=vectors[index],
            coverage=[
                {
                    "sequence_id": sequence_id,
                    "window_index": 0,
                    "start_bp": 0,
                    "input_base_count": len(sequence),
                    "newly_covered_base_count": len(sequence),
                    "wrapped_input_base_count": 0,
                    "input_token_count": len(sequence),
                    "content_token_count": len(sequence),
                    "special_token_count": 0,
                    "out_of_vocabulary_token_count": 0,
                }
            ],
            elapsed_seconds=0.01,
        )


def test_invariance_node_records_complete_transform_and_geometry_evidence(monkeypatch) -> None:
    panel = _panel()
    panel_hash = dataframe_content_sha256(
        panel,
        sort_columns=["sequence_id"],
        value_columns=fixed_representation.PANEL_HASH_COLUMNS,
    )
    panel_manifest = _panel_manifest(panel_hash)
    smoke_manifest = _smoke_manifest()
    monkeypatch.setattr(nodes, "FrozenDnaEncoder", _FakeEncoder)
    monkeypatch.setattr(nodes, "_configure_determinism", lambda _seed: None)
    monkeypatch.setattr(nodes, "_runtime_provenance", lambda: {"runtime": "test"})

    features, coverage, similarities, diagnostics, manifest = nodes.run_invariance_check(
        panel,
        panel_manifest,
        smoke_manifest,
        _smoke_params(),
        _params(panel_hash),
        "candidate",
    )

    assert len(features) == 20
    assert len(coverage) == 20
    assert len(similarities) == 16
    assert set(features["variant_id"]) == {
        "original",
        "rotate_25",
        "rotate_50",
        "rotate_75",
        "reverse_complement",
    }
    assert np.allclose(similarities["cosine_to_original"], 1.0)
    assert diagnostics["status"] == "passed_invariance_check"
    assert diagnostics["coverage_pass"]
    assert diagnostics["geometry"]["passed_effective_rank"]
    assert manifest["decision"] == {
        "status": "passed_invariance_check",
        "candidate_selected": False,
        "retrieval_metrics_computed": False,
        "validation_outcomes_read": False,
        "test_rows_read": False,
    }
    readback = validate_invariance_outputs(
        features,
        coverage,
        similarities,
        diagnostics,
        manifest,
        expected_configuration=_params(panel_hash),
    )
    assert readback["status"] == "passed_independent_readback"
    assert readback["maximum_cosine_absolute_error"] == 0.0

    changed_similarities = similarities.copy()
    changed_similarities.loc[0, "cosine_to_original"] = 0.5
    with pytest.raises(ValueError, match="persisted cosine changed"):
        validate_invariance_outputs(
            features,
            coverage,
            changed_similarities,
            diagnostics,
            manifest,
            expected_configuration=_params(panel_hash),
        )

    changed_diagnostics = {**diagnostics, "geometry": {**diagnostics["geometry"]}}
    changed_diagnostics["geometry"]["effective_rank"] = 0.0
    changed_manifest = {**manifest, "diagnostic_summary": changed_diagnostics}
    with pytest.raises(ValueError, match="effective_rank differs"):
        validate_invariance_outputs(
            features,
            coverage,
            similarities,
            changed_diagnostics,
            changed_manifest,
            expected_configuration=_params(panel_hash),
        )

    for correlation_name in (
        "pairwise_cosine_length_difference_pearson",
        "pairwise_cosine_gc_difference_pearson",
    ):
        changed_diagnostics = {**diagnostics, "geometry": {**diagnostics["geometry"]}}
        changed_diagnostics["geometry"][correlation_name] += 0.1
        changed_manifest = {**manifest, "diagnostic_summary": changed_diagnostics}
        with pytest.raises(ValueError, match=f"{correlation_name} differs"):
            validate_invariance_outputs(
                features,
                coverage,
                similarities,
                changed_diagnostics,
                changed_manifest,
                expected_configuration=_params(panel_hash),
            )

    changed_manifest = {
        **manifest,
        "resolved_invariance_configuration": {
            **manifest["resolved_invariance_configuration"],
            "minimum_median_transform_cosine": 0.5,
        },
    }
    with pytest.raises(ValueError, match="frozen configuration"):
        validate_invariance_outputs(
            features,
            coverage,
            similarities,
            diagnostics,
            changed_manifest,
            expected_configuration=_params(panel_hash),
        )


def test_invariance_node_rejects_a_candidate_without_a_passing_smoke_manifest() -> None:
    panel = _panel()
    panel_hash = dataframe_content_sha256(
        panel,
        sort_columns=["sequence_id"],
        value_columns=fixed_representation.PANEL_HASH_COLUMNS,
    )

    with pytest.raises(ValueError, match="did not pass"):
        fixed_representation_invariance.validate_invariance_recipe(
            panel,
            _panel_manifest(panel_hash),
            _smoke_manifest("technical_failure"),
            _smoke_params(),
            _params(panel_hash),
            "candidate",
        )


def test_invariance_recipe_rejects_a_changed_recipe_after_smoke() -> None:
    panel = _panel()
    panel_hash = dataframe_content_sha256(
        panel,
        sort_columns=["sequence_id"],
        value_columns=fixed_representation.PANEL_HASH_COLUMNS,
    )
    changed_params = _smoke_params()
    changed_params["candidates"]["candidate"]["model_id"] = "organization/changed-model"

    with pytest.raises(ValueError, match="recipe does not match"):
        fixed_representation_invariance.validate_invariance_recipe(
            panel,
            _panel_manifest(panel_hash),
            _smoke_manifest(),
            changed_params,
            _params(panel_hash),
            "candidate",
        )


def test_invariance_recipe_uses_the_recorded_runtime_for_a_legacy_smoke_manifest() -> None:
    panel = _panel()
    panel_hash = dataframe_content_sha256(
        panel,
        sort_columns=["sequence_id"],
        value_columns=fixed_representation.PANEL_HASH_COLUMNS,
    )
    panel_manifest = _panel_manifest(panel_hash)
    smoke_manifest = _smoke_manifest()
    smoke_manifest["candidate"] = {**smoke_manifest["candidate"]}
    smoke_manifest["candidate"].pop("transformers_version")
    smoke_manifest["runtime"] = {"packages": {"transformers": "5.12.1"}}
    params = _params(panel_hash)
    params["accepted_smoke_artifacts"]["candidate"] = {
        **params["accepted_smoke_artifacts"]["candidate"],
        "smoke_manifest_sha256": fixed_representation_invariance.json_content_sha256(
            smoke_manifest
        ),
        "transformers_version_source": "runtime.packages.transformers",
    }

    recipe = fixed_representation_invariance.validate_invariance_recipe(
        panel,
        panel_manifest,
        smoke_manifest,
        _smoke_params(),
        params,
        "candidate",
    )

    assert recipe.transformers_version == "5.12.1"


@pytest.mark.parametrize("artifact_name", ["panel_manifest", "smoke_manifest"])
def test_invariance_recipe_rejects_smoke_content_without_the_accepted_hash(
    artifact_name: str,
) -> None:
    panel = _panel()
    panel_hash = dataframe_content_sha256(
        panel,
        sort_columns=["sequence_id"],
        value_columns=fixed_representation.PANEL_HASH_COLUMNS,
    )
    panel_manifest = _panel_manifest(panel_hash)
    smoke_manifest = _smoke_manifest()
    params = _params(panel_hash)
    if artifact_name == "panel_manifest":
        panel_manifest = {**panel_manifest, "unaccepted_attempt": "later-run"}
    else:
        smoke_manifest = {**smoke_manifest, "unaccepted_attempt": "later-run"}

    with pytest.raises(ValueError, match="accepted content hashes"):
        fixed_representation_invariance.validate_invariance_recipe(
            panel,
            panel_manifest,
            smoke_manifest,
            _smoke_params(),
            params,
            "candidate",
        )


@pytest.mark.parametrize("changed_field", ["sequence_sha256", "length_bp"])
def test_invariance_recipe_rejects_changed_panel_sequence_identity(changed_field: str) -> None:
    panel = _panel()
    panel_hash = dataframe_content_sha256(
        panel,
        sort_columns=["sequence_id"],
        value_columns=fixed_representation.PANEL_HASH_COLUMNS,
    )
    changed_panel = panel.copy()
    changed_panel.loc[0, changed_field] = "declared" if changed_field == "sequence_sha256" else 99

    with pytest.raises(ValueError, match="SHA-256 mismatch|length mismatch"):
        fixed_representation_invariance.validate_invariance_recipe(
            changed_panel,
            _panel_manifest(panel_hash),
            _smoke_manifest(),
            _smoke_params(),
            _params(panel_hash),
            "candidate",
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_invariance_compute_authorization_rejects_non_finite_values(value: float) -> None:
    params = _params("unused")
    params["compute_authorization"]["instance_hour_limit"] = value

    with pytest.raises(ValueError, match="must be finite"):
        fixed_representation_invariance.validated_compute_authorization(params)


def test_invariance_pipeline_and_versioned_readback_run_through_the_test_catalog(
    tmp_path, monkeypatch, capsys
) -> None:
    panel = _panel()
    panel_hash = dataframe_content_sha256(
        panel,
        sort_columns=["sequence_id"],
        value_columns=fixed_representation.PANEL_HASH_COLUMNS,
    )
    panel_manifest = _panel_manifest(panel_hash)
    smoke_manifest = _smoke_manifest()
    monkeypatch.setenv("VEC2VEC_TEST_ROOT", str(tmp_path))
    monkeypatch.setattr(nodes, "FrozenDnaEncoder", _FakeEncoder)
    monkeypatch.setattr(nodes, "_configure_determinism", lambda _seed: None)
    monkeypatch.setattr(nodes, "_runtime_provenance", lambda: {"runtime": "test"})
    bootstrap_project(PROJECT_ROOT)
    runtime_params = {
        "fixed_representation_smoke": _smoke_params(),
        "fixed_representation_invariance": _params(panel_hash),
        "fixed_representation_invariance_candidate": "candidate",
    }

    with KedroSession.create(
        project_path=PROJECT_ROOT,
        env="test",
        runtime_params=runtime_params,
    ) as session:
        catalog = session.load_context().catalog
        catalog.save("e02_fixed_representation_smoke_panel", panel)
        catalog.save("e02_fixed_representation_smoke_panel_manifest", panel_manifest)
        catalog.save("e02_fixed_representation_smoke_manifest", smoke_manifest)
        session.run(pipeline_name="fixed_representation_invariance")

        persisted_manifest = catalog.load("e02_fixed_representation_invariance_manifest")
        persisted_similarities = catalog.load("e02_fixed_representation_invariance_similarities")

    assert persisted_manifest["decision"]["status"] == "passed_invariance_check"
    assert len(persisted_similarities) == 16
    feature_root = tmp_path / "lake/04_feature/e02/fixed_representation_invariance_features.parquet"
    versions = sorted(path.name for path in feature_root.iterdir() if path.is_dir())
    assert len(versions) == 1
    capsys.readouterr()
    monkeypatch.setattr(
        validate_fixed_representation_invariance_artifacts,
        "_expected_configuration",
        lambda _all_params: _params(panel_hash),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_fixed_representation_invariance_artifacts.py",
            "--version",
            versions[0],
            "--env",
            "test",
        ],
    )
    validate_fixed_representation_invariance_artifacts.main()
    output = capsys.readouterr().out
    readback = json.loads(output[output.index("{") :])
    assert readback["status"] == "passed_independent_readback"
    assert readback["artifact_version"] == versions[0]


def test_git_provenance_uses_the_pipeline_worktree_from_an_unrelated_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=nodes.PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    provenance = nodes._git_provenance()

    assert provenance["commit"] == expected_commit
