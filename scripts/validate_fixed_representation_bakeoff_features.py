#!/usr/bin/env python3
"""Read and independently validate one persisted E02b neural feature version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fsspec
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from vec2vec.lib import fixed_representation_bakeoff
from vec2vec.lib.fixed_representation_bakeoff_validation import (
    validate_neural_dna_features,
    validate_text_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = {
    "dna": {
        "features": "e02b_dna_features",
        "coverage": "e02b_dna_coverage",
        "manifest": "e02b_dna_manifest",
    },
    "text": {
        "features": "e02b_text_features",
        "manifest": "e02b_text_manifest",
    },
}


def main() -> None:
    """Load one exact feature version and print its acceptance record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(ARTIFACT_NAMES), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--env", default=None)
    arguments = parser.parse_args()
    kind = str(arguments.kind)
    candidate_id = str(arguments.candidate)
    version = str(arguments.version)
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(
        project_path=PROJECT_ROOT,
        env=arguments.env,
        save_on_close=False,
    ) as session:
        context = session.load_context()
        configuration = context.params["fixed_representation_bakeoff"]
        input_version = str(configuration["accepted_input_artifact"]["version"])
        catalog = context.catalog
        pairs = catalog.load("e02b_pairs", version=input_version)
        input_manifest = catalog.load("e02b_input_manifest", version=input_version)
        artifacts = {
            label: catalog.load(dataset_name, version=version)
            for label, dataset_name in ARTIFACT_NAMES[kind].items()
        }
        queries = catalog.load("e02b_queries", version=input_version) if kind == "text" else None
        invariance_manifest = (
            _load_invariance_manifest(catalog, configuration, candidate_id)
            if kind == "dna"
            else None
        )
        smoke_manifest = (
            _load_smoke_manifest(catalog, invariance_manifest) if kind == "dna" else None
        )
        catalog_config, _, _, _ = catalog.to_config()

    expected_configuration = {
        "device": str(configuration["device"]),
        "precision": str(configuration["precision"]),
        "window_overlap_fraction": float(configuration["window_overlap_fraction"]),
        "seed": int(configuration["seed"]),
    }
    if kind == "dna":
        accepted_invariance = configuration["accepted_invariance_artifacts"][candidate_id]
        expected_compute = _expected_compute_authorization(
            configuration, kind=kind, candidate_id=candidate_id
        )
        report = validate_neural_dna_features(
            pairs,
            input_manifest,
            invariance_manifest,
            smoke_manifest,
            artifacts["features"],
            artifacts["coverage"],
            artifacts["manifest"],
            expected_candidate_id=candidate_id,
            expected_candidate=dict(configuration["dna_candidates"][candidate_id]),
            expected_invariance_manifest_sha256=str(accepted_invariance["manifest_sha256"]),
            expected_configuration=expected_configuration,
            accepted_input_artifact=dict(configuration["accepted_input_artifact"]),
            expected_compute_authorization=expected_compute,
        )
    else:
        if queries is None:
            raise RuntimeError("text validation did not load E02b queries")
        expected_compute = _expected_compute_authorization(
            configuration, kind=kind, candidate_id=candidate_id
        )
        report = validate_text_features(
            pairs,
            queries,
            input_manifest,
            artifacts["features"],
            artifacts["manifest"],
            expected_candidate_id=candidate_id,
            expected_candidate=dict(configuration["text_candidates"][candidate_id]),
            expected_configuration=expected_configuration,
            accepted_input_artifact=dict(configuration["accepted_input_artifact"]),
            expected_compute_authorization=expected_compute,
        )
    sizes = {
        label: _file_size(_versioned_path(str(catalog_config[name]["filepath"]), version))
        for label, name in ARTIFACT_NAMES[kind].items()
    }
    sizes["total"] = sum(sizes.values())
    report["artifact_version"] = version
    report["persisted_artifact_bytes"] = sizes
    report["accepted_feature_artifact"] = {
        "version": version,
        "manifest_sha256": report["manifest_sha256"],
        "features_sha256": report["output_hashes"]["features_sha256"],
        "extraction_gpu_hours": report["extraction_gpu_hours"],
        "persisted_bytes": sizes["total"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _expected_compute_authorization(
    configuration: dict[str, Any], *, kind: str, candidate_id: str
) -> dict[str, Any]:
    accepted = configuration.get("accepted_feature_artifacts", {}).get(kind, {}).get(candidate_id)
    if isinstance(accepted, dict) and isinstance(accepted.get("compute_authorization"), dict):
        return dict(accepted["compute_authorization"])
    return fixed_representation_bakeoff.approved_compute_authorization(
        configuration, stage=f"{kind}_features:{candidate_id}"
    )


def _load_invariance_manifest(
    catalog: Any,
    configuration: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    accepted = configuration.get("accepted_invariance_artifacts", {}).get(candidate_id)
    if not isinstance(accepted, dict) or not str(accepted.get("version", "")):
        raise ValueError(f"candidate {candidate_id} lacks an accepted invariance version")
    return catalog.load(
        "e02_fixed_representation_invariance_manifest", version=str(accepted["version"])
    )


def _load_smoke_manifest(catalog: Any, invariance_manifest: dict[str, Any]) -> dict[str, Any]:
    accepted = invariance_manifest.get("accepted_numerical_smoke_artifact")
    if not isinstance(accepted, dict) or not str(accepted.get("configured_version", "")):
        raise ValueError("accepted invariance manifest lacks its numerical-smoke version")
    return catalog.load(
        "e02_fixed_representation_smoke_manifest",
        version=str(accepted["configured_version"]),
    )


def _versioned_path(filepath: str, version: str) -> str:
    filename = filepath.rstrip("/").rsplit("/", 1)[-1]
    return f"{filepath.rstrip('/')}/{version}/{filename}"


def _file_size(path: str) -> int:
    filesystem, filesystem_path = fsspec.core.url_to_fs(path)
    return int(filesystem.info(filesystem_path)["size"])


if __name__ == "__main__":
    main()
