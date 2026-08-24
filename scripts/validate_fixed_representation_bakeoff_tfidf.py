#!/usr/bin/env python3
"""Read and validate one persisted E02b TF-IDF/SVD feature version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fsspec
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from vec2vec.lib.fixed_representation_bakeoff_validation import validate_tfidf_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ARTIFACTS = {
    "features": "e02b_dna_features",
    "vocabulary": "e02b_tfidf_vocabulary",
    "svd_state": "e02b_tfidf_svd_state",
    "manifest": "e02b_dna_manifest",
}


def main() -> None:
    """Load one exact baseline version and print independent validation evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--env", default=None)
    arguments = parser.parse_args()
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
            for label, dataset_name in FEATURE_ARTIFACTS.items()
        }
        catalog_config, _, _, _ = catalog.to_config()
    report = validate_tfidf_features(
        pairs,
        input_manifest,
        artifacts["features"],
        artifacts["vocabulary"],
        artifacts["svd_state"],
        artifacts["manifest"],
        expected_candidate_id=str(configuration["tfidf"]["candidate_id"]),
        expected_dimension=int(configuration["tfidf"]["svd_components"]),
        expected_training_rows=int(configuration["training_rows"]),
    )
    sizes = {
        label: _file_size(_versioned_path(str(catalog_config[name]["filepath"]), version))
        for label, name in FEATURE_ARTIFACTS.items()
    }
    sizes["total"] = sum(sizes.values())
    report["artifact_version"] = version
    report["persisted_artifact_bytes"] = sizes
    report["accepted_feature_artifact"] = {
        "version": version,
        "manifest_sha256": report["manifest_sha256"],
        "features_sha256": report["output_hashes"]["features_sha256"],
        "extraction_gpu_hours": 0.0,
        "persisted_bytes": sizes["total"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _versioned_path(filepath: str, version: str) -> str:
    filename = filepath.rstrip("/").rsplit("/", 1)[-1]
    return f"{filepath.rstrip('/')}/{version}/{filename}"


def _file_size(path: str) -> int:
    filesystem, filesystem_path = fsspec.core.url_to_fs(path)
    return int(filesystem.info(filesystem_path)["size"])


if __name__ == "__main__":
    main()
