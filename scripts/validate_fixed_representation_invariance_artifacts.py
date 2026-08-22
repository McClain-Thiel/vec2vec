#!/usr/bin/env python3
"""Read versioned Gate 1 invariance outputs and print an independent validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fsspec
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from vec2vec.lib.fixed_representation_validation import validate_invariance_outputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = {
    "features": "e02_fixed_representation_invariance_features",
    "coverage": "e02_fixed_representation_invariance_coverage",
    "similarities": "e02_fixed_representation_invariance_similarities",
    "diagnostics": "e02_fixed_representation_invariance_diagnostics",
    "manifest": "e02_fixed_representation_invariance_manifest",
}


def main() -> None:
    """Load one exact candidate version and validate its persisted contents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--env",
        default=None,
        help="Optional Kedro environment. Use test only with local test artifacts.",
    )
    arguments = parser.parse_args()
    version = str(arguments.version)
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(
        project_path=PROJECT_ROOT,
        env=arguments.env,
        save_on_close=False,
    ) as session:
        context = session.load_context()
        catalog = context.catalog
        expected_configuration = _expected_configuration(context.params)
        artifacts = {
            label: catalog.load(dataset_name, version=version)
            for label, dataset_name in ARTIFACT_NAMES.items()
        }
        catalog_config, _, _, _ = catalog.to_config()
    report = validate_invariance_outputs(
        artifacts["features"],
        artifacts["coverage"],
        artifacts["similarities"],
        artifacts["diagnostics"],
        artifacts["manifest"],
        expected_configuration=expected_configuration,
    )
    report["artifact_version"] = version
    report["persisted_artifact_bytes"] = {
        label: _file_size(_versioned_path(str(catalog_config[dataset_name]["filepath"]), version))
        for label, dataset_name in ARTIFACT_NAMES.items()
    }
    report["persisted_artifact_bytes"]["total"] = sum(report["persisted_artifact_bytes"].values())
    print(json.dumps(report, indent=2, sort_keys=True))


def _expected_configuration(params: dict[str, object]) -> dict[str, object]:
    configuration = params.get("fixed_representation_invariance")
    if not isinstance(configuration, dict):
        raise ValueError("fixed_representation_invariance configuration must be a mapping")
    return configuration


def _versioned_path(filepath: str, version: str) -> str:
    filename = filepath.rstrip("/").rsplit("/", 1)[-1]
    return f"{filepath.rstrip('/')}/{version}/{filename}"


def _file_size(path: str) -> int:
    filesystem, filesystem_path = fsspec.core.url_to_fs(path)
    return int(filesystem.info(filesystem_path)["size"])


if __name__ == "__main__":
    main()
