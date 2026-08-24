#!/usr/bin/env python3
"""Read and independently validate one persisted E02b input artifact version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fsspec
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from vec2vec.lib.fixed_representation_bakeoff_validation import validate_bakeoff_inputs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_NAMES = {
    "pairs": "e02b_pairs",
    "exclusions": "e02b_exclusions",
    "queries": "e02b_queries",
    "query_states": "e02b_query_states",
    "manifest": "e02b_input_manifest",
}


def main() -> None:
    """Load one exact version and print its independent validation report."""
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
        catalog = context.catalog
        artifacts = {
            label: catalog.load(dataset_name, version=version)
            for label, dataset_name in ARTIFACT_NAMES.items()
        }
        catalog_config, _, _, _ = catalog.to_config()
    report = validate_bakeoff_inputs(
        artifacts["pairs"],
        artifacts["exclusions"],
        artifacts["queries"],
        artifacts["query_states"],
        artifacts["manifest"],
        expected_protocol_version=str(configuration["protocol_version"]),
        expected_training_rows=int(configuration["training_rows"]),
    )
    sizes = {
        label: _file_size(_versioned_path(str(catalog_config[name]["filepath"]), version))
        for label, name in ARTIFACT_NAMES.items()
    }
    sizes["total"] = sum(sizes.values())
    report["artifact_version"] = version
    report["persisted_artifact_bytes"] = sizes
    print(json.dumps(report, indent=2, sort_keys=True))


def _versioned_path(filepath: str, version: str) -> str:
    filename = filepath.rstrip("/").rsplit("/", 1)[-1]
    return f"{filepath.rstrip('/')}/{version}/{filename}"


def _file_size(path: str) -> int:
    filesystem, filesystem_path = fsspec.core.url_to_fs(path)
    return int(filesystem.info(filesystem_path)["size"])


if __name__ == "__main__":
    main()
