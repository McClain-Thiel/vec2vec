"""Read the versioned S3 graph outputs and print an independent validation report."""

from __future__ import annotations

import argparse
import json

import fsspec
import pandas as pd

from vec2vec.lib.similarity_graph_validation import validate_similarity_graph_outputs

BUCKET_ROOT = "s3://plasmidclip/kedro/08_reporting/e00"
EXPECTED_ROWS = 115_120
EXPECTED_POPULATION_SHA256 = "7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5"
EXPECTED_RETRIEVAL_VERSION = "2026-08-04T09.02.10.007Z"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    version = str(args.version)

    tables = {
        name: pd.read_parquet(_artifact_path(name, "parquet", version))
        for name in ("edges", "nodes", "components", "query_profile", "runs")
    }
    with fsspec.open(_artifact_path("manifest", "json", version), mode="rt") as handle:
        manifest = json.load(handle)
    report = validate_similarity_graph_outputs(
        tables["edges"],
        tables["nodes"],
        tables["components"],
        tables["query_profile"],
        tables["runs"],
        manifest,
        expected_rows=EXPECTED_ROWS,
        expected_population_sha256=EXPECTED_POPULATION_SHA256,
        expected_retrieval_version=EXPECTED_RETRIEVAL_VERSION,
    )
    report["artifact_version"] = version
    print(json.dumps(report, indent=2, sort_keys=True))


def _artifact_path(name: str, extension: str, version: str) -> str:
    filename = f"similarity_graph_{name}.{extension}"
    return f"{BUCKET_ROOT}/{filename}/{version}/{filename}"


if __name__ == "__main__":
    main()
