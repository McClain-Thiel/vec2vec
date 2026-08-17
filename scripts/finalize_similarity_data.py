"""Validate graph and split, then build and validate the frozen Gate 0 benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import fsspec
import pandas as pd

from vec2vec.lib.query_benchmark_validation import validate_query_benchmark_outputs
from vec2vec.lib.similarity_graph_validation import validate_similarity_graph_outputs
from vec2vec.lib.similarity_split_validation import validate_similarity_split_outputs

BUCKET = "plasmidclip"
RETRIEVAL_VERSION = "2026-08-04T09.02.10.007Z"
CONSTRAINT_STATE_VERSION = "2026-08-06T13.27.47.937Z"
POPULATION_SHA256 = "7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5"
EXPECTED_ROWS = 115_120
BACKUP_PREFIX = (
    "s3://plasmidclip/research-backups/vec2vec/e00/global_similarity_graph_v0.1/"
    "2026-08-17T08-48-41Z/validation"
)
SCRATCH_ROOT = Path("data/09_scratch/similarity_graph_calibration")

GRAPH_ARTIFACTS = {
    "edges": ("kedro/08_reporting/e00", "similarity_graph_edges.parquet"),
    "nodes": ("kedro/08_reporting/e00", "similarity_graph_nodes.parquet"),
    "components": ("kedro/08_reporting/e00", "similarity_graph_components.parquet"),
    "query_profile": ("kedro/08_reporting/e00", "similarity_graph_query_profile.parquet"),
    "runs": ("kedro/08_reporting/e00", "similarity_graph_runs.parquet"),
    "manifest": ("kedro/08_reporting/e00", "similarity_graph_manifest.json"),
}

SPLIT_ARTIFACTS = {
    "mapping": ("kedro/04_feature/e00", "split_grouped_v2.parquet"),
    "components": ("kedro/08_reporting/e00", "split_grouped_v2_components.parquet"),
    "cross_edges": ("kedro/08_reporting/e00", "split_grouped_v2_cross_edges.parquet"),
    "manifest": ("kedro/08_reporting/e00", "split_grouped_v2_manifest.json"),
}

BENCHMARK_ARTIFACTS = {
    "query_catalog": ("kedro/05_model_input/e00", "query_catalog.parquet"),
    "galleries": ("kedro/05_model_input/e00", "candidate_galleries.parquet"),
    "query_states": ("kedro/05_model_input/e00", "query_candidate_state.parquet"),
    "base_masses": ("kedro/05_model_input/e00", "candidate_base_mass.parquet"),
    "rankings": ("kedro/08_reporting/e00", "benchmark_control_rankings.parquet"),
    "metrics": ("kedro/08_reporting/e00", "benchmark_control_metrics.parquet"),
    "manifest": ("kedro/08_reporting/e00", "query_benchmark_manifest.json"),
}

CONSTRAINT_STATE_ARTIFACT = (
    "kedro/05_model_input/e00",
    "plasmid_constraint_state.parquet",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-version")
    args = parser.parse_args()
    filesystem = fsspec.filesystem("s3")

    graph_version = args.graph_version or _latest_common_version(filesystem, GRAPH_ARTIFACTS)
    graph_tables = {
        name: pd.read_parquet(_uri(root, filename, graph_version))
        for name, (root, filename) in GRAPH_ARTIFACTS.items()
        if name != "manifest"
    }
    graph_manifest = _load_json(
        filesystem,
        _key(*GRAPH_ARTIFACTS["manifest"], graph_version),
    )
    graph_report = validate_similarity_graph_outputs(
        graph_tables["edges"],
        graph_tables["nodes"],
        graph_tables["components"],
        graph_tables["query_profile"],
        graph_tables["runs"],
        graph_manifest,
        expected_rows=EXPECTED_ROWS,
        expected_population_sha256=POPULATION_SHA256,
        expected_retrieval_version=RETRIEVAL_VERSION,
    )
    graph_report["artifact_version"] = graph_version
    graph_report_path = _write_and_upload_report(
        "similarity_graph_readback",
        graph_version,
        graph_report,
    )

    graph_load_versions = ",".join(
        [
            f"retrieval_dataset@split_audit:{RETRIEVAL_VERSION}",
            f"e00_similarity_graph_nodes:{graph_version}",
            f"e00_similarity_graph_edges:{graph_version}",
            f"e00_similarity_graph_manifest:{graph_version}",
        ]
    )
    subprocess.run(
        [
            "./.venv/bin/kedro",
            "run",
            "--pipeline",
            "similarity_split",
            "--load-versions",
            graph_load_versions,
            "--params",
            f"similarity_split.input_graph_artifact_version={graph_version}",
        ],
        check=True,
    )

    split_version = _latest_common_version(filesystem, SPLIT_ARTIFACTS)
    split_tables = {
        name: pd.read_parquet(_uri(root, filename, split_version))
        for name, (root, filename) in SPLIT_ARTIFACTS.items()
        if name != "manifest"
    }
    split_manifest = _load_json(
        filesystem,
        _key(*SPLIT_ARTIFACTS["manifest"], split_version),
    )
    split_report = validate_similarity_split_outputs(
        split_tables["mapping"],
        split_tables["components"],
        split_tables["cross_edges"],
        split_manifest,
        expected_rows=EXPECTED_ROWS,
        expected_graph_artifact_version=graph_version,
    )
    split_report["artifact_version"] = split_version
    split_report["input_graph_artifact_version"] = graph_version
    split_report_path = _write_and_upload_report(
        "split_grouped_v2_readback",
        split_version,
        split_report,
    )

    benchmark_load_versions = ",".join(
        [
            f"retrieval_dataset@query_benchmark:{RETRIEVAL_VERSION}",
            f"e00_split_grouped_v2:{split_version}",
            f"e00_similarity_graph_edges:{graph_version}",
            f"e00_similarity_graph_manifest:{graph_version}",
            f"e00_split_grouped_v2_manifest:{split_version}",
            f"e00_constraint_vocabulary:{CONSTRAINT_STATE_VERSION}",
            f"e00_plasmid_constraint_state:{CONSTRAINT_STATE_VERSION}",
            f"e00_constraint_state_manifest:{CONSTRAINT_STATE_VERSION}",
        ]
    )
    benchmark_params = ",".join(
        [
            f"query_benchmark.input_graph_artifact_version={graph_version}",
            f"query_benchmark.input_split_artifact_version={split_version}",
        ]
    )
    subprocess.run(
        [
            "./.venv/bin/kedro",
            "run",
            "--pipeline",
            "query_benchmark",
            "--load-versions",
            benchmark_load_versions,
            "--params",
            benchmark_params,
        ],
        check=True,
    )

    benchmark_version = _latest_common_version(filesystem, BENCHMARK_ARTIFACTS)
    benchmark_tables = {
        name: pd.read_parquet(_uri(root, filename, benchmark_version))
        for name, (root, filename) in BENCHMARK_ARTIFACTS.items()
        if name != "manifest"
    }
    benchmark_manifest = _load_json(
        filesystem,
        _key(*BENCHMARK_ARTIFACTS["manifest"], benchmark_version),
    )
    source_states = pd.read_parquet(_uri(*CONSTRAINT_STATE_ARTIFACT, CONSTRAINT_STATE_VERSION))
    benchmark_report = validate_query_benchmark_outputs(
        benchmark_tables["query_catalog"],
        benchmark_tables["galleries"],
        benchmark_tables["query_states"],
        benchmark_tables["base_masses"],
        benchmark_tables["rankings"],
        benchmark_tables["metrics"],
        benchmark_manifest,
        source_states,
        expected_rows=EXPECTED_ROWS,
        expected_retrieval_version=RETRIEVAL_VERSION,
        expected_graph_artifact_version=graph_version,
        expected_split_artifact_version=split_version,
        expected_constraint_state_artifact_version=CONSTRAINT_STATE_VERSION,
    )
    benchmark_report["artifact_version"] = benchmark_version
    benchmark_report_path = _write_and_upload_report(
        "query_benchmark_readback",
        benchmark_version,
        benchmark_report,
    )

    print(
        json.dumps(
            {
                "status": "gate0_data_complete",
                "graph_artifact_version": graph_version,
                "graph_validation_report": str(graph_report_path),
                "split_artifact_version": split_version,
                "split_validation_report": str(split_report_path),
                "query_benchmark_artifact_version": benchmark_version,
                "query_benchmark_validation_report": str(benchmark_report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _latest_common_version(filesystem: Any, artifacts: dict[str, tuple[str, str]]) -> str:
    versions = [
        _artifact_versions(filesystem, root, filename) for root, filename in artifacts.values()
    ]
    common = set.intersection(*versions)
    if not common:
        counts = [len(values) for values in versions]
        raise RuntimeError(f"artifacts do not have a common version; version_counts={counts}")
    return max(common)


def _artifact_versions(filesystem: Any, root: str, filename: str) -> set[str]:
    matches = filesystem.glob(f"{BUCKET}/{root}/{filename}/*/{filename}")
    return {Path(match).parts[-2] for match in matches}


def _uri(root: str, filename: str, version: str) -> str:
    return f"s3://{BUCKET}/{_key(root, filename, version)}"


def _key(root: str, filename: str, version: str) -> str:
    return f"{root}/{filename}/{version}/{filename}"


def _load_json(filesystem: Any, key: str) -> dict[str, Any]:
    with filesystem.open(f"{BUCKET}/{key}", mode="rt") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact is not an object: {key}")
    return value


def _write_and_upload_report(name: str, version: str, report: dict[str, Any]) -> Path:
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    safe_version = version.replace(":", "-")
    path = SCRATCH_ROOT / f"{name}_{safe_version}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(path),
            f"{BACKUP_PREFIX}/{path.name}",
            "--sse",
            "AES256",
            "--checksum-algorithm",
            "SHA256",
            "--no-progress",
        ],
        check=True,
    )
    return path


if __name__ == "__main__":
    main()
