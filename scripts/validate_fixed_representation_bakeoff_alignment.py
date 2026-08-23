#!/usr/bin/env python3
"""Read and independently validate one persisted E02b alignment version."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fsspec
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from vec2vec.lib import fixed_representation_bakeoff
from vec2vec.lib.fixed_representation_alignment_validation import (
    validate_alignment_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DATASETS = {
    "whitening_state": "e02b_whitening_state",
    "probe_checkpoints": "e02b_probe_checkpoints",
    "training_history": "e02b_training_history",
    "paired_metrics": "e02b_paired_metrics",
    "query_rankings": "e02b_query_rankings",
    "query_metrics": "e02b_query_metrics",
    "query_summaries": "e02b_query_summaries",
    "bootstrap_draws": "e02b_bootstrap_draws",
    "selection_report": "e02b_selection_report",
}
FEATURE_DATASETS = {
    "dna": {
        "tfidf_6mer_svd_512": "e02b_dna_features_tfidf_6mer_svd_512",
        "carbon_500m": "e02b_dna_features_carbon_500m",
        "generanno_prokaryote_500m": "e02b_dna_features_generanno_prokaryote_500m",
        "generator_v2_prokaryote_1_2b": "e02b_dna_features_generator_v2_prokaryote_1_2b",
    },
    "text": {
        "bge_base_en_v1_5": "e02b_text_features_bge_base_en_v1_5",
        "gte_modernbert_base": "e02b_text_features_gte_modernbert_base",
        "qwen3_embedding_0_6b": "e02b_text_features_qwen3_embedding_0_6b",
    },
}


def main() -> None:
    """Load one exact alignment version and print its acceptance record."""
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
        queries = catalog.load("e02b_queries", version=input_version)
        query_states = catalog.load("e02b_query_states", version=input_version)
        input_manifest = catalog.load("e02b_input_manifest", version=input_version)
        outputs = {
            label: catalog.load(dataset_name, version=version)
            for label, dataset_name in OUTPUT_DATASETS.items()
        }
        features = {
            feature_kind: {
                candidate_id: catalog.load(
                    dataset_name,
                    version=str(
                        configuration["accepted_feature_artifacts"][feature_kind][candidate_id][
                            "version"
                        ]
                    ),
                )
                for candidate_id, dataset_name in datasets.items()
            }
            for feature_kind, datasets in FEATURE_DATASETS.items()
        }
        catalog_config, _, _, _ = catalog.to_config()

    report = validate_alignment_outputs(
        pairs,
        queries,
        query_states,
        input_manifest,
        features["dna"],
        features["text"],
        outputs["whitening_state"],
        outputs["probe_checkpoints"],
        outputs["training_history"],
        outputs["paired_metrics"],
        outputs["query_rankings"],
        outputs["query_metrics"],
        outputs["query_summaries"],
        outputs["bootstrap_draws"],
        outputs["selection_report"],
        params=configuration,
        expected_compute_authorization=fixed_representation_bakeoff.approved_compute_authorization(
            configuration, stage="alignment_probe"
        ),
    )
    sizes = {
        label: _file_size(_versioned_path(str(catalog_config[name]["filepath"]), version))
        for label, name in OUTPUT_DATASETS.items()
    }
    sizes["total"] = sum(sizes.values())
    report["artifact_version"] = version
    report["persisted_artifact_bytes"] = sizes
    report["accepted_alignment_artifact"] = {
        "version": version,
        "selection_report_sha256": report["selection_report_sha256"],
        "output_hashes": report["output_hashes"],
        "persisted_bytes": sizes["total"],
        "selected_pair": report["selected_pair"],
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
