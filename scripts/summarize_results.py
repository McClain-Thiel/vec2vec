"""Regenerate the compact vec2vec evidence tables from accepted S3 reports."""

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import fsspec
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

E02B_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e02b/selection_report.json/"
    "2026-08-24T16.34.48.358Z/selection_report.json"
)
GATE2_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e03e04/report.json/2026-08-25T10.52.39.447Z/report.json"
)
E05_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e05/composition_report.json/"
    "2026-08-26T11.49.24.525Z/composition_report.json"
)
E06_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e06/composition_report.json/"
    "2026-08-26T13.06.51.803Z/composition_report.json"
)
E07_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e07/additive_report.json/"
    "2026-08-27T15.33.47.015Z/additive_report.json"
)
E08_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e08/natural_parameter_report.json/"
    "2026-08-27T16.17.31.598Z/natural_parameter_report.json"
)
E09_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e09/natural_parameter_report.json/"
    "2026-08-27T17.30.52.908Z/natural_parameter_report.json"
)
E10_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e10/report.json/2026-08-27T18.46.12.234Z/report.json"
)
E11_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e11/atomic_classifier_report.json/"
    "2026-08-28T10.19.38.951Z/atomic_classifier_report.json"
)
E12_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e12/compositional_report.json/"
    "2026-08-28T11.09.04.400Z/compositional_report.json"
)
E13_REPORT = (
    "s3://plasmidclip/kedro/08_reporting/e13/report.json/2026-08-28T11.22.25.340Z/report.json"
)
REPORT_HASHES = {
    E02B_REPORT: "a675a3a3fac1b87827749764caeea07a395debf86c0ee886998417fd9a5b8d25",
    GATE2_REPORT: "32914b0f7dec4a0c9c893dbb0eecd80a16dce049d46e642405f22f61cbbee9b2",
    E05_REPORT: "182fb0dd75a1bd3159bc24b488e4921ff7dac1c6292352a408c8ff6d2d44082d",
    E06_REPORT: "800e23597f209197835b9648c4663cc3d35e686d0ac59e04c50bf1838e015230",
    E07_REPORT: "a9a1ca17eab82fca3c4774326013034552dbf34583239fddcb19c5017515e275",
    E08_REPORT: "178eb98d922296c89839c536f84773453c60fa0768eb1205d53815a0736da4c7",
    E09_REPORT: "f1e844788ed5bc6b8f063e991cee2f50094f0c5c9bb0bef8680e60952ea86bab",
    E10_REPORT: "3b84fd0a20a377165a15f470e17365400756c5790526cef09169cb5d8b04f6b8",
    E11_REPORT: "8ea38fa831993052fa5f71b6dd66bb107d5092febc8172b25985dda81626e4c4",
    E12_REPORT: "f3b8b23bca6b651aaaf59c0b775589448e854480556c65deeddbf95f63d96638",
    E13_REPORT: "cf7b2352d6dab807bb63add83742f073348d488ce3b7ab8138136a9113dd0011",
}


def _read_report(path, expected_sha256):
    with fsspec.open(path, "rb") as handle:
        content = handle.read()
    report = json.loads(content)
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    observed = hashlib.sha256(canonical).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"report hash mismatch for {path}: {observed}")
    return report


def _encoder_rows(report):
    selected = report["selection"]["selected_pair"]
    artifacts = report["accepted_feature_artifacts"]
    rows = []
    for rank, result in enumerate(report["selection"]["leaderboard"], start=1):
        dna = result["dna_candidate_id"]
        text = result["text_candidate_id"]
        rows.append(
            {
                "rank": rank,
                "dna_encoder": dna,
                "text_encoder": text,
                "utility_at_10": result["mean_utility_at_10"],
                "interval_lower": result["bootstrap_lower"],
                "interval_upper": result["bootstrap_upper"],
                "dna_version": artifacts["dna"][dna]["version"],
                "text_version": artifacts["text"][text]["version"],
                "selected": dna == selected["dna_candidate_id"]
                and text == selected["text_candidate_id"],
            }
        )
    return rows


def _supervision_rows(report):
    result = report["comparison"]
    tracking = report["tracking"]
    urls = {
        objective: " ".join(row["url"] for row in tracking if row["objective"] == objective)
        for objective in ("paired_identity", "verified_set")
    }
    return [
        {
            "objective": "paired_identity",
            "pair_utility_at_10": result["paired_identity"],
            "difference_vs_paired": 0.0,
            "difference_interval_lower": None,
            "difference_interval_upper": None,
            "decision": "baseline",
            "wandb_runs": urls["paired_identity"],
        },
        {
            "objective": "verified_set",
            "pair_utility_at_10": result["verified_set"],
            "difference_vs_paired": result["verified_set_minus_paired_identity"],
            "difference_interval_lower": result["paired_component_bootstrap_95_interval"][0],
            "difference_interval_upper": result["paired_component_bootstrap_95_interval"][1],
            "decision": "accepted" if result["supports_set_supervision"] else "rejected",
            "wandb_runs": urls["verified_set"],
        },
    ]


def _composition_rows(report, *, experiment=None):
    rows = []
    for row in _supervision_rows(report):
        result = {
            "objective": row["objective"],
            "unseen_pair_utility_at_10": row["pair_utility_at_10"],
            "difference_vs_paired": row["difference_vs_paired"],
            "difference_interval_lower": row["difference_interval_lower"],
            "difference_interval_upper": row["difference_interval_upper"],
            "decision": row["decision"],
            "wandb_runs": row["wandb_runs"],
        }
        if experiment is not None:
            result = {
                "experiment": experiment,
                "training_rows": report["population"]["training_rows"],
                "query_representation": "direct_text",
                **result,
            }
        rows.append(result)
    return rows


def _additive_rows(report):
    comparison = report["additive_comparison"]
    return [
        {
            "direct_text_utility_at_10": comparison["direct_text"],
            "atomic_sum_utility_at_10": comparison["atomic_sum"],
            "atomic_sum_minus_direct_text": comparison["atomic_sum_minus_direct_text"],
            "difference_interval_lower": comparison["paired_component_bootstrap_95_interval"][0],
            "difference_interval_upper": comparison["paired_component_bootstrap_95_interval"][1],
            "atomic_sum_interval_lower": comparison["atomic_sum_component_bootstrap_95_interval"][
                0
            ],
            "atomic_sum_interval_upper": comparison["atomic_sum_component_bootstrap_95_interval"][
                1
            ],
            "mean_jensen_shannon_divergence": comparison["mean_jensen_shannon_divergence"],
        }
    ]


def _additive_composition_rows(report):
    comparison = report["additive_comparison"]
    urls = " ".join(row["url"] for row in report["tracking"] if row["objective"] == "verified_set")
    common = {
        "experiment": "E07",
        "training_rows": report["population"]["training_rows"],
        "objective": "verified_set",
    }
    return [
        {
            **common,
            "query_representation": "direct_text",
            "unseen_pair_utility_at_10": comparison["direct_text"],
            "difference_vs_paired": 0.0,
            "difference_interval_lower": None,
            "difference_interval_upper": None,
            "decision": "baseline",
            "wandb_runs": urls,
        },
        {
            **common,
            "query_representation": "atomic_sum",
            "unseen_pair_utility_at_10": comparison["atomic_sum"],
            "difference_vs_paired": comparison["atomic_sum_minus_direct_text"],
            "difference_interval_lower": comparison["paired_component_bootstrap_95_interval"][0],
            "difference_interval_upper": comparison["paired_component_bootstrap_95_interval"][1],
            "decision": "positive_not_distinguishable_from_direct",
            "wandb_runs": urls,
        },
    ]


def _natural_parameter_rows(report, *, experiment, decision, learning_rate):
    comparison = report["comparison"]
    rows = []
    for base_measure in ("uniform_plasmid", "uniform_v2_component"):
        atomic_sum = comparison[base_measure]
        additive = comparison["direct_vs_atomic_sum"][base_measure]
        common = {
            "experiment": experiment,
            "learning_rate": learning_rate,
            "base_measure": base_measure,
            "query_count": report["population"]["evaluation_queries"],
            "decision": decision,
        }
        rows.extend(
            [
                {
                    **common,
                    "query_representation": "direct_text",
                    "utility_at_10": atomic_sum - additive["atomic_sum_minus_direct_text"],
                    "reference_representation": "",
                    "difference_from_reference": 0.0,
                    "difference_interval_lower": None,
                    "difference_interval_upper": None,
                },
                {
                    **common,
                    "query_representation": "atomic_sum",
                    "utility_at_10": atomic_sum,
                    "reference_representation": "direct_text",
                    "difference_from_reference": additive["atomic_sum_minus_direct_text"],
                    "difference_interval_lower": additive["paired_component_bootstrap_95_interval"][
                        0
                    ],
                    "difference_interval_upper": additive["paired_component_bootstrap_95_interval"][
                        1
                    ],
                },
            ]
        )
    return rows


def _weak_annotation_rows(report):
    comparison = report["comparison"]
    common = {
        "experiment": "E10",
        "learning_rate": report["resolved_configuration"]["probe"]["learning_rate"],
        "base_measure": "uniform_v2_component",
        "query_count": report["population"]["held_out_conjunction_queries"],
        "decision": "exploratory_weak_labels",
    }
    return [
        {
            **common,
            "query_representation": "direct_text",
            "utility_at_10": comparison["direct_text"],
            "reference_representation": "",
            "difference_from_reference": 0.0,
            "difference_interval_lower": None,
            "difference_interval_upper": None,
        },
        {
            **common,
            "query_representation": "atomic_sum",
            "utility_at_10": comparison["atomic_sum"],
            "reference_representation": "direct_text",
            "difference_from_reference": comparison["atomic_sum_minus_direct_text"],
            "difference_interval_lower": comparison["difference_95_interval"][0],
            "difference_interval_upper": comparison["difference_95_interval"][1],
        },
    ]


def _atomic_classifier_rows(report):
    comparison = report["comparison"]
    common = {
        "experiment": "E11",
        "learning_rate": report["resolved_configuration"]["classifier"]["learning_rate"],
        "base_measure": "empirical_training_prevalence",
        "query_count": report["population"]["held_out_conjunction_queries"],
    }
    rows = []
    for representation, values in comparison["e11_representations"].items():
        primary = representation == comparison["primary_representation"]
        rows.append(
            {
                **common,
                "decision": "preregistered_primary" if primary else "diagnostic",
                "query_representation": representation,
                "utility_at_10": values["utility_at_10"],
                "reference_representation": "E10_atomic_sum" if primary else "",
                "difference_from_reference": (comparison["primary_minus_e10"] if primary else None),
                "difference_interval_lower": (
                    comparison["primary_minus_e10_query_bootstrap_95_interval"][0]
                    if primary
                    else None
                ),
                "difference_interval_upper": (
                    comparison["primary_minus_e10_query_bootstrap_95_interval"][1]
                    if primary
                    else None
                ),
            }
        )
    return rows


def _compositional_measurement_rows(report):
    primary = report["summary"]["primary_k"]
    return [
        {
            "experiment": "E12",
            "training_rows": 88474,
            "query_representation": report["resolved_configuration"]["score_representation"],
            "objective": "strict_adherence_and_component_coverage",
            "unseen_pair_utility_at_10": primary["signed_strict_utility"]["mean"],
            "difference_vs_paired": None,
            "difference_interval_lower": None,
            "difference_interval_upper": None,
            "decision": "accepted_measurement_contract",
            "wandb_runs": report["tracking"]["url"],
        }
    ]


def _text_conditioned_rows(report):
    return [
        {
            "experiment": "E13",
            "training_rows": report["population"]["training_rows"],
            "query_representation": "held_atom_text_conditioned_head",
            "objective": "nested_training_head_reconstruction",
            "unseen_pair_utility_at_10": report["comparison"]["e13"],
            "difference_vs_paired": None,
            "difference_interval_lower": None,
            "difference_interval_upper": None,
            "decision": "accepted_negative_result",
            "wandb_runs": report["tracking"]["url"],
        }
    ]


def _artifact_rows(e02b, gate2, e05, e06, e07, e08, e09, e10, e11, e12, e13):
    rows = []
    for kind, candidates in e02b["accepted_feature_artifacts"].items():
        for candidate, artifact in candidates.items():
            rows.append(
                {
                    "artifact": candidate,
                    "kind": f"{kind}_features",
                    "status": "accepted",
                    "version": artifact["version"],
                    "sha256": artifact["features_sha256"],
                    "location": f"s3://plasmidclip/kedro/04_feature/e02b/{kind}_features.parquet",
                    "note": "",
                }
            )
    input_versions = gate2["resolved_configuration"]["input_versions"]
    rows.extend(
        [
            {
                "artifact": "retrieval_dataset",
                "kind": "model_input",
                "status": "accepted",
                "version": "2026-08-04T09.02.10.007Z",
                "sha256": "7e54ca3f9a3fe9f5e4afbffbdc458437665caf781e729ec33655f96381e446a5",
                "location": "s3://plasmidclip/kedro/04_feature/retrieval_dataset.parquet",
                "note": "115,120-row population identity",
            },
            {
                "artifact": "constraint_state",
                "kind": "model_input",
                "status": "accepted",
                "version": "2026-08-06T13.27.47.937Z",
                "sha256": "571b45e807e21f74699a5400faf7b20678df7d9595de3ef591a981e2a34d3208",
                "location": "s3://plasmidclip/kedro/05_model_input/e00/",
                "note": "constraint vocabulary e60a91d7cf2e415d",
            },
            {
                "artifact": "similarity_graph",
                "kind": "model_input",
                "status": "accepted",
                "version": "2026-08-17T22.59.04.326Z",
                "sha256": "6d2691a54153150d68100c20fe6cf9a9e5bb7dc8ebd028ef39c96d261253598e",
                "location": "s3://plasmidclip/kedro/08_reporting/e00/",
                "note": "edge table; accepted complete-under-configured-caps graph",
            },
            {
                "artifact": "split_grouped_v2",
                "kind": "model_input",
                "status": "accepted",
                "version": "2026-08-17T23.49.47.355Z",
                "sha256": "403c5bf6fec3afcc47d71ea302796c737609eefc47b49ea85ebd3c1ffaba5628",
                "location": "s3://plasmidclip/kedro/04_feature/e00/split_grouped_v2.parquet",
                "note": "strict primary-similarity-closed mapping",
            },
            {
                "artifact": "e02b_inputs",
                "kind": "model_input",
                "status": "accepted",
                "version": input_versions["e02b_inputs"],
                "sha256": e02b["input_manifest_sha256"],
                "location": "s3://plasmidclip/kedro/05_model_input/e02b/",
                "note": "20,000 training rows; 10,852 validation rows; 108 queries",
            },
            {
                "artifact": "query_benchmark",
                "kind": "model_input",
                "status": "accepted",
                "version": input_versions["query_benchmark"],
                "sha256": "72e2ed5576c10647c17ede29b4c45e6fd9cf7c19c2bb195279e6a329a54a2fd1",
                "location": "s3://plasmidclip/kedro/05_model_input/e00/",
                "note": "frozen controlled queries and verified/contradicted states",
            },
            {
                "artifact": "e02b_selection_report",
                "kind": "report",
                "status": "accepted",
                "version": "2026-08-24T16.34.48.358Z",
                "sha256": REPORT_HASHES[E02B_REPORT],
                "location": E02B_REPORT,
                "note": "validation-only encoder selection; no test rows read",
            },
            {
                "artifact": "e03e04_set_supervision_report",
                "kind": "report",
                "status": "accepted",
                "version": "2026-08-25T10.52.39.447Z",
                "sha256": REPORT_HASHES[GATE2_REPORT],
                "location": GATE2_REPORT,
                "note": "validation-only; pair queries were present during training",
            },
            {
                "artifact": "e05_unseen_composition_report",
                "kind": "report",
                "status": "accepted",
                "version": e05["execution"]["report_version"],
                "sha256": REPORT_HASHES[E05_REPORT],
                "location": E05_REPORT,
                "note": "atomic-only training; 80 unseen conjunction queries",
            },
            {
                "artifact": "e06_inputs",
                "kind": "model_input",
                "status": "accepted",
                "version": "2026-08-26T12.24.14.212Z",
                "sha256": "880cea9088d64720032fdd3b6ef70aa8d99e006908168f539fd432924e8c9362",
                "location": "s3://plasmidclip/kedro/05_model_input/e06/",
                "note": "88,474 train; 10,852 validation; no test rows",
            },
            {
                "artifact": "e06_tfidf_6mer_svd_512",
                "kind": "dna_features",
                "status": "accepted",
                "version": "2026-08-26T12.43.28.764Z",
                "sha256": "4376e4e0cec03dcfa6665239436f396818648aab5ef2d7c9bfd518ad537e6fe0",
                "location": "s3://plasmidclip/kedro/04_feature/e06/dna_features.parquet",
                "note": "TF-IDF/SVD refit on the full eligible training population",
            },
            {
                "artifact": "e06_qwen3_embedding_0_6b",
                "kind": "text_features",
                "status": "accepted",
                "version": "2026-08-26T12.43.28.764Z",
                "sha256": "9c131e45a457163f141e840056faf383a2ee0a7a84fc9a967e361d41aa5c2fce",
                "location": "s3://plasmidclip/kedro/04_feature/e06/text_features.parquet",
                "note": "expanded documents plus the unchanged query set",
            },
            {
                "artifact": "e06_population_scale_report",
                "kind": "report",
                "status": "accepted",
                "version": e06["execution"]["report_version"],
                "sha256": REPORT_HASHES[E06_REPORT],
                "location": E06_REPORT,
                "note": "88,474-row atomic-only training; 80 unseen conjunction queries",
            },
            {
                "artifact": "e07_additive_retrieval_report",
                "kind": "report",
                "status": "accepted",
                "version": e07["execution"]["report_version"],
                "sha256": REPORT_HASHES[E07_REPORT],
                "location": E07_REPORT,
                "note": "direct conjunction text versus summed projected atomic queries",
            },
            {
                "artifact": "e08_natural_parameter_report",
                "kind": "report",
                "status": "accepted_negative_result",
                "version": e08["execution"]["artifact_versions"]["e08_natural_parameter_report"],
                "sha256": REPORT_HASHES[E08_REPORT],
                "location": E08_REPORT,
                "note": "exact max-entropy fit diverged; four closed-world conjunctions",
            },
            {
                "artifact": "e09_natural_parameter_report",
                "kind": "report",
                "status": "accepted",
                "version": e09["execution"]["artifact_versions"]["e09_natural_parameter_report"],
                "sha256": REPORT_HASHES[E09_REPORT],
                "location": E09_REPORT,
                "note": "training-only stability selection; four closed-world conjunctions",
            },
            {
                "artifact": "e10_weak_annotation_report",
                "kind": "report",
                "status": "accepted_exploratory",
                "version": e10["execution"]["artifact_versions"]["e10_weak_annotation_report"],
                "sha256": REPORT_HASHES[E10_REPORT],
                "location": E10_REPORT,
                "note": "64 weak-label atoms; 128 unseen conjunctions; no test rows",
            },
            {
                "artifact": "e11_atomic_classifier_report",
                "kind": "report",
                "status": "accepted_exploratory",
                "version": e11["execution"]["artifact_versions"]["e11_atomic_classifier_report"],
                "sha256": REPORT_HASHES[E11_REPORT],
                "location": E11_REPORT,
                "note": "known-atom classifier ceiling; calibrated probability-product AND",
            },
            {
                "artifact": "e12_compositional_report",
                "kind": "report",
                "status": "accepted_exploratory",
                "version": e12["execution"]["artifact_versions"]["e12_compositional_report"],
                "sha256": REPORT_HASHES[E12_REPORT],
                "location": E12_REPORT,
                "note": "strict adherence and non-redundant component coverage; no test rows",
            },
            {
                "artifact": "e13_text_conditioned_report",
                "kind": "report",
                "status": "accepted_negative_result",
                "version": e13["execution"]["artifact_versions"]["e13_text_conditioned_report"],
                "sha256": REPORT_HASHES[E13_REPORT],
                "location": E13_REPORT,
                "note": "held-atom text-to-head mapping failed; no test rows",
            },
            {
                "artifact": "final_model_v1",
                "kind": "model",
                "status": "accepted",
                "version": "78565560b8473b9d1145cc9818084af63dfe0702",
                "sha256": "284a1315ae1c39b2624f09f78ef3ec0f18e8f40fd8f0c0e11d96d274b61c877e",
                "location": (
                    "hf://buckets/McClain/plasmidclip-train-ckpts/models/vec2vec-final-v1/"
                    "78565560b8473b9d1145cc9818084af63dfe0702"
                ),
                "note": "110,267-row final fit; W&B m4eeei4w; no evaluation",
            },
            {
                "artifact": "e06_first_feature_attempt",
                "kind": "run",
                "status": "failed",
                "version": "2026-08-26",
                "sha256": "",
                "location": "/home/ubuntu/Projects/vec2vec-e06/e06-*-20260826.log",
                "note": (
                    "provenance capture rejected root-owned systemd context; no feature artifacts"
                ),
            },
            {
                "artifact": "generanno_partial",
                "kind": "dna_features",
                "status": "rejected",
                "version": "2026-08-23T16.23.06.763Z",
                "sha256": "",
                "location": "s3://plasmidclip/kedro/04_feature/e02b/dna_features.parquet",
                "note": "features existed but coverage and manifest were absent; never reused",
            },
            {
                "artifact": "gate2_first_wrapper",
                "kind": "run",
                "status": "failed",
                "version": "2026-08-25",
                "sha256": "",
                "location": (
                    "/home/ubuntu/Projects/vec2vec-e02b/e03e04-set-supervision-20260825.log"
                ),
                "note": (
                    "failed before data or GPU work because the child script was not executable"
                ),
            },
        ]
    )
    return rows


def _csv_text(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_or_check(path, content, check):
    if check:
        if not path.exists() or path.read_text() != content:
            raise ValueError(f"stale result table: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _generate_tables(args, *, check):
    e02b_hash = REPORT_HASHES.get(args.e02b_report)
    gate2_hash = REPORT_HASHES.get(args.gate2_report)
    e05_hash = REPORT_HASHES.get(args.e05_report)
    e06_hash = REPORT_HASHES.get(args.e06_report)
    e07_hash = REPORT_HASHES.get(args.e07_report)
    e08_hash = REPORT_HASHES.get(args.e08_report)
    e09_hash = REPORT_HASHES.get(args.e09_report)
    e10_hash = REPORT_HASHES.get(args.e10_report)
    e11_hash = REPORT_HASHES.get(args.e11_report)
    e12_hash = REPORT_HASHES.get(args.e12_report)
    e13_hash = REPORT_HASHES.get(args.e13_report)
    if any(
        value is None
        for value in (
            e02b_hash,
            gate2_hash,
            e05_hash,
            e06_hash,
            e07_hash,
            e08_hash,
            e09_hash,
            e10_hash,
            e11_hash,
            e12_hash,
            e13_hash,
        )
    ):
        raise ValueError(
            "custom report paths require adding their accepted SHA-256 to REPORT_HASHES"
        )

    e02b = _read_report(args.e02b_report, e02b_hash)
    gate2 = _read_report(args.gate2_report, gate2_hash)
    e05 = _read_report(args.e05_report, e05_hash)
    e06 = _read_report(args.e06_report, e06_hash)
    e07 = _read_report(args.e07_report, e07_hash)
    e08 = _read_report(args.e08_report, e08_hash)
    e09 = _read_report(args.e09_report, e09_hash)
    e10 = _read_report(args.e10_report, e10_hash)
    e11 = _read_report(args.e11_report, e11_hash)
    e12 = _read_report(args.e12_report, e12_hash)
    e13 = _read_report(args.e13_report, e13_hash)
    tables = {
        "encoders.csv": _encoder_rows(e02b),
        "supervision.csv": _supervision_rows(gate2),
        "composition.csv": [
            *_composition_rows(e05, experiment="E05"),
            *_composition_rows(e06, experiment="E06"),
            *_additive_composition_rows(e07),
            *_compositional_measurement_rows(e12),
            *_text_conditioned_rows(e13),
        ],
        "natural_parameters.csv": [
            *_natural_parameter_rows(
                e08,
                experiment="E08",
                decision="optimization_failed",
                learning_rate=0.001,
            ),
            *_natural_parameter_rows(
                e09,
                experiment="E09",
                decision="stable_selected",
                learning_rate=e09["calibration"]["selected_learning_rate"],
            ),
            *_weak_annotation_rows(e10),
            *_atomic_classifier_rows(e11),
        ],
        "artifacts.csv": _artifact_rows(e02b, gate2, e05, e06, e07, e08, e09, e10, e11, e12, e13),
    }
    for name, rows in tables.items():
        _write_or_check(args.output_dir / name, _csv_text(rows), check)


def _authorization(args):
    values = {
        "approval_reference": str(args.approval_reference or "").strip(),
        "region": str(args.region or "").strip(),
        "instance_type": str(args.instance_type or "").strip(),
        "instance_hour_limit": float(args.instance_hour_limit or math.nan),
        "observed_instance_price_usd_per_hour": float(
            args.observed_instance_price_usd_per_hour or math.nan
        ),
    }
    for name in ("approval_reference", "region", "instance_type"):
        if not values[name]:
            raise ValueError(f"--{name.replace('_', '-')} is required for reproduction")
    for name in ("instance_hour_limit", "observed_instance_price_usd_per_hour"):
        if not math.isfinite(values[name]) or values[name] <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if values["instance_hour_limit"] * 3600.0 <= 30.0:
        raise ValueError("reproduction needs more than 30 seconds of authorized instance time")
    return values


def _run_reproduction_parent(args, authorization):
    command = [sys.executable, *sys.argv, "--internal-child"]
    timeout = authorization["instance_hour_limit"] * 3600.0 - 30.0
    maximum_cost = (
        authorization["instance_hour_limit"] * authorization["observed_instance_price_usd_per_hour"]
    )
    print(
        json.dumps(
            {
                "status": "starting_authorized_reproduction",
                "stage": args.reproduce,
                "maximum_cost_usd": maximum_cost,
                **authorization,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    started = time.perf_counter()
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        timeout=timeout,
        start_new_session=True,
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "status": "reproduction_complete",
                "stage": args.reproduce,
                "elapsed_seconds": elapsed,
                "observed_cost_usd": (
                    elapsed / 3600.0 * authorization["observed_instance_price_usd_per_hour"]
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_reproduction(stage, authorization, output_dir):
    from kedro.framework.session import KedroSession
    from kedro.framework.startup import bootstrap_project

    from vec2vec.lib import fixed_representation_alignment, set_supervision

    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT, save_on_close=False) as session:
        context = session.load_context()
        config = context.params["result_reproduction"]
        _validate_runtime(config["runtime"])
        catalog = context.catalog
        scale = (
            config.get("scale")
            if stage in {"scale", "additive", "natural-parameters", "natural-parameter-calibration"}
            else None
        )
        inputs = scale["inputs"] if scale else config["inputs"]
        input_version = str(inputs["panel_version"] if scale else inputs["e02b_version"])
        input_prefix = "e06" if scale else "e02b"
        pairs = catalog.load(f"{input_prefix}_pairs", version=input_version)
        queries = catalog.load(f"{input_prefix}_queries", version=input_version)
        validation_states = catalog.load(f"{input_prefix}_query_states", version=input_version)
        input_manifest = catalog.load(f"{input_prefix}_input_manifest", version=input_version)
        deadline = time.monotonic() + authorization["instance_hour_limit"] * 3600.0

        if stage == "alignment":
            dna_features, dna_manifests = _load_features(catalog, config, kind="dna")
            text_features, text_manifests = _load_features(catalog, config, kind="text")
            *_, report = fixed_representation_alignment.run_factorial_alignment(
                pairs,
                queries,
                validation_states,
                input_manifest,
                dna_features,
                dna_manifests,
                text_features,
                text_manifests,
                _alignment_params(config),
                deadline_monotonic=deadline,
            )
            _verify_output_hashes(
                report["output_hashes"], config["alignment"]["expected_output_hashes"]
            )
            _verify_result_table(output_dir / "encoders.csv", _encoder_rows(report))
            print(
                json.dumps(
                    {
                        "status": "accepted_alignment_reproduced",
                        "output_hashes": report["output_hashes"],
                        "selection": report["selection"]["selected_pair"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

        if scale:
            dna_features, dna_manifests = _load_scale_features(catalog, scale, kind="dna")
            text_features, text_manifests = _load_scale_features(catalog, scale, kind="text")
        else:
            dna_features, dna_manifests = _load_features(
                catalog, config, kind="dna", candidates=("tfidf_6mer_svd_512",)
            )
            text_features, text_manifests = _load_features(
                catalog, config, kind="text", candidates=("qwen3_embedding_0_6b",)
            )
        query_version = str(inputs["query_benchmark_version"])
        all_query_states = catalog.load("e00_query_candidate_state", version=query_version)
        query_manifest = catalog.load("e00_query_benchmark_manifest", version=query_version)
        section_name = (
            "natural_parameters"
            if stage == "natural-parameters"
            else "natural_parameter_calibration"
            if stage == "natural-parameter-calibration"
            else stage
            if stage in {"composition", "scale", "additive"}
            else "supervision"
        )
        if stage in {
            "composition",
            "scale",
            "additive",
            "natural-parameters",
            "natural-parameter-calibration",
        }:
            _validate_frozen_authorization(
                authorization, config[section_name]["compute_authorization"]
            )
            git_commit, git_dirty = _git_state()
            if git_dirty:
                raise RuntimeError(f"{stage} must start from a clean Git checkout")
        stage_started = time.perf_counter()
        if stage == "natural-parameter-calibration":
            checkpoints, history, summary, report = (
                set_supervision.run_natural_parameter_calibration(
                    pairs,
                    queries,
                    validation_states,
                    all_query_states,
                    query_manifest,
                    input_manifest,
                    dna_features["tfidf_6mer_svd_512"],
                    dna_manifests["tfidf_6mer_svd_512"],
                    text_features["qwen3_embedding_0_6b"],
                    text_manifests["qwen3_embedding_0_6b"],
                    _natural_parameter_params(
                        config,
                        section_name="natural_parameter_calibration",
                        run_name_prefix="e09",
                    ),
                    deadline_monotonic=deadline,
                )
            )
            elapsed = time.perf_counter() - stage_started
            failed_tracking = [row for row in report["tracking"] if row.get("status") != "complete"]
            if failed_tracking:
                raise RuntimeError(f"W&B tracking did not complete: {failed_tracking}")
            expected_hashes = config["natural_parameter_calibration"].get("expected_output_hashes")
            if expected_hashes is not None:
                _verify_output_hashes(report["output_hashes"], expected_hashes)
            versions = {
                name: catalog.get(name).resolve_save_version()
                for name in (
                    "e09_natural_parameter_checkpoints",
                    "e09_natural_parameter_history",
                    "e09_natural_parameter_summary",
                    "e09_natural_parameter_report",
                )
            }
            report["execution"] = {
                **authorization,
                "maximum_cost_usd": (
                    authorization["instance_hour_limit"]
                    * authorization["observed_instance_price_usd_per_hour"]
                ),
                "stage_elapsed_seconds": elapsed,
                "stage_cost_usd": (
                    elapsed / 3600.0 * authorization["observed_instance_price_usd_per_hour"]
                ),
                "git_commit": git_commit,
                "git_dirty": False,
                "runtime": config["runtime"],
                "artifact_versions": versions,
            }
            catalog.save("e09_natural_parameter_checkpoints", checkpoints)
            catalog.save("e09_natural_parameter_history", history)
            catalog.save("e09_natural_parameter_summary", summary)
            catalog.save("e09_natural_parameter_report", report)
            print(
                json.dumps(
                    {
                        "status": "e09_natural_parameter_calibration_complete",
                        "calibration": report["calibration"],
                        "comparison": report["comparison"],
                        "output_hashes": report["output_hashes"],
                        "tracking": report["tracking"],
                        "execution": report["execution"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        if stage == "natural-parameters":
            checkpoints, history, summary, report = (
                set_supervision.run_natural_parameter_comparison(
                    pairs,
                    queries,
                    validation_states,
                    all_query_states,
                    query_manifest,
                    input_manifest,
                    dna_features["tfidf_6mer_svd_512"],
                    dna_manifests["tfidf_6mer_svd_512"],
                    text_features["qwen3_embedding_0_6b"],
                    text_manifests["qwen3_embedding_0_6b"],
                    _natural_parameter_params(config),
                    deadline_monotonic=deadline,
                )
            )
            elapsed = time.perf_counter() - stage_started
            failed_tracking = [row for row in report["tracking"] if row.get("status") != "complete"]
            if failed_tracking:
                raise RuntimeError(f"W&B tracking did not complete: {failed_tracking}")
            expected_hashes = config["natural_parameters"].get("expected_output_hashes")
            if expected_hashes is not None:
                _verify_output_hashes(report["output_hashes"], expected_hashes)
            required_unchanged = (
                config["natural_parameters"]
                .get("technical_retry", {})
                .get("required_unchanged_output_hashes", {})
            )
            changed = {
                name: {"expected": expected, "observed": report["output_hashes"].get(name)}
                for name, expected in required_unchanged.items()
                if report["output_hashes"].get(name) != expected
            }
            if changed:
                raise ValueError(f"E08 technical retry changed model outputs: {changed}")
            versions = {
                name: catalog.get(name).resolve_save_version()
                for name in (
                    "e08_natural_parameter_checkpoints",
                    "e08_natural_parameter_history",
                    "e08_natural_parameter_summary",
                    "e08_natural_parameter_report",
                )
            }
            report["execution"] = {
                **authorization,
                "maximum_cost_usd": (
                    authorization["instance_hour_limit"]
                    * authorization["observed_instance_price_usd_per_hour"]
                ),
                "stage_elapsed_seconds": elapsed,
                "stage_cost_usd": (
                    elapsed / 3600.0 * authorization["observed_instance_price_usd_per_hour"]
                ),
                "git_commit": git_commit,
                "git_dirty": False,
                "runtime": config["runtime"],
                "artifact_versions": versions,
            }
            catalog.save("e08_natural_parameter_checkpoints", checkpoints)
            catalog.save("e08_natural_parameter_history", history)
            catalog.save("e08_natural_parameter_summary", summary)
            catalog.save("e08_natural_parameter_report", report)
            print(
                json.dumps(
                    {
                        "status": "e08_natural_parameter_validation_complete",
                        "comparison": report["comparison"],
                        "output_hashes": report["output_hashes"],
                        "tracking": report["tracking"],
                        "execution": report["execution"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return
        supervision_params = (
            _additive_params(config)
            if stage == "additive"
            else _supervision_params(config, section_name=section_name)
        )
        outputs = set_supervision.run_set_supervision_comparison(
            pairs,
            queries,
            validation_states,
            all_query_states,
            query_manifest,
            input_manifest,
            dna_features["tfidf_6mer_svd_512"],
            dna_manifests["tfidf_6mer_svd_512"],
            text_features["qwen3_embedding_0_6b"],
            text_manifests["qwen3_embedding_0_6b"],
            supervision_params,
            deadline_monotonic=deadline,
        )
        *_, report = outputs
        elapsed = time.perf_counter() - stage_started
        failed_tracking = [row for row in report["tracking"] if row.get("status") != "complete"]
        if failed_tracking:
            raise RuntimeError(f"W&B tracking did not complete: {failed_tracking}")

        if stage in {"composition", "scale", "additive"}:
            expected_hashes = config[section_name].get("expected_output_hashes")
            if expected_hashes is not None:
                _verify_output_hashes(report["output_hashes"], expected_hashes)
            if stage == "additive":
                expected_direct = float(config["additive"]["expected_direct_text_utility_at_10"])
                if not math.isclose(
                    report["comparison"]["verified_set"],
                    expected_direct,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "additive audit did not reproduce the accepted direct-text result"
                    )
                summary = pd.DataFrame(_additive_rows(report))
                result_prefix = "e07"
                summary_dataset = "e07_additive_summary"
                report_dataset = "e07_additive_report"
            else:
                summary = pd.DataFrame(_composition_rows(report))
                result_prefix = "e06" if stage == "scale" else "e05"
                summary_dataset = f"{result_prefix}_composition_summary"
                report_dataset = f"{result_prefix}_composition_report"
            summary_version = catalog.get(summary_dataset).resolve_save_version()
            report_version = catalog.get(report_dataset).resolve_save_version()
            report["execution"] = {
                **authorization,
                "maximum_cost_usd": (
                    authorization["instance_hour_limit"]
                    * authorization["observed_instance_price_usd_per_hour"]
                ),
                "stage_elapsed_seconds": elapsed,
                "stage_cost_usd": (
                    elapsed / 3600.0 * authorization["observed_instance_price_usd_per_hour"]
                ),
                "git_commit": git_commit,
                "git_dirty": False,
                "runtime": config["runtime"],
                "summary_version": summary_version,
                "report_version": report_version,
            }
            catalog.save(summary_dataset, summary)
            catalog.save(report_dataset, report)
            if stage == "composition":
                _write_or_check(
                    output_dir / "composition.csv", _csv_text(_composition_rows(report)), False
                )
            print(
                json.dumps(
                    {
                        "status": (
                            "e07_additive_retrieval_audit_complete"
                            if stage == "additive"
                            else f"{result_prefix}_unseen_composition_complete"
                        ),
                        "comparison": report["comparison"],
                        "additive_comparison": report.get("additive_comparison"),
                        "output_hashes": report["output_hashes"],
                        "tracking": report["tracking"],
                        "execution": report["execution"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

        _verify_output_hashes(
            report["output_hashes"], config["supervision"]["expected_output_hashes"]
        )
        _verify_result_table(
            output_dir / "supervision.csv",
            _supervision_rows(report),
            ignored_columns=("wandb_runs",),
        )
        print(
            json.dumps(
                {
                    "status": "accepted_supervision_result_reproduced",
                    "output_hashes": report["output_hashes"],
                    "comparison": report["comparison"],
                    "tracking": report["tracking"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _load_features(catalog, config, *, kind, candidates=None):
    accepted = config["features"][kind]
    candidate_ids = tuple(candidates or sorted(accepted))
    dataset = (
        "e02b_dna_features_tfidf_6mer_svd_512"
        if kind == "dna"
        else "e02b_text_features_qwen3_embedding_0_6b"
    )
    manifest_dataset = (
        "e02b_dna_manifest_tfidf_6mer_svd_512"
        if kind == "dna"
        else "e02b_text_manifest_qwen3_embedding_0_6b"
    )
    features = {}
    manifests = {}
    for candidate_id in candidate_ids:
        version = str(accepted[candidate_id]["version"])
        features[candidate_id] = catalog.load(dataset, version=version)
        manifests[candidate_id] = catalog.load(manifest_dataset, version=version)
    return features, manifests


def _load_scale_features(catalog, scale, *, kind):
    candidate_id = "tfidf_6mer_svd_512" if kind == "dna" else "qwen3_embedding_0_6b"
    dataset = (
        "e06_dna_features_tfidf_6mer_svd_512"
        if kind == "dna"
        else "e06_text_features_qwen3_embedding_0_6b"
    )
    manifest_dataset = (
        "e06_dna_manifest_tfidf_6mer_svd_512"
        if kind == "dna"
        else "e06_text_manifest_qwen3_embedding_0_6b"
    )
    accepted = scale["features"][kind]
    version = str(accepted["version"])
    return (
        {candidate_id: catalog.load(dataset, version=version)},
        {candidate_id: catalog.load(manifest_dataset, version=version)},
    )


def _alignment_params(config):
    section = config["alignment"]
    features = config["features"]
    return {
        "protocol_version": section["protocol_version"],
        "training_rows": section["training_rows"],
        "device": section["device"],
        "precision": section.get("precision", "float32"),
        "tfidf": {"candidate_id": "tfidf_6mer_svd_512"},
        "dna_candidates": {
            candidate_id: {}
            for candidate_id in features["dna"]
            if candidate_id != "tfidf_6mer_svd_512"
        },
        "text_candidates": {candidate_id: {} for candidate_id in features["text"]},
        "accepted_input_artifact": {
            "manifest_sha256": config["inputs"]["manifest_sha256"],
            "pairs_sha256": config["inputs"]["pairs_sha256"],
        },
        "accepted_feature_artifacts": features,
        "probe": section["probe"],
    }


def _supervision_params(config, *, section_name="supervision"):
    section = config[section_name]
    is_scale = section_name == "scale"
    inputs = section["inputs"] if is_scale else config["inputs"]
    dna = (
        section["features"]["dna"] if is_scale else config["features"]["dna"]["tfidf_6mer_svd_512"]
    )
    text = (
        section["features"]["text"]
        if is_scale
        else config["features"]["text"]["qwen3_embedding_0_6b"]
    )
    params = {
        "protocol_version": section["protocol_version"],
        "protocol_path": "studies/set_valued_compositional_embeddings/EXPERIMENT_LOG.md",
        "input_versions": {
            "e06_inputs" if is_scale else "e02b_inputs": (
                inputs["panel_version"] if is_scale else inputs["e02b_version"]
            ),
            "query_benchmark": inputs["query_benchmark_version"],
            "dna_features": dna["version"],
            "text_features": text["version"],
        },
        "training_rows": section["training_rows"],
        "minimum_training_verified_rows": section["minimum_training_verified_rows"],
        "expected_training_query_states_sha256": inputs["training_query_states_sha256"],
        "accepted_input_artifact": {
            "manifest_sha256": inputs["manifest_sha256"],
            "pairs_sha256": inputs["pairs_sha256"],
        },
        "accepted_feature_artifacts": {
            "dna": {"candidate_id": "tfidf_6mer_svd_512", **dna},
            "text": {"candidate_id": "qwen3_embedding_0_6b", **text},
        },
        "objectives": section["objectives"],
        "device": section["device"],
        "primary_k": section["primary_k"],
        "minimum_practical_improvement": section["minimum_practical_improvement"],
        "probe": section["probe"],
        "tracking": section["tracking"],
    }
    if section_name in {"composition", "scale"}:
        params.update(
            training_query_kind=section["training_query_kind"],
            evaluation_query_kind=section["evaluation_query_kind"],
            expected_training_queries=section["expected_training_queries"],
            expected_evaluation_queries=section["expected_evaluation_queries"],
            expected_evaluation_controlled_split=section["expected_evaluation_controlled_split"],
            completion_status=f"{section_name}_unseen_composition_complete",
            run_name_prefix="e06" if is_scale else "e05",
        )
    return params


def _additive_params(config):
    params = _supervision_params(config, section_name="scale")
    additive = config["additive"]
    params.update(
        protocol_version=additive["protocol_version"],
        tracking=additive["tracking"],
        query_representations=additive["query_representations"],
        audit_atomic_sum=True,
        completion_status="additive_retrieval_audit_complete",
        run_name_prefix="e07-additive",
    )
    return params


def _natural_parameter_params(config, *, section_name="natural_parameters", run_name_prefix="e08"):
    params = _supervision_params(config, section_name="scale")
    section = config[section_name]
    params.update(
        protocol_version=section["protocol_version"],
        training_semantic_query_ids=section["training_semantic_query_ids"],
        evaluation_semantic_query_ids=section["evaluation_semantic_query_ids"],
        expected_training_queries=section["expected_training_queries"],
        expected_evaluation_queries=section["expected_evaluation_queries"],
        minimum_training_state_rows=section["minimum_training_state_rows"],
        base_measures=section["base_measures"],
        device=section["device"],
        precision=section["precision"],
        primary_k=section["primary_k"],
        minimum_practical_improvement=section["minimum_practical_improvement"],
        probe=section["probe"],
        tracking=section["tracking"],
        query_representations=["direct_text", "atomic_sum"],
        run_name_prefix=run_name_prefix,
    )
    if "stability" in section:
        params["stability"] = section["stability"]
    return params


def _validate_frozen_authorization(observed, expected):
    differences = {
        name: {"expected": expected.get(name), "observed": observed.get(name)}
        for name in expected
        if name != "instance_hour_limit"
        if observed.get(name) != expected.get(name)
    }
    observed_limit = float(observed.get("instance_hour_limit", math.nan))
    expected_limit = float(expected["instance_hour_limit"])
    if not math.isfinite(observed_limit) or observed_limit > expected_limit:
        differences["instance_hour_limit"] = {
            "expected_maximum": expected_limit,
            "observed": observed_limit,
        }
    if differences:
        raise ValueError(f"compute authorization differs from frozen contract: {differences}")


def _git_state():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return commit, bool(status.strip())


def _validate_runtime(expected):
    observed = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "packages": {name: importlib.metadata.version(name) for name in expected["packages"]},
    }
    mismatches = []
    for name in ("python", "machine"):
        if str(observed[name]) != str(expected[name]):
            mismatches.append(f"{name}: expected {expected[name]}, observed {observed[name]}")
    for name, version in expected["packages"].items():
        if observed["packages"][name] != str(version):
            mismatches.append(f"{name}: expected {version}, observed {observed['packages'][name]}")
    import torch

    if not torch.cuda.is_available():
        mismatches.append("CUDA is unavailable")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        if gpu_name != str(expected["gpu_name"]):
            mismatches.append(f"GPU: expected {expected['gpu_name']}, observed {gpu_name}")
    if mismatches:
        raise RuntimeError(
            "reproduction runtime differs from accepted run: " + "; ".join(mismatches)
        )


def _verify_output_hashes(observed, expected):
    if observed != expected:
        differences = {
            name: {"expected": expected.get(name), "observed": observed.get(name)}
            for name in sorted(set(expected) | set(observed))
            if expected.get(name) != observed.get(name)
        }
        raise ValueError(f"recomputed artifact hashes differ: {differences}")


def _verify_result_table(path, rows, *, ignored_columns=()):
    if not path.exists():
        raise ValueError(f"accepted result table is missing: {path}")
    expected = list(csv.DictReader(path.open()))
    observed = list(csv.DictReader(io.StringIO(_csv_text(rows))))
    for table in (expected, observed):
        for row in table:
            for column in ignored_columns:
                row.pop(column, None)
    if observed != expected:
        raise ValueError(f"recomputed scientific result differs from {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e02b-report", default=E02B_REPORT)
    parser.add_argument("--gate2-report", default=GATE2_REPORT)
    parser.add_argument("--e05-report", default=E05_REPORT)
    parser.add_argument("--e06-report", default=E06_REPORT)
    parser.add_argument("--e07-report", default=E07_REPORT)
    parser.add_argument("--e08-report", default=E08_REPORT)
    parser.add_argument("--e09-report", default=E09_REPORT)
    parser.add_argument("--e10-report", default=E10_REPORT)
    parser.add_argument("--e11-report", default=E11_REPORT)
    parser.add_argument("--e12-report", default=E12_REPORT)
    parser.add_argument("--e13-report", default=E13_REPORT)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--reproduce",
        choices=(
            "alignment",
            "supervision",
            "composition",
            "scale",
            "additive",
            "natural-parameters",
            "natural-parameter-calibration",
        ),
    )
    parser.add_argument("--approval-reference")
    parser.add_argument("--region")
    parser.add_argument("--instance-type")
    parser.add_argument("--instance-hour-limit", type=float)
    parser.add_argument("--observed-instance-price-usd-per-hour", type=float)
    parser.add_argument("--internal-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    _generate_tables(args, check=True if args.reproduce else args.check)
    if not args.reproduce:
        return
    authorization = _authorization(args)
    if args.internal_child:
        _run_reproduction(args.reproduce, authorization, args.output_dir)
        return
    _run_reproduction_parent(args, authorization)


if __name__ == "__main__":
    main()
