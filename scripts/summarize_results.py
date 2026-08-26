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
REPORT_HASHES = {
    E02B_REPORT: "a675a3a3fac1b87827749764caeea07a395debf86c0ee886998417fd9a5b8d25",
    GATE2_REPORT: "32914b0f7dec4a0c9c893dbb0eecd80a16dce049d46e642405f22f61cbbee9b2",
    E05_REPORT: "182fb0dd75a1bd3159bc24b488e4921ff7dac1c6292352a408c8ff6d2d44082d",
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


def _composition_rows(report):
    return [
        {
            "objective": row["objective"],
            "unseen_pair_utility_at_10": row["pair_utility_at_10"],
            "difference_vs_paired": row["difference_vs_paired"],
            "difference_interval_lower": row["difference_interval_lower"],
            "difference_interval_upper": row["difference_interval_upper"],
            "decision": row["decision"],
            "wandb_runs": row["wandb_runs"],
        }
        for row in _supervision_rows(report)
    ]


def _artifact_rows(e02b, gate2, e05):
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
    if e02b_hash is None or gate2_hash is None or e05_hash is None:
        raise ValueError(
            "custom report paths require adding their accepted SHA-256 to REPORT_HASHES"
        )

    e02b = _read_report(args.e02b_report, e02b_hash)
    gate2 = _read_report(args.gate2_report, gate2_hash)
    e05 = _read_report(args.e05_report, e05_hash)
    tables = {
        "encoders.csv": _encoder_rows(e02b),
        "supervision.csv": _supervision_rows(gate2),
        "composition.csv": _composition_rows(e05),
        "artifacts.csv": _artifact_rows(e02b, gate2, e05),
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
        scale = config.get("scale") if stage == "scale" else None
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
        section_name = stage if stage in {"composition", "scale"} else "supervision"
        if stage in {"composition", "scale"}:
            _validate_frozen_authorization(
                authorization, config[section_name]["compute_authorization"]
            )
            git_commit, git_dirty = _git_state()
            if git_dirty:
                raise RuntimeError(f"{stage} must start from a clean Git checkout")
        stage_started = time.perf_counter()
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
            _supervision_params(config, section_name=section_name),
            deadline_monotonic=deadline,
        )
        *_, report = outputs
        elapsed = time.perf_counter() - stage_started
        failed_tracking = [row for row in report["tracking"] if row.get("status") != "complete"]
        if failed_tracking:
            raise RuntimeError(f"W&B tracking did not complete: {failed_tracking}")

        if stage in {"composition", "scale"}:
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
                        "status": f"{result_prefix}_unseen_composition_complete",
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
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--reproduce", choices=("alignment", "supervision", "composition", "scale"))
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
