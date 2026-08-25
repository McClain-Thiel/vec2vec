#!/usr/bin/env python3
"""Run the approved Gate 2 comparison under a hard wall-clock limit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Validate the frozen approval and run Gate 2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--instance-hour-limit", required=True, type=float)
    parser.add_argument("--observed-instance-price-usd-per-hour", required=True, type=float)
    parser.add_argument("--internal-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    authorization = {
        "approval_reference": args.approval_reference,
        "region": args.region,
        "instance_type": args.instance_type,
        "instance_hour_limit": args.instance_hour_limit,
        "observed_instance_price_usd_per_hour": args.observed_instance_price_usd_per_hour,
    }
    configuration = _configuration()
    if authorization != configuration.get("approved_compute_authorization"):
        parser.error("command authorization differs from the frozen Gate 2 approval")
    if args.internal_child:
        _run(authorization, configuration)
        return

    command = [*sys.argv, "--internal-child"]
    started = time.perf_counter()
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        timeout=args.instance_hour_limit * 3600.0 - 30.0,
        start_new_session=True,
    )
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": elapsed,
                "observed_instance_cost_usd": (
                    elapsed / 3600.0 * args.observed_instance_price_usd_per_hour
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _configuration() -> dict:
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT, save_on_close=False) as session:
        configuration = session.load_context().params.get("set_supervision")
    if not isinstance(configuration, dict):
        raise ValueError("set_supervision configuration must be a mapping")
    return configuration


def _run(authorization: dict, configuration: dict) -> None:
    versions = dict(configuration["input_versions"])
    load_versions = {
        "e02b_pairs": versions["e02b_inputs"],
        "e02b_queries": versions["e02b_inputs"],
        "e02b_query_states": versions["e02b_inputs"],
        "e02b_input_manifest": versions["e02b_inputs"],
        "e00_query_candidate_state": versions["query_benchmark"],
        "e00_query_benchmark_manifest": versions["query_benchmark"],
        "e02b_dna_features_tfidf_6mer_svd_512": versions["dna_features"],
        "e02b_dna_manifest_tfidf_6mer_svd_512": versions["dna_features"],
        "e02b_text_features_qwen3_embedding_0_6b": versions["text_features"],
        "e02b_text_manifest_qwen3_embedding_0_6b": versions["text_features"],
    }
    runtime = {"set_supervision": {**configuration, "compute_authorization": authorization}}
    with KedroSession.create(project_path=PROJECT_ROOT, runtime_params=runtime) as session:
        session.run(pipeline_name="set_supervision", load_versions=load_versions)


if __name__ == "__main__":
    main()
