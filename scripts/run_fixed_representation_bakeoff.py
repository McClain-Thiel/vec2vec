#!/usr/bin/env python3
"""Run one approved paid E02b stage under an external wall-clock limit."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

from vec2vec.lib.fixed_representation_bakeoff import approved_compute_authorization

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAID_STAGES = ("dna_features", "text_features", "alignment_probe")


def main() -> None:
    """Validate authorization, then run the complete Kedro child under one deadline."""
    parser = _parser()
    arguments = parser.parse_args()
    _validate_arguments(parser, arguments)
    if arguments.internal_child:
        _run_stage(arguments)
        return

    timeout_seconds = _paid_timeout_seconds(
        arguments.instance_hour_limit,
        shutdown_reserve_seconds=arguments.shutdown_reserve_seconds,
    )
    command = _child_command(arguments)
    started = time.perf_counter()
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            timeout=timeout_seconds,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired as error:
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "status": "stopped_at_authorized_deadline",
                    "stage": _authorized_stage(arguments),
                    "elapsed_seconds": elapsed,
                    "instance_hour_limit": arguments.instance_hour_limit,
                    "shutdown_reserve_seconds": arguments.shutdown_reserve_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise TimeoutError(
            f"E02b {_authorized_stage(arguments)} child was stopped before the paid deadline"
        ) from error
    elapsed = time.perf_counter() - started
    print(
        json.dumps(
            {
                "status": "completed_within_authorized_deadline",
                "stage": _authorized_stage(arguments),
                "approval_reference": arguments.approval_reference,
                "region": arguments.region,
                "instance_type": arguments.instance_type,
                "instance_hour_limit": arguments.instance_hour_limit,
                "elapsed_seconds": elapsed,
                "observed_instance_cost_usd": (
                    elapsed / 3600.0 * arguments.observed_instance_price_usd_per_hour
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=PAID_STAGES, required=True)
    parser.add_argument(
        "--candidate",
        help="Candidate key for a DNA or text feature stage. Omit for alignment.",
    )
    parser.add_argument("--approval-reference", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--instance-hour-limit", required=True, type=float)
    parser.add_argument("--observed-instance-price-usd-per-hour", required=True, type=float)
    parser.add_argument("--shutdown-reserve-seconds", default=30.0, type=float)
    parser.add_argument("--internal-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _validate_arguments(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    feature_stage = arguments.stage in {"dna_features", "text_features"}
    if feature_stage and not str(arguments.candidate or "").strip():
        parser.error(f"{arguments.stage} requires --candidate")
    if not feature_stage and arguments.candidate is not None:
        parser.error("alignment_probe does not accept --candidate")
    for name in ("approval_reference", "region", "instance_type"):
        if not str(getattr(arguments, name)).strip():
            parser.error(f"--{name.replace('_', '-')} must not be empty")
    for name in ("instance_hour_limit", "observed_instance_price_usd_per_hour"):
        value = float(getattr(arguments, name))
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(arguments.shutdown_reserve_seconds):
        parser.error("--shutdown-reserve-seconds must be finite")
    if arguments.shutdown_reserve_seconds < 0.0:
        parser.error("--shutdown-reserve-seconds must not be negative")
    if arguments.shutdown_reserve_seconds >= arguments.instance_hour_limit * 3600.0:
        parser.error("shutdown reserve must be shorter than the instance-hour limit")


def _paid_timeout_seconds(
    instance_hour_limit: float,
    *,
    shutdown_reserve_seconds: float,
) -> float:
    if not math.isfinite(instance_hour_limit) or instance_hour_limit <= 0.0:
        raise ValueError("instance hour limit must be finite and positive")
    if not math.isfinite(shutdown_reserve_seconds) or shutdown_reserve_seconds < 0.0:
        raise ValueError("shutdown reserve must be finite and non-negative")
    timeout = instance_hour_limit * 3600.0 - shutdown_reserve_seconds
    if timeout <= 0.0:
        raise ValueError("shutdown reserve leaves no paid stage execution time")
    return timeout


def _authorized_stage(arguments: argparse.Namespace) -> str:
    if arguments.stage == "alignment_probe":
        return "alignment_probe"
    return f"{arguments.stage}:{arguments.candidate}"


def _child_command(arguments: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        str(arguments.stage),
        "--approval-reference",
        str(arguments.approval_reference),
        "--region",
        str(arguments.region),
        "--instance-type",
        str(arguments.instance_type),
        "--instance-hour-limit",
        str(arguments.instance_hour_limit),
        "--observed-instance-price-usd-per-hour",
        str(arguments.observed_instance_price_usd_per_hour),
        "--shutdown-reserve-seconds",
        "0",
        "--internal-child",
    ]
    if arguments.candidate is not None:
        command.extend(["--candidate", str(arguments.candidate)])
    return command


def _run_stage(arguments: argparse.Namespace) -> None:
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT, save_on_close=False) as session:
        base_params = session.load_context().params
    configuration = base_params.get("fixed_representation_bakeoff")
    if not isinstance(configuration, dict):
        raise ValueError("fixed_representation_bakeoff configuration must be a mapping")
    compute_authorization = {
        "stage": _authorized_stage(arguments),
        "approval_reference": str(arguments.approval_reference),
        "region": str(arguments.region),
        "instance_type": str(arguments.instance_type),
        "instance_hour_limit": float(arguments.instance_hour_limit),
        "observed_instance_price_usd_per_hour": float(
            arguments.observed_instance_price_usd_per_hour
        ),
    }
    expected_authorization = approved_compute_authorization(
        configuration, stage=_authorized_stage(arguments)
    )
    if compute_authorization != expected_authorization:
        raise ValueError(
            "command authorization differs from the frozen E02b approval: "
            f"observed={compute_authorization}, expected={expected_authorization}"
        )
    runtime_params = {
        "fixed_representation_bakeoff": {
            **configuration,
            "compute_authorization": compute_authorization,
        }
    }
    if arguments.candidate is not None:
        runtime_params["fixed_representation_bakeoff_feature_candidate"] = str(arguments.candidate)
    pipeline_name, load_versions = _stage_contract(arguments, configuration)
    with KedroSession.create(
        project_path=PROJECT_ROOT,
        runtime_params=runtime_params,
    ) as session:
        session.run(pipeline_name=pipeline_name, load_versions=load_versions)


def _stage_contract(
    arguments: argparse.Namespace,
    configuration: dict[str, Any],
) -> tuple[str, dict[str, str]]:
    accepted_input = configuration.get("accepted_input_artifact")
    if not isinstance(accepted_input, dict) or not str(accepted_input.get("version", "")):
        raise ValueError("accepted_input_artifact with a version is required for paid E02b work")
    input_version = str(accepted_input["version"])
    common_versions = {
        "e02b_pairs": input_version,
        "e02b_input_manifest": input_version,
    }
    candidate = str(arguments.candidate or "")
    if arguments.stage == "dna_features":
        if candidate not in configuration["dna_candidates"]:
            raise ValueError(f"unknown neural DNA candidate: {candidate}")
        invariance = configuration["accepted_invariance_artifacts"].get(candidate)
        if not isinstance(invariance, dict) or not str(invariance.get("version", "")):
            raise ValueError(f"candidate {candidate} lacks an accepted invariance version")
        return (
            "fixed_representation_bakeoff_dna_features",
            {
                **common_versions,
                "e02_fixed_representation_invariance_manifest": str(invariance["version"]),
            },
        )
    if arguments.stage == "text_features":
        if candidate not in configuration["text_candidates"]:
            raise ValueError(f"unknown text candidate: {candidate}")
        return (
            "fixed_representation_bakeoff_text_features",
            {
                **common_versions,
                "e02b_queries": input_version,
            },
        )
    accepted_features = configuration.get("accepted_feature_artifacts")
    if not isinstance(accepted_features, dict):
        raise ValueError("accepted_feature_artifacts are required for alignment")
    versions = {
        **common_versions,
        "e02b_queries": input_version,
        "e02b_query_states": input_version,
    }
    aliases = {
        "dna": {
            "tfidf_6mer_svd_512": "tfidf_6mer_svd_512",
            "carbon_500m": "carbon_500m",
            "generanno_prokaryote_500m": "generanno_prokaryote_500m",
            "generator_v2_prokaryote_1_2b": "generator_v2_prokaryote_1_2b",
        },
        "text": {
            "bge_base_en_v1_5": "bge_base_en_v1_5",
            "gte_modernbert_base": "gte_modernbert_base",
            "qwen3_embedding_0_6b": "qwen3_embedding_0_6b",
        },
    }
    for feature_kind, candidates in aliases.items():
        accepted_kind = accepted_features.get(feature_kind)
        if not isinstance(accepted_kind, dict):
            raise ValueError(f"accepted {feature_kind} feature artifacts are missing")
        for candidate_id, suffix in candidates.items():
            record = accepted_kind.get(candidate_id)
            if not isinstance(record, dict) or not str(record.get("version", "")):
                raise ValueError(f"accepted feature version is missing for {candidate_id}")
            version = str(record["version"])
            versions[f"e02b_{feature_kind}_features_{suffix}"] = version
            versions[f"e02b_{feature_kind}_manifest_{suffix}"] = version
    return "fixed_representation_bakeoff_alignment", versions


if __name__ == "__main__":
    main()
