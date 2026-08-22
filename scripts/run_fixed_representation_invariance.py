#!/usr/bin/env python3
"""Run approved Gate 1 DNA invariance candidates under an explicit compute cap."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run each requested candidate as a separate versioned Kedro run."""
    batch_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        required=True,
        help="Accepted candidate key. Repeat to run more than one candidate.",
    )
    parser.add_argument(
        "--approval-reference",
        required=True,
        help="Durable user-approval reference for this paid run.",
    )
    parser.add_argument("--region", required=True, help="AWS region used for pricing and launch.")
    parser.add_argument("--instance-type", required=True, help="EC2 instance type for this run.")
    parser.add_argument(
        "--instance-hour-limit",
        required=True,
        type=float,
        help="Maximum authorized instance hours across all requested candidates.",
    )
    parser.add_argument(
        "--observed-instance-price-usd-per-hour",
        required=True,
        type=float,
        help="Current observed on-demand Linux price in US dollars per instance-hour.",
    )
    parser.add_argument(
        "--shutdown-reserve-seconds",
        default=30.0,
        type=float,
        help="Time reserved to stop the child process and exit before the total hour cap.",
    )
    parser.add_argument(
        "--internal-candidate-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-instance-hour-limit",
        type=float,
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if not math.isfinite(arguments.instance_hour_limit):
        parser.error("instance hour limit must be finite")
    if arguments.instance_hour_limit <= 0.0:
        parser.error("instance hour limit must be greater than zero")
    if not math.isfinite(arguments.observed_instance_price_usd_per_hour):
        parser.error("observed instance price must be finite")
    if arguments.observed_instance_price_usd_per_hour <= 0.0:
        parser.error("observed instance price must be greater than zero")
    if not math.isfinite(arguments.shutdown_reserve_seconds):
        parser.error("shutdown reserve must be finite")
    if arguments.shutdown_reserve_seconds < 0.0:
        parser.error("shutdown reserve must not be negative")
    if arguments.shutdown_reserve_seconds >= arguments.instance_hour_limit * 3600.0:
        parser.error("shutdown reserve must be shorter than the instance-hour limit")
    if arguments.internal_candidate_child and arguments.batch_instance_hour_limit is None:
        parser.error("internal candidate child requires the original batch hour limit")
    bootstrap_project(PROJECT_ROOT)

    with KedroSession.create(project_path=PROJECT_ROOT, save_on_close=False) as session:
        context_params = session.load_context().params
        smoke_params = context_params["fixed_representation_smoke"]
        invariance_params = context_params["fixed_representation_invariance"]
    accepted_artifacts = invariance_params["accepted_smoke_artifacts"]
    if not isinstance(accepted_artifacts, dict) or not accepted_artifacts:
        parser.error("accepted_smoke_artifacts must be a non-empty mapping")
    candidates = tuple(arguments.candidates)
    unknown = sorted(set(candidates).difference(accepted_artifacts))
    if unknown:
        parser.error(
            f"candidate(s) {unknown} have no accepted smoke result; "
            f"accepted candidates are {tuple(accepted_artifacts)}"
        )
    if len(candidates) != len(set(candidates)):
        parser.error("candidate arguments must be unique")
    try:
        _required_transformers_version(candidates, smoke_params["candidates"])
    except ValueError as error:
        parser.error(str(error))

    if arguments.internal_candidate_child:
        if len(candidates) != 1:
            parser.error("internal candidate child requires exactly one candidate")
        _run_candidate(
            candidate=candidates[0],
            approval_reference=arguments.approval_reference,
            region=arguments.region,
            instance_type=arguments.instance_type,
            candidate_instance_hour_limit=arguments.instance_hour_limit,
            batch_instance_hour_limit=float(arguments.batch_instance_hour_limit),
            observed_instance_price_usd_per_hour=(arguments.observed_instance_price_usd_per_hour),
            smoke_params=smoke_params,
            invariance_params=invariance_params,
        )
        return

    candidate_runs: list[dict[str, Any]] = []
    for candidate in candidates:
        elapsed_seconds = time.perf_counter() - batch_started
        candidate_timeout_seconds = _remaining_candidate_seconds(
            arguments.instance_hour_limit,
            elapsed_seconds=elapsed_seconds,
            shutdown_reserve_seconds=arguments.shutdown_reserve_seconds,
        )
        command = _candidate_child_command(
            candidate=candidate,
            approval_reference=arguments.approval_reference,
            region=arguments.region,
            instance_type=arguments.instance_type,
            candidate_instance_hour_limit=candidate_timeout_seconds / 3600.0,
            batch_instance_hour_limit=arguments.instance_hour_limit,
            observed_instance_price_usd_per_hour=(arguments.observed_instance_price_usd_per_hour),
        )
        candidate_started = time.perf_counter()
        try:
            _run_candidate_child(command, timeout_seconds=candidate_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            elapsed_seconds = time.perf_counter() - batch_started
            print(
                json.dumps(
                    {
                        "status": "stopped_at_authorized_batch_deadline",
                        "candidate_id": candidate,
                        "batch_elapsed_seconds": elapsed_seconds,
                        "batch_instance_hour_limit": arguments.instance_hour_limit,
                        "shutdown_reserve_seconds": arguments.shutdown_reserve_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise TimeoutError(
                "fixed-representation invariance child was stopped before the authorized "
                f"batch deadline while running {candidate!r}"
            ) from error
        candidate_runs.append(
            {
                "candidate_id": candidate,
                "wall_seconds_including_setup_and_persistence": (
                    time.perf_counter() - candidate_started
                ),
            }
        )

    batch_elapsed_seconds = time.perf_counter() - batch_started
    print(
        json.dumps(
            {
                "status": "completed_within_authorized_batch_deadline",
                "approval_reference": arguments.approval_reference,
                "region": arguments.region,
                "instance_type": arguments.instance_type,
                "batch_instance_hour_limit": arguments.instance_hour_limit,
                "shutdown_reserve_seconds": arguments.shutdown_reserve_seconds,
                "batch_elapsed_seconds": batch_elapsed_seconds,
                "observed_batch_instance_cost_usd": (
                    batch_elapsed_seconds / 3600.0 * arguments.observed_instance_price_usd_per_hour
                ),
                "candidate_runs": candidate_runs,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _run_candidate(
    *,
    candidate: str,
    approval_reference: str,
    region: str,
    instance_type: str,
    candidate_instance_hour_limit: float,
    batch_instance_hour_limit: float,
    observed_instance_price_usd_per_hour: float,
    smoke_params: dict[str, Any],
    invariance_params: dict[str, Any],
) -> None:
    accepted_artifact = invariance_params["accepted_smoke_artifacts"][candidate]
    accepted_version = str(accepted_artifact["version"])
    compute_authorization: dict[str, Any] = {
        "approval_reference": approval_reference,
        "region": region,
        "instance_type": instance_type,
        "instance_hour_limit": candidate_instance_hour_limit,
        "batch_instance_hour_limit": batch_instance_hour_limit,
        "observed_instance_price_usd_per_hour": observed_instance_price_usd_per_hour,
    }
    runtime_params = {
        "fixed_representation_smoke": smoke_params,
        "fixed_representation_invariance": {
            **invariance_params,
            "compute_authorization": compute_authorization,
        },
        "fixed_representation_invariance_candidate": candidate,
    }
    load_versions = {
        "e02_fixed_representation_smoke_panel": accepted_version,
        "e02_fixed_representation_smoke_panel_manifest": accepted_version,
        "e02_fixed_representation_smoke_manifest": accepted_version,
    }
    with KedroSession.create(
        project_path=PROJECT_ROOT,
        runtime_params=runtime_params,
    ) as session:
        session.run(
            pipeline_name="fixed_representation_invariance",
            load_versions=load_versions,
        )


def _candidate_child_command(
    *,
    candidate: str,
    approval_reference: str,
    region: str,
    instance_type: str,
    candidate_instance_hour_limit: float,
    batch_instance_hour_limit: float,
    observed_instance_price_usd_per_hour: float,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--candidate",
        candidate,
        "--approval-reference",
        approval_reference,
        "--region",
        region,
        "--instance-type",
        instance_type,
        "--instance-hour-limit",
        str(candidate_instance_hour_limit),
        "--batch-instance-hour-limit",
        str(batch_instance_hour_limit),
        "--observed-instance-price-usd-per-hour",
        str(observed_instance_price_usd_per_hour),
        "--shutdown-reserve-seconds",
        "0",
        "--internal-candidate-child",
    ]


def _run_candidate_child(command: list[str], *, timeout_seconds: float) -> None:
    """Run setup, inference, and artifact persistence inside one external deadline."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("candidate child timeout must be finite and positive")
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        timeout=timeout_seconds,
        start_new_session=True,
    )


def _remaining_instance_hours(total_hours: float, *, elapsed_seconds: float) -> float:
    if not math.isfinite(total_hours):
        raise ValueError("total_hours must be finite")
    if total_hours <= 0.0:
        raise ValueError("total_hours must be positive")
    if not math.isfinite(elapsed_seconds):
        raise ValueError("elapsed_seconds must be finite")
    if elapsed_seconds < 0.0:
        raise ValueError("elapsed_seconds must not be negative")
    remaining_hours = total_hours - elapsed_seconds / 3600.0
    if remaining_hours <= 0.0:
        raise TimeoutError(
            "fixed-representation invariance batch reached its authorized instance-hour limit"
        )
    return remaining_hours


def _remaining_candidate_seconds(
    total_hours: float,
    *,
    elapsed_seconds: float,
    shutdown_reserve_seconds: float,
) -> float:
    remaining_seconds = (
        _remaining_instance_hours(total_hours, elapsed_seconds=elapsed_seconds) * 3600.0
    )
    if not math.isfinite(shutdown_reserve_seconds):
        raise ValueError("shutdown_reserve_seconds must be finite")
    if shutdown_reserve_seconds < 0.0:
        raise ValueError("shutdown_reserve_seconds must not be negative")
    candidate_seconds = remaining_seconds - shutdown_reserve_seconds
    if candidate_seconds <= 0.0:
        raise TimeoutError(
            "fixed-representation invariance batch reached its shutdown reserve before "
            "starting another candidate"
        )
    return candidate_seconds


def _required_transformers_version(
    candidates: tuple[str, ...],
    recipes: dict[str, Any],
) -> str:
    missing = sorted(set(candidates).difference(recipes))
    if missing:
        raise ValueError(f"candidate(s) {missing} have no configured encoder recipe")
    versions: set[str] = set()
    for candidate in candidates:
        recipe = recipes[candidate]
        if not isinstance(recipe, dict) or not str(recipe.get("transformers_version", "")).strip():
            raise ValueError(f"candidate {candidate!r} has no configured Transformers version")
        versions.add(str(recipe["transformers_version"]))
    if len(versions) != 1:
        raise ValueError(
            "one invariance command cannot mix candidate Transformers runtimes: "
            f"observed {sorted(versions)}"
        )
    return versions.pop()


if __name__ == "__main__":
    main()
