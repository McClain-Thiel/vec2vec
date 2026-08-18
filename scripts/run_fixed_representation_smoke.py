#!/usr/bin/env python3
"""Run one or more pinned Gate 1 DNA numerical smoke candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run each requested candidate as a separate versioned Kedro run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidates",
        help="Candidate key from parameters_fixed_representation_smoke.yml. Repeat as needed.",
    )
    parser.add_argument("--instance-type", help="Observed EC2 instance type for this run.")
    parser.add_argument(
        "--instance-hour-limit",
        type=float,
        help="Maximum instance hours authorized for this run.",
    )
    parser.add_argument(
        "--observed-instance-price-usd-per-hour",
        type=float,
        help="Observed on-demand Linux price in US dollars per instance-hour.",
    )
    arguments = parser.parse_args()
    bootstrap_project(PROJECT_ROOT)

    with KedroSession.create(project_path=PROJECT_ROOT, save_on_close=False) as session:
        params = session.load_context().params["fixed_representation_smoke"]
    configured_candidates = tuple(params["candidates"])
    candidates = tuple(arguments.candidates or configured_candidates)
    unknown = sorted(set(candidates).difference(configured_candidates))
    if unknown:
        parser.error(
            f"unknown candidate(s) {unknown}; configured candidates are {configured_candidates}"
        )
    if len(candidates) != len(set(candidates)):
        parser.error("candidate arguments must be unique")

    compute_override = (
        arguments.instance_type,
        arguments.instance_hour_limit,
        arguments.observed_instance_price_usd_per_hour,
    )
    if any(value is not None for value in compute_override) and not all(
        value is not None for value in compute_override
    ):
        parser.error("instance type, hour limit, and observed price must be overridden together")
    if arguments.instance_hour_limit is not None and arguments.instance_hour_limit <= 0:
        parser.error("instance hour limit must be greater than zero")
    if (
        arguments.observed_instance_price_usd_per_hour is not None
        and arguments.observed_instance_price_usd_per_hour <= 0
    ):
        parser.error("observed instance price must be greater than zero")

    resolved_params = dict(params)
    if arguments.instance_type is not None:
        resolved_params.update(
            {
                "instance_type": arguments.instance_type,
                "instance_hour_limit": arguments.instance_hour_limit,
                "observed_instance_price_usd_per_hour": (
                    arguments.observed_instance_price_usd_per_hour
                ),
            }
        )

    load_versions = {
        "retrieval_dataset@fixed_representation_smoke": str(params["input_retrieval_version"]),
        "e00_split_grouped_v2": str(params["input_split_version"]),
    }
    for candidate in candidates:
        runtime_params = {
            "fixed_representation_smoke": resolved_params,
            "fixed_representation_smoke_candidate": candidate,
        }
        with KedroSession.create(
            project_path=PROJECT_ROOT,
            runtime_params=runtime_params,
        ) as session:
            session.run(
                pipeline_name="fixed_representation_smoke",
                load_versions=load_versions,
            )


if __name__ == "__main__":
    main()
