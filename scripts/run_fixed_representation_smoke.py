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

    load_versions = {
        "retrieval_dataset@fixed_representation_smoke": str(params["input_retrieval_version"]),
        "e00_split_grouped_v2": str(params["input_split_version"]),
    }
    for candidate in candidates:
        runtime_params = {"fixed_representation_smoke_candidate": candidate}
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
