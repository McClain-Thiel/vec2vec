"""Gate 2 set-supervision node."""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd

from vec2vec.lib import set_supervision
from vec2vec.pipelines.fixed_representation_bakeoff.nodes import (
    _git_provenance,
    _runtime_provenance,
)


def run_comparison(
    pairs: pd.DataFrame,
    queries: pd.DataFrame,
    validation_states: pd.DataFrame,
    all_query_states: pd.DataFrame,
    query_manifest: dict[str, Any],
    input_manifest: dict[str, Any],
    dna_features: pd.DataFrame,
    dna_manifest: dict[str, Any],
    text_features: pd.DataFrame,
    text_manifest: dict[str, Any],
    params: dict[str, Any],
) -> tuple[Any, ...]:
    """Run the complete approved comparison and attach execution provenance."""
    compute = _validated_compute_authorization(params)
    deadline = time.monotonic() + float(compute["instance_hour_limit"]) * 3600.0
    started = time.perf_counter()
    outputs = set_supervision.run_set_supervision_comparison(
        pairs,
        queries,
        validation_states,
        all_query_states,
        query_manifest,
        input_manifest,
        dna_features,
        dna_manifest,
        text_features,
        text_manifest,
        params,
        deadline_monotonic=deadline,
    )
    elapsed = time.perf_counter() - started
    *tables, report = outputs
    report["compute_authorization"] = compute
    report["runtime"] = _runtime_provenance()
    report["git"] = _git_provenance()
    report["elapsed_seconds"] = elapsed
    report["estimated_compute_cost_usd"] = (
        elapsed / 3600.0 * float(compute["observed_instance_price_usd_per_hour"])
    )
    return (*tables, report)


def _validated_compute_authorization(params: dict[str, Any]) -> dict[str, Any]:
    expected = params.get("approved_compute_authorization")
    observed = params.get("compute_authorization")
    if not isinstance(expected, dict) or not isinstance(observed, dict):
        raise ValueError("Gate 2 requires frozen and runtime compute authorization")
    normalized = {
        "approval_reference": str(observed.get("approval_reference", "")),
        "region": str(observed.get("region", "")),
        "instance_type": str(observed.get("instance_type", "")),
        "instance_hour_limit": float(observed.get("instance_hour_limit", math.nan)),
        "observed_instance_price_usd_per_hour": float(
            observed.get("observed_instance_price_usd_per_hour", math.nan)
        ),
    }
    if normalized != expected:
        raise ValueError(f"runtime Gate 2 compute authorization differs: {normalized}")
    if not math.isfinite(normalized["instance_hour_limit"]) or not math.isfinite(
        normalized["observed_instance_price_usd_per_hour"]
    ):
        raise ValueError("Gate 2 compute limits must be finite")
    if normalized["instance_hour_limit"] <= 0.0:
        raise ValueError("Gate 2 instance-hour limit must be positive")
    if normalized["observed_instance_price_usd_per_hour"] <= 0.0:
        raise ValueError("Gate 2 instance price must be positive")
    return normalized
