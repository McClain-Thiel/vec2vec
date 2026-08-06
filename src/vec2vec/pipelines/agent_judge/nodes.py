"""Kedro nodes for explicit agent-assisted rule checks."""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pandas as pd

from vec2vec.lib import agent_judge, openrouter

DECISION_COLUMNS = (
    "pilot_index",
    "audit_row_id",
    "evidence_packet_sha256",
    "status",
    "semantic_support",
    "benchmark_scope",
    "semantic_reason",
    "scope_reason",
    "evidence_used_json",
    "suggested_canonical_values_json",
    "decision_json",
    "raw_response",
    "error",
    "judge_model",
    "requested_provider",
    "requested_seed",
    "requested_temperature",
    "requested_max_tokens",
    "requested_max_retries",
    "input_packet_output_version",
    "prompt_version",
    "prompt_hash",
    "messages_sha256",
    "upstream_generation_id",
    "upstream_model",
    "upstream_provider",
    "reasoning_enabled",
    "requested_reasoning_effort",
    "structured_output",
    "cost_usd",
    "cumulative_cost_usd",
    "human_review_required",
    "accepted_label_created",
)


def build_targeted_packets(sample: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Build deterministic packets for named revised mappings and controls."""
    return agent_judge.build_targeted_packets(sample, params)


def build_constraint_benchmark_packets(
    sample: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Freeze the complete rule-derived validation sample as judge packets."""
    return agent_judge.build_constraint_benchmark_packets(sample, params)


def select_smoke_packets(packets: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Select an explicit, fixed subset of prepared packets for a paid smoke run."""
    indices = [int(value) for value in params["pilot_indices"]]
    if len(indices) != len(set(indices)) or len(indices) != int(params["max_rows"]):
        raise ValueError("pilot_indices must be unique and contain max_rows values")
    selected = packets.loc[packets["pilot_index"].isin(indices)].copy()
    if len(selected) != len(indices):
        missing = sorted(set(indices) - set(selected["pilot_index"].astype(int)))
        raise ValueError(f"agent-judge smoke selection is missing pilot indices: {missing}")
    if selected["stratum"].nunique() != len(selected):
        raise ValueError("agent-judge smoke selection must contain one row per stratum")
    return selected.sort_values("pilot_index", kind="stable").reset_index(drop=True)


def judge_packets(
    packets: pd.DataFrame, params: dict[str, Any], credentials: dict[str, Any]
) -> pd.DataFrame:
    """Judge packets in order, retaining invalid responses and request failures."""
    api_key = credentials.get("api_key")
    if not api_key:
        raise ValueError("OpenRouter credentials are missing an 'api_key' entry")
    if packets.empty:
        raise ValueError("agent-judge packet table is empty")
    agent_judge.validate_pilot_packets(packets, params)

    model = str(params["model"])
    cap = float(params["cost_cap_usd"])
    if cap <= 0:
        raise ValueError("cost_cap_usd must be positive")

    spent = 0.0
    records: list[dict[str, Any]] = []
    with httpx.Client() as client:
        for row in packets.sort_values("pilot_index", kind="stable").to_dict("records"):
            base = {
                "pilot_index": int(row["pilot_index"]),
                "audit_row_id": str(row["audit_row_id"]),
                "evidence_packet_sha256": str(row["evidence_packet_sha256"]),
                "judge_model": model,
                "requested_provider": params["provider"],
                "requested_seed": params["seed"],
                "requested_temperature": params["temperature"],
                "requested_max_tokens": int(params["max_tokens"]),
                "requested_max_retries": int(params.get("max_retries", 0)),
                "input_packet_output_version": params.get("input_packet_output_version"),
                "prompt_version": str(row["prompt_version"]),
                "prompt_hash": str(row["prompt_hash"]),
                "messages_sha256": str(row["messages_sha256"]),
                "reasoning_enabled": bool(params["reasoning_enabled"]),
                "requested_reasoning_effort": params.get("reasoning_effort"),
                "structured_output": bool(params["structured_output"]),
                "human_review_required": True,
                "accepted_label_created": False,
            }
            if spent >= cap:
                records.append(
                    {
                        **base,
                        "status": "not_run_cost_cap",
                        "error": f"reported cumulative cost reached the ${cap:.2f} cap",
                        "cost_usd": 0.0,
                        "cumulative_cost_usd": round(spent, 6),
                    }
                )
                continue

            raw_response: str | None = None
            request_cost = 0.0
            generation_id: str | None = None
            upstream_model: str | None = None
            upstream_provider: str | None = None
            try:
                messages = json.loads(str(row["messages_json"]))
                completion = openrouter.complete(
                    client,
                    messages,
                    model=model,
                    api_key=str(api_key),
                    max_tokens=int(params["max_tokens"]),
                    temperature=(
                        None if params["temperature"] is None else float(params["temperature"])
                    ),
                    provider=params["provider"],
                    seed=None if params["seed"] is None else int(params["seed"]),
                    reasoning=(
                        {"effort": str(params["reasoning_effort"])}
                        if params.get("reasoning_effort") is not None
                        else {"enabled": bool(params["reasoning_enabled"])}
                    ),
                    response_format=(
                        {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "judge_decision",
                                "strict": True,
                                "schema": agent_judge.decision_json_schema(
                                    base["audit_row_id"], base["evidence_packet_sha256"]
                                ),
                            },
                        }
                        if params["structured_output"]
                        else None
                    ),
                    require_parameters=bool(params.get("require_parameters", True)),
                    timeout=float(params.get("request_timeout_seconds", 90.0)),
                    max_retries=int(params.get("max_retries", 0)),
                )
                raw_response = completion.text
                request_cost = completion.cost_usd
                generation_id = completion.generation_id
                upstream_model = completion.upstream_model
                upstream_provider = completion.upstream_provider
                spent += request_cost
                decision = agent_judge.parse_decision(
                    raw_response,
                    audit_row_id=base["audit_row_id"],
                    packet_sha256=base["evidence_packet_sha256"],
                )
                records.append(
                    {
                        **base,
                        "status": "valid",
                        "semantic_support": decision.semantic_support,
                        "benchmark_scope": decision.benchmark_scope,
                        "semantic_reason": decision.semantic_reason,
                        "scope_reason": decision.scope_reason,
                        "evidence_used_json": json.dumps(decision.evidence_used),
                        "suggested_canonical_values_json": json.dumps(
                            decision.suggested_canonical_values
                        ),
                        "decision_json": decision.model_dump_json(),
                        "raw_response": raw_response,
                        "error": None,
                        "upstream_generation_id": generation_id,
                        "upstream_model": upstream_model,
                        "upstream_provider": upstream_provider,
                        "cost_usd": round(request_cost, 6),
                        "cumulative_cost_usd": round(spent, 6),
                        "human_review_required": (
                            decision.semantic_support != "supported"
                            or decision.benchmark_scope != "in_scope"
                            if params.get("packet_protocol") == "constraint_benchmark"
                            else True
                        ),
                    }
                )
            except Exception as error:  # noqa: BLE001 - failures are explicit research output
                unrecorded_cost = float(getattr(error, "cost_usd", 0.0))
                if request_cost == 0.0 and unrecorded_cost:
                    request_cost = unrecorded_cost
                    spent += unrecorded_cost
                generation_id = generation_id or getattr(error, "generation_id", None)
                upstream_model = upstream_model or getattr(error, "upstream_model", None)
                upstream_provider = upstream_provider or getattr(error, "upstream_provider", None)
                records.append(
                    {
                        **base,
                        "status": (
                            "invalid_response" if raw_response is not None else "request_error"
                        ),
                        "raw_response": raw_response,
                        "error": f"{type(error).__name__}: {error}",
                        "upstream_generation_id": generation_id,
                        "upstream_model": upstream_model,
                        "upstream_provider": upstream_provider,
                        "cost_usd": round(request_cost, 6),
                        "cumulative_cost_usd": round(spent, 6),
                    }
                )

    return pd.DataFrame.from_records(records, columns=list(DECISION_COLUMNS))


def summarize(
    packets: pd.DataFrame, decisions: pd.DataFrame, params: dict[str, Any]
) -> dict[str, Any]:
    """Summarize completeness, validation failures, verdicts, and spend."""
    if len(packets) != len(decisions):
        raise ValueError("each evidence packet must have one result row")
    if decisions["audit_row_id"].duplicated().any():
        raise ValueError("agent-judge results contain duplicate audit_row_id values")
    expected = packets.set_index("audit_row_id")["evidence_packet_sha256"].astype(str)
    observed = decisions.set_index("audit_row_id")["evidence_packet_sha256"].astype(str)
    if not expected.sort_index().equals(observed.sort_index()):
        raise ValueError("agent-judge result packet identities do not match the inputs")

    status_counts = decisions["status"].value_counts().sort_index()
    semantic_counts = decisions["semantic_support"].dropna().value_counts().sort_index()
    scope_counts = decisions["benchmark_scope"].dropna().value_counts().sort_index()
    total_cost = float(pd.to_numeric(decisions["cost_usd"], errors="coerce").fillna(0).sum())
    prompt_versions = packets["prompt_version"].astype(str).unique().tolist()
    prompt_hashes = packets["prompt_hash"].astype(str).unique().tolist()
    if len(prompt_versions) != 1 or len(prompt_hashes) != 1:
        raise ValueError("agent-judge packets contain inconsistent prompt identities")
    summary = {
        "judge_version": str(params["judge_version"]),
        "input_audit_version": str(params["input_audit_version"]),
        "input_audit_output_version": str(params["input_audit_output_version"]),
        "input_packet_output_version": params.get("input_packet_output_version"),
        "model": str(params["model"]),
        "requested_provider": params["provider"],
        "requested_seed": params["seed"],
        "requested_temperature": params["temperature"],
        "requested_max_tokens": int(params["max_tokens"]),
        "requested_max_retries": int(params.get("max_retries", 0)),
        "packet_protocol": str(params.get("packet_protocol", "facet_audit")),
        "prompt_version": prompt_versions[0],
        "prompt_hash": prompt_hashes[0],
        "packet_rows": len(packets),
        "status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "semantic_support_counts": {str(key): int(value) for key, value in semantic_counts.items()},
        "benchmark_scope_counts": {str(key): int(value) for key, value in scope_counts.items()},
        "reported_total_cost_usd": round(total_cost, 6),
        "configured_cost_cap_usd": float(params["cost_cap_usd"]),
        "reasoning_enabled": bool(params["reasoning_enabled"]),
        "requested_reasoning_effort": params.get("reasoning_effort"),
        "structured_output": bool(params["structured_output"]),
        "require_parameters": bool(params.get("require_parameters", True)),
        "request_timeout_seconds": float(params.get("request_timeout_seconds", 90.0)),
        "response_schema_profile": "full",
        "manual_review_rows": int(decisions["human_review_required"].fillna(True).sum()),
        "all_rows_require_human_review": bool(
            decisions["human_review_required"].fillna(True).all()
        ),
        "accepted_labels_created": False,
        "evidence_scope": params.get(
            "evidence_scope",
            "frozen Addgene metadata and source description; no browsing or annotations",
        ),
    }
    if params.get("packet_protocol") == "constraint_benchmark":
        summary["preliminary_accuracy"] = _constraint_accuracy(packets, decisions)
    return summary


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    """Return the two-sided 95% Wilson binomial interval."""
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return [round(centre - half_width, 6), round(centre + half_width, 6)]


def _constraint_accuracy(packets: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, Any]:
    """Summarize judge pass rates without converting them into accepted labels."""
    joined = packets[["audit_row_id", "selection_group"]].merge(
        decisions[["audit_row_id", "status", "semantic_support", "benchmark_scope"]],
        on="audit_row_id",
        how="left",
        validate="one_to_one",
    )
    joined["judge_pass"] = (
        joined["status"].eq("valid")
        & joined["semantic_support"].eq("supported")
        & joined["benchmark_scope"].eq("in_scope")
    )

    def measure(frame: pd.DataFrame) -> dict[str, Any]:
        valid = frame["status"].eq("valid")
        valid_rows = int(valid.sum())
        passes = int(frame.loc[valid, "judge_pass"].sum())
        return {
            "rows": int(len(frame)),
            "valid_rows": valid_rows,
            "pass_rows": passes,
            "pass_fraction_of_valid": round(passes / valid_rows, 6) if valid_rows else None,
            "pass_fraction_95pct_wilson": _wilson_interval(passes, valid_rows),
        }

    return {
        "overall": measure(joined),
        "by_facet": {
            str(facet): measure(frame)
            for facet, frame in joined.groupby("selection_group", sort=True)
        },
        "pass_definition": "valid_and_semantically_supported_and_in_scope",
        "reference_is_model_judgment": True,
        "accepted_labels_created": False,
    }
