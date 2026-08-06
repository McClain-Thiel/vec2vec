"""Tests for the paid judge node, with OpenRouter replaced by a local stub."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from vec2vec.lib import agent_judge, openrouter
from vec2vec.pipeline_registry import register_pipelines
from vec2vec.pipelines.agent_judge import nodes

PARAMS = {
    "judge_version": "judge-v1",
    "input_audit_version": "audit-v1",
    "input_audit_output_version": "2026-01-01T00.00.00.000Z",
    "model": "test/model",
    "provider": None,
    "max_tokens": 500,
    "max_retries": 0,
    "temperature": 0.0,
    "seed": 17,
    "reasoning_enabled": False,
    "reasoning_effort": None,
    "structured_output": True,
    "cost_cap_usd": 1.0,
    "max_rows": 3,
}


def _hash(number: int) -> str:
    return f"{number:064x}"


@pytest.fixture
def packets() -> pd.DataFrame:
    records = []
    for index in range(1, 4):
        evidence = {"audit_row_id": _hash(index), "source_value_json": f'"value {index}"'}
        evidence_json = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
        evidence_hash = hashlib.sha256(evidence_json.encode()).hexdigest()
        messages = agent_judge.build_messages(evidence, evidence_hash)
        messages_json = json.dumps(messages, sort_keys=True, separators=(",", ":"))
        records.append(
            {
                "pilot_index": index,
                "selection_group": "fixture",
                "audit_row_id": _hash(index),
                "evidence_packet_json": evidence_json,
                "evidence_packet_sha256": evidence_hash,
                "messages_json": messages_json,
                "messages_sha256": hashlib.sha256(messages_json.encode()).hexdigest(),
                "prompt_version": agent_judge.PROMPT_VERSION,
                "prompt_hash": agent_judge.prompt_hash(),
                "input_audit_version": PARAMS["input_audit_version"],
                "input_audit_output_version": PARAMS["input_audit_output_version"],
                "accepted_label_created": False,
            }
        )
    return pd.DataFrame.from_records(records)


def _valid_response(row: dict, semantic_support: str = "supported") -> str:
    return agent_judge.JudgeDecision(
        audit_row_id=row["audit_row_id"],
        evidence_packet_sha256=row["evidence_packet_sha256"],
        semantic_support=semantic_support,
        benchmark_scope="in_scope",
        semantic_reason="The exact source value supports the proposed biological meaning.",
        scope_reason="The proposed meaning is part of the stated benchmark rule.",
        evidence_used=["source_value_json"],
        suggested_canonical_values=[],
    ).model_dump_json()


def test_judge_retains_valid_serialized_decisions(monkeypatch, packets):
    calls = 0

    def complete(client, messages, **kwargs):
        nonlocal calls
        row = packets.iloc[calls].to_dict()
        calls += 1
        return openrouter.Completion(text=_valid_response(row), cost_usd=0.01)

    monkeypatch.setattr(nodes.openrouter, "complete", complete)
    decisions = nodes.judge_packets(packets.head(2), {**PARAMS, "max_rows": 2}, {"api_key": "test"})

    assert set(decisions["status"]) == {"valid"}
    assert set(decisions["semantic_support"]) == {"supported"}
    assert set(decisions["benchmark_scope"]) == {"in_scope"}
    assert decisions["human_review_required"].all()
    assert not decisions["accepted_label_created"].any()
    assert decisions["cost_usd"].sum() == 0.02
    serialized = json.loads(decisions.iloc[0]["decision_json"])
    assert serialized["semantic_support"] == "supported"
    assert serialized["benchmark_scope"] == "in_scope"


def test_smoke_selection_is_explicit_and_has_one_row_per_stratum(packets):
    packets["stratum"] = ["copy", "growth", "selection"]
    selected = nodes.select_smoke_packets(packets, {"pilot_indices": [1, 3], "max_rows": 2})
    assert selected["pilot_index"].tolist() == [1, 3]

    with pytest.raises(ValueError, match="one row per stratum"):
        nodes.select_smoke_packets(
            packets.assign(stratum="same"), {"pilot_indices": [1, 3], "max_rows": 2}
        )


def test_invalid_response_is_retained_and_its_cost_is_counted(monkeypatch, packets):
    monkeypatch.setattr(
        nodes.openrouter,
        "complete",
        lambda *args, **kwargs: openrouter.Completion(text="not JSON", cost_usd=0.03),
    )
    one_row_params = {**PARAMS, "max_rows": 1}
    decisions = nodes.judge_packets(packets.head(1), one_row_params, {"api_key": "test"})
    summary = nodes.summarize(packets.head(1), decisions, one_row_params)

    assert decisions.iloc[0]["status"] == "invalid_response"
    assert decisions.iloc[0]["raw_response"] == "not JSON"
    assert decisions.iloc[0]["cost_usd"] == 0.03
    assert summary["reported_total_cost_usd"] == 0.03
    assert summary["accepted_labels_created"] is False


def test_extraction_failure_retains_upstream_cost_and_identity(monkeypatch, packets):
    def fail(*args, **kwargs):
        raise openrouter.ResponseExtractionError(
            "no final content", cost_usd=0.04, generation_id="generation-123"
        )

    monkeypatch.setattr(nodes.openrouter, "complete", fail)
    params = {**PARAMS, "max_rows": 1}
    decisions = nodes.judge_packets(packets.head(1), params, {"api_key": "test"})

    assert decisions.iloc[0]["status"] == "request_error"
    assert decisions.iloc[0]["cost_usd"] == 0.04
    assert decisions.iloc[0]["cumulative_cost_usd"] == 0.04
    assert decisions.iloc[0]["upstream_generation_id"] == "generation-123"


def test_cost_cap_marks_remaining_packets_without_calling(monkeypatch, packets):
    calls = 0

    def complete(client, messages, **kwargs):
        nonlocal calls
        row = packets.iloc[calls].to_dict()
        calls += 1
        return openrouter.Completion(text=_valid_response(row), cost_usd=0.6)

    monkeypatch.setattr(nodes.openrouter, "complete", complete)
    decisions = nodes.judge_packets(packets, PARAMS, {"api_key": "test"})

    assert calls == 2
    assert decisions["status"].tolist() == ["valid", "valid", "not_run_cost_cap"]
    assert decisions["cost_usd"].sum() == 1.2


def test_judge_requires_an_api_key(packets):
    with pytest.raises(ValueError, match="api_key"):
        nodes.judge_packets(packets, PARAMS, {})


def test_paid_judge_pipeline_is_not_in_the_default_pipeline():
    pipelines = register_pipelines()
    default_nodes = {node.name for node in pipelines["__default__"].nodes}
    targeted_nodes = {node.name for node in pipelines["agent_judge_targeted"].nodes}
    assert targeted_nodes
    assert targeted_nodes.isdisjoint(default_nodes)
    benchmark_nodes = {node.name for node in pipelines["constraint_benchmark_judge"].nodes}
    assert benchmark_nodes.isdisjoint(default_nodes)


def test_constraint_benchmark_summarizes_model_reference_accuracy(monkeypatch):
    sample = pd.DataFrame(
        {
            "benchmark_index": [1, 2, 3],
            "benchmark_sample_version": ["benchmark-v1"] * 3,
            "evidence_version": ["evidence-v1"] * 3,
            "rule_contract_sha256": [_hash(90)] * 3,
            "mapping_application_id": [_hash(1), _hash(2), _hash(3)],
            "split_grouped": ["val"] * 3,
            "rule_id": ["copy.v1", "growth.v1", "selection.v1"],
            "facet": ["copy", "growth", "selection"],
            "relation": ["reported_as", "reported_at", "reported_selection_includes"],
            "source_field": ["plasmid_copy", "growth_temp", "bacterial_resistance"],
            "source_value_json": ['"High Copy"', '"37"', '"Kanamycin"'],
            "canonical_values_json": ['["high"]', '["37_c"]', '["kanamycin"]'],
            "mapping_section": ["included"] * 3,
            "mapping_note": [None] * 3,
            "addgene_id": [1, 2, 3],
            "url": [f"https://example.test/{index}" for index in range(1, 4)],
            "source_description": [None] * 3,
            "plannotate_features_json": ["[]"] * 3,
            "plannotate_evidence_state": ["missing"] * 3,
            "benchmark_label_created": [False] * 3,
        }
    )
    params = {
        **PARAMS,
        "packet_protocol": "constraint_benchmark",
        "input_audit_version": "evidence-v1",
        "benchmark_sample_version": "benchmark-v1",
    }
    benchmark_packets = agent_judge.build_constraint_benchmark_packets(sample, params)
    calls = 0

    def complete(client, messages, **kwargs):
        nonlocal calls
        row = benchmark_packets.iloc[calls].to_dict()
        support = "not_supported" if calls == 1 else "supported"
        calls += 1
        return openrouter.Completion(text=_valid_response(row, support), cost_usd=0.01)

    monkeypatch.setattr(nodes.openrouter, "complete", complete)
    decisions = nodes.judge_packets(benchmark_packets, params, {"api_key": "test"})
    summary = nodes.summarize(benchmark_packets, decisions, params)

    assert decisions["human_review_required"].tolist() == [False, True, False]
    accuracy = summary["preliminary_accuracy"]["overall"]
    assert accuracy["valid_rows"] == 3
    assert accuracy["pass_rows"] == 2
    assert accuracy["pass_fraction_of_valid"] == 0.666667
    assert summary["manual_review_rows"] == 1
