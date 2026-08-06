"""Tests for the serialized agent-judge contract and packet selection."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from vec2vec.lib import agent_judge


def _hash(number: int) -> str:
    return f"{number:064x}"


def sample_frame() -> pd.DataFrame:
    """Build three small, ordered pilot strata."""
    records = []
    strata = [
        "copy_class:missing",
        "growth_temperature:held_out",
        "bacterial_selection:proposed_exclude",
    ]
    for stratum_index, stratum in enumerate(strata, start=1):
        for rank in range(1, 4):
            number = stratum_index * 10 + rank
            records.append(
                {
                    "audit_row_id": _hash(number),
                    "audit_version": "audit-v1",
                    "rule_id": "rule-v1",
                    "facet": stratum.split(":")[0],
                    "relation": "reported_as",
                    "stratum": stratum,
                    "source_field": "field",
                    "source_value_json": json.dumps(f"value {rank}"),
                    "classified_source_values_json": "[]",
                    "canonical_values_json": "[]",
                    "proposed_claims_json": "[]",
                    "mapping_status": stratum.split(":")[1],
                    "proposed_evidence_state": "unknown",
                    "mapping_note": None,
                    "exclusion_reason": "pilot proposal",
                    "addgene_id": number,
                    "url": f"https://example.test/{number}",
                    "source_description": f"source description {number}",
                    "generated_description": f"do not show model-generated text {number}",
                    "split_grouped": "train",
                    "selection_rank": rank,
                    "selection_hash": _hash(100 - number),
                }
            )
    return pd.DataFrame.from_records(records)


def pilot_params() -> dict:
    """Return a six-row packet-selection configuration."""
    return {
        "max_rows": 6,
        "stratum_counts": {
            "copy_class:missing": 2,
            "growth_temperature:held_out": 2,
            "bacterial_selection:proposed_exclude": 2,
        },
        "input_audit_version": "audit-v1",
        "input_audit_output_version": "2026-01-01T00.00.00.000Z",
    }


def test_decision_round_trips_and_forbids_unknown_fields():
    decision = agent_judge.JudgeDecision(
        audit_row_id=_hash(1),
        evidence_packet_sha256=_hash(2),
        semantic_support="uncertain",
        benchmark_scope="out_of_scope",
        semantic_reason="The source does not resolve the biological mapping.",
        scope_reason="The stated rule excludes this value from version 1.",
        evidence_used=["source_value_json"],
        suggested_canonical_values=[],
    )
    restored = agent_judge.JudgeDecision.model_validate_json(decision.model_dump_json())
    assert restored == decision

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        agent_judge.JudgeDecision.model_validate({**decision.model_dump(), "confidence": 0.9})

    with pytest.raises(ValidationError, match="only semantically not_supported"):
        agent_judge.JudgeDecision.model_validate(
            {**decision.model_dump(), "suggested_canonical_values": ["some_value"]}
        )


def test_response_schema_binds_packet_identity():
    schema = agent_judge.decision_json_schema(_hash(1), _hash(2))
    assert schema["properties"]["audit_row_id"]["const"] == _hash(1)
    assert schema["properties"]["evidence_packet_sha256"]["const"] == _hash(2)


def test_constraint_benchmark_prompt_hash_matches_the_recorded_contract():
    assert (
        agent_judge.constraint_benchmark_prompt_hash()
        == "491cd43a849cb74d624f0c00c4ab1b6b740d6d3107f1f43c7108f728140019c4"
    )


def test_targeted_packets_select_exact_values_and_control_strata():
    params = {
        **pilot_params(),
        "max_rows": 3,
        "selectors": [
            {
                "name": "exact_value",
                "source_field": "field",
                "source_value": "value 1",
                "count": 2,
            },
            {
                "name": "control",
                "stratum": "copy_class:missing",
                "count": 1,
            },
        ],
    }
    packets = agent_judge.build_targeted_packets(sample_frame(), params)

    assert packets["selection_group"].tolist() == ["exact_value", "exact_value", "control"]
    exact_values = packets.loc[packets["selection_group"].eq("exact_value"), "evidence_packet_json"]
    assert all(json.loads(value)["source_value_json"] == '"value 1"' for value in exact_values)
    assert not packets["evidence_packet_json"].str.contains("model-generated").any()
    assert not packets["messages_json"].str.contains("model-generated").any()
    agent_judge.validate_pilot_packets(packets, params)

    changed = packets.copy()
    changed.loc[0, "evidence_packet_json"] = "{}"
    with pytest.raises(ValueError, match="evidence packet hash"):
        agent_judge.validate_pilot_packets(changed, params)


def test_targeted_packets_reject_overlapping_selectors():
    params = {
        **pilot_params(),
        "max_rows": 2,
        "selectors": [
            {"name": "first", "stratum": "copy_class:missing", "count": 1},
            {"name": "second", "stratum": "copy_class:missing", "count": 1},
        ],
    }
    with pytest.raises(ValueError, match="duplicate audit rows"):
        agent_judge.build_targeted_packets(sample_frame(), params)

    params = {
        **pilot_params(),
        "max_rows": 1,
        "selectors": [{"name": "control", "stratum": "copy_class:missing", "count": 1}],
    }
    params["input_audit_version"] = "different-audit"
    with pytest.raises(ValueError, match="does not match input_audit_version"):
        agent_judge.build_targeted_packets(sample_frame(), params)


def test_parse_decision_binds_the_response_to_the_packet():
    decision = agent_judge.JudgeDecision(
        audit_row_id=_hash(1),
        evidence_packet_sha256=_hash(2),
        semantic_support="supported",
        benchmark_scope="in_scope",
        semantic_reason="The exact source value supports the biological mapping.",
        scope_reason="The exact mapping is part of the stated rule.",
        evidence_used=["source_value_json"],
        suggested_canonical_values=[],
    )
    assert (
        agent_judge.parse_decision(
            decision.model_dump_json(), audit_row_id=_hash(1), packet_sha256=_hash(2)
        )
        == decision
    )
    with pytest.raises(ValueError, match="packet_sha256"):
        agent_judge.parse_decision(
            decision.model_dump_json(), audit_row_id=_hash(1), packet_sha256=_hash(3)
        )


def test_constraint_benchmark_packets_are_fixed_and_use_the_separate_prompt():
    sample = pd.DataFrame(
        {
            "benchmark_index": [1, 2],
            "benchmark_sample_version": ["benchmark-v1"] * 2,
            "evidence_version": ["evidence-v1"] * 2,
            "rule_contract_sha256": [_hash(90)] * 2,
            "mapping_application_id": [_hash(1), _hash(2)],
            "split_grouped": ["val", "val"],
            "rule_id": ["copy.v1", "selection.v1"],
            "facet": ["copy_class", "bacterial_selection_marker"],
            "relation": ["reported_as", "reported_selection_includes"],
            "source_field": ["plasmid_copy", "bacterial_resistance"],
            "source_value_json": ['"High Copy"', '"Kanamycin"'],
            "canonical_values_json": ['["high"]', '["kanamycin"]'],
            "mapping_section": ["included", "included"],
            "mapping_note": [None, None],
            "addgene_id": [1, 2],
            "url": ["https://example.test/1", "https://example.test/2"],
            "source_description": [None, "Kanamycin selection"],
            "plannotate_features_json": ["[]", '[{"feature":"KanR"}]'],
            "plannotate_evidence_state": ["missing", "present"],
            "benchmark_label_created": [False, False],
        }
    )
    params = {
        "packet_protocol": "constraint_benchmark",
        "max_rows": 2,
        "input_audit_version": "evidence-v1",
        "input_audit_output_version": "2026-01-01T00.00.00.000Z",
        "benchmark_sample_version": "benchmark-v1",
    }

    packets = agent_judge.build_constraint_benchmark_packets(sample, params)
    shuffled = agent_judge.build_constraint_benchmark_packets(
        sample.sample(frac=1, random_state=3), params
    )

    pd.testing.assert_frame_equal(packets, shuffled)
    assert set(packets["prompt_version"]) == {agent_judge.CONSTRAINT_BENCHMARK_PROMPT_VERSION}
    evidence = json.loads(packets.iloc[1]["evidence_packet_json"])
    assert evidence["source_value"] == "Kanamycin"
    assert evidence["canonical_values"] == ["kanamycin"]
    assert evidence["plannotate_features"] == [{"feature": "KanR"}]
    agent_judge.validate_pilot_packets(packets, params)
