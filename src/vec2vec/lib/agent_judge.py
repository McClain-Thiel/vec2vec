"""Serializable decisions and stable evidence packets for agent-assisted checks."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vec2vec.lib.serialization import stable_json
from vec2vec.lib.text import sha256_text

PROMPT_VERSION = "agent-judge-v4-semantic-scope"
CONSTRAINT_BENCHMARK_PROMPT_VERSION = "constraint-benchmark-judge-v1"

SYSTEM_PROMPT = """You are reviewing one proposed metadata treatment for a plasmid retrieval
benchmark.

Use only the supplied evidence. The Addgene source field and source description are primary
evidence.
The URL identifies the source record, but you cannot browse it. Do not infer facts from a missing
statement. Do not treat missing evidence as negative evidence.

Make two separate judgments.

Semantic support asks whether the source supports the proposed biological meaning. For an excluded
value, also consider whether the source supports a useful direct biological meaning that the
proposal omits.
- supported: the source supports the proposed meaning or a useful direct meaning;
- not_supported: the proposed mapping or interpretation is contradicted or misleading;
- uncertain: the biological meaning is incomplete, conflicting, or needs domain review.

Benchmark scope asks whether the meaning belongs in the stated frozen benchmark rule.
- in_scope: the rule includes this type of meaning;
- out_of_scope: the meaning can be valid but is intentionally outside this benchmark version;
- uncertain: the supplied rule does not establish the scope clearly.

Do not change semantic support to match benchmark scope. A source value can have supported
biological meaning and still be out of scope.

The treatment applies to classified_source_values_json. The full source_value_json provides
context and can contain other values that a separate treatment handles. Do not say that the whole
record is excluded when only one classified value is excluded.

Keep both reasons short and factual. Name the evidence fields you used. Suggested canonical values
must be short snake_case strings and must be empty unless the proposed biological mapping is not
supported and a different direct mapping is justified. Return one JSON object that matches the
supplied schema. Return no Markdown or other text."""

CONSTRAINT_BENCHMARK_SYSTEM_PROMPT = """You are evaluating one rule-derived metadata mapping for
a plasmid retrieval training set.

Use only the supplied evidence. The Addgene source field is primary evidence for what Addgene
reports. The source description can corroborate or conflict with it, but silence is not negative
evidence. pLannotate features are predicted supplementary evidence. Their absence or naming
ambiguity is not a contradiction. Do not infer biological performance from a metadata tag.

Make two separate judgments.

Semantic support asks whether canonical_values faithfully represent source_value under the stated
facet and relation.
- supported: every proposed canonical value is a faithful direct normalization and no relevant
  meaning was added or removed;
- not_supported: the mapping changes the meaning, adds an unsupported value, or omits a direct
  value that the relation requires;
- uncertain: the source value is ambiguous or the evidence conflicts.

Benchmark scope asks whether this exact type of reported metadata belongs in the stated facet.
- in_scope: the facet and relation preserve the limited metadata claim;
- out_of_scope: the meaning can be valid but belongs outside this benchmark facet;
- uncertain: the supplied rule does not establish scope clearly.

Judge the mapping application, not the complete plasmid. A record can contain other features and
markers without contradicting the proposed value. Keep both reasons short and factual. Name the
evidence fields you used. Suggested canonical values must be short snake_case strings and must be
empty unless semantic support is not_supported and a different direct mapping is justified. Return
one JSON object that matches the supplied schema. Return no Markdown or other text."""

_EVIDENCE_COLUMNS = (
    "audit_row_id",
    "audit_version",
    "rule_id",
    "facet",
    "relation",
    "stratum",
    "source_field",
    "source_value_json",
    "classified_source_values_json",
    "canonical_values_json",
    "proposed_claims_json",
    "mapping_status",
    "proposed_evidence_state",
    "mapping_note",
    "exclusion_reason",
    "addgene_id",
    "url",
    "source_description",
)


class JudgeDecision(BaseModel):
    """One validated judge response.

    This model is the source of truth for valid decisions. Raw or invalid model
    responses are retained separately and never converted into a decision.
    """

    model_config = ConfigDict(extra="forbid")

    audit_row_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_support: Literal["supported", "not_supported", "uncertain"]
    benchmark_scope: Literal["in_scope", "out_of_scope", "uncertain"]
    semantic_reason: str = Field(min_length=1, max_length=500)
    scope_reason: str = Field(min_length=1, max_length=500)
    evidence_used: list[str] = Field(min_length=1, max_length=8)
    suggested_canonical_values: list[str] = Field(max_length=8)

    @field_validator("semantic_reason", "scope_reason")
    @classmethod
    def strip_reasons(cls, value: str) -> str:
        """Reject an effectively empty explanation."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("reasons must contain visible text")
        return stripped

    @field_validator("evidence_used", "suggested_canonical_values")
    @classmethod
    def normalize_short_lists(cls, values: list[str]) -> list[str]:
        """Strip list values and reject empty or duplicate entries."""
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("list entries must contain visible text")
        if len(normalized) != len(set(normalized)):
            raise ValueError("list entries must be unique")
        return normalized

    @model_validator(mode="after")
    def check_suggestions(self) -> JudgeDecision:
        """Keep alternative mappings only on a rejected proposed treatment."""
        if self.semantic_support != "not_supported" and self.suggested_canonical_values:
            raise ValueError(
                "only semantically not_supported decisions can suggest canonical values"
            )
        if any(
            re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", value) is None
            for value in self.suggested_canonical_values
        ):
            raise ValueError("suggested canonical values must use snake_case")
        return self


def prompt_hash() -> str:
    """Identify the complete fixed prompt contract, including its JSON schema."""
    contract = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "response_schema": JudgeDecision.model_json_schema(),
    }
    return sha256_text(stable_json(contract))


def constraint_benchmark_prompt_hash() -> str:
    """Identify the fixed constraint-accuracy prompt and response contract."""
    contract = {
        "prompt_version": CONSTRAINT_BENCHMARK_PROMPT_VERSION,
        "system_prompt": CONSTRAINT_BENCHMARK_SYSTEM_PROMPT,
        "response_schema": JudgeDecision.model_json_schema(),
    }
    return sha256_text(stable_json(contract))


def decision_json_schema(audit_row_id: str, packet_sha256: str) -> dict[str, Any]:
    """Bind the response schema to the identity of one requested packet."""
    schema = deepcopy(JudgeDecision.model_json_schema())
    schema["properties"]["audit_row_id"]["const"] = audit_row_id
    schema["properties"]["evidence_packet_sha256"]["const"] = packet_sha256
    return schema


def build_messages(evidence: dict[str, Any], packet_sha256: str) -> list[dict[str, str]]:
    """Build the exact messages for one evidence packet."""
    request = {
        "task": "Judge whether the proposed treatment is supported by the evidence.",
        "evidence_packet_sha256": packet_sha256,
        "evidence": evidence,
        "response_schema": decision_json_schema(str(evidence["audit_row_id"]), packet_sha256),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": stable_json(request)},
    ]


def build_constraint_benchmark_messages(
    evidence: dict[str, Any], packet_sha256: str
) -> list[dict[str, str]]:
    """Build the exact messages for one constraint benchmark application."""
    request = {
        "task": "Judge the accuracy and scope of this rule-derived mapping application.",
        "evidence_packet_sha256": packet_sha256,
        "evidence": evidence,
        "response_schema": decision_json_schema(str(evidence["audit_row_id"]), packet_sha256),
    }
    return [
        {"role": "system", "content": CONSTRAINT_BENCHMARK_SYSTEM_PROMPT},
        {"role": "user", "content": stable_json(request)},
    ]


def _packet_protocol(params: Mapping[str, Any]) -> str:
    protocol = str(params.get("packet_protocol", "facet_audit"))
    if protocol not in {"facet_audit", "constraint_benchmark"}:
        raise ValueError(f"unsupported agent-judge packet protocol: {protocol!r}")
    return protocol


def _protocol_identity(params: Mapping[str, Any]) -> tuple[str, str]:
    if _packet_protocol(params) == "constraint_benchmark":
        return CONSTRAINT_BENCHMARK_PROMPT_VERSION, constraint_benchmark_prompt_hash()
    return PROMPT_VERSION, prompt_hash()


def _protocol_messages(
    evidence: dict[str, Any], packet_sha256: str, params: Mapping[str, Any]
) -> list[dict[str, str]]:
    if _packet_protocol(params) == "constraint_benchmark":
        return build_constraint_benchmark_messages(evidence, packet_sha256)
    return build_messages(evidence, packet_sha256)


def _validate_sample(sample: pd.DataFrame, params: dict[str, Any]) -> None:
    """Validate the common audit-sample contract before packet selection."""
    missing = set(_EVIDENCE_COLUMNS).difference(sample.columns)
    if missing:
        raise ValueError(f"facet audit sample is missing columns: {sorted(missing)}")
    if sample.empty:
        raise ValueError("facet audit sample is empty")
    if sample["audit_row_id"].isna().any() or sample["audit_row_id"].duplicated().any():
        raise ValueError("facet audit sample needs unique, non-missing audit_row_id values")
    if "split_grouped" in sample and sample["split_grouped"].astype(str).eq("test").any():
        raise ValueError("agent-judge packets must not contain test rows")
    audit_versions = sample["audit_version"].astype(str).unique().tolist()
    if audit_versions != [str(params["input_audit_version"])]:
        raise ValueError(
            "facet audit sample version does not match input_audit_version: "
            f"observed {audit_versions!r}"
        )


def _serialize_packets(chosen: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Freeze selected rows as hashed evidence and message packets."""
    if chosen.empty:
        raise ValueError("agent-judge packet selection is empty")
    if chosen["audit_row_id"].duplicated().any():
        raise ValueError("agent-judge packet selection contains duplicate audit rows")

    records: list[dict[str, Any]] = []
    for pilot_index, row in enumerate(chosen.to_dict("records"), start=1):
        evidence = {
            column: None if pd.isna(row[column]) else row[column] for column in _EVIDENCE_COLUMNS
        }
        evidence_json = stable_json(evidence)
        evidence_sha256 = sha256_text(evidence_json)
        messages_json = stable_json(build_messages(evidence, evidence_sha256))
        records.append(
            {
                "pilot_index": pilot_index,
                "selection_group": str(row["_packet_selection_group"]),
                "audit_row_id": str(row["audit_row_id"]),
                "stratum": str(row["stratum"]),
                "evidence_packet_json": evidence_json,
                "evidence_packet_sha256": evidence_sha256,
                "messages_json": messages_json,
                "messages_sha256": sha256_text(messages_json),
                "prompt_version": PROMPT_VERSION,
                "prompt_hash": prompt_hash(),
                "input_audit_version": str(params["input_audit_version"]),
                "input_audit_output_version": str(params["input_audit_output_version"]),
                "accepted_label_created": False,
            }
        )
    return pd.DataFrame.from_records(records)


def build_targeted_packets(sample: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Select named exact values or control strata and freeze their model inputs."""
    _validate_sample(sample, params)
    selectors = params["selectors"]
    if not isinstance(selectors, list) or not selectors:
        raise ValueError("selectors must be a non-empty list")

    selected: list[pd.DataFrame] = []
    names: set[str] = set()
    requested_rows = 0
    for selector in selectors:
        name = str(selector["name"]).strip()
        count = int(selector["count"])
        has_stratum = "stratum" in selector
        has_exact_value = "source_field" in selector and "source_value" in selector
        allowed_fields = (
            {"name", "count", "stratum"}
            if has_stratum
            else {"name", "count", "source_field", "source_value"}
        )
        if (
            not name
            or name in names
            or count < 1
            or has_stratum == has_exact_value
            or set(selector) != allowed_fields
        ):
            raise ValueError(f"invalid targeted selector: {selector!r}")
        names.add(name)
        requested_rows += count

        if has_stratum:
            available = sample.loc[sample["stratum"].astype(str).eq(str(selector["stratum"]))]
        else:
            available = sample.loc[
                sample["source_field"].astype(str).eq(str(selector["source_field"]))
                & sample["source_value_json"].astype(str).eq(stable_json(selector["source_value"]))
            ]
        available = available.sort_values(["selection_hash", "audit_row_id"], kind="stable")
        if len(available) < count:
            raise ValueError(
                f"targeted selector {name!r} has {len(available)} rows; it requires {count}"
            )
        chosen_group = available.head(count).copy()
        chosen_group["_packet_selection_group"] = name
        selected.append(chosen_group)

    if requested_rows != int(params["max_rows"]):
        raise ValueError("targeted selector counts must sum to max_rows")
    return _serialize_packets(pd.concat(selected, ignore_index=True), params)


def build_constraint_benchmark_packets(
    sample: pd.DataFrame, params: dict[str, Any]
) -> pd.DataFrame:
    """Freeze every application in the fixed validation benchmark."""
    required = {
        "benchmark_index",
        "benchmark_sample_version",
        "evidence_version",
        "rule_contract_sha256",
        "mapping_application_id",
        "split_grouped",
        "rule_id",
        "facet",
        "relation",
        "source_field",
        "source_value_json",
        "canonical_values_json",
        "mapping_section",
        "mapping_note",
        "addgene_id",
        "url",
        "source_description",
        "plannotate_features_json",
        "plannotate_evidence_state",
        "benchmark_label_created",
    }
    missing = required.difference(sample.columns)
    if missing:
        raise ValueError(f"constraint benchmark sample is missing columns: {sorted(missing)}")
    if len(sample) != int(params["max_rows"]):
        raise ValueError("constraint benchmark sample count does not match max_rows")
    if (
        sample["mapping_application_id"].isna().any()
        or sample["mapping_application_id"].duplicated().any()
    ):
        raise ValueError("constraint benchmark needs unique mapping_application_id values")
    if set(sample["split_grouped"].astype(str)) != {"val"}:
        raise ValueError("constraint benchmark packets require only validation rows")
    if sample["benchmark_label_created"].astype(bool).any():
        raise ValueError("constraint benchmark input must not contain benchmark labels")
    observed_evidence = sample["evidence_version"].astype(str).unique().tolist()
    if observed_evidence != [str(params["input_audit_version"])]:
        raise ValueError("constraint benchmark evidence version does not match configuration")
    observed_sample = sample["benchmark_sample_version"].astype(str).unique().tolist()
    if observed_sample != [str(params["benchmark_sample_version"])]:
        raise ValueError("constraint benchmark sample version does not match configuration")
    if _packet_protocol(params) != "constraint_benchmark":
        raise ValueError("constraint benchmark packets need packet_protocol=constraint_benchmark")

    prompt_version, fixed_prompt_hash = _protocol_identity(params)
    records: list[dict[str, Any]] = []
    ordered = sample.sort_values("benchmark_index", kind="stable")
    for row in ordered.to_dict("records"):
        audit_row_id = str(row["mapping_application_id"])
        evidence = {
            "audit_row_id": audit_row_id,
            "evidence_version": str(row["evidence_version"]),
            "benchmark_sample_version": str(row["benchmark_sample_version"]),
            "rule_contract_sha256": str(row["rule_contract_sha256"]),
            "rule_id": str(row["rule_id"]),
            "facet": str(row["facet"]),
            "relation": str(row["relation"]),
            "source_field": str(row["source_field"]),
            "source_value": json.loads(str(row["source_value_json"])),
            "canonical_values": json.loads(str(row["canonical_values_json"])),
            "mapping_section": str(row["mapping_section"]),
            "mapping_note": None if pd.isna(row["mapping_note"]) else str(row["mapping_note"]),
            "addgene_id": int(row["addgene_id"]),
            "url": str(row["url"]),
            "source_description": (
                None if pd.isna(row["source_description"]) else str(row["source_description"])
            ),
            "plannotate_features": json.loads(str(row["plannotate_features_json"])),
            "plannotate_evidence_state": str(row["plannotate_evidence_state"]),
        }
        evidence_json = stable_json(evidence)
        evidence_sha256 = sha256_text(evidence_json)
        messages_json = stable_json(_protocol_messages(evidence, evidence_sha256, params))
        records.append(
            {
                "pilot_index": int(row["benchmark_index"]),
                "selection_group": str(row["facet"]),
                "audit_row_id": audit_row_id,
                "stratum": str(row["facet"]),
                "evidence_packet_json": evidence_json,
                "evidence_packet_sha256": evidence_sha256,
                "messages_json": messages_json,
                "messages_sha256": sha256_text(messages_json),
                "prompt_version": prompt_version,
                "prompt_hash": fixed_prompt_hash,
                "input_audit_version": str(params["input_audit_version"]),
                "input_audit_output_version": str(params["input_audit_output_version"]),
                "accepted_label_created": False,
            }
        )
    return pd.DataFrame.from_records(records)


def validate_pilot_packets(packets: pd.DataFrame, params: dict[str, Any]) -> None:
    """Verify persisted packet identity before a paid request can run."""
    required = {
        "pilot_index",
        "selection_group",
        "audit_row_id",
        "evidence_packet_json",
        "evidence_packet_sha256",
        "messages_json",
        "messages_sha256",
        "prompt_version",
        "prompt_hash",
        "input_audit_version",
        "input_audit_output_version",
        "accepted_label_created",
    }
    missing = required.difference(packets.columns)
    if missing:
        raise ValueError(f"agent-judge packet table is missing columns: {sorted(missing)}")
    if len(packets) != int(params["max_rows"]):
        raise ValueError("agent-judge packet count does not match max_rows")
    if packets["audit_row_id"].duplicated().any():
        raise ValueError("agent-judge packet table contains duplicate audit_row_id values")
    if packets["accepted_label_created"].astype(bool).any():
        raise ValueError("agent-judge input must not contain accepted labels")

    prompt_version, fixed_prompt_hash = _protocol_identity(params)
    expected_constants = {
        "prompt_version": prompt_version,
        "prompt_hash": fixed_prompt_hash,
        "input_audit_version": str(params["input_audit_version"]),
        "input_audit_output_version": str(params["input_audit_output_version"]),
    }
    for column, expected in expected_constants.items():
        observed = packets[column].astype(str).unique().tolist()
        if observed != [expected]:
            raise ValueError(f"agent-judge packet {column} does not match configuration")

    for row in packets.to_dict("records"):
        evidence_json = str(row["evidence_packet_json"])
        evidence_sha256 = sha256_text(evidence_json)
        if evidence_sha256 != str(row["evidence_packet_sha256"]):
            raise ValueError("agent-judge evidence packet hash does not match its content")
        if "generated_description" in json.loads(evidence_json):
            raise ValueError("agent-judge evidence must not contain a generated description")
        expected_messages = stable_json(
            _protocol_messages(json.loads(evidence_json), evidence_sha256, params)
        )
        if expected_messages != str(row["messages_json"]):
            raise ValueError("agent-judge messages do not match the evidence and prompt")
        if sha256_text(expected_messages) != str(row["messages_sha256"]):
            raise ValueError("agent-judge message hash does not match its content")


def parse_decision(text: str, *, audit_row_id: str, packet_sha256: str) -> JudgeDecision:
    """Validate one raw model response and bind it to the requested packet."""
    decision = JudgeDecision.model_validate_json(text)
    if decision.audit_row_id != audit_row_id:
        raise ValueError("judge response audit_row_id does not match the request")
    if decision.evidence_packet_sha256 != packet_sha256:
        raise ValueError("judge response evidence_packet_sha256 does not match the request")
    return decision
