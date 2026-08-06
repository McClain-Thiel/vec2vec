"""Typed human decisions and model-free review exports for the facet audit."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vec2vec.lib.serialization import stable_json
from vec2vec.lib.text import sha256_text

_REVIEW_COLUMNS = (
    "audit_version",
    "audit_row_id",
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
    "leakage_component",
    "sequence_id",
    "addgene_id",
    "url",
    "split_grouped",
    "source_description",
    "second_review_sample",
)


class HumanAuditDecision(BaseModel):
    """One explicit source-review decision.

    Model outputs do not populate this object. It is reserved for a named human
    reviewer when model disagreement or source ambiguity needs adjudication.
    """

    model_config = ConfigDict(extra="forbid")

    audit_version: str = Field(min_length=1)
    audit_row_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(min_length=1)
    reviewed_at: datetime
    verdict: Literal["supported", "not_supported", "ambiguous", "source_unavailable"]
    reason: str = Field(max_length=1000)
    previous_decision_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("audit_version", "reviewer_id", "reason")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """Remove formatting-only whitespace."""
        return value.strip()

    @model_validator(mode="after")
    def check_review(self) -> HumanAuditDecision:
        """Require traceable time and reasons for every non-supported result."""
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("reviewed_at must include a time zone")
        if self.verdict != "supported" and not self.reason:
            raise ValueError("non-supported decisions require a reason")
        return self

    def decision_id(self) -> str:
        """Return a stable identity for the serialized decision."""
        payload = self.model_dump(mode="json")
        return sha256_text(stable_json(payload))


def build_blinded_review_table(sample: pd.DataFrame) -> pd.DataFrame:
    """Create an exportable audit table without generated or judge conclusions."""
    missing = set(_REVIEW_COLUMNS).difference(sample.columns)
    if missing:
        raise ValueError(f"facet audit sample is missing review columns: {sorted(missing)}")
    if sample.empty:
        raise ValueError("facet audit sample is empty")
    if sample["audit_row_id"].isna().any() or sample["audit_row_id"].duplicated().any():
        raise ValueError("review export needs unique, non-missing audit_row_id values")
    if sample["split_grouped"].astype(str).eq("test").any():
        raise ValueError("review export must not contain test rows")

    review = sample.loc[:, list(_REVIEW_COLUMNS)].copy()
    review = review.sort_values(["facet", "stratum", "audit_row_id"], kind="stable").reset_index(
        drop=True
    )
    review.insert(0, "review_index", range(1, len(review) + 1))
    review["decision_status"] = "unreviewed"
    review["reviewer_id"] = ""
    review["reviewed_at"] = ""
    review["human_verdict"] = ""
    review["human_reason"] = ""
    review["previous_decision_id"] = ""
    review["model_outputs_visible"] = False
    review["accepted_label_created"] = False
    return review
