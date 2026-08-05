"""Tests for typed human decisions and the model-free review export."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from vec2vec.lib.audit_decision import HumanAuditDecision, build_blinded_review_table


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "audit_version": ["audit-v1"],
            "audit_row_id": ["1" * 64],
            "rule_id": ["rule-v1"],
            "facet": ["selection"],
            "relation": [None],
            "stratum": ["selection:include"],
            "source_field": ["bacterial_resistance"],
            "source_value_json": ['"Ampicillin"'],
            "classified_source_values_json": ['["Ampicillin"]'],
            "canonical_values_json": ['["ampicillin"]'],
            "proposed_claims_json": ["[]"],
            "mapping_status": ["included"],
            "proposed_evidence_state": ["verified"],
            "mapping_note": [None],
            "exclusion_reason": [None],
            "leakage_component": ["component-1"],
            "sequence_id": ["sequence-1"],
            "addgene_id": [1],
            "url": ["https://www.addgene.org/1/"],
            "split_grouped": ["train"],
            "source_description": ["Source text"],
            "generated_description": ["Generated text must stay hidden"],
            "second_review_sample": [False],
        }
    )


def test_human_decision_serializes_with_stable_identity():
    decision = HumanAuditDecision(
        audit_version="audit-v1",
        audit_row_id="1" * 64,
        reviewer_id="reviewer-1",
        reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
        verdict="supported",
        reason="",
    )

    assert len(decision.decision_id()) == 64
    assert decision.decision_id() == decision.decision_id()


def test_non_supported_human_decision_requires_reason():
    with pytest.raises(ValidationError, match="require a reason"):
        HumanAuditDecision(
            audit_version="audit-v1",
            audit_row_id="1" * 64,
            reviewer_id="reviewer-1",
            reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
            verdict="ambiguous",
            reason="  ",
        )


def test_blinded_review_export_excludes_generated_and_model_outputs():
    review = build_blinded_review_table(_sample())

    assert len(review) == 1
    assert "generated_description" not in review
    assert "validator_verdict" not in review
    assert review.iloc[0]["human_verdict"] == ""
    assert not bool(review.iloc[0]["model_outputs_visible"])
    assert not bool(review.iloc[0]["accepted_label_created"])


def test_blinded_review_export_rejects_test_rows():
    with pytest.raises(ValueError, match="test rows"):
        build_blinded_review_table(_sample().assign(split_grouped="test"))
