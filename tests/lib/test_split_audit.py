"""Tests for split concentration and global-similarity result handling."""

from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.lib import split_audit
from vec2vec.lib.constraint_state import retrieval_population_sha256
from vec2vec.lib.sequences import sequence_sha256


def _retrieval() -> pd.DataFrame:
    sequences = ["ACGT" * 25, "TGCA" * 25, "AAGC" * 25, "TTGC" * 25]
    return pd.DataFrame(
        {
            "sequence_id": ["s1", "s2", "s3", "s4"],
            "sequence": sequences,
            "sequence_sha256": [sequence_sha256(sequence) for sequence in sequences],
            "family_key": ["backbone::x", "backbone::x", "id::s3", "id::s4"],
            "leakage_component": [10, 10, 20, 30],
            "split_grouped": ["train", "train", "val", "test"],
            "length_bp": [len(sequence) for sequence in sequences],
        }
    )


def _rules() -> tuple[split_audit.SimilarityRule, split_audit.SimilarityRule]:
    return (
        split_audit.SimilarityRule(0.99, 0.95, 0.95, 0.95),
        split_audit.SimilarityRule(0.95, 0.90, 0.90, 0.90),
    )


def _tokens(
    *,
    token: str,
    sequence_id: str,
    split: str,
    length: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token": [token],
            "sequence_id": [sequence_id],
            "split_grouped": [split],
            "length_bp": [length],
        }
    )


def _paf(**overrides) -> pd.DataFrame:
    record = {
        "qname": "q1",
        "qlen": 200,
        "qstart": 75,
        "qend": 175,
        "strand": "+",
        "tname": "s1",
        "tlen": 100,
        "tstart": 0,
        "tend": 100,
        "matching_bases": 99,
        "alignment_block_length": 100,
        "mapq": 60,
    }
    record.update(overrides)
    return pd.DataFrame.from_records([record], columns=split_audit.PAF_COLUMNS)


def test_retrieval_validation_and_concentration_are_content_stable():
    retrieval = _retrieval()
    expected_hash = retrieval_population_sha256(retrieval)
    first = split_audit.validate_retrieval(
        retrieval,
        expected_population_sha256=expected_hash,
    )
    second = split_audit.validate_retrieval(
        retrieval.sample(frac=1, random_state=4),
        expected_population_sha256=expected_hash,
    )

    assert first == second
    assert first["groups_crossing_existing_split"] == {
        "family_key": 0,
        "sequence_sha256": 0,
        "leakage_component": 0,
    }
    profile, summary = split_audit.profile_split_concentration(retrieval)
    assert profile["rows"].sum() == 4
    assert summary["train"]["effective_component_count"] == 1.0
    assert summary["train"]["component_macro_required"] is True


def test_retrieval_validation_rejects_duplicate_hash_and_split_invariants():
    retrieval = _retrieval()
    expected_hash = retrieval_population_sha256(retrieval)

    duplicate = pd.concat([retrieval, retrieval.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate sequence_id"):
        split_audit.validate_retrieval(
            duplicate,
            expected_population_sha256=expected_hash,
        )

    bad_hash = retrieval.copy()
    bad_hash.loc[0, "sequence_sha256"] = "wrong"
    with pytest.raises(ValueError, match="do not match sequence content"):
        split_audit.validate_retrieval(
            bad_hash,
            expected_population_sha256=expected_hash,
        )

    impure = retrieval.copy()
    impure.loc[1, "split_grouped"] = "test"
    impure_hash = retrieval_population_sha256(impure)
    with pytest.raises(ValueError, match="family_key groups cross"):
        split_audit.validate_retrieval(
            impure,
            expected_population_sha256=impure_hash,
        )


def test_minimap_paf_classification_uses_the_same_inclusive_rules():
    primary, sensitivity = _rules()
    edges = split_audit.classify_minimap_alignments(
        _paf(),
        query_tokens=_tokens(token="q1", sequence_id="query", split="val", length=100),
        subject_tokens=_tokens(token="s1", sequence_id="subject", split="train", length=100),
        search_pair="val_vs_train",
        query_repeat=2,
        primary_rule=primary,
        sensitivity_rule=sensitivity,
    )

    assert len(edges) == 1
    assert edges.loc[0, "identity"] == 0.99
    assert edges.loc[0, "query_coverage"] == 1.0
    assert edges.loc[0, "subject_coverage"] == 1.0
    assert bool(edges.loc[0, "primary_near_duplicate"])


def test_minimap_paf_rejects_invalid_coordinates():
    primary, sensitivity = _rules()
    with pytest.raises(ValueError, match="query coordinates are out of bounds"):
        split_audit.classify_minimap_alignments(
            _paf(qend=201),
            query_tokens=_tokens(token="q1", sequence_id="query", split="val", length=100),
            subject_tokens=_tokens(token="s1", sequence_id="subject", split="train", length=100),
            search_pair="val_vs_train",
            query_repeat=2,
            primary_rule=primary,
            sensitivity_rule=sensitivity,
        )


def test_augmented_summary_reports_cross_split_component_merges():
    retrieval = _retrieval()
    edges = pd.DataFrame(
        {
            "query_sequence_id": ["s3"],
            "subject_sequence_id": ["s1"],
            "primary_near_duplicate": [True],
            "sensitivity_near_duplicate": [True],
        }
    )
    summary = split_audit.augmented_component_summary(
        retrieval,
        edges,
        edge_flag="primary_near_duplicate",
    )

    assert summary["qualifying_edges"] == 1
    assert summary["augmented_components_merging_current_components"] == 1
    assert summary["augmented_components_crossing_original_splits"] == 1
    assert summary["largest_augmented_component_rows"] == 3
