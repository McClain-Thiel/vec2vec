"""Tests for the frozen prompt and its reproducibility hashes."""

from __future__ import annotations

import json

from vec2vec.lib import prompts, qc


def record(**overrides):
    base = {
        "sequence_id": "addgene_1",
        "addgene_id": 1,
        "length_bp": 1200,
        "name": "pTest",
        "description": "A test construct",
        "bacterial_resistance": "Ampicillin",
        "plasmid_copy": "High Copy",
        "growth_strain": None,
        "origin": None,
        "backbone": "pUC19",
        "vector_types": ["Bacterial Expression"],
        "insert_species": [],
        "annotation_features": ["AmpR", "ori"],
    }
    return {**base, **overrides}


def test_compact_metadata_drops_empty_fields_and_keeps_field_order():
    compact = prompts.compact_metadata(record())
    assert "growth_strain" not in compact and "insert_species" not in compact
    assert list(compact) == [field for field in prompts.METADATA_FIELDS if field in compact]
    assert compact["accession"] == 1
    assert compact["features"] == ["AmpR", "ori"]


def test_compact_metadata_caps_the_feature_list():
    many = [f"feature{index}" for index in range(50)]
    compact = prompts.compact_metadata(record(annotation_features=many))
    assert compact["features"] == many[: prompts.MAX_FEATURES]


def test_input_hash_tracks_the_payload_not_untouched_fields():
    baseline = prompts.input_hash(record())
    assert prompts.input_hash(record()) == baseline
    assert prompts.input_hash(record(sequence_id="addgene_999")) == baseline
    assert prompts.input_hash(record(bacterial_resistance="Kanamycin")) != baseline


def test_prompt_hash_is_stable_across_calls():
    assert prompts.prompt_hash() == prompts.prompt_hash()


def test_messages_embed_the_metadata_payload():
    messages = prompts.build_messages(record())
    assert [message["role"] for message in messages] == ["system", "user"]
    payload = messages[1]["content"].split("Metadata:\n", 1)[1]
    assert json.loads(payload) == prompts.compact_metadata(record())


def test_qc_flags_an_origin_the_metadata_never_mentions():
    flags = qc.unsupported_entity_flags(
        [
            ("addgene_1", "A pUC19 vector with a ColE1 origin.", record()),
            ("addgene_2", "A pUC19 vector with an origin of replication.", record()),
        ]
    )
    assert flags[0]["sequence_id"] == "addgene_1"
    assert flags[0]["invented_origins"] == ["ColE1"]
    assert all(not flag["invented_origins"] for flag in flags[1:])


def test_qc_field_coverage_counts_only_eligible_records():
    coverage = qc.field_coverage(
        [
            ("An Ampicillin vector.", record()),
            ("A vector.", record()),
        ]
    )
    assert coverage["bacterial_resistance"] == 0.5
    assert "growth_strain" not in coverage


def test_qc_duplicate_stats_normalize_whitespace_and_case():
    stats = qc.duplicate_stats(["A vector.", "a  VECTOR.", "Another."])
    assert stats["unique"] == 2
    assert stats["duplicate_groups"] == 1
    assert stats["records_in_duplicates"] == 2


def test_qc_length_stats_flag_the_tails():
    stats = qc.length_stats(["short.", "x" * 900])
    assert stats["n"] == 2
    assert stats["very_short_lt80"] == 1
    assert stats["very_long_gt800"] == 1
