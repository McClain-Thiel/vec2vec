"""Tests for flattening raw Addgene plasmids."""

from __future__ import annotations

import pyarrow as pa
import pytest

from tests.conftest import raw_plasmid
from vec2vec.lib import addgene


def test_to_record_matches_the_declared_schema(raw_plasmids):
    rows = [addgene.to_record(plasmid) for plasmid in raw_plasmids[:5]]
    table = pa.Table.from_pylist(rows, schema=addgene.RECORD_SCHEMA)
    assert table.num_rows == 5
    assert table.schema == addgene.RECORD_SCHEMA


def test_to_record_flattens_identity_and_inserts(raw_plasmids):
    record = addgene.to_record(raw_plasmids[0])
    assert record["sequence_id"] == "addgene_1"
    assert record["sequence_kind"] == "full"
    assert record["length_bp"] == len(record["sequence"]) == 1200
    assert record["backbone"] == "pUC19"
    assert record["vector_types"] == ["Bacterial Expression"]
    assert record["insert_genes"] == ["GFP"]
    assert record["insert_gene_aliases"] == ["gfp10"]
    assert record["insert_species"] == ["Aequorea victoria"]
    assert record["insert_promoters"] == ["T7"]
    assert record["url"] == "https://www.addgene.org/1/"


def test_partial_sequences_are_excluded_by_default(raw_plasmids):
    fragment = raw_plasmids[-1]
    with pytest.raises(LookupError):
        addgene.to_record(fragment)
    assert addgene.to_record(fragment, include_partial=True)["sequence_kind"] == "partial"


def test_full_sequences_win_over_partial_ones():
    plasmid = raw_plasmid(9, name="pBoth", sequence="ACGT")
    plasmid["sequences"]["public_addgene_partial_sequences"] = [{"sequence": "TTTT"}]
    sequence, kind = addgene.extract_sequence(plasmid, include_partial=True)
    assert (sequence, kind) == ("ACGT", "full")


def test_sequences_are_normalized_to_the_iupac_alphabet():
    plasmid = raw_plasmid(10, name="pMessy", sequence=" acg\nt xx N ")
    assert addgene.to_record(plasmid)["sequence"] == "ACGTN"


def test_records_without_an_id_or_sequence_are_rejected():
    with pytest.raises(LookupError):
        addgene.to_record(raw_plasmid(11, name="pEmpty", sequence=""))
    no_id = raw_plasmid(12, name="pNoId", sequence="ACGT")
    no_id["id"] = None
    with pytest.raises(LookupError):
        addgene.to_record(no_id)


def test_insert_fields_are_always_present_and_deduplicated():
    plasmid = raw_plasmid(13, name="pDup", sequence="ACGT")
    plasmid["inserts"] = [
        {"name": "GFP; gfp", "entrez_gene": [{"gene": "GFP"}, {"gene": "gfp"}]},
        {"name": "GFP"},
    ]
    record = addgene.to_record(plasmid)
    assert set(addgene.INSERT_FIELDS).issubset(record)
    assert record["insert_names"] == ["GFP", "gfp"] or record["insert_names"] == ["GFP"]
    assert record["insert_genes"] == ["GFP"]
    assert record["insert_tags"] == []


def test_malformed_nesting_raises_with_context():
    plasmid = raw_plasmid(14, name="pBad", sequence="ACGT")
    plasmid["inserts"] = {"not": "a list"}
    with pytest.raises(TypeError, match="record.inserts"):
        addgene.to_record(plasmid)
