"""Extraction of canonical plasmid records from the raw Addgene JSON export.

The raw export nests sequences, cloning details, publication references and a
variable-length ``inserts`` array under each plasmid. This module flattens one
raw plasmid object into a single typed row, and nothing else: I/O, streaming and
persistence belong to the Kedro datasets and nodes that call it.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from vec2vec.lib.sequences import clean_sequence
from vec2vec.lib.text import clean_text, split_delimited, unique_preserving_order

FULL_SEQUENCE_KEY = "public_addgene_full_sequences"
PARTIAL_SEQUENCE_KEY = "public_addgene_partial_sequences"

#: Structured insert constraints lifted out of the nested ``inserts`` array.
INSERT_FIELDS = (
    "insert_names",
    "insert_alt_names",
    "insert_genes",
    "insert_gene_aliases",
    "insert_mutations",
    "insert_tags",
    "insert_promoters",
    "insert_species",
)

_LIST = pa.list_(pa.string())

#: Explicit schema for the processed record table.
#:
#: Declared up front rather than inferred, so every Parquet shard written from a
#: streaming node shares one schema even when a shard happens to contain only
#: empty list values.
RECORD_SCHEMA = pa.schema(
    [
        ("sequence_id", pa.string()),
        ("addgene_id", pa.int64()),
        ("sequence", pa.string()),
        ("sequence_kind", pa.string()),
        ("length_bp", pa.int32()),
        ("name", pa.string()),
        ("description", pa.string()),
        ("bacterial_resistance", pa.string()),
        ("plasmid_copy", pa.string()),
        ("growth_strain", pa.string()),
        ("growth_temp", pa.string()),
        ("origin", pa.string()),
        ("backbone", pa.string()),
        ("vector_types", _LIST),
        ("article_doi", pa.string()),
        ("article_pubmed_id", pa.string()),
        ("url", pa.string()),
        *[(field, _LIST) for field in INSERT_FIELDS],
    ]
)


def _as_object(value: Any, context: str) -> dict[str, Any]:
    """Narrow a decoded JSON value to an object, treating null as empty."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object, got {type(value).__name__}")
    return value


def _as_list(value: Any, context: str) -> list[Any]:
    """Narrow a decoded JSON value to a list, treating null as empty."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array, got {type(value).__name__}")
    return value


def extract_sequence(record: dict[str, Any], *, include_partial: bool = False) -> tuple[str, str]:
    """Return the plasmid's ``(sequence, kind)``, preferring full sequences.

    Partial sequences are inserts, guides or other fragments. Treating them as
    complete plasmids creates invalid sequence-description pairs, so they are
    excluded unless *include_partial* is explicitly enabled.

    Raises:
        LookupError: when no usable sequence is present.
    """
    sequences = _as_object(record.get("sequences"), "record.sequences")
    keys = [FULL_SEQUENCE_KEY] + ([PARTIAL_SEQUENCE_KEY] if include_partial else [])
    for key in keys:
        entries = _as_list(sequences.get(key), f"record.sequences.{key}")
        if not entries:
            continue
        payload = _as_object(entries[0], f"record.sequences.{key}[0]")
        raw = payload.get("sequence", "")
        if not isinstance(raw, str):
            raise TypeError(f"record.sequences.{key}[0].sequence must be a string")
        sequence = clean_sequence(raw)
        if sequence:
            return sequence, ("full" if key == FULL_SEQUENCE_KEY else "partial")
    raise LookupError("record has no usable sequence")


def extract_insert_fields(record: dict[str, Any]) -> dict[str, list[str]]:
    """Collect structured insert constraints from the nested ``inserts`` array.

    Every :data:`INSERT_FIELDS` key is always present so that downstream schemas
    stay stable; absent constraints are empty lists.
    """
    collected: dict[str, list[str]] = {field: [] for field in INSERT_FIELDS}

    for index, entry in enumerate(_as_list(record.get("inserts"), "record.inserts")):
        insert = _as_object(entry, f"record.inserts[{index}]")
        collected["insert_names"] += split_delimited(insert.get("name"))
        collected["insert_alt_names"] += split_delimited(insert.get("alt_names"))
        collected["insert_mutations"] += split_delimited(insert.get("mutation"))
        collected["insert_tags"] += split_delimited(insert.get("tags"))

        cloning = _as_object(insert.get("cloning"), f"record.inserts[{index}].cloning")
        collected["insert_promoters"] += split_delimited(cloning.get("promoter"))

        genes = insert.get("entrez_gene")
        gene_entries = [genes] if isinstance(genes, dict) else _as_list(genes, "entrez_gene")
        for gene_index, gene_entry in enumerate(gene_entries):
            gene = _as_object(gene_entry, f"record.inserts[{index}].entrez_gene[{gene_index}]")
            collected["insert_genes"] += split_delimited(gene.get("gene"))
            collected["insert_gene_aliases"] += split_delimited(gene.get("aliases"))

        # Species arrive either as a bare name or as a ``[rank, name]`` pair.
        for species in _as_list(insert.get("species"), f"record.inserts[{index}].species"):
            if isinstance(species, list) and len(species) >= 2:
                collected["insert_species"] += split_delimited(species[1])
            elif isinstance(species, str):
                collected["insert_species"] += split_delimited(species)

    return {field: unique_preserving_order(values) for field, values in collected.items()}


def to_record(record: dict[str, Any], *, include_partial: bool = False) -> dict[str, Any]:
    """Flatten one raw Addgene plasmid into a row matching :data:`RECORD_SCHEMA`.

    Raises:
        LookupError: when the plasmid has no identifier or no usable sequence.
    """
    raw_id = record.get("id")
    if raw_id is None or raw_id == "":
        raise LookupError("record has no id")
    addgene_id = int(raw_id)

    sequence, sequence_kind = extract_sequence(record, include_partial=include_partial)
    cloning = _as_object(record.get("cloning"), "record.cloning")
    article = _as_object(record.get("article"), "record.article")

    return {
        "sequence_id": f"addgene_{addgene_id}",
        "addgene_id": addgene_id,
        "sequence": sequence,
        "sequence_kind": sequence_kind,
        "length_bp": len(sequence),
        "name": clean_text(record.get("name")),
        "description": clean_text(record.get("description")),
        "bacterial_resistance": clean_text(record.get("bacterial_resistance")),
        "plasmid_copy": clean_text(record.get("plasmid_copy")),
        "growth_strain": clean_text(record.get("growth_strain")),
        "growth_temp": clean_text(record.get("growth_temp")),
        "origin": clean_text(record.get("origin")),
        "backbone": clean_text(cloning.get("backbone")),
        "vector_types": split_delimited(cloning.get("vector_types")),
        "article_doi": clean_text(article.get("doi")),
        "article_pubmed_id": clean_text(article.get("pubmed_id")),
        "url": f"https://www.addgene.org/{addgene_id}/",
        **extract_insert_fields(record),
    }
