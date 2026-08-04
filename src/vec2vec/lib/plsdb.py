"""Extraction of canonical plasmid records from the PLSDB release files.

PLSDB ships sequences as one compressed FASTA keyed by accession, with metadata
and taxonomy in separate CSVs. This module joins one FASTA record to its
metadata; streaming and persistence belong to the datasets and nodes.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from vec2vec.lib.sequences import clean_sequence
from vec2vec.lib.text import clean_text

RECORD_SCHEMA = pa.schema(
    [
        ("sequence_id", pa.string()),
        ("accession", pa.string()),
        ("sequence", pa.string()),
        ("length_bp", pa.int32()),
        ("header", pa.string()),
        ("description", pa.string()),
        ("completeness", pa.string()),
        ("genome", pa.string()),
        ("gc_content", pa.float64()),
        ("source_db", pa.string()),
        ("topology", pa.string()),
        ("organism", pa.string()),
        ("taxonomy_superkingdom", pa.string()),
        ("taxonomy_phylum", pa.string()),
        ("taxonomy_class", pa.string()),
        ("taxonomy_order", pa.string()),
        ("taxonomy_family", pa.string()),
        ("taxonomy_genus", pa.string()),
        ("taxonomy_species", pa.string()),
    ]
)

_TAXONOMY_RANKS = (
    "superkingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
)


def parse_accession(header: str) -> str:
    """Return the accession, which is the first whitespace-delimited header token."""
    return header.split()[0] if header.strip() else ""


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_record(
    header: str,
    sequence: str,
    nuccore: dict[str, Any],
    taxonomy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Flatten one PLSDB entry into a row matching :data:`RECORD_SCHEMA`.

    Raises:
        LookupError: when the header carries no accession or the sequence is empty.
    """
    accession = parse_accession(header)
    if not accession:
        raise LookupError("FASTA header has no accession")
    cleaned = clean_sequence(sequence)
    if not cleaned:
        raise LookupError(f"{accession} has an empty sequence")

    taxonomy = taxonomy or {}
    return {
        "sequence_id": f"plsdb_{accession}",
        "accession": accession,
        "sequence": cleaned,
        "length_bp": len(cleaned),
        "header": header,
        "description": clean_text(nuccore.get("NUCCORE_Description")),
        "completeness": clean_text(nuccore.get("NUCCORE_Completeness")),
        "genome": clean_text(nuccore.get("NUCCORE_Genome")),
        "gc_content": _float(nuccore.get("NUCCORE_GC")),
        "source_db": clean_text(nuccore.get("NUCCORE_Source")),
        "topology": clean_text(nuccore.get("NUCCORE_Topology")) or "unknown",
        "organism": clean_text(taxonomy.get("TAXONOMY_taxon_name")),
        **{
            f"taxonomy_{rank}": clean_text(taxonomy.get(f"TAXONOMY_{rank}"))
            for rank in _TAXONOMY_RANKS
        },
    }
