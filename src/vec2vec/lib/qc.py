"""Quality control for generated plasmid descriptions.

Four screens, each answering a question the dataset would otherwise hide:

- **Length** — did generations truncate or run away?
- **Field coverage** — is the structured metadata actually reaching the text?
- **Duplicates** — are many plasmids collapsing to the same sentence?
- **Unsupported specificity** — does a description name entities the source
  metadata never mentions? This is a heuristic screen that produces a ranked
  list to eyeball, not a proof of error. Named replication origins are called
  out separately because they are the failure mode the prompt exists to prevent.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from statistics import mean, median
from typing import Any

from vec2vec.lib.prompts import METADATA_FIELDS, compact_metadata

#: Replication origins a model may invent when metadata only says a generic "ori".
KNOWN_ORIGINS = (
    "ColE1",
    "pUC",
    "pMB1",
    "p15A",
    "R6K",
    "pSC101",
    "SV40",
    "f1",
    "oriV",
    "oriP",
    "RSF1030",
    "CloDF13",
)

#: Capitalized tokens that are structural English, units, or ubiquitous terms.
STOPWORDS = frozenset(
    {
        "The", "This", "It", "A", "An", "Addgene", "DNA", "RNA", "PCR", "ORF",
        "CDS", "GFP", "Escherichia", "coli", "Homo", "sapiens", "bp", "kb",
    }
)  # fmt: skip

_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9]{1,}(?:[-_/][A-Za-z0-9]+)*\b")


def candidate_entities(text: str) -> set[str]:
    """Capitalized or alphanumeric tokens that look like named entities."""
    return {
        token
        for token in _ENTITY_PATTERN.findall(text)
        if token not in STOPWORDS and not token.isdigit()
    }


def sequence_length_summary(lengths: Any) -> dict[str, float | int]:
    """Distribution of plasmid lengths, shared by the processing and dataset reports."""
    return {
        "min": int(lengths.min()),
        "median": float(lengths.median()),
        "max": int(lengths.max()),
        "under_1kb": int((lengths < 1000).sum()),
    }


def length_stats(descriptions: list[str]) -> dict[str, float]:
    """Character and sentence distributions across all descriptions."""
    lengths = [len(text) for text in descriptions]
    sentences = [max(text.count("."), 1) for text in descriptions]
    return {
        "n": len(descriptions),
        "chars_mean": round(mean(lengths), 1),
        "chars_median": median(lengths),
        "chars_min": min(lengths),
        "chars_max": max(lengths),
        "sentences_mean": round(mean(sentences), 2),
        "very_short_lt80": sum(1 for length in lengths if length < 80),
        "very_long_gt800": sum(1 for length in lengths if length > 800),
    }


def field_coverage(pairs: list[tuple[str, dict[str, Any]]]) -> dict[str, float]:
    """Fraction of descriptions that surface each metadata field's value.

    Args:
        pairs: ``(description, record_row)`` for every described plasmid.
    """
    hits: Counter[str] = Counter()
    eligible: Counter[str] = Counter()
    for description, row in pairs:
        compact = compact_metadata(row)
        lowered = description.lower()
        for field in METADATA_FIELDS:
            value = compact.get(field)
            if value in (None, "", [], {}):
                continue
            eligible[field] += 1
            tokens = value if isinstance(value, list) else [value]
            if any(str(token).lower() in lowered for token in tokens if str(token).strip()):
                hits[field] += 1
    return {
        field: round(hits[field] / eligible[field], 3)
        for field in METADATA_FIELDS
        if eligible[field]
    }


def duplicate_stats(descriptions: list[str]) -> dict[str, Any]:
    """Exact-after-normalization duplicate groups and the worst offenders."""
    normalized = [re.sub(r"\s+", " ", text.lower()).strip() for text in descriptions]
    counts = Counter(normalized)
    duplicates = {text: count for text, count in counts.items() if count > 1}
    return {
        "unique": len(counts),
        "total": len(normalized),
        "duplicate_groups": len(duplicates),
        "records_in_duplicates": sum(duplicates.values()),
        "top_duplicates": [
            {"count": count, "text": text[:160]}
            for text, count in sorted(duplicates.items(), key=lambda item: -item[1])[:5]
        ],
    }


def unsupported_entity_flags(
    pairs: list[tuple[str, str, dict[str, Any]]],
    *,
    max_entities: int = 12,
) -> list[dict[str, Any]]:
    """Rank records by description entities absent from their source metadata.

    Args:
        pairs: ``(sequence_id, description, record_row)`` for every plasmid.
    """
    flagged: list[dict[str, Any]] = []
    for sequence_id, description, row in pairs:
        metadata_blob = json.dumps(compact_metadata(row), sort_keys=True).lower()
        unsupported = sorted(
            entity
            for entity in candidate_entities(description)
            if entity.lower() not in metadata_blob
            and entity.lower().rstrip("s") not in metadata_blob
        )
        invented_origins = sorted(
            origin
            for origin in KNOWN_ORIGINS
            if origin.lower() in description.lower() and origin.lower() not in metadata_blob
        )
        if unsupported or invented_origins:
            flagged.append(
                {
                    "sequence_id": sequence_id,
                    "n_unsupported": len(unsupported),
                    "invented_origins": invented_origins,
                    "unsupported_entities": unsupported[:max_entities],
                    "description": description,
                }
            )
    flagged.sort(
        key=lambda flag: (len(flag["invented_origins"]), flag["n_unsupported"]), reverse=True
    )
    return flagged
