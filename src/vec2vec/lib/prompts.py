"""The frozen description-generation prompt and its reproducibility hashes.

Every generated description records the hash of the prompt that produced it and
the hash of the metadata payload that fed it, so any row can be traced back to
an exact, reproducible input. Changing anything in this module changes those
hashes: bump :data:`PROMPT_VERSION` whenever you do.
"""

from __future__ import annotations

import json
from typing import Any

from vec2vec.lib.text import as_list, sha256_text

PROMPT_VERSION = "desc-v2"

SYSTEM_PROMPT = (
    "You write concise natural-language descriptions for plasmid sequence-text "
    "training pairs. Stay faithful to the supplied metadata. Do not invent genes, "
    "hosts, functions, diseases, or applications. If metadata is sparse, say only "
    "what is supported. Never name a specific replication origin system (for example "
    "ColE1, pUC, pMB1, p15A, R6K, oriV) unless that exact name appears in the "
    "metadata; if the metadata only lists a generic 'ori' feature, describe it as an "
    "'origin of replication' without naming a family."
)

USER_TEMPLATE = """\
Write one description for this plasmid in 1-3 sentences.

Requirements:
- Include source identity, organism/host if available, topology/length if useful.
- Mention salient vector traits, resistance marker, insert species, and annotated
  features only when present.
- Do not name a specific origin-of-replication family (ColE1, pUC, pMB1, etc.)
  unless that name is explicitly in the metadata; otherwise write "origin of
  replication".
- Avoid sales/catalog phrasing and avoid unsupported claims.
- Return plain text only.

Metadata:
{metadata_json}"""

#: Fields lifted into the prompt, in a stable order. Also the field list that
#: :mod:`vec2vec.lib.qc` measures description coverage against.
METADATA_FIELDS = (
    "source",
    "length_bp",
    "name",
    "description",
    "organism",
    "taxonomy",
    "topology",
    "gc_content",
    "bacterial_resistance",
    "plasmid_copy",
    "growth_strain",
    "origin",
    "vector_types",
    "backbone",
    "insert_species",
    "features",
    "accession",
)

#: Number of annotated feature names shown to the model.
MAX_FEATURES = 16

_EMPTY = (None, "", [], {})


def compact_metadata(row: dict[str, Any]) -> dict[str, Any]:
    """Project one record row onto the prompt payload, dropping empty fields.

    *row* is a processed record (see :data:`vec2vec.lib.addgene.RECORD_SCHEMA`)
    optionally carrying an ``annotation_features`` list.
    """
    features = as_list(row.get("annotation_features"))[:MAX_FEATURES]
    values: dict[str, Any] = {
        "source": row.get("source", "addgene"),
        "length_bp": row.get("length_bp"),
        "name": row.get("name"),
        "description": row.get("description"),
        "organism": row.get("organism"),
        "taxonomy": row.get("taxonomy"),
        "topology": row.get("topology"),
        "gc_content": row.get("gc_content"),
        "bacterial_resistance": row.get("bacterial_resistance"),
        "plasmid_copy": row.get("plasmid_copy"),
        "growth_strain": row.get("growth_strain"),
        "origin": row.get("origin"),
        "vector_types": as_list(row.get("vector_types")),
        "backbone": row.get("backbone"),
        "insert_species": as_list(row.get("insert_species")),
        "features": features,
        "accession": row.get("accession") or row.get("addgene_id"),
    }
    return {field: values[field] for field in METADATA_FIELDS if values.get(field) not in _EMPTY}


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build the chat messages for one record using the frozen prompt."""
    metadata_json = json.dumps(compact_metadata(row), indent=2, sort_keys=True)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(metadata_json=metadata_json)},
    ]


def prompt_hash() -> str:
    """Stable hash of the frozen prompt template and its field recipe."""
    payload = json.dumps(
        {
            "version": PROMPT_VERSION,
            "system": SYSTEM_PROMPT,
            "user_template": USER_TEMPLATE,
            "fields": METADATA_FIELDS,
            "annotation_sources": ("plannotate", "plasmidkit"),
            "max_features": MAX_FEATURES,
        },
        sort_keys=True,
    )
    return sha256_text(payload)


def input_hash(row: dict[str, Any]) -> str:
    """Stable hash of the compacted metadata that actually feeds the model."""
    return sha256_text(json.dumps(compact_metadata(row), sort_keys=True))
