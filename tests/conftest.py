"""Shared fixtures: a tiny synthetic Addgene release the whole suite runs on."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest


def raw_plasmid(
    plasmid_id: int,
    *,
    name: str,
    sequence: str,
    backbone: str | None = "pUC19",
    resistance: str = "Ampicillin",
    partial_only: bool = False,
    **overrides: Any,
) -> dict[str, Any]:
    """Build one raw Addgene plasmid object in the shape the export uses."""
    key = "public_addgene_partial_sequences" if partial_only else "public_addgene_full_sequences"
    return {
        "id": plasmid_id,
        "name": name,
        "description": f"{name} test construct",
        "bacterial_resistance": resistance,
        "plasmid_copy": "High Copy",
        "growth_strain": "DH5alpha",
        "growth_temp": 37,
        "origin": "ori",
        "sequences": {key: [{"sequence": sequence}]},
        "cloning": {"backbone": backbone, "vector_types": "Bacterial Expression"},
        "article": {"doi": "10.1000/test", "pubmed_id": "12345"},
        "inserts": [
            {
                "name": "GFP",
                "alt_names": "eGFP",
                "mutation": None,
                "tags": "His6",
                "cloning": {"promoter": "T7"},
                "entrez_gene": {"gene": "GFP", "aliases": "gfp10"},
                "species": [["genus", "Aequorea victoria"]],
            }
        ],
        **overrides,
    }


@pytest.fixture
def raw_plasmids() -> list[dict[str, Any]]:
    """Six plasmids covering full/partial, shared backbones, and duplicate sequences."""
    return [
        raw_plasmid(1, name="pTest_alpha", sequence="ACGT" * 300),
        raw_plasmid(2, name="pTest_beta", sequence="ACGTT" * 240),
        # Same sequence as plasmid 1: must never land in a different split.
        raw_plasmid(3, name="pOther_one", sequence="ACGT" * 300, backbone="pET28a"),
        raw_plasmid(4, name="pOther_two", sequence="GGCCAT" * 200, backbone="pET28a"),
        raw_plasmid(5, name="pSolo", sequence="TTGCA" * 150, backbone=None),
        # Fragment only: excluded from the paired dataset.
        raw_plasmid(6, name="pFragment", sequence="AAGG" * 50, partial_only=True),
    ]


@pytest.fixture
def annotations() -> pd.DataFrame:
    """A normalized annotation table covering both sources."""
    return pd.DataFrame(
        {
            "sequence_id": ["addgene_1", "addgene_1", "addgene_1", "addgene_2"],
            "addgene_id": [1, 1, 1, 2],
            "source": ["plannotate", "plasmidkit", "plannotate", "plasmidkit"],
            "feature": ["AmpR", "GFP", "AmpR", "KanR"],
            "feature_type": ["CDS", "CDS", "CDS", "CDS"],
            "description": [None, None, None, None],
            "start": [1, 100, 1, 5],
            "end": [50, 200, 50, 60],
            "strand": ["+", "-", "+", "+"],
            "confidence": [0.99, 0.9, 0.99, 0.8],
        }
    )
