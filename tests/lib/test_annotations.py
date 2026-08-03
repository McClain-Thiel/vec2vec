"""Tests for normalizing the two annotation sources onto one schema."""

from __future__ import annotations

import pandas as pd

from vec2vec.lib import annotations as annotations_lib


def test_plannotate_maps_onto_the_shared_schema():
    raw = pd.DataFrame(
        {
            "plasmid_id": [1, 2],
            "Feature": ["AmpR", "ori"],
            "Type": ["CDS", "rep_origin"],
            "Description": ["beta-lactamase", "origin"],
            "qstart": [10, 500],
            "qend": [900, 1100],
            "sframe": [1, -1],
            "pident": [99.5, 100.0],
        }
    )
    normalized = annotations_lib.normalize_plannotate(raw)
    assert list(normalized.columns) == list(annotations_lib.ANNOTATION_COLUMNS)
    assert normalized["sequence_id"].tolist() == ["addgene_1", "addgene_2"]
    assert normalized["source"].unique().tolist() == ["plannotate"]
    assert normalized["strand"].tolist() == ["+", "-"]
    assert normalized["confidence"].tolist() == [0.995, 1.0]
    # Published coordinates are preserved, not reinterpreted.
    assert normalized["start"].tolist() == [10, 500]


def test_plasmidkit_maps_onto_the_shared_schema():
    raw = pd.DataFrame(
        {
            "plasmid_id": [3],
            "Feature": ["GFP"],
            "Type": ["CDS"],
            "method": ["blast"],
            "start": [1],
            "end": [717],
            "strand": ["+"],
            "confidence": [0.9],
        }
    )
    normalized = annotations_lib.normalize_plasmidkit(raw)
    assert list(normalized.columns) == list(annotations_lib.ANNOTATION_COLUMNS)
    assert normalized.loc[0, "sequence_id"] == "addgene_3"
    assert normalized.loc[0, "source"] == "plasmidkit"


def test_rows_without_coordinates_are_dropped():
    raw = pd.DataFrame(
        {
            "plasmid_id": [1, 2],
            "Feature": ["AmpR", "ori"],
            "Type": ["CDS", "rep_origin"],
            "method": ["blast", "blast"],
            "start": [10, None],
            "end": [90, 100],
            "strand": ["+", "+"],
            "confidence": [1.0, 1.0],
        }
    )
    assert len(annotations_lib.normalize_plasmidkit(raw)) == 1


def test_feature_lists_deduplicate_and_order_by_source(annotations):
    features = annotations_lib.feature_lists(annotations)
    by_id = dict(zip(features["sequence_id"], features["annotation_features"], strict=True))
    # AmpR appears twice under plannotate; plannotate is listed before plasmidkit.
    assert by_id["addgene_1"] == ["AmpR", "GFP"]
    assert by_id["addgene_2"] == ["KanR"]


def test_feature_lists_respect_a_cap(annotations):
    features = annotations_lib.feature_lists(annotations, max_features=1)
    assert all(len(value) <= 1 for value in features["annotation_features"])
