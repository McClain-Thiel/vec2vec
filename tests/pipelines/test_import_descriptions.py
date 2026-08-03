"""Tests for adopting already-published descriptions."""

from __future__ import annotations

import pandas as pd
import pytest

from vec2vec.pipelines.descriptions.import_nodes import import_published_descriptions


def published(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sequence_id": ["addgene_1", "addgene_2", "addgene_404"],
            "sequence": ["ACGT", "TTTT", "GGGG"],
            "description": ["first", "second", "orphan"],
            "generation_model": ["moonshotai/kimi-k2"] * 3,
            "prompt_version": ["desc-v2"] * 3,
            "prompt_hash": ["hash"] * 3,
            "input_hash": ["a", "b", "c"],
            "cost_usd": [0.001] * 3,
            "split_grouped": ["train", "val", "test"],
        }
    )
    return frame.assign(**overrides)


METADATA = pd.DataFrame({"sequence_id": ["addgene_1", "addgene_2", "addgene_3"]})


def test_import_projects_to_the_description_columns():
    imported = import_published_descriptions(published(), METADATA, {})
    assert list(imported.columns) == [
        "sequence_id",
        "description",
        "generation_model",
        "prompt_version",
        "prompt_hash",
        "input_hash",
        "cost_usd",
    ]


def test_descriptions_without_a_current_source_record_are_dropped():
    imported = import_published_descriptions(published(), METADATA, {})
    assert imported["sequence_id"].tolist() == ["addgene_1", "addgene_2"]


def test_a_prompt_version_mismatch_is_refused():
    with pytest.raises(ValueError, match="prompt versions"):
        import_published_descriptions(
            published(prompt_version="desc-v1"), METADATA, {"expect_prompt_version": "desc-v2"}
        )


def test_a_matching_prompt_version_is_accepted():
    imported = import_published_descriptions(
        published(), METADATA, {"expect_prompt_version": "desc-v2"}
    )
    assert len(imported) == 2


def test_a_dataset_without_provenance_columns_is_refused():
    with pytest.raises(ValueError, match="missing columns"):
        import_published_descriptions(published().drop(columns=["input_hash"]), METADATA, {})


def test_duplicate_sequence_ids_collapse():
    duplicated = pd.concat([published(), published()], ignore_index=True)
    assert len(import_published_descriptions(duplicated, METADATA, {})) == 2
