"""Tests for the paid description-generation step, with the API stubbed out."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import record_frame
from vec2vec.lib import openrouter
from vec2vec.pipelines.descriptions import nodes

# Mirrors the full `descriptions` block in conf/base/parameters.yml: nodes index
# params directly so the config file stays the only source of defaults.
PARAMS = {
    "model": "test/model",
    "provider": None,
    "max_tokens": 256,
    "temperature": 0.2,
    "concurrency": 2,
    "batch_size": 2,
    "cost_cap_usd": None,
    "limit": None,
}
QC_PARAMS = {"report_examples": 5}


@pytest.fixture
def metadata() -> pd.DataFrame:
    return record_frame(5)


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame({"sequence_id": ["addgene_1"], "annotation_features": [["AmpR"]]})


@pytest.fixture
def stub_api(monkeypatch):
    """Replace the OpenRouter call with a deterministic, cost-charging stub."""
    calls: list[str] = []

    def complete(client, messages, **kwargs):
        calls.append(messages[1]["content"])
        return openrouter.Completion(text=f"description {len(calls)}", cost_usd=0.01)

    monkeypatch.setattr(nodes.openrouter, "complete", complete)
    return calls


def materialize(partitions: dict) -> pd.DataFrame:
    """Save-time behaviour: call each lazy partition in order, as Kedro would."""
    frames = [build() for _, build in sorted(partitions.items())]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def test_generation_batches_and_records_provenance(metadata, features, stub_api):
    partitions = nodes.generate_descriptions(metadata, features, {}, PARAMS, {"api_key": "k"})
    assert sorted(partitions) == ["part-00000", "part-00001", "part-00002"]

    written = materialize(partitions)
    assert len(written) == 5
    assert len(stub_api) == 5
    assert list(written.columns) == list(nodes.DESCRIPTION_COLUMNS)
    assert written["generation_model"].unique().tolist() == ["test/model"]
    assert written["input_hash"].nunique() == 5
    assert written["prompt_hash"].nunique() == 1


def test_generation_skips_plasmids_that_already_have_a_description(metadata, features, stub_api):
    completed = {"part-00000": lambda: pd.DataFrame({"sequence_id": ["addgene_1", "addgene_2"]})}
    partitions = nodes.generate_descriptions(
        metadata, features, completed, PARAMS, {"api_key": "k"}
    )
    written = materialize(partitions)

    assert set(written["sequence_id"]) == {"addgene_3", "addgene_4", "addgene_5"}
    assert len(stub_api) == 3
    # Names start past the existing partitions, so a resume cannot overwrite them.
    assert min(partitions) == "part-00001"


def test_generation_stops_spending_at_the_cost_cap(metadata, features, stub_api):
    partitions = nodes.generate_descriptions(
        metadata, features, {}, {**PARAMS, "cost_cap_usd": 0.015}, {"api_key": "k"}
    )
    written = materialize(partitions)
    # The first batch of two exceeds the cap; nothing after it is charged.
    assert len(written) == 2
    assert len(stub_api) == 2


def test_generation_survives_a_failing_plasmid(metadata, features, monkeypatch):
    def complete(client, messages, **kwargs):
        if "pTest1" in messages[1]["content"]:
            raise RuntimeError("upstream refused")
        return openrouter.Completion(text="ok", cost_usd=0.0)

    monkeypatch.setattr(nodes.openrouter, "complete", complete)
    written = materialize(
        nodes.generate_descriptions(metadata, features, {}, PARAMS, {"api_key": "k"})
    )
    assert set(written["sequence_id"]) == {f"addgene_{index}" for index in range(2, 6)}


def test_generation_requires_a_key(metadata, features):
    with pytest.raises(ValueError, match="api_key"):
        nodes.generate_descriptions(metadata, features, {}, PARAMS, {})


def test_merge_deduplicates_across_partitions():
    frame = pd.DataFrame(
        {
            "sequence_id": ["addgene_1", "addgene_2"],
            "description": ["first", "second"],
        }
    )
    merged = nodes.merge_descriptions(
        {"part-00000": lambda: frame, "part-00001": lambda: frame.head(1)}
    )
    assert merged["sequence_id"].tolist() == ["addgene_1", "addgene_2"]


def test_merge_rejects_an_empty_prefix():
    with pytest.raises(ValueError, match="no description partitions"):
        nodes.merge_descriptions({})


def test_qc_reports_coverage_duplicates_and_invented_origins(metadata, features):
    descriptions = pd.DataFrame(
        {
            "sequence_id": ["addgene_1", "addgene_2", "addgene_3"],
            "description": [
                "An Ampicillin plasmid with a ColE1 origin.",
                "An Ampicillin plasmid.",
                "An Ampicillin plasmid.",
            ],
        }
    )
    report, flagged = nodes.check_descriptions(descriptions, metadata, features, QC_PARAMS)
    assert report["length"]["n"] == 3
    assert report["duplicates"]["duplicate_groups"] == 1
    assert report["field_coverage"]["bacterial_resistance"] == 1.0
    assert report["unsupported_specificity"]["n_with_invented_origin"] == 1
    assert "addgene_1" in set(flagged["sequence_id"])


def test_qc_skips_and_reports_descriptions_without_a_source_record(metadata, features):
    mixed = pd.DataFrame(
        {
            "sequence_id": ["addgene_1", "addgene_999"],
            "description": ["An Ampicillin plasmid.", "An orphan."],
        }
    )
    report, _ = nodes.check_descriptions(mixed, metadata, features, QC_PARAMS)
    assert report["described"] == 1
    assert report["orphaned_descriptions"] == 1


def test_qc_rejects_input_where_nothing_matches(metadata, features):
    orphan = pd.DataFrame({"sequence_id": ["addgene_999"], "description": ["x"]})
    with pytest.raises(ValueError, match="no description matches"):
        nodes.check_descriptions(orphan, metadata, features, QC_PARAMS)


def test_merge_ignores_empty_partitions():
    frame = pd.DataFrame({"sequence_id": ["addgene_1"], "description": ["first"]})
    empty = pd.DataFrame(columns=list(nodes.DESCRIPTION_COLUMNS))
    merged = nodes.merge_descriptions({"part-00000": lambda: frame, "part-00001": lambda: empty})
    assert merged["sequence_id"].tolist() == ["addgene_1"]


def test_merge_rejects_partitions_that_are_all_empty():
    empty = pd.DataFrame(columns=list(nodes.DESCRIPTION_COLUMNS))
    with pytest.raises(ValueError, match="every description partition is empty"):
        nodes.merge_descriptions({"part-00000": lambda: empty})
