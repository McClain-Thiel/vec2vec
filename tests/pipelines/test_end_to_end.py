"""Run the whole data pipeline on a local fixture through a real Kedro session.

This exercises the catalog, the custom datasets, node wiring and the pipeline
registry together, so a change that only breaks in composition still fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def lake(tmp_path, monkeypatch, raw_plasmids, annotations) -> Path:
    """Materialize a fixture release on disk and point the `test` env at it."""
    raw = tmp_path / "raw" / "addgene" / "clean"
    (raw / "raw").mkdir(parents=True)
    (raw / "annotations").mkdir(parents=True)
    (raw / "raw" / "addgene_plasmids.json").write_text(json.dumps({"plasmids": raw_plasmids}))

    plannotate = annotations.loc[annotations["source"] == "plannotate"]
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "plasmid_id": plannotate["addgene_id"],
                    "Feature": plannotate["feature"],
                    "Type": plannotate["feature_type"],
                    "Description": plannotate["description"],
                    "qstart": plannotate["start"],
                    "qend": plannotate["end"],
                    "sframe": 1,
                    "pident": plannotate["confidence"] * 100,
                }
            ),
            preserve_index=False,
        ),
        raw / "annotations" / "plannotate_annotations.parquet",
    )
    plasmidkit = annotations.loc[annotations["source"] == "plasmidkit"]
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "plasmid_id": plasmidkit["addgene_id"],
                    "Feature": plasmidkit["feature"],
                    "Type": plasmidkit["feature_type"],
                    "method": "blast",
                    "start": plasmidkit["start"],
                    "end": plasmidkit["end"],
                    "strand": plasmidkit["strand"],
                    "confidence": plasmidkit["confidence"],
                }
            ),
            preserve_index=False,
        ),
        raw / "annotations" / "plasmidkit_annotations.parquet",
    )

    monkeypatch.setenv("VEC2VEC_TEST_ROOT", str(tmp_path))
    return tmp_path / "lake"


def run(pipeline: str | None = None) -> dict:
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(project_path=PROJECT_ROOT, env="test") as session:
        return session.run(pipeline_name=pipeline)


def seed_descriptions(lake: Path, sequence_ids: list[str]) -> None:
    """Stand in for the paid generation step with deterministic descriptions."""
    directory = lake / "03_primary"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "description": [
                "An Ampicillin High Copy plasmid propagated in DH5alpha with an AmpR feature."
                for _ in sequence_ids
            ],
            "generation_model": "fixture/model",
            "prompt_version": "desc-v2",
            "prompt_hash": "fixturehash",
            "input_hash": [f"input-{sequence_id}" for sequence_id in sequence_ids],
            "cost_usd": 0.0,
        }
    ).to_parquet(directory / "plasmid_descriptions.parquet", index=False)


def test_processing_produces_records_annotations_and_a_report(lake):
    run("processing")

    records = pd.read_parquet(lake / "02_intermediate" / "addgene_records.parquet")
    assert len(records) == 5  # the fragment-only plasmid is excluded
    assert set(records["sequence_kind"]) == {"full"}
    assert records["sequence_id"].is_unique

    annotations = pd.read_parquet(lake / "02_intermediate" / "addgene_annotations.parquet")
    assert set(annotations["source"]) == {"plannotate", "plasmidkit"}

    features = pd.read_parquet(lake / "02_intermediate" / "addgene_annotation_features.parquet")
    assert dict(zip(features["sequence_id"], features["annotation_features"], strict=True))[
        "addgene_1"
    ].tolist() == ["AmpR", "GFP"]

    report = json.loads((lake / "08_reporting" / "addgene_processing_report.json").read_text())
    assert report["records"]["rows"] == 5
    assert report["annotations"]["annotated_sequences"] == 2


def test_dataset_and_audit_produce_a_leakage_safe_retrieval_set(lake):
    run("processing")
    records = pd.read_parquet(lake / "02_intermediate" / "addgene_records.parquet")
    seed_descriptions(lake, records["sequence_id"].tolist())

    run("dataset")

    versions = sorted((lake / "04_feature" / "retrieval_dataset.parquet").iterdir())
    dataset = pd.read_parquet(versions[-1] / "retrieval_dataset.parquet")
    assert len(dataset) == 5
    assert {"split_grouped", "split_random", "surfaced_constraints_json"} <= set(dataset.columns)

    # Plasmids 1 and 3 share a sequence and must share a split despite different
    # backbones; plasmids 3 and 4 share a backbone and must share one too.
    by_id = dataset.set_index("sequence_id")["split_grouped"]
    assert by_id["addgene_1"] == by_id["addgene_3"] == by_id["addgene_4"]

    surfaced = json.loads(dataset.loc[0, "surfaced_constraints_json"])
    assert surfaced["fields"]["bacterial_resistance"] == ["ampicillin"]
    assert surfaced["group_count"] >= 2

    audit_versions = sorted((lake / "08_reporting" / "retrieval_dataset_audit.json").iterdir())
    audit = json.loads((audit_versions[-1] / "retrieval_dataset_audit.json").read_text())
    assert audit["rows"] == 5
    assert audit["components_straddling_grouped_split"] == 0


def test_audit_measures_hard_negative_yield(lake):
    run("processing")
    records = pd.read_parquet(lake / "02_intermediate" / "addgene_records.parquet")
    seed_descriptions(lake, records["sequence_id"].tolist())
    run("dataset")

    # The fixture is small, so audit whichever split the seed actually filled.
    versions = sorted((lake / "04_feature" / "retrieval_dataset.parquet").iterdir())
    dataset = pd.read_parquet(versions[-1] / "retrieval_dataset.parquet")
    split = dataset["split_grouped"].value_counts().idxmax()

    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(
        project_path=PROJECT_ROOT, env="test", runtime_params={"audit.split": split}
    ) as session:
        session.run(pipeline_name="audit")

    summary = json.loads((lake / "08_reporting" / "hard_negative_summary.json").read_text())
    assert summary["split"] == split
    assert summary["overall"]["queries"] > 0

    yields = pd.read_parquet(lake / "08_reporting" / "hard_negative_yield.parquet")
    assert (yields["order"] >= 1).all()
    assert (yields["known_hard_negative_count"] >= 0).all()
