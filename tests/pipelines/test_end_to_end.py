"""Run the whole data pipeline on a local fixture through a real Kedro session.

This exercises the catalog, the custom datasets, node wiring and the pipeline
registry together, so a change that only breaks in composition still fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DataCatalog

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


def run(pipeline: str | None = None, **runtime_params: Any) -> DataCatalog:
    """Run a pipeline in the `test` environment and return its catalog.

    Assertions read results back through the catalog by dataset name, so the
    tests never restate a filepath, a layer name, or Kedro's versioned-directory
    layout — that all stays in conf/base/catalog.yml where it belongs.
    """
    bootstrap_project(PROJECT_ROOT)
    with KedroSession.create(
        project_path=PROJECT_ROOT, env="test", runtime_params=runtime_params or None
    ) as session:
        context = session.load_context()
        session.run(pipeline_name=pipeline)
        return context.catalog


def seed_descriptions(catalog: DataCatalog, sequence_ids: list[str]) -> None:
    """Stand in for the paid generation step with deterministic descriptions."""
    catalog.save(
        "plasmid_descriptions",
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
        ),
    )


def test_processing_produces_records_annotations_and_a_report(lake):
    catalog = run("processing")

    records = catalog.load("addgene_records@full")
    assert len(records) == 5  # the fragment-only plasmid is excluded
    assert set(records["sequence_kind"]) == {"full"}
    assert records["sequence_id"].is_unique

    annotations = catalog.load("addgene_annotations")
    assert set(annotations["source"]) == {"plannotate", "plasmidkit"}

    features = catalog.load("addgene_annotation_features")
    by_id = dict(zip(features["sequence_id"], features["annotation_features"], strict=True))
    assert list(by_id["addgene_1"]) == ["AmpR", "GFP"]

    report = catalog.load("addgene_processing_report")
    assert report["records"]["rows"] == 5
    assert report["annotations"]["annotated_sequences"] == 2


def test_the_metadata_view_projects_away_the_sequence_column(lake):
    catalog = run("processing")
    assert "sequence" not in catalog.load("addgene_records@metadata").columns
    assert "sequence" in catalog.load("addgene_records@full").columns


def test_dataset_and_audit_produce_a_leakage_safe_retrieval_set(lake):
    catalog = run("processing")
    seed_descriptions(catalog, catalog.load("addgene_records@metadata")["sequence_id"].tolist())

    catalog = run("dataset")

    dataset = catalog.load("retrieval_dataset@full")
    assert len(dataset) == 5
    assert {"split_grouped", "split_random", "surfaced_constraints_json"} <= set(dataset.columns)

    # Plasmids 1 and 3 share a sequence and must share a split despite different
    # backbones; plasmids 3 and 4 share a backbone and must share one too.
    by_id = dataset.set_index("sequence_id")["split_grouped"]
    assert by_id["addgene_1"] == by_id["addgene_3"] == by_id["addgene_4"]

    surfaced = json.loads(dataset.loc[0, "surfaced_constraints_json"])
    assert surfaced["fields"]["bacterial_resistance"] == ["ampicillin"]
    assert surfaced["group_count"] >= 2

    audit = catalog.load("retrieval_dataset_audit")
    assert audit["rows"] == 5
    assert audit["components_straddling_grouped_split"] == 0


def test_audit_measures_hard_negative_yield(lake):
    catalog = run("processing")
    seed_descriptions(catalog, catalog.load("addgene_records@metadata")["sequence_id"].tolist())
    catalog = run("dataset")

    # The fixture is small, so audit whichever split the seed actually filled.
    split = catalog.load("retrieval_dataset@full")["split_grouped"].value_counts().idxmax()
    catalog = run("audit", **{"audit.split": split})

    summary = catalog.load("hard_negative_summary")
    assert summary["split"] == split
    assert summary["overall"]["queries"] > 0

    yields = catalog.load("hard_negative_yield")
    assert (yields["order"] >= 1).all()
    assert (yields["known_hard_negative_count"] >= 0).all()


def test_constraint_semantics_profiles_fields_components_and_plannotate_only(lake):
    catalog = run("processing")
    seed_descriptions(catalog, catalog.load("addgene_records@metadata")["sequence_id"].tolist())
    run("dataset")

    catalog = run("constraint_semantics")

    fields = catalog.load("e00_constraint_field_profile").set_index("field")
    assert fields.loc["bacterial_resistance", "known_rows"] == 5
    assert fields.loc["plasmid_copy", "normalized_value_count"] == 1

    components = catalog.load("e00_split_component_profile")
    split = catalog.load("e00_split_profile")
    assert components["rows"].sum() == 5
    assert split["components_crossing_grouped_split"] == 0

    plannotate = catalog.load("e00_plannotate_profile")
    assert plannotate["source"] == "plannotate"
    assert plannotate["annotation_rows_all"] == 2
    assert plannotate["retrieval_sequences_with_annotations"] == 1
    assert plannotate["retrieval_sequences_without_annotations"] == 4
    assert plannotate["provenance_complete"] is False


def test_facet_audit_sample_is_label_free_and_excludes_test_rows(lake):
    catalog = run("processing")
    seed_descriptions(catalog, catalog.load("addgene_records@metadata")["sequence_id"].tolist())
    catalog = run("dataset")

    # The five-row fixture has only a train split. Override the eligible set for
    # this smoke run. Kedro replaces top-level parameter blocks, so pass the
    # complete block rather than a partial mapping.
    fixture_params = dict(catalog.load("params:facet_audit"))
    fixture_params["eligible_splits"] = ["train"]
    catalog = run("facet_audit_sample", facet_audit=fixture_params)

    sample = catalog.load("e00_facet_audit_sample")
    vocabulary = catalog.load("e00_facet_audit_vocabulary")
    manifest = catalog.load("e00_facet_audit_manifest")
    assert set(sample["split_grouped"]) == {"train"}
    assert sample["audit_row_id"].is_unique
    assert {"plasmid_copy", "growth_temp", "bacterial_resistance", "vector_types"} <= set(
        vocabulary["source_field"]
    )
    assert manifest["accepted_labels_created"] is False
    assert manifest["test_metadata_used_for_sampling"] is False


def test_constraint_evidence_builds_train_labels_and_validation_sample(lake):
    catalog = run("processing")
    seed_descriptions(catalog, catalog.load("addgene_records@metadata")["sequence_id"].tolist())
    catalog = run("dataset")

    # The five-row split fixture intentionally rounds to training only. Rewrite
    # this temporary artifact with one validation and one test component so this
    # test exercises both the benchmark path and the catalog-level test filter.
    retrieval_path = next((lake / "04_feature" / "retrieval_dataset.parquet").glob("*/*.parquet"))
    retrieval = pd.read_parquet(retrieval_path)
    sequence_components = {
        value: index for index, value in enumerate(retrieval["sequence_sha256"].drop_duplicates())
    }
    retrieval["leakage_component"] = retrieval["sequence_sha256"].map(sequence_components)
    retrieval["split_grouped"] = "train"
    retrieval.loc[retrieval["leakage_component"].eq(0), "split_grouped"] = "val"
    retrieval.loc[retrieval["leakage_component"].eq(1), "split_grouped"] = "test"
    retrieval.to_parquet(retrieval_path, index=False)

    evidence_params = dict(catalog.load("params:constraint_evidence"))
    evidence_params["benchmark"] = {"target_applications": 10, "minimum_per_facet": 0}
    catalog = run("constraint_evidence", constraint_evidence=evidence_params)

    training = catalog.load("e00_training_constraint_evidence")
    benchmark = catalog.load("e00_constraint_benchmark_sample")
    manifest = catalog.load("e00_constraint_evidence_manifest")
    assert set(training["split_grouped"]) == {"train"}
    assert set(benchmark["split_grouped"]) == {"val"}
    assert training["training_label_created"].all()
    assert not benchmark["benchmark_label_created"].any()
    assert manifest["test_rows_loaded"] == 0
    assert manifest["annotation_source"] == "plannotate"
    assert manifest["plasmidkit_fallback_used"] is False
