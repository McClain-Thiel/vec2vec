from __future__ import annotations

import pytest
from scripts import finalize_similarity_data


class FakeFilesystem:
    def __init__(self, matches):
        self.matches = matches

    def glob(self, pattern):
        return self.matches.get(pattern, [])


def test_latest_common_version_requires_every_artifact():
    artifacts = {
        "first": ("root", "first.parquet"),
        "second": ("root", "second.json"),
    }
    filesystem = FakeFilesystem(
        {
            "plasmidclip/root/first.parquet/*/first.parquet": [
                "plasmidclip/root/first.parquet/v1/first.parquet",
                "plasmidclip/root/first.parquet/v2/first.parquet",
            ],
            "plasmidclip/root/second.json/*/second.json": [
                "plasmidclip/root/second.json/v2/second.json"
            ],
        }
    )

    assert finalize_similarity_data._latest_common_version(filesystem, artifacts) == "v2"


def test_latest_common_version_rejects_disjoint_versions():
    artifacts = {
        "first": ("root", "first.parquet"),
        "second": ("root", "second.json"),
    }
    filesystem = FakeFilesystem(
        {
            "plasmidclip/root/first.parquet/*/first.parquet": [
                "plasmidclip/root/first.parquet/v1/first.parquet"
            ],
            "plasmidclip/root/second.json/*/second.json": [
                "plasmidclip/root/second.json/v2/second.json"
            ],
        }
    )

    with pytest.raises(RuntimeError, match="common version"):
        finalize_similarity_data._latest_common_version(filesystem, artifacts)


def test_artifact_uri_contains_versioned_kedro_layout():
    assert finalize_similarity_data._uri("kedro/layer", "table.parquet", "v1") == (
        "s3://plasmidclip/kedro/layer/table.parquet/v1/table.parquet"
    )


def test_merge_param_overrides_keeps_untouched_sibling_keys():
    # Kedro replaces a whole top-level parameter block on a runtime
    # override instead of deep-merging into it. Passing only the changed
    # leaf key would silently drop every sibling (e.g. train_fraction),
    # so the merge must restate the complete base block.
    base_params = {
        "similarity_split": {
            "protocol_version": "split_grouped_v2_v0.1",
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "seed": 42,
            "input_graph_artifact_version": None,
        }
    }

    merged = finalize_similarity_data.merge_param_overrides(
        base_params, {"similarity_split.input_graph_artifact_version": "v2"}
    )

    assert merged == {
        "similarity_split": {
            "protocol_version": "split_grouped_v2_v0.1",
            "train_fraction": 0.8,
            "val_fraction": 0.1,
            "seed": 42,
            "input_graph_artifact_version": "v2",
        }
    }


def test_merge_param_overrides_groups_multiple_leaves_by_block():
    base_params = {
        "query_benchmark": {
            "input_graph_artifact_version": None,
            "input_split_artifact_version": None,
            "top_k": [1, 5, 10, 50],
        }
    }

    merged = finalize_similarity_data.merge_param_overrides(
        base_params,
        {
            "query_benchmark.input_graph_artifact_version": "g1",
            "query_benchmark.input_split_artifact_version": "s1",
        },
    )

    assert merged == {
        "query_benchmark": {
            "input_graph_artifact_version": "g1",
            "input_split_artifact_version": "s1",
            "top_k": [1, 5, 10, 50],
        }
    }
