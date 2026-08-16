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
