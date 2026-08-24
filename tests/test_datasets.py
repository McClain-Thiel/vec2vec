"""Tests for the project's custom Kedro datasets."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow as pa
import pytest
from kedro.io import DatasetError

from vec2vec.datasets import (
    ChunkedParquetDataset,
    JSONStreamDataset,
    OptionalPartitionedDataset,
)


def test_json_stream_yields_objects_under_the_prefix(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"plasmids": [{"id": 1}, {"id": 2}], "other": [{"id": 3}]}))
    dataset = JSONStreamDataset(filepath=str(path), prefix="plasmids.item")
    assert [item["id"] for item in dataset.load()] == [1, 2]
    assert dataset.exists()


def test_json_stream_rejects_non_objects(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"plasmids": ["not an object"]}))
    dataset = JSONStreamDataset(filepath=str(path), prefix="plasmids.item")
    with pytest.raises(DatasetError, match="not a JSON object"):
        list(dataset.load())


def test_json_stream_is_read_only(tmp_path):
    with pytest.raises(DatasetError, match="read-only"):
        JSONStreamDataset(filepath=str(tmp_path / "x.json")).save([{"id": 1}])


def test_chunked_parquet_writes_one_file_from_many_chunks(tmp_path):
    path = tmp_path / "nested" / "records.parquet"
    schema = pa.schema([("id", pa.int64()), ("tags", pa.list_(pa.string()))])
    chunks = [
        pa.Table.from_pylist([{"id": 1, "tags": []}], schema=schema),
        pa.Table.from_pylist([{"id": 2, "tags": ["a"]}], schema=schema),
    ]
    dataset = ChunkedParquetDataset(filepath=str(path))
    dataset.save(iter(chunks))

    loaded = dataset.load()
    assert loaded["id"].tolist() == [1, 2]
    assert loaded["tags"].tolist()[1].tolist() == ["a"]
    # One physical file, one row group per chunk.
    assert path.is_file()


def test_chunked_parquet_accepts_a_single_frame_and_projects_columns(tmp_path):
    path = tmp_path / "records.parquet"
    ChunkedParquetDataset(filepath=str(path)).save(
        pd.DataFrame({"id": [1, 2], "sequence": ["ACGT", "TTTT"]})
    )
    narrow = ChunkedParquetDataset(filepath=str(path), load_args={"columns": ["id"]})
    assert list(narrow.load().columns) == ["id"]


def test_chunked_parquet_refuses_to_write_nothing(tmp_path):
    with pytest.raises(DatasetError, match="empty dataset"):
        ChunkedParquetDataset(filepath=str(tmp_path / "empty.parquet")).save(iter([]))


def test_chunked_parquet_preserves_the_previous_file_when_a_later_chunk_is_invalid(tmp_path):
    path = tmp_path / "records.parquet"
    dataset = ChunkedParquetDataset(filepath=str(path))
    dataset.save(pd.DataFrame({"id": [1]}))

    invalid_chunks = iter([pd.DataFrame({"id": [2]}), pd.DataFrame({"id": ["not-an-integer"]})])
    with pytest.raises(DatasetError, match="not-an-integer"):
        dataset.save(invalid_chunks)

    assert dataset.load()["id"].tolist() == [1]
    assert list(tmp_path.glob("records.parquet.tmp-*")) == []


def test_optional_partitioned_reads_an_absent_prefix_as_empty(tmp_path):
    dataset = OptionalPartitionedDataset(
        path=str(tmp_path / "missing"),
        dataset="pandas.ParquetDataset",
        filename_suffix=".parquet",
    )
    assert dataset.load() == {}


def test_optional_partitioned_reads_written_partitions(tmp_path):
    path = tmp_path / "partitions"
    dataset = OptionalPartitionedDataset(
        path=str(path), dataset="pandas.ParquetDataset", filename_suffix=".parquet"
    )
    dataset.save({"part-00000": pd.DataFrame({"sequence_id": ["addgene_1"]})})
    loaded = dataset.load()
    assert set(loaded) == {"part-00000"}
    assert loaded["part-00000"]()["sequence_id"].tolist() == ["addgene_1"]
