"""Kedro datasets for sources the built-in dataset library does not cover.

Each exists for one reason: the upstream artifact is too large to materialize
whole, or a step needs to resume without re-doing paid work.
"""

from vec2vec.datasets.chunked_parquet import ChunkedParquetDataset
from vec2vec.datasets.json_stream import JSONStreamDataset
from vec2vec.datasets.optional_partitioned import OptionalPartitionedDataset

__all__ = [
    "ChunkedParquetDataset",
    "JSONStreamDataset",
    "OptionalPartitionedDataset",
]
