"""A Parquet dataset that can be written from a stream of chunks."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from kedro.io import AbstractDataset, DatasetError

from vec2vec.datasets._base import FsspecDataset

Chunk = pd.DataFrame | pa.Table
Saveable = Chunk | Iterable[Chunk]


class ChunkedParquetDataset(FsspecDataset, AbstractDataset[Saveable, pd.DataFrame]):
    """Write one Parquet file from an iterable of chunks; read it as a frame.

    Ingestion nodes turn multi-gigabyte sources into tables that are far smaller
    than their inputs but still too large to hold twice in memory. Yielding
    chunks lets such a node stay bounded while producing a single well-formed
    Parquet file with one row group per chunk.

    The first chunk's schema is authoritative and later chunks are cast to it,
    so producers should pass explicitly typed :class:`pyarrow.Table` chunks when
    a column can be empty in some chunks and populated in others.

    ``load_args`` are forwarded to :func:`pandas.read_parquet`, which makes a
    column projection (``columns: [...]``) a catalog-level concern: a second
    entry pointing at the same ``filepath`` gives a cheap narrow view of a wide
    table without a second copy of the data.

    Example catalog entry:

    .. code-block:: yaml

        addgene_records:
          type: vec2vec.datasets.ChunkedParquetDataset
          filepath: s3://plasmidclip/kedro/02_intermediate/addgene_records.parquet
          credentials: s3
          save_args:
            compression: zstd
    """

    DEFAULT_SAVE_ARGS: dict[str, Any] = {"compression": "zstd"}

    def __init__(
        self,
        *,
        filepath: str,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            filepath=filepath, credentials=credentials, fs_args=fs_args, metadata=metadata
        )
        self._load_args = copy.deepcopy(load_args) or {}
        self._save_args = {**self.DEFAULT_SAVE_ARGS, **(copy.deepcopy(save_args) or {})}

    def load(self) -> pd.DataFrame:
        """Read the Parquet file into a DataFrame, honouring ``load_args``."""
        return pd.read_parquet(
            self._filepath, storage_options=self._storage_options or None, **self._load_args
        )

    def save(self, data: Saveable) -> None:
        """Write *data*, which may be one chunk or an iterable of chunks."""
        chunks = iter([data]) if isinstance(data, pd.DataFrame | pa.Table) else iter(data)
        first = next(chunks, None)
        if first is None:
            raise DatasetError(f"refusing to write an empty dataset to {self._filepath}")

        table = _as_table(first)
        filesystem, path = self._resolve()
        parent, _, _ = path.rpartition("/")
        if parent:
            filesystem.makedirs(parent, exist_ok=True)
        temporary_path = f"{path}.tmp-{uuid4().hex}"
        try:
            with filesystem.open(temporary_path, "wb") as handle:
                with pq.ParquetWriter(handle, table.schema, **self._save_args) as writer:
                    writer.write_table(table)
                    for chunk in chunks:
                        writer.write_table(_as_table(chunk).cast(table.schema))
            filesystem.move(temporary_path, path)
        except BaseException:
            if filesystem.exists(temporary_path):
                filesystem.rm(temporary_path)
            raise

    def _describe(self) -> dict[str, Any]:
        return {
            **super()._describe(),
            "load_args": self._load_args,
            "save_args": self._save_args,
        }


def _as_table(chunk: Chunk) -> pa.Table:
    """Coerce one chunk to an Arrow table without carrying a pandas index."""
    if isinstance(chunk, pa.Table):
        return chunk
    if isinstance(chunk, pd.DataFrame):
        return pa.Table.from_pandas(chunk, preserve_index=False)
    raise DatasetError(f"expected a DataFrame or Arrow table, got {type(chunk).__name__}")
