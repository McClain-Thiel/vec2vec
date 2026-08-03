"""A Kedro dataset that streams records out of a (optionally compressed) FASTA."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator
from typing import Any

import fsspec
from kedro.io import AbstractDataset, DatasetError


def parse_fasta(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` pairs from FASTA lines.

    The header excludes the leading ``>``; sequence lines are concatenated with
    no separator.
    """
    header = ""
    chunks: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header:
                yield header, "".join(chunks)
            header, chunks = line[1:].strip(), []
        else:
            chunks.append(line)
    if header:
        yield header, "".join(chunks)


class FastaDataset(AbstractDataset[None, Iterator[tuple[str, str]]]):
    """Stream ``(header, sequence)`` pairs from a FASTA file.

    PLSDB publishes its sequences as a single bzip2-compressed FASTA of several
    gigabytes, so records are yielded lazily rather than materialized.

    Read-only: this dataset describes an upstream source of truth.

    Example catalog entry:

    .. code-block:: yaml

        plsdb_sequences:
          type: vec2vec.datasets.FastaDataset
          filepath: s3://plasmidclip/data/raw/plsdb/2024_05_31_v2/sequences.fasta.bz2
          compression: bz2
          credentials: s3
    """

    def __init__(
        self,
        *,
        filepath: str,
        compression: str | None = "infer",
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._filepath = filepath
        self._compression = compression
        self._storage_options: dict[str, Any] = {
            **(copy.deepcopy(fs_args) or {}),
            **(copy.deepcopy(credentials) or {}),
        }
        self.metadata = metadata

    def load(self) -> Iterator[tuple[str, str]]:
        """Yield each ``(header, sequence)`` pair in file order."""

        def stream() -> Iterator[tuple[str, str]]:
            with fsspec.open(
                self._filepath,
                "rt",
                encoding="utf-8",
                compression=self._compression,
                **self._storage_options,
            ) as handle:
                yield from parse_fasta(handle)

        return stream()

    def save(self, data: None) -> None:
        raise DatasetError(f"{self.__class__.__name__} is read-only")

    def _exists(self) -> bool:
        filesystem, path = fsspec.core.url_to_fs(self._filepath, **self._storage_options)
        return bool(filesystem.exists(path))

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath, "compression": self._compression}
