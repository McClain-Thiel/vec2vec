"""A Kedro dataset that streams records out of a (optionally compressed) FASTA."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import fsspec
from kedro.io import AbstractDataset

from vec2vec.datasets._base import ReadOnlyDataset


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


class FastaDataset(ReadOnlyDataset, AbstractDataset[None, Iterator[tuple[str, str]]]):
    """Stream ``(header, sequence)`` pairs from a FASTA file.

    PLSDB publishes its sequences as a single bzip2-compressed FASTA of several
    gigabytes, so records are yielded lazily rather than materialized.

    Example catalog entry:

    .. code-block:: yaml

        plsdb_sequences:
          type: vec2vec.datasets.FastaDataset
          filepath: s3://plasmidclip/data/raw/plsdb/2024_05_31_v2/sequences.fasta.bz2
          compression: bz2
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
        super().__init__(
            filepath=filepath, credentials=credentials, fs_args=fs_args, metadata=metadata
        )
        self._compression = compression

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

    def _describe(self) -> dict[str, Any]:
        return {**super()._describe(), "compression": self._compression}
