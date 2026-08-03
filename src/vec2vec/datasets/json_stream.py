"""A Kedro dataset that streams objects out of a very large JSON document."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import fsspec
import ijson
from kedro.io import AbstractDataset, DatasetError


class JSONStreamDataset(AbstractDataset[None, Iterator[dict[str, Any]]]):
    """Incrementally yield JSON objects matching an ``ijson`` prefix.

    The raw Addgene export is a single multi-gigabyte JSON document whose
    plasmids live under ``plasmids``. Parsing it whole would need far more
    memory than the records themselves, so this dataset yields one plasmid
    object at a time and the consuming node stays bounded.

    Read-only: this dataset describes an upstream source of truth.

    Example catalog entry:

    .. code-block:: yaml

        addgene_raw:
          type: vec2vec.datasets.JSONStreamDataset
          filepath: s3://plasmidclip/data/raw/addgene/clean/raw/addgene_plasmids.json
          prefix: plasmids.item
          credentials: s3
    """

    def __init__(
        self,
        *,
        filepath: str,
        prefix: str = "item",
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._filepath = filepath
        self._prefix = prefix
        self._storage_options: dict[str, Any] = {
            **(copy.deepcopy(fs_args) or {}),
            **(copy.deepcopy(credentials) or {}),
        }
        self.metadata = metadata

    def load(self) -> Iterator[dict[str, Any]]:
        """Yield each JSON object under the configured prefix."""

        def stream() -> Iterator[dict[str, Any]]:
            with fsspec.open(self._filepath, "rb", **self._storage_options) as handle:
                for index, value in enumerate(ijson.items(handle, self._prefix)):
                    if not isinstance(value, dict):
                        raise DatasetError(
                            f"{self._filepath}:{self._prefix}[{index}] is not a JSON object"
                        )
                    yield value

        return stream()

    def save(self, data: None) -> None:
        raise DatasetError(f"{self.__class__.__name__} is read-only")

    def _exists(self) -> bool:
        filesystem, path = fsspec.core.url_to_fs(self._filepath, **self._storage_options)
        return bool(filesystem.exists(path))

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath, "prefix": self._prefix}
