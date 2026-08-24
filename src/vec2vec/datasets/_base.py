"""Shared fsspec plumbing for the project's custom datasets."""

from __future__ import annotations

import copy
from typing import Any

import fsspec
from kedro.io import DatasetError


class FsspecDataset:
    """Filepath, storage options and existence checks over any fsspec backend.

    Kedro's own datasets each re-derive this; the ones here are custom enough to
    need their own ``load``/``save`` but share exactly this much.
    """

    def __init__(
        self,
        *,
        filepath: str,
        credentials: dict[str, Any] | None = None,
        fs_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._filepath = filepath
        self._storage_options: dict[str, Any] = {
            **(copy.deepcopy(fs_args) or {}),
            **(copy.deepcopy(credentials) or {}),
        }
        self.metadata = metadata

    def _resolve(self) -> tuple[Any, str]:
        """Return the filesystem and backend-native path for this dataset."""
        return fsspec.core.url_to_fs(self._filepath, **self._storage_options)

    def _exists(self) -> bool:
        filesystem, path = self._resolve()
        return bool(filesystem.exists(path))

    def _describe(self) -> dict[str, Any]:
        return {"filepath": self._filepath}


class ReadOnlyDataset(FsspecDataset):
    """An ``FsspecDataset`` describing an upstream source this project never writes."""

    def save(self, data: Any) -> None:
        raise DatasetError(f"{type(self).__name__} is read-only")
