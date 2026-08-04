"""A partitioned dataset that reads as empty before anything has been written."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kedro_datasets.partitions import PartitionedDataset


class OptionalPartitionedDataset(PartitionedDataset):
    """:class:`PartitionedDataset` that yields ``{}`` when no partitions exist.

    Description generation is a paid, long-running step, so it writes one
    partition per batch and reads back whatever it already produced in order to
    skip that work on a re-run. The stock partitioned dataset raises when its
    prefix is empty, which would make the very first run fail; this subclass
    lets the resume input start out empty and fill in over time.
    """

    def load(self) -> dict[str, Callable[[], Any]]:
        """Return the partition loaders, or ``{}`` when the prefix is empty.

        Asks whether any partition exists rather than inferring it from the
        wording of the base class's error, so a rephrased message upstream
        cannot turn a first run into a crash.
        """
        self._invalidate_caches()
        if not self._list_partitions():
            return {}
        return super().load()
