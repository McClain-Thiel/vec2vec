"""A partitioned dataset that reads as empty before anything has been written."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kedro.io import DatasetError
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
        """Return the partition loaders, or ``{}`` when the prefix is empty."""
        try:
            return super().load()
        except DatasetError as error:
            if "No partitions found" in str(error):
                return {}
            raise
