"""Adopt descriptions that were already generated and published.

Generation is the one paid step here, and ~158k descriptions already exist in
the published Hugging Face dataset. Importing them costs nothing and keeps the
provenance columns intact, so the rest of the pipeline can be rebuilt from
scratch without paying twice. Only plasmids missing from the import need to go
through :mod:`vec2vec.pipelines.descriptions.nodes`.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from vec2vec.pipelines.descriptions.nodes import DESCRIPTION_COLUMNS

logger = logging.getLogger(__name__)


def import_published_descriptions(
    published: pd.DataFrame,
    metadata: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Project a published paired dataset down to its description columns.

    Rows whose ``sequence_id`` is absent from the current records are dropped:
    the published dataset was built from an older Addgene snapshot, and a
    description with no source record cannot be quality-checked or paired.
    """
    missing = set(DESCRIPTION_COLUMNS).difference(published.columns)
    if missing:
        raise ValueError(f"published dataset is missing columns: {sorted(missing)}")

    descriptions = (
        published.loc[:, list(DESCRIPTION_COLUMNS)]
        .drop_duplicates(subset="sequence_id", keep="first")
        .reset_index(drop=True)
    )
    known = descriptions["sequence_id"].isin(metadata["sequence_id"])
    orphans = int((~known).sum())
    if orphans:
        logger.warning("Dropped %s imported descriptions with no source record", f"{orphans:,}")

    imported = descriptions.loc[known].reset_index(drop=True)
    expected_version = params.get("expect_prompt_version")
    if expected_version is not None:
        versions = set(imported["prompt_version"].dropna().unique())
        if versions != {expected_version}:
            raise ValueError(
                f"imported descriptions carry prompt versions {sorted(versions)}, "
                f"expected {expected_version!r}"
            )

    logger.info(
        "Imported %s of %s published descriptions, covering %.1f%% of current records",
        f"{len(imported):,}",
        f"{len(published):,}",
        100 * len(imported) / max(len(metadata), 1),
    )
    return imported
