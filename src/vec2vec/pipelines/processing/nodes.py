"""Turn raw Addgene and PLSDB releases into canonical record tables."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pyarrow as pa

from vec2vec.lib import addgene, plsdb, qc
from vec2vec.lib import annotations as annotations_lib

logger = logging.getLogger(__name__)


def _chunks(
    rows: Iterator[dict[str, Any]],
    schema: pa.Schema,
    size: int,
    limit: int | None = None,
) -> Iterator[pa.Table]:
    """Group flattened rows into Arrow tables of at most *size* rows.

    The schema is passed explicitly rather than inferred, so every chunk shares
    one schema even when a column happens to be empty throughout a chunk.
    """
    if size < 1:
        raise ValueError("chunk_size must be positive")
    batch: list[dict[str, Any]] = []
    for count, row in enumerate(rows, start=1):
        batch.append(row)
        if len(batch) >= size:
            yield pa.Table.from_pylist(batch, schema=schema)
            batch = []
        if limit is not None and count >= limit:
            break
    if batch:
        yield pa.Table.from_pylist(batch, schema=schema)


def process_addgene_records(
    plasmids: Iterator[dict[str, Any]],
    params: dict[str, Any],
) -> Iterator[pa.Table]:
    """Flatten the raw Addgene JSON stream into canonical record chunks.

    Returns a generator rather than a frame so a multi-gigabyte source is never
    held in memory. Plasmids without an id or without a usable sequence are
    dropped and counted; ``include_partial`` decides whether fragment sequences
    count as usable.
    """
    include_partial = bool(params["include_partial"])
    limit = params["limit"]

    def rows() -> Iterator[dict[str, Any]]:
        kept = 0
        skipped = 0
        for plasmid in plasmids:
            try:
                row = addgene.to_record(plasmid, include_partial=include_partial)
            except LookupError:
                skipped += 1
                continue
            kept += 1
            if kept % 25_000 == 0:
                logger.info("Flattened %s Addgene records", f"{kept:,}")
            yield row
        logger.info(
            "Addgene records: %s kept, %s dropped for missing id or sequence (include_partial=%s)",
            f"{kept:,}",
            f"{skipped:,}",
            include_partial,
        )

    return _chunks(
        rows(),
        addgene.RECORD_SCHEMA,
        int(params["chunk_size"]),
        None if limit is None else int(limit),
    )


def process_plsdb_records(
    sequences: Iterator[tuple[str, str]],
    nuccore: pd.DataFrame,
    taxonomy: pd.DataFrame,
    params: dict[str, Any],
) -> Iterator[pa.Table]:
    """Join the PLSDB FASTA stream to its metadata and taxonomy tables.

    The metadata tables are small enough to index in memory; the FASTA is not,
    so it stays a stream.
    """
    limit = params["limit"]
    nuccore_index = nuccore.set_index("NUCCORE_ACC").to_dict("index")
    taxonomy_index = taxonomy.set_index("TAXONOMY_UID").to_dict("index")
    logger.info(
        "Indexed %s nuccore rows and %s taxonomy rows",
        f"{len(nuccore_index):,}",
        f"{len(taxonomy_index):,}",
    )

    def rows() -> Iterator[dict[str, Any]]:
        kept = 0
        unmatched = 0
        for header, sequence in sequences:
            metadata = nuccore_index.get(plsdb.parse_accession(header))
            if metadata is None:
                unmatched += 1
                continue
            try:
                row = plsdb.to_record(
                    header,
                    sequence,
                    metadata,
                    taxonomy_index.get(metadata.get("TAXONOMY_UID")),
                )
            except LookupError:
                unmatched += 1
                continue
            kept += 1
            if kept % 25_000 == 0:
                logger.info("Flattened %s PLSDB records", f"{kept:,}")
            yield row
        logger.info(
            "PLSDB records: %s kept, %s dropped for missing metadata or sequence",
            f"{kept:,}",
            f"{unmatched:,}",
        )

    return _chunks(
        rows(),
        plsdb.RECORD_SCHEMA,
        int(params["chunk_size"]),
        None if limit is None else int(limit),
    )


def normalize_annotations(plannotate: pd.DataFrame, plasmidkit: pd.DataFrame) -> pd.DataFrame:
    """Merge both annotation sources into one long table with a shared schema."""
    merged = pd.concat(
        [
            annotations_lib.normalize_plannotate(plannotate),
            annotations_lib.normalize_plasmidkit(plasmidkit),
        ],
        ignore_index=True,
    )
    logger.info(
        "Normalized %s annotations across %s plasmids",
        f"{len(merged):,}",
        f"{merged['sequence_id'].nunique():,}",
    )
    return merged


def build_annotation_features(
    annotations: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Reduce annotations to one de-duplicated feature-name list per sequence."""
    max_features = params["max_features"]
    features = annotations_lib.feature_lists(
        annotations, max_features=None if max_features is None else int(max_features)
    )
    logger.info("Built feature lists for %s sequences", f"{len(features):,}")
    return features


def summarize_records(
    records: pd.DataFrame,
    annotations: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    """Profile the processed tables so a run can be checked without loading them."""
    return {
        "records": {
            "rows": len(records),
            "unique_sequence_ids": int(records["sequence_id"].nunique()),
            "sequence_kind": records["sequence_kind"].value_counts().to_dict(),
            "length_bp": qc.sequence_length_summary(records["length_bp"]),
            "null_fraction": {
                column: round(float(records[column].isna().mean()), 4)
                for column in ("name", "backbone", "bacterial_resistance", "plasmid_copy")
            },
        },
        "annotations": {
            "rows": len(annotations),
            "by_source": annotations["source"].value_counts().to_dict(),
            "annotated_sequences": len(features),
            "records_with_features": int(
                records["sequence_id"].isin(features["sequence_id"]).sum()
            ),
        },
    }
