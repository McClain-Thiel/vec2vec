"""Normalization of the two Addgene annotation sources into one long table.

pLannotate and plasmidkit describe the same thing — a named feature occupying an
interval of a plasmid — with different column names and different coordinate
conventions. The upstream project embedded both raw shapes verbatim inside each
record's metadata blob, which forced every consumer to branch per source and
made the record table an order of magnitude larger than the data it carried.

Here both sources are mapped onto one schema and stored as their own dataset.
Interval values are preserved exactly as published: the ``source`` column tells a
consumer which convention applies, and no coordinates are silently rewritten.
"""

from __future__ import annotations

import pandas as pd

from vec2vec.lib.text import unique_preserving_order

#: Annotation sources in the order their features are surfaced to consumers.
ANNOTATION_SOURCES = ("plannotate", "plasmidkit")

ANNOTATION_COLUMNS = (
    "sequence_id",
    "addgene_id",
    "source",
    "feature",
    "feature_type",
    "description",
    "start",
    "end",
    "strand",
    "confidence",
)


def _strand_from_frame(frame: pd.Series) -> pd.Series:
    """Map a pLannotate reading frame (``+1``/``-1``) onto a strand symbol."""
    numeric = pd.to_numeric(frame, errors="coerce")
    return numeric.map(lambda value: "." if pd.isna(value) else ("-" if value < 0 else "+"))


def _finalize(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    """Apply the shared identity columns and dtypes to one normalized source."""
    frame = frame.dropna(subset=["addgene_id", "start", "end"]).copy()
    frame["addgene_id"] = frame["addgene_id"].astype("int64")
    frame["sequence_id"] = "addgene_" + frame["addgene_id"].astype(str)
    frame["source"] = source
    frame["start"] = frame["start"].astype("int64")
    frame["end"] = frame["end"].astype("int64")
    for column in ("feature", "feature_type", "description", "strand"):
        frame[column] = frame[column].astype("string").str.strip()
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce").astype("float64")
    return frame.loc[:, list(ANNOTATION_COLUMNS)]


def normalize_plannotate(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the pLannotate BLAST-style table onto :data:`ANNOTATION_COLUMNS`.

    ``qstart``/``qend`` are query coordinates on the plasmid; ``pident`` is a
    percentage identity, rescaled here to a 0-1 confidence.
    """
    return _finalize(
        pd.DataFrame(
            {
                "addgene_id": raw["plasmid_id"],
                "feature": raw["Feature"],
                "feature_type": raw["Type"],
                "description": raw["Description"],
                "start": raw["qstart"],
                "end": raw["qend"],
                "strand": _strand_from_frame(raw["sframe"]),
                "confidence": pd.to_numeric(raw["pident"], errors="coerce") / 100.0,
            }
        ),
        "plannotate",
    )


def normalize_plasmidkit(raw: pd.DataFrame) -> pd.DataFrame:
    """Map the plasmidkit annotation table onto :data:`ANNOTATION_COLUMNS`."""
    return _finalize(
        pd.DataFrame(
            {
                "addgene_id": raw["plasmid_id"],
                "feature": raw["Feature"],
                "feature_type": raw["Type"],
                "description": raw["method"],
                "start": raw["start"],
                "end": raw["end"],
                "strand": raw["strand"],
                "confidence": raw["confidence"],
            }
        ),
        "plasmidkit",
    )


def feature_lists(annotations: pd.DataFrame, *, max_features: int | None = None) -> pd.DataFrame:
    """Reduce annotations to one de-duplicated feature-name list per sequence.

    Sources are visited in :data:`ANNOTATION_SOURCES` order so the list is
    reproducible, and comparison for de-duplication is case-insensitive.

    Returns:
        A frame indexed by position with ``sequence_id`` and
        ``annotation_features`` columns.
    """
    ordered = annotations.assign(
        _source_rank=annotations["source"].map(
            {source: rank for rank, source in enumerate(ANNOTATION_SOURCES)}
        )
    ).sort_values(["sequence_id", "_source_rank"], kind="stable")

    grouped = ordered.groupby("sequence_id", sort=True)["feature"].apply(
        lambda values: unique_preserving_order(
            value for value in values.tolist() if isinstance(value, str)
        )[:max_features]
    )
    return grouped.rename("annotation_features").reset_index()
