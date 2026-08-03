"""Measure how much supervision the retrieval dataset can actually support.

Structured queries are only useful for training if the dataset contains peers
that *provably* fail them. This audit builds the query curriculum for every
eligible row and counts the hard negatives each query yields, so a shortage
shows up here rather than as a quiet plateau during training.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import queries
from vec2vec.lib.relevance import RelevanceIndex

logger = logging.getLogger(__name__)

_POOL_COLUMNS = (
    "same_backbone_count",
    "alternative_positive_count",
    "known_hard_negative_count",
    "strict_near_miss_count",
)

_HARD_NEGATIVE_THRESHOLDS = (1, 2, 4, 8)


def _summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    """Central tendency and tail of each pool size across a set of queries."""
    summary: dict[str, float | int] = {"queries": len(frame)}
    for column in _POOL_COLUMNS:
        counts = frame[column].to_numpy(dtype=np.int64)
        prefix = column.removesuffix("_count")
        summary[f"median_{prefix}"] = float(np.median(counts))
        summary[f"p90_{prefix}"] = float(np.quantile(counts, 0.9))
        summary[f"mean_{prefix}"] = float(np.mean(counts))
    hard = frame["known_hard_negative_count"].to_numpy(dtype=np.int64)
    for threshold in _HARD_NEGATIVE_THRESHOLDS:
        summary[f"fraction_with_{threshold}+_hard_negatives"] = float(np.mean(hard >= threshold))
    return summary


def audit_hard_negatives(
    dataset: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the query curriculum for one split and measure its negative yield.

    Returns:
        Per-query pool sizes, the richest example queries per order, and a summary.
    """
    split = str(params.get("split", "train"))
    max_order = int(params.get("max_order", 3))
    seed = int(params.get("seed", 17))

    frame = dataset.reset_index(drop=True)
    fields = ("backbone", *queries.QUERY_FIELDS)
    missing = set(fields).difference(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing audit columns: {sorted(missing)}")

    relevance = RelevanceIndex.from_frame(frame, fields=fields, min_constraint_groups=1)
    source_indices = frame.index[frame["split_grouped"].eq(split)].tolist()
    if not source_indices:
        raise ValueError(f"split {split!r} contains no rows")
    eligible = set(source_indices)

    records: list[dict[str, Any]] = []
    for source_index in source_indices:
        for query in queries.build_query_family(
            relevance, source_index, max_order=max_order, seed=seed
        ):
            pools = queries.same_backbone_hard_negatives(relevance, query, eligible)
            constraints = {field: sorted(values)[0] for field, values in query.constraints.fields}
            records.append(
                {
                    "query_id": query.query_id,
                    "source_index": query.source_index,
                    "source_sequence_id": str(frame.at[query.source_index, "sequence_id"]),
                    "query_text": query.text,
                    "order": query.order,
                    "added_field": query.field_names[-1],
                    "fields_json": json.dumps(query.field_names),
                    "constraints_json": json.dumps(constraints, sort_keys=True),
                    "same_backbone_count": len(pools.same_backbone),
                    "alternative_positive_count": len(pools.alternative_positives),
                    "known_hard_negative_count": len(pools.known_hard_negatives),
                    "strict_near_miss_count": len(pools.strict_near_misses),
                    "example_hard_negatives_json": json.dumps(pools.known_hard_negatives[:4]),
                }
            )

    results = pd.DataFrame.from_records(records)
    if results.empty:
        raise RuntimeError(f"no structured queries could be built for split {split!r}")

    examples = (
        results.sort_values(
            ["strict_near_miss_count", "known_hard_negative_count", "query_id"],
            ascending=[False, False, True],
        )
        .groupby("order", sort=True)
        .head(int(params.get("examples_per_order", 20)))
        .reset_index(drop=True)
    )
    summary = {
        "split": split,
        "source_rows": len(source_indices),
        "source_rows_with_queries": int(results["source_index"].nunique()),
        "max_order": max_order,
        "seed": seed,
        "query_fields": list(queries.QUERY_FIELDS),
        "overall": _summarize(results),
        "by_order": {
            str(order): _summarize(group) for order, group in results.groupby("order", sort=True)
        },
        "by_added_field": {
            str(field): _summarize(group)
            for field, group in results.groupby("added_field", sort=True)
        },
    }
    logger.info(
        "Audited %s queries over %s %r rows; median hard negatives %.1f",
        f"{len(results):,}",
        f"{len(source_indices):,}",
        split,
        summary["overall"]["median_known_hard_negative"],
    )
    return results, examples, summary
