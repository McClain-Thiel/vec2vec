"""Reproducible profiling for the E00 constraint-semantics gate.

These functions describe the current data. They do not decide which values are valid constraints
and they do not convert missing metadata into negative evidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from vec2vec.lib.serialization import stable_json
from vec2vec.lib.text import normalize_values


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    """Fail when *frame* does not contain every required column."""
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _validate_retrieval_identity(frame: pd.DataFrame) -> None:
    """Check the row identity needed by all E00 profiles."""
    _require_columns(frame, {"sequence_id"}, "retrieval dataset")
    if frame.empty:
        raise ValueError("retrieval dataset is empty")
    if frame["sequence_id"].isna().any():
        raise ValueError("retrieval dataset contains missing sequence_id values")
    if frame["sequence_id"].duplicated().any():
        raise ValueError("retrieval dataset contains duplicate sequence_id values")


def profile_constraint_fields(
    frame: pd.DataFrame,
    *,
    fields: Sequence[str],
    split_labels: Sequence[str],
    minimum_rows: int,
    minimum_components: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile normalized values without assigning constraint semantics.

    Returns:
        A one-row-per-field profile and a one-row-per-normalized-value profile. Raw cell variants
        remain attached to each normalized value for later manual review.
    """
    if not fields:
        raise ValueError("fields cannot be empty")
    if len(set(fields)) != len(fields):
        raise ValueError("fields cannot contain duplicates")
    if not split_labels or len(set(split_labels)) != len(split_labels):
        raise ValueError("split_labels must be non-empty and unique")
    if minimum_rows < 1 or minimum_components < 1:
        raise ValueError("support thresholds must be positive")

    _validate_retrieval_identity(frame)
    _require_columns(
        frame,
        {"leakage_component", "split_grouped", *fields},
        "retrieval dataset",
    )
    if frame[["leakage_component", "split_grouped"]].isna().any().any():
        raise ValueError("retrieval dataset contains missing split or component values")

    expected_splits = tuple(str(label) for label in split_labels)
    observed_splits = set(frame["split_grouped"].astype(str))
    unexpected = observed_splits.difference(expected_splits)
    if unexpected:
        raise ValueError(f"retrieval dataset contains unexpected splits: {sorted(unexpected)}")

    field_records: list[dict[str, Any]] = []
    value_records: list[dict[str, Any]] = []
    splits = frame["split_grouped"].astype(str).tolist()
    components = frame["leakage_component"].astype(str).tolist()

    for field in fields:
        row_support: Counter[str] = Counter()
        component_support: dict[str, set[str]] = defaultdict(set)
        split_row_support: dict[str, Counter[str]] = {split: Counter() for split in expected_splits}
        split_component_support: dict[str, dict[str, set[str]]] = {
            split: defaultdict(set) for split in expected_splits
        }
        raw_variants: dict[str, Counter[str]] = defaultdict(Counter)
        known_by_split: Counter[str] = Counter()

        for raw, split, component in zip(frame[field].tolist(), splits, components, strict=True):
            normalized = normalize_values(raw)
            if normalized:
                known_by_split[split] += 1
            raw_cell_json = stable_json(raw)
            for value in normalized:
                row_support[value] += 1
                component_support[value].add(component)
                split_row_support[split][value] += 1
                split_component_support[split][value].add(component)
                raw_variants[value][raw_cell_json] += 1

        for value in sorted(row_support):
            record: dict[str, Any] = {
                "field": field,
                "normalized_value": value,
                "row_support": row_support[value],
                "component_support": len(component_support[value]),
                "raw_variants_json": stable_json(
                    [
                        {"raw_cell_json": raw_json, "rows": count}
                        for raw_json, count in sorted(
                            raw_variants[value].items(), key=lambda item: (-item[1], item[0])
                        )
                    ],
                ),
            }
            for split in expected_splits:
                record[f"{split}_row_support"] = split_row_support[split][value]
                record[f"{split}_component_support"] = len(split_component_support[split][value])
            value_records.append(record)

        train_rows = split_row_support.get("train", Counter())
        train_components = split_component_support.get("train", {})
        field_record: dict[str, Any] = {
            "field": field,
            "rows": len(frame),
            "known_rows": sum(known_by_split.values()),
            "unknown_rows": len(frame) - sum(known_by_split.values()),
            "coverage": sum(known_by_split.values()) / len(frame) if len(frame) else 0.0,
            "normalized_value_count": len(row_support),
            "singleton_value_count": sum(count == 1 for count in row_support.values()),
            "values_meeting_total_support": sum(
                row_support[value] >= minimum_rows
                and len(component_support[value]) >= minimum_components
                for value in row_support
            ),
            "values_meeting_train_support": sum(
                train_rows[value] >= minimum_rows
                and len(train_components.get(value, set())) >= minimum_components
                for value in row_support
            ),
            "minimum_rows": minimum_rows,
            "minimum_components": minimum_components,
        }
        for split in expected_splits:
            split_size = splits.count(split)
            field_record[f"{split}_rows"] = split_size
            field_record[f"{split}_known_rows"] = known_by_split[split]
            field_record[f"{split}_coverage"] = (
                known_by_split[split] / split_size if split_size else 0.0
            )
        field_records.append(field_record)

    return pd.DataFrame.from_records(field_records), pd.DataFrame.from_records(value_records)


def profile_split_components(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Profile component purity and concentration without changing the split."""
    _validate_retrieval_identity(frame)
    required = {
        "sequence_sha256",
        "family_key",
        "leakage_component",
        "split_grouped",
    }
    _require_columns(frame, required, "retrieval dataset")
    if frame[list(required)].isna().any().any():
        raise ValueError("retrieval dataset contains missing split identity values")

    grouped = frame.groupby("leakage_component", sort=True, dropna=False)
    split_counts = grouped["split_grouped"].nunique()
    impure = split_counts[split_counts > 1]
    if not impure.empty:
        raise ValueError(f"{len(impure)} leakage components cross grouped splits")

    profile = grouped.agg(
        split_grouped=("split_grouped", "first"),
        rows=("sequence_id", "size"),
        family_count=("family_key", "nunique"),
        exact_sequence_count=("sequence_sha256", "nunique"),
    ).reset_index()
    profile = profile.sort_values(
        ["split_grouped", "rows", "leakage_component"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    by_split: dict[str, Any] = {}
    for split, split_frame in profile.groupby("split_grouped", sort=True):
        sizes = split_frame["rows"].astype(int)
        row_count = int(sizes.sum())
        by_split[str(split)] = {
            "rows": row_count,
            "components": len(split_frame),
            "singleton_components": int(sizes.eq(1).sum()),
            "largest_component_rows": int(sizes.max()),
            "largest_component_row_fraction": float(sizes.max() / row_count),
            "ten_largest_component_rows": int(sizes.nlargest(10).sum()),
            "ten_largest_component_row_fraction": float(sizes.nlargest(10).sum() / row_count),
        }

    family_sources = (
        frame["family_key"].astype(str).str.partition("::", expand=True)[0].value_counts()
    )
    summary = {
        "rows": len(frame),
        "sequence_ids": int(frame["sequence_id"].nunique()),
        "exact_sequence_groups": int(frame["sequence_sha256"].nunique()),
        "families": int(frame["family_key"].nunique()),
        "leakage_components": len(profile),
        "components_crossing_grouped_split": 0,
        "family_key_source_rows": {
            str(source): int(count) for source, count in family_sources.sort_index().items()
        },
        "by_split": by_split,
    }
    return profile, summary


def profile_plannotate(
    retrieval: pd.DataFrame,
    annotations: pd.DataFrame,
    *,
    expected_source: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Profile pLannotate coverage and raw coordinates for the retrieval population."""
    _validate_retrieval_identity(retrieval)
    _require_columns(retrieval, {"length_bp"}, "retrieval dataset")
    annotation_columns = {
        "sequence_id",
        "source",
        "feature",
        "feature_type",
        "start",
        "end",
        "strand",
        "confidence",
    }
    _require_columns(annotations, annotation_columns, "annotation dataset")
    if annotations.empty:
        raise ValueError("annotation dataset is empty")
    if annotations["sequence_id"].isna().any() or annotations["source"].isna().any():
        raise ValueError("annotation dataset contains missing identity or source values")

    observed_sources = set(annotations["source"].astype(str))
    if observed_sources != {expected_source}:
        message = (
            f"expected only annotation source {expected_source!r}, "
            f"observed {sorted(observed_sources)}"
        )
        raise ValueError(message)
    for column in ("start", "end", "confidence"):
        if not pd.api.types.is_numeric_dtype(annotations[column]):
            raise ValueError(f"annotation column {column!r} is not numeric")

    retrieval_ids = set(retrieval["sequence_id"].astype(str))
    annotation_ids = set(annotations["sequence_id"].astype(str))
    matched_ids = retrieval_ids & annotation_ids
    population = annotations.loc[annotations["sequence_id"].astype(str).isin(retrieval_ids)].copy()
    lengths = retrieval.set_index("sequence_id")["length_bp"]
    population["_length_bp"] = population["sequence_id"].map(lengths)
    if population["_length_bp"].isna().any():
        raise RuntimeError("matched pLannotate rows failed to resolve a sequence length")

    start = population["start"]
    end = population["end"]
    confidence = population["confidence"]
    length = population["_length_bp"]
    feature_types = population["feature_type"].fillna("<missing>").astype(str).value_counts()
    strands = population["strand"].fillna("<missing>").astype(str).value_counts()

    provenance_record = dict(provenance)
    required_provenance = ("software_version", "database_version", "circular_setting")
    missing_provenance = [
        field
        for field in required_provenance
        if field not in provenance_record or provenance_record[field] is None
    ]

    return {
        "source": expected_source,
        "retrieval_rows": len(retrieval),
        "annotation_rows_all": len(annotations),
        "annotation_rows_in_retrieval": len(population),
        "annotation_rows_outside_retrieval": len(annotations) - len(population),
        "retrieval_sequences_with_annotations": len(matched_ids),
        "retrieval_sequences_without_annotations": len(retrieval_ids - annotation_ids),
        "retrieval_sequence_coverage": len(matched_ids) / len(retrieval_ids),
        "annotation_sequences_outside_retrieval": len(annotation_ids - retrieval_ids),
        "distinct_features_in_retrieval": int(population["feature"].nunique(dropna=True)),
        "feature_type_rows": {
            str(value): int(count) for value, count in feature_types.sort_index().items()
        },
        "strand_rows": {str(value): int(count) for value, count in strands.sort_index().items()},
        "raw_coordinate_checks": {
            "missing_start": int(start.isna().sum()),
            "missing_end": int(end.isna().sum()),
            "start_greater_than_end": int(start.gt(end).sum()),
            "start_less_than_zero": int(start.lt(0).sum()),
            "end_less_than_zero": int(end.lt(0).sum()),
            "start_equal_zero": int(start.eq(0).sum()),
            "end_equal_zero": int(end.eq(0).sum()),
            "start_greater_than_sequence_length": int(start.gt(length).sum()),
            "end_greater_than_sequence_length": int(end.gt(length).sum()),
        },
        "confidence_checks": {
            "missing": int(confidence.isna().sum()),
            "outside_zero_one": int(((confidence < 0) | (confidence > 1)).sum()),
        },
        "provenance": provenance_record,
        "missing_provenance_fields": missing_provenance,
        "provenance_complete": not missing_provenance,
        "interpretation": (
            "Raw source coordinates are profiled but not normalized or interpreted. "
            "No interval, coverage, masking, or edit measurement is produced."
        ),
    }
