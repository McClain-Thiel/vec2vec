"""Kedro nodes for the E00 constraint-semantics profile."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vec2vec.lib import constraint_semantics


def profile_constraint_values(
    dataset: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profile configured metadata fields without assigning semantic labels."""
    support = params["support"]
    return constraint_semantics.profile_constraint_fields(
        dataset,
        fields=tuple(params["fields"]),
        split_labels=tuple(params["split_labels"]),
        minimum_rows=int(support["minimum_rows"]),
        minimum_components=int(support["minimum_components"]),
    )


def profile_components(dataset: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Profile grouped-split purity and component concentration."""
    return constraint_semantics.profile_split_components(dataset)


def profile_primary_annotations(
    dataset: pd.DataFrame,
    annotations: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Profile the explicit pLannotate-only catalog view."""
    annotation_params = params["plannotate"]
    return constraint_semantics.profile_plannotate(
        dataset,
        annotations,
        expected_source=str(annotation_params["expected_source"]),
        provenance={
            "software_version": annotation_params["software_version"],
            "database_version": annotation_params["database_version"],
            "circular_setting": annotation_params["circular_setting"],
            "coordinate_convention": annotation_params["coordinate_convention"],
        },
    )
