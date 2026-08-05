"""Kedro nodes for the E00 manual facet-audit sample."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vec2vec.lib import audit_decision, facet_audit


def build_sample(
    dataset: pd.DataFrame, params: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build proposed review rows, vocabulary classifications, and a manifest."""
    return facet_audit.build_facet_audit_sample(dataset, params)


def build_review_export(sample: pd.DataFrame) -> pd.DataFrame:
    """Build a model-free table for source review and later human decisions."""
    return audit_decision.build_blinded_review_table(sample)
