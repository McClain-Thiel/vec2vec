"""Kedro nodes for rule-derived training evidence and benchmark sampling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vec2vec.lib import constraint_evidence


def build_evidence(
    retrieval: pd.DataFrame,
    plannotate: pd.DataFrame,
    evidence_params: dict[str, Any],
    facet_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build training claims and a label-free validation sample."""
    return constraint_evidence.build_constraint_evidence(
        retrieval,
        plannotate,
        evidence_params,
        facet_params,
    )
