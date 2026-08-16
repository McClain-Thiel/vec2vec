"""Kedro nodes for frozen plasmid-constraint states."""

from __future__ import annotations

from typing import Any

import pandas as pd

from vec2vec.lib import constraint_state


def build_states(
    retrieval: pd.DataFrame,
    state_params: dict[str, Any],
    evidence_params: dict[str, Any],
    facet_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the constraint vocabulary and sparse verified/contradicted states."""
    return constraint_state.build_constraint_state_tables(
        retrieval,
        state_params,
        evidence_params,
        facet_params,
    )
