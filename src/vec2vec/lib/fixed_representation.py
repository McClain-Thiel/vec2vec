"""Deterministic panels and circular windows for fixed DNA representations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib.constraint_state import retrieval_population_sha256
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.similarity_graph import dataframe_content_sha256
from vec2vec.lib.text import sha256_text

_RETRIEVAL_COLUMNS = {
    "sequence_id",
    "sequence",
    "sequence_sha256",
    "description",
    "length_bp",
    "leakage_component",
    "split_grouped",
}
_SPLIT_COLUMNS = {
    "sequence_id",
    "similarity_component_primary",
    "leakage_component_v2",
    "split_grouped_v2",
}
PANEL_HASH_COLUMNS = [
    "sequence_id",
    "sequence_sha256",
    "description_sha256",
    "similarity_component_primary",
    "leakage_component_v2",
    "length_bp",
    "length_decile",
    "selection_sha256",
    "in_numerical_smoke_panel",
]


@dataclass(frozen=True)
class CircularWindow:
    """One model input window over a circular DNA sequence."""

    index: int
    start_bp: int
    input_base_count: int
    newly_covered_base_count: int
    wrapped_input_base_count: int


def build_fixed_representation_panels(
    retrieval: pd.DataFrame,
    split_mapping: pd.DataFrame,
    *,
    expected_population_sha256: str,
    invariance_rows: int,
    numerical_smoke_rows: int,
    length_strata: int,
    selection_salt: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the frozen training-only invariance and numerical smoke panels."""
    _validate_input_tables(retrieval, split_mapping)
    observed_population_sha256 = retrieval_population_sha256(retrieval)
    if observed_population_sha256 != expected_population_sha256:
        raise ValueError(
            "retrieval population changed: "
            f"expected {expected_population_sha256}, observed {observed_population_sha256}"
        )
    joined = retrieval.merge(
        split_mapping.loc[:, sorted(_SPLIT_COLUMNS)],
        on="sequence_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"split mapping has {unmatched} unmatched retrieval rows")
    joined = joined.drop(columns="_merge")
    train = joined.loc[joined["split_grouped_v2"].eq("train")].copy()
    if train.empty:
        raise ValueError("split_grouped_v2 has no training rows")
    _validate_sequences(train)
    try:
        train["length_decile"] = pd.qcut(
            train["length_bp"], q=length_strata, labels=False, duplicates="raise"
        ).astype("int16")
    except ValueError as error:
        raise ValueError(f"cannot create {length_strata} distinct length strata") from error
    observed_strata = sorted(train["length_decile"].unique().tolist())
    if observed_strata != list(range(length_strata)):
        raise ValueError(f"length strata are incomplete: observed {observed_strata}")

    train["description_sha256"] = train["description"].map(lambda value: sha256_text(str(value)))
    train["selection_sha256"] = train.apply(
        lambda row: sha256_text(
            "|".join(
                [
                    selection_salt,
                    str(row["sequence_id"]),
                    str(row["sequence_sha256"]),
                    str(row["similarity_component_primary"]),
                ]
            )
        ),
        axis=1,
    )
    invariance = _select_invariance_panel(
        train,
        rows=invariance_rows,
        strata=length_strata,
    )
    smoke_ids = set(
        _select_smoke_panel(
            invariance,
            rows=numerical_smoke_rows,
            strata=length_strata,
        )["sequence_id"]
    )
    invariance["in_numerical_smoke_panel"] = invariance["sequence_id"].isin(smoke_ids)
    invariance = invariance.sort_values(
        ["length_decile", "length_bp", "selection_sha256"], kind="stable"
    ).reset_index(drop=True)
    panel_hash = dataframe_content_sha256(
        invariance,
        sort_columns=["sequence_id"],
        value_columns=PANEL_HASH_COLUMNS,
    )
    summary = {
        "input_population_sha256": observed_population_sha256,
        "panel_sha256": panel_hash,
        "selection_salt": selection_salt,
        "invariance_rows": int(len(invariance)),
        "numerical_smoke_rows": int(invariance["in_numerical_smoke_panel"].sum()),
        "primary_components": int(invariance["similarity_component_primary"].nunique()),
        "length_strata": int(length_strata),
        "strata": {
            str(int(stratum)): {
                "invariance_rows": int(len(group)),
                "numerical_smoke_rows": int(group["in_numerical_smoke_panel"].sum()),
                "minimum_length_bp": int(group["length_bp"].min()),
                "maximum_length_bp": int(group["length_bp"].max()),
            }
            for stratum, group in invariance.groupby("length_decile", sort=True)
        },
    }
    return invariance, summary


def circular_window_plan(
    sequence_length_bp: int,
    *,
    maximum_content_bp: int,
    tokenizer_unit_bp: int,
    overlap_fraction: float,
) -> list[CircularWindow]:
    """Plan token-aligned windows that cover each circular base at least once."""
    if sequence_length_bp < 1:
        raise ValueError("sequence_length_bp must be positive")
    if tokenizer_unit_bp < 1:
        raise ValueError("tokenizer_unit_bp must be positive")
    if maximum_content_bp < tokenizer_unit_bp:
        raise ValueError("maximum_content_bp must fit one tokenizer unit")
    if maximum_content_bp % tokenizer_unit_bp:
        raise ValueError("maximum_content_bp must be a multiple of tokenizer_unit_bp")
    if not 0.0 <= overlap_fraction < 1.0:
        raise ValueError("overlap_fraction must be in [0, 1)")

    input_base_count = min(
        maximum_content_bp,
        _round_up(sequence_length_bp, tokenizer_unit_bp),
    )
    if sequence_length_bp <= maximum_content_bp:
        return [
            CircularWindow(
                index=0,
                start_bp=0,
                input_base_count=input_base_count,
                newly_covered_base_count=sequence_length_bp,
                wrapped_input_base_count=max(0, input_base_count - sequence_length_bp),
            )
        ]

    stride = math.floor(maximum_content_bp * (1.0 - overlap_fraction))
    stride -= stride % tokenizer_unit_bp
    if stride < tokenizer_unit_bp:
        raise ValueError("overlap_fraction leaves a zero-length token-aligned stride")
    regular_starts = list(range(0, sequence_length_bp - maximum_content_bp + 1, stride))
    if not regular_starts:
        regular_starts = [0]
    starts = regular_starts.copy()
    covered = np.zeros(sequence_length_bp, dtype=bool)
    windows: list[CircularWindow] = []

    while True:
        start = starts[len(windows)]
        positions = (start + np.arange(maximum_content_bp, dtype=np.int64)) % sequence_length_bp
        new_count = int((~covered[positions]).sum())
        covered[positions] = True
        wrapped = max(0, start + maximum_content_bp - sequence_length_bp)
        windows.append(
            CircularWindow(
                index=len(windows),
                start_bp=start,
                input_base_count=maximum_content_bp,
                newly_covered_base_count=new_count,
                wrapped_input_base_count=wrapped,
            )
        )
        if covered.all():
            break
        if len(windows) == len(starts):
            starts.append(starts[-1] + stride)
        if len(windows) > math.ceil(sequence_length_bp / stride) + 1:
            raise RuntimeError("circular window planner failed to cover the sequence")

    if sum(window.newly_covered_base_count for window in windows) != sequence_length_bp:
        raise RuntimeError("circular window weights do not sum to the sequence length")
    return windows


def circular_subsequence(sequence: str, window: CircularWindow) -> str:
    """Return one window, wrapping across the recorded origin when necessary."""
    if not sequence:
        raise ValueError("sequence must not be empty")
    return "".join(
        sequence[(window.start_bp + offset) % len(sequence)]
        for offset in range(window.input_base_count)
    )


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of an unambiguous DNA sequence."""
    invalid = sorted(set(sequence).difference("ACGT"))
    if invalid:
        raise ValueError(f"sequence contains unsupported bases: {invalid}")
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def _validate_input_tables(retrieval: pd.DataFrame, split_mapping: pd.DataFrame) -> None:
    missing_retrieval = _RETRIEVAL_COLUMNS.difference(retrieval.columns)
    missing_split = _SPLIT_COLUMNS.difference(split_mapping.columns)
    if missing_retrieval:
        raise ValueError(f"retrieval dataset is missing columns: {sorted(missing_retrieval)}")
    if missing_split:
        raise ValueError(f"split mapping is missing columns: {sorted(missing_split)}")
    if retrieval.empty or split_mapping.empty:
        raise ValueError("retrieval dataset and split mapping must not be empty")
    for name, frame in (("retrieval dataset", retrieval), ("split mapping", split_mapping)):
        if frame["sequence_id"].isna().any() or frame["sequence_id"].duplicated().any():
            raise ValueError(f"{name} needs unique, non-missing sequence_id values")
    if len(retrieval) != len(split_mapping):
        raise ValueError(
            "retrieval and split row counts differ: "
            f"retrieval={len(retrieval)}, split={len(split_mapping)}"
        )


def _validate_sequences(frame: pd.DataFrame) -> None:
    missing = frame[["sequence", "sequence_sha256", "length_bp"]].isna().any(axis=1)
    if missing.any():
        missing_rows = int(missing.sum())
        raise ValueError(f"training rows have {missing_rows} missing sequence identity values")
    for row in frame.itertuples(index=False):
        sequence = str(row.sequence)
        if len(sequence) != int(row.length_bp):
            raise ValueError(
                f"sequence {row.sequence_id} length mismatch: "
                f"recorded={row.length_bp}, observed={len(sequence)}"
            )
        observed_sha256 = sequence_sha256(sequence)
        if observed_sha256 != str(row.sequence_sha256):
            raise ValueError(f"sequence {row.sequence_id} SHA-256 mismatch")


def _select_invariance_panel(
    train: pd.DataFrame,
    *,
    rows: int,
    strata: int,
) -> pd.DataFrame:
    targets = _balanced_targets(rows, strata)
    selected: list[int] = []
    used_components: set[str] = set()

    for stratum in range(strata):
        group = train.loc[train["length_decile"].eq(stratum)]
        minimum = group.sort_values(["length_bp", "selection_sha256"], kind="stable").index.tolist()
        maximum = group.sort_values(
            ["length_bp", "selection_sha256"], ascending=[False, True], kind="stable"
        ).index.tolist()
        for candidates in (minimum, maximum):
            index = _first_unused_component(train, candidates, used_components)
            selected.append(index)
            used_components.add(str(train.at[index, "similarity_component_primary"]))

    for stratum in range(strata):
        existing = sum(int(train.at[index, "length_decile"]) == stratum for index in selected)
        candidates = (
            train.loc[train["length_decile"].eq(stratum)]
            .sort_values("selection_sha256", kind="stable")
            .index.tolist()
        )
        while existing < targets[stratum]:
            index = _first_unused_component(train, candidates, used_components, selected)
            selected.append(index)
            used_components.add(str(train.at[index, "similarity_component_primary"]))
            existing += 1

    result = train.loc[selected].copy()
    if len(result) != rows:
        raise RuntimeError(f"invariance panel has {len(result)} rows, expected {rows}")
    if result["similarity_component_primary"].duplicated().any():
        raise RuntimeError("invariance panel repeats a primary component")
    return result


def _select_smoke_panel(
    invariance: pd.DataFrame,
    *,
    rows: int,
    strata: int,
) -> pd.DataFrame:
    targets = _balanced_targets(rows, strata)
    selected: list[int] = []
    for stratum in range(strata):
        group = invariance.loc[invariance["length_decile"].eq(stratum)]
        if targets[stratum] < 2:
            raise ValueError("numerical smoke target must allow two extrema per stratum")
        minimum = group.sort_values(["length_bp", "selection_sha256"], kind="stable").index[0]
        maximum_candidates = group.sort_values(
            ["length_bp", "selection_sha256"], ascending=[False, True], kind="stable"
        ).index.tolist()
        maximum = next(index for index in maximum_candidates if index != minimum)
        chosen = [minimum, maximum]
        remaining = (
            group.drop(index=chosen).sort_values("selection_sha256", kind="stable").index.tolist()
        )
        chosen.extend(remaining[: targets[stratum] - 2])
        selected.extend(chosen)
    result = invariance.loc[selected].copy()
    if len(result) != rows or result["sequence_id"].duplicated().any():
        raise RuntimeError("numerical smoke panel selection is invalid")
    return result


def _balanced_targets(rows: int, strata: int) -> list[int]:
    if strata < 1 or rows < strata:
        raise ValueError("panel rows must be at least the number of strata")
    quotient, remainder = divmod(rows, strata)
    return [quotient + int(stratum < remainder) for stratum in range(strata)]


def _first_unused_component(
    frame: pd.DataFrame,
    candidates: Sequence[int],
    used_components: set[str],
    selected: Sequence[int] = (),
) -> int:
    selected_set = set(selected)
    for index in candidates:
        component = str(frame.at[index, "similarity_component_primary"])
        if index not in selected_set and component not in used_components:
            return int(index)
    raise ValueError("not enough distinct primary components to fill a panel stratum")


def _round_up(value: int, unit: int) -> int:
    return ((value + unit - 1) // unit) * unit


def embedding_sha256(vector: np.ndarray) -> str:
    """Hash a normalized embedding in a fixed little-endian float32 representation."""
    array = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()
