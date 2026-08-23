"""Deterministic panels and circular windows for fixed DNA representations."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib.constraint_state import retrieval_population_sha256
from vec2vec.lib.sequences import DNA_ALPHABET, sequence_sha256
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
    eligible_sequence_alphabet: str,
    expected_prior_panel_sha256: str | None,
    expected_panel_sha256: str | None,
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
    prior_invariance = _select_invariance_panel(
        train,
        rows=invariance_rows,
        strata=length_strata,
    )
    prior_panel = _finalize_panel(
        prior_invariance,
        numerical_smoke_rows=numerical_smoke_rows,
        strata=length_strata,
    )
    prior_panel_hash = dataframe_content_sha256(
        prior_panel,
        sort_columns=["sequence_id"],
        value_columns=PANEL_HASH_COLUMNS,
    )
    if expected_prior_panel_sha256 is not None and prior_panel_hash != expected_prior_panel_sha256:
        raise ValueError(
            "prior invariance panel changed: "
            f"expected {expected_prior_panel_sha256}, observed {prior_panel_hash}"
        )

    eligible_train, sequence_eligibility = _eligible_training_rows(
        train,
        eligible_sequence_alphabet=eligible_sequence_alphabet,
    )
    invariance, amendment = _replace_ineligible_panel_rows(
        prior_invariance,
        eligible_train,
        rows=invariance_rows,
        strata=length_strata,
        eligible_sequence_alphabet=eligible_sequence_alphabet,
    )
    invariance = _finalize_panel(
        invariance,
        numerical_smoke_rows=numerical_smoke_rows,
        strata=length_strata,
    )
    panel_hash = dataframe_content_sha256(
        invariance,
        sort_columns=["sequence_id"],
        value_columns=PANEL_HASH_COLUMNS,
    )
    if expected_panel_sha256 is not None and panel_hash != expected_panel_sha256:
        raise ValueError(
            "amended invariance panel changed: "
            f"expected {expected_panel_sha256}, observed {panel_hash}"
        )
    summary = {
        "input_population_sha256": observed_population_sha256,
        "panel_sha256": panel_hash,
        "selection_salt": selection_salt,
        "invariance_rows": int(len(invariance)),
        "numerical_smoke_rows": int(invariance["in_numerical_smoke_panel"].sum()),
        "primary_components": int(invariance["similarity_component_primary"].nunique()),
        "length_strata": int(length_strata),
        "sequence_eligibility": sequence_eligibility,
        "panel_amendment": {
            "policy": "preserve_prior_eligible_rows_and_replace_ineligible_v0.1",
            "prior_panel_sha256": prior_panel_hash,
            **amendment,
        },
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


def circular_rotate(sequence: str, fraction: float) -> str:
    """Rotate a circular DNA sequence by the recorded fractional offset."""
    if not sequence:
        raise ValueError("sequence must not be empty")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"rotation fraction must be in [0, 1], got {fraction}")
    offset = int(round(len(sequence) * fraction)) % len(sequence)
    return sequence[offset:] + sequence[:offset]


def gc_fraction(sequence: str) -> float:
    """Return the G/C fraction of a validated, non-empty DNA sequence."""
    if not sequence:
        raise ValueError("sequence must not be empty")
    invalid = sorted(set(sequence).difference("ACGT"))
    if invalid:
        raise ValueError(f"sequence contains unsupported bases: {invalid}")
    return (sequence.count("G") + sequence.count("C")) / len(sequence)


def effective_rank(embeddings: np.ndarray) -> float:
    """Return entropy effective rank from centered singular values."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("effective rank requires a two-dimensional matrix with at least two rows")
    if not np.isfinite(matrix).all():
        raise ValueError("effective rank requires finite embeddings")
    singular_values = np.linalg.svd(matrix - matrix.mean(axis=0, keepdims=True), compute_uv=False)
    total = float(singular_values.sum())
    if total == 0.0:
        return 0.0
    probabilities = singular_values / total
    positive = probabilities > 0.0
    entropy = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
    return float(np.exp(entropy))


def representation_geometry(
    embeddings: np.ndarray,
    *,
    lengths_bp: Sequence[int],
    gc_fractions: Sequence[float],
) -> dict[str, float | int | str | None]:
    """Measure collapse and length/G+C confounding in original-sequence vectors."""
    matrix = np.asarray(embeddings, dtype=np.float64)
    lengths = np.asarray(lengths_bp, dtype=np.float64)
    gc_values = np.asarray(gc_fractions, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 2:
        raise ValueError("representation geometry requires at least two embedding rows")
    if len(lengths) != len(matrix) or len(gc_values) != len(matrix):
        raise ValueError("embedding, length, and G+C rows must have equal counts")
    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(lengths).all()
        or not np.isfinite(gc_values).all()
    ):
        raise ValueError("representation geometry requires finite inputs")
    if np.any(lengths <= 0.0):
        raise ValueError("sequence lengths must be positive")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("representation geometry rejects zero vectors")
    normalized = matrix / norms[:, None]
    pair_rows, pair_columns = np.triu_indices(len(normalized), k=1)
    pairwise_cosines = np.sum(
        normalized[pair_rows] * normalized[pair_columns],
        axis=1,
    )
    length_differences = np.abs(np.log2(lengths[pair_rows] / lengths[pair_columns]))
    gc_differences = np.abs(gc_values[pair_rows] - gc_values[pair_columns])
    rank = effective_rank(matrix)
    length_correlation, length_correlation_status = _pearson(pairwise_cosines, length_differences)
    gc_correlation, gc_correlation_status = _pearson(pairwise_cosines, gc_differences)
    return {
        "rows": int(len(matrix)),
        "embedding_dimension": int(matrix.shape[1]),
        "effective_rank": rank,
        "effective_rank_fraction": rank / matrix.shape[1],
        "mean_pairwise_cosine": float(pairwise_cosines.mean()),
        "median_pairwise_cosine": float(np.median(pairwise_cosines)),
        "pairwise_cosine_length_difference_pearson": length_correlation,
        "pairwise_cosine_length_difference_pearson_status": length_correlation_status,
        "pairwise_cosine_gc_difference_pearson": gc_correlation,
        "pairwise_cosine_gc_difference_pearson_status": gc_correlation_status,
    }


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
        invalid_symbols = sorted(set(sequence).difference(DNA_ALPHABET))
        if invalid_symbols:
            raise ValueError(
                f"sequence {row.sequence_id} contains non-IUPAC symbols: {invalid_symbols}"
            )


def _eligible_training_rows(
    train: pd.DataFrame,
    *,
    eligible_sequence_alphabet: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not eligible_sequence_alphabet:
        raise ValueError("eligible_sequence_alphabet must not be empty")
    if eligible_sequence_alphabet != eligible_sequence_alphabet.upper():
        raise ValueError("eligible_sequence_alphabet must use uppercase symbols")
    if len(set(eligible_sequence_alphabet)) != len(eligible_sequence_alphabet):
        raise ValueError("eligible_sequence_alphabet must not repeat symbols")
    allowed_symbols = set(eligible_sequence_alphabet)
    invalid_allowed_symbols = sorted(allowed_symbols.difference(DNA_ALPHABET))
    if invalid_allowed_symbols:
        raise ValueError(
            f"eligible_sequence_alphabet contains non-IUPAC symbols: {invalid_allowed_symbols}"
        )

    eligible_indices: list[int] = []
    excluded_rows: list[dict[str, Any]] = []
    excluded_symbol_counts: Counter[str] = Counter()
    for row in train.itertuples(index=True):
        sequence = str(row.sequence)
        unsupported_symbol_counts = Counter(
            symbol for symbol in sequence if symbol not in allowed_symbols
        )
        if not unsupported_symbol_counts:
            eligible_indices.append(int(row.Index))
            continue
        excluded_symbol_counts.update(unsupported_symbol_counts)
        excluded_rows.append(
            {
                "sequence_id": str(row.sequence_id),
                "sequence_sha256": str(row.sequence_sha256),
                "unsupported_symbol_counts": dict(sorted(unsupported_symbol_counts.items())),
            }
        )

    eligible = train.loc[eligible_indices].copy()
    if eligible.empty:
        raise ValueError("no training rows satisfy eligible_sequence_alphabet")
    return eligible, {
        "allowed_alphabet": eligible_sequence_alphabet,
        "eligible_training_rows": int(len(eligible)),
        "excluded_training_rows": int(len(excluded_rows)),
        "excluded_symbol_counts": dict(sorted(excluded_symbol_counts.items())),
        "excluded_rows_sha256": sha256_text(
            "\n".join(
                "|".join(
                    [
                        str(row["sequence_id"]),
                        str(row["sequence_sha256"]),
                        ",".join(
                            f"{symbol}:{count}"
                            for symbol, count in row["unsupported_symbol_counts"].items()
                        ),
                    ]
                )
                for row in sorted(excluded_rows, key=lambda row: row["sequence_id"])
            )
        ),
    }


def _replace_ineligible_panel_rows(
    prior_panel: pd.DataFrame,
    eligible_train: pd.DataFrame,
    *,
    rows: int,
    strata: int,
    eligible_sequence_alphabet: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    targets = _balanced_targets(rows, strata)
    eligible_indices = set(eligible_train.index)
    selected = [int(index) for index in prior_panel.index if index in eligible_indices]
    removed_indices = [int(index) for index in prior_panel.index if index not in eligible_indices]
    used_components = {
        str(eligible_train.at[index, "similarity_component_primary"]) for index in selected
    }
    replacement_indices: list[int] = []

    for stratum in range(strata):
        prior_group = prior_panel.loc[prior_panel["length_decile"].eq(stratum)]
        eligible_group = eligible_train.loc[eligible_train["length_decile"].eq(stratum)]
        existing = sum(
            int(eligible_train.at[index, "length_decile"]) == stratum for index in selected
        )
        if existing == targets[stratum]:
            continue

        prior_minimum = int(
            prior_group.sort_values(["length_bp", "selection_sha256"], kind="stable").index[0]
        )
        prior_maximum = int(
            prior_group.sort_values(
                ["length_bp", "selection_sha256"],
                ascending=[False, True],
                kind="stable",
            ).index[0]
        )
        priority_orders: list[list[int]] = []
        if prior_minimum not in eligible_indices:
            priority_orders.append(
                eligible_group.sort_values(
                    ["length_bp", "selection_sha256"], kind="stable"
                ).index.tolist()
            )
        if prior_maximum not in eligible_indices and prior_maximum != prior_minimum:
            priority_orders.append(
                eligible_group.sort_values(
                    ["length_bp", "selection_sha256"],
                    ascending=[False, True],
                    kind="stable",
                ).index.tolist()
            )
        priority_orders.append(
            eligible_group.sort_values("selection_sha256", kind="stable").index.tolist()
        )

        for candidates in priority_orders:
            if existing >= targets[stratum]:
                break
            index = _first_unused_component(
                eligible_train,
                candidates,
                used_components,
                selected,
            )
            selected.append(index)
            replacement_indices.append(index)
            used_components.add(str(eligible_train.at[index, "similarity_component_primary"]))
            existing += 1
        while existing < targets[stratum]:
            index = _first_unused_component(
                eligible_train,
                priority_orders[-1],
                used_components,
                selected,
            )
            selected.append(index)
            replacement_indices.append(index)
            used_components.add(str(eligible_train.at[index, "similarity_component_primary"]))
            existing += 1

    result = eligible_train.loc[selected].copy()
    if len(result) != rows:
        raise RuntimeError(f"amended invariance panel has {len(result)} rows, expected {rows}")
    if result["similarity_component_primary"].duplicated().any():
        raise RuntimeError("amended invariance panel repeats a primary component")
    if len(removed_indices) != len(replacement_indices):
        raise RuntimeError("amended invariance panel replacement count changed")
    allowed_symbols = set(eligible_sequence_alphabet)
    observed_targets = (
        result.groupby("length_decile").size().reindex(range(strata), fill_value=0).tolist()
    )
    if observed_targets != targets:
        raise RuntimeError(
            f"amended invariance panel stratum counts changed: observed {observed_targets}"
        )
    if not result["sequence"].map(lambda sequence: set(str(sequence)) <= allowed_symbols).all():
        raise RuntimeError("amended invariance panel contains an ineligible sequence")
    removed_rows = [
        {
            "sequence_id": str(prior_panel.at[index, "sequence_id"]),
            "sequence_sha256": str(prior_panel.at[index, "sequence_sha256"]),
            "length_decile": int(prior_panel.at[index, "length_decile"]),
            "unsupported_symbol_counts": dict(
                sorted(
                    Counter(
                        symbol
                        for symbol in str(prior_panel.at[index, "sequence"])
                        if symbol not in allowed_symbols
                    ).items()
                )
            ),
        }
        for index in removed_indices
    ]
    replacement_rows = [
        {
            "sequence_id": str(eligible_train.at[index, "sequence_id"]),
            "sequence_sha256": str(eligible_train.at[index, "sequence_sha256"]),
            "similarity_component_primary": str(
                eligible_train.at[index, "similarity_component_primary"]
            ),
            "length_bp": int(eligible_train.at[index, "length_bp"]),
            "length_decile": int(eligible_train.at[index, "length_decile"]),
        }
        for index in replacement_indices
    ]
    removed_by_stratum = Counter(row["length_decile"] for row in removed_rows)
    replacement_by_stratum = Counter(row["length_decile"] for row in replacement_rows)
    if removed_by_stratum != replacement_by_stratum:
        raise RuntimeError("amended invariance panel changed a length-stratum quota")
    return result, {
        "preserved_prior_rows": int(len(selected) - len(replacement_indices)),
        "replaced_rows": int(len(replacement_indices)),
        "removed_rows": sorted(removed_rows, key=lambda row: row["sequence_id"]),
        "replacement_rows": sorted(
            replacement_rows,
            key=lambda row: row["sequence_id"],
        ),
    }


def _finalize_panel(
    invariance: pd.DataFrame,
    *,
    numerical_smoke_rows: int,
    strata: int,
) -> pd.DataFrame:
    result = invariance.copy()
    smoke_ids = set(
        _select_smoke_panel(
            result,
            rows=numerical_smoke_rows,
            strata=strata,
        )["sequence_id"]
    )
    result["in_numerical_smoke_panel"] = result["sequence_id"].isin(smoke_ids)
    return result.sort_values(
        ["length_decile", "length_bp", "selection_sha256"], kind="stable"
    ).reset_index(drop=True)


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


def _pearson(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str]:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson correlation requires at least two paired values")
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None, "undefined_constant_input"
    return float(np.corrcoef(left, right)[0, 1]), "calculated"


def embedding_sha256(vector: np.ndarray) -> str:
    """Hash a normalized embedding in a fixed little-endian float32 representation."""
    array = np.asarray(vector, dtype="<f4")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()
