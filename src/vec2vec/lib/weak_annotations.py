"""Deterministic weak-label annotation vocabulary and retrieval benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib.serialization import stable_json


@dataclass(frozen=True)
class WeakAnnotationBenchmark:
    """Frozen queries and masks in the sorted train and validation row order."""

    queries: pd.DataFrame
    train_verified: np.ndarray
    train_known: np.ndarray
    validation_verified: np.ndarray


def normalize_annotation(value: str) -> str:
    """Normalize display variants without claiming that biological aliases are equivalent."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(
        "".join(character if character.isalnum() else " " for character in normalized).split()
    )


def build_weak_annotation_benchmark(
    pairs: pd.DataFrame,
    annotation_rows: pd.DataFrame,
    params: dict[str, Any],
) -> WeakAnnotationBenchmark:
    """Select annotation atoms and unseen conjunctions, then construct weak-label masks."""
    train, validation, calls = _validated_calls(pairs, annotation_rows, params)
    train_calls = calls.loc[calls["panel_role"].eq("alignment_train")]
    validation_calls = calls.loc[calls["panel_role"].eq("validation_gallery")]
    support = _annotation_support(train_calls, validation_calls)
    atoms = _select_atoms(train_calls, support, params)
    train_sets = _call_sets(train_calls, atoms)
    validation_sets = _call_sets(validation_calls, atoms)
    display_names = _display_names(train_calls, atoms)
    atom_rows = _atomic_query_rows(atoms, support, display_names, params)
    pair_rows = _pair_query_rows(
        atoms,
        train_sets,
        validation_sets,
        train,
        validation,
        display_names,
        params,
    )
    queries = pd.DataFrame([*atom_rows, *pair_rows]).sort_values(
        ["query_kind", "query_id"], kind="stable", ignore_index=True
    )
    expected_queries = int(params["vocabulary"]["atoms"]) + int(params["conjunctions"]["pairs"])
    if len(queries) != expected_queries or queries["query_id"].duplicated().any():
        raise RuntimeError("weak-annotation query construction is incomplete or non-unique")

    atomic = queries.loc[queries["query_kind"].eq("atomic")]
    conjunctions = queries.loc[queries["query_kind"].eq("pair_conjunction")]
    train_verified, train_known = _training_masks(train, atomic, train_sets, params)
    validation_verified = _validation_masks(validation, queries, validation_sets)
    if np.any(
        validation_verified[: len(atomic)].sum(axis=1)
        < int(params["vocabulary"]["minimum_validation_rows"])
    ):
        raise RuntimeError("an atomic validation query fell below its frozen support")
    if np.any(
        validation_verified[len(atomic) :].sum(axis=1)
        < int(params["conjunctions"]["minimum_validation_rows"])
    ):
        raise RuntimeError("a conjunction validation query fell below its frozen support")
    if not set(conjunctions["controlled_split"].astype(str)) == {"atoms_seen_conjunction_unseen"}:
        raise RuntimeError("weak-annotation conjunctions changed controlled split")
    return WeakAnnotationBenchmark(queries, train_verified, train_known, validation_verified)


def retrieval_metrics(
    scores: np.ndarray,
    queries: pd.DataFrame,
    positive_mask: np.ndarray,
    *,
    seed: int,
    representation: str,
    cutoffs: tuple[int, ...],
) -> pd.DataFrame:
    """Compute exact weak-label precision and signed utility at each cutoff."""
    values = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positive_mask)
    if values.shape != positives.shape or values.shape[0] != len(queries):
        raise ValueError("retrieval scores, queries, and positive mask must align")
    if positives.dtype != np.bool_ or not np.isfinite(values).all():
        raise ValueError("retrieval inputs must be finite scores and boolean labels")
    if not cutoffs or min(cutoffs) < 1 or max(cutoffs) > values.shape[1]:
        raise ValueError("retrieval cutoffs must fit the gallery")
    rows: list[dict[str, Any]] = []
    for position, query in enumerate(queries.itertuples(index=False)):
        order = np.argsort(-values[position], kind="stable")
        first_positive_rank = int(np.flatnonzero(positives[position, order])[0] + 1)
        for cutoff in cutoffs:
            positive_hits = int(positives[position, order[:cutoff]].sum())
            positive_fraction = positive_hits / cutoff
            rows.append(
                {
                    "query_id": str(query.query_id),
                    "query_kind": str(query.query_kind),
                    "seed": seed,
                    "representation": representation,
                    "k": cutoff,
                    "positive_hits": positive_hits,
                    "positive_fraction": positive_fraction,
                    "weak_negative_fraction": 1.0 - positive_fraction,
                    "utility": 2.0 * positive_fraction - 1.0,
                    "first_positive_rank": first_positive_rank,
                }
            )
    return pd.DataFrame(rows)


def fuse_atomic_classifier_scores(
    raw_logits: np.ndarray,
    calibrated_logits: np.ndarray,
    atomic_queries: pd.DataFrame,
    pair_queries: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Compose atomic classifier evidence using three fixed AND scoring rules."""
    raw = np.asarray(raw_logits, dtype=np.float64)
    calibrated = np.asarray(calibrated_logits, dtype=np.float64)
    if raw.shape != calibrated.shape or raw.shape[1] != len(atomic_queries):
        raise ValueError("atomic classifier scores and atomic queries must align")
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise ValueError("atomic classifier scores must be finite")
    by_key = {
        json.loads(str(row.annotation_keys_json))[0]: position
        for position, row in enumerate(atomic_queries.itertuples(index=False))
    }
    if len(by_key) != len(atomic_queries):
        raise ValueError("atomic classifier queries repeat an annotation key")
    pair_positions = []
    for row in pair_queries.itertuples(index=False):
        keys = json.loads(str(row.annotation_keys_json))
        if len(keys) != 2 or any(key not in by_key for key in keys):
            raise ValueError("pair query does not contain two known atomic keys")
        pair_positions.append((by_key[keys[0]], by_key[keys[1]]))
    left = np.asarray([positions[0] for positions in pair_positions])
    right = np.asarray([positions[1] for positions in pair_positions])
    left_calibrated = calibrated[:, left]
    right_calibrated = calibrated[:, right]
    left_log_probability = -np.logaddexp(0.0, -left_calibrated)
    right_log_probability = -np.logaddexp(0.0, -right_calibrated)
    return {
        "raw_logit_sum": (raw[:, left] + raw[:, right]).T,
        "calibrated_log_probability_sum": (left_log_probability + right_log_probability).T,
        "calibrated_min_logit": np.minimum(left_calibrated, right_calibrated).T,
    }


def paired_query_bootstrap(
    metrics: pd.DataFrame,
    *,
    k: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap held-out conjunction queries after averaging over probe seeds."""
    selected = metrics.loc[metrics["query_kind"].eq("pair_conjunction") & metrics["k"].eq(k)]
    per_query = (
        selected.groupby(["query_id", "representation"], sort=True)["utility"].mean().unstack()
    )
    expected = {"atomic_sum", "direct_text"}
    if set(per_query.columns) != expected or per_query.isna().any(axis=None):
        raise ValueError("paired bootstrap lacks one conjunction representation")
    generator = np.random.default_rng(seed)
    positions = generator.integers(0, len(per_query), size=(draws, len(per_query)))
    direct = per_query["direct_text"].to_numpy()[positions].mean(axis=1)
    additive = per_query["atomic_sum"].to_numpy()[positions].mean(axis=1)
    difference = additive - direct
    return {
        "draws": draws,
        "resampling_unit": "held_out_conjunction_query",
        "direct_text": float(per_query["direct_text"].mean()),
        "atomic_sum": float(per_query["atomic_sum"].mean()),
        "atomic_sum_minus_direct_text": float(
            per_query["atomic_sum"].mean() - per_query["direct_text"].mean()
        ),
        "direct_text_95_interval": _interval(direct),
        "atomic_sum_95_interval": _interval(additive),
        "difference_95_interval": _interval(difference),
    }


def _validated_calls(
    pairs: pd.DataFrame,
    annotation_rows: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_columns = {
        "sequence_id",
        "sequence_sha256",
        "panel_role",
        "leakage_component_v2",
    }
    annotation_columns = {"sequence_id", "sequence_sha256", "annotations"}
    if pair_columns.difference(pairs) or annotation_columns.difference(annotation_rows):
        raise ValueError("weak-annotation inputs are missing required columns")
    if pairs["sequence_id"].duplicated().any() or annotation_rows["sequence_id"].duplicated().any():
        raise ValueError("weak-annotation inputs repeat sequence identifiers")
    selected_pairs = pairs.loc[
        pairs["panel_role"].isin(["alignment_train", "validation_gallery"]), list(pair_columns)
    ].copy()
    joined = selected_pairs.merge(
        annotation_rows.loc[:, list(annotation_columns)],
        on="sequence_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_annotation"),
        indicator=True,
    )
    if not joined["_merge"].eq("both").all() or joined["annotations"].isna().any():
        raise ValueError("HF annotations do not cover the complete E06 population")
    if not joined["sequence_sha256"].eq(joined["sequence_sha256_annotation"]).all():
        raise ValueError("HF and E06 sequence hashes differ")
    joined = joined.drop(columns=["_merge", "sequence_sha256_annotation"])
    train = joined.loc[joined["panel_role"].eq("alignment_train")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    validation = joined.loc[joined["panel_role"].eq("validation_gallery")].sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    if len(train) != int(params["inputs"]["training_rows"]) or len(validation) != int(
        params["inputs"]["validation_rows"]
    ):
        raise ValueError("E06 population row counts changed")

    calls = joined.loc[
        :, ["sequence_id", "panel_role", "leakage_component_v2", "annotations"]
    ].explode("annotations")
    calls = calls.rename(columns={"annotations": "raw_annotation"}).dropna(
        subset=["raw_annotation"]
    )
    calls["raw_annotation"] = calls["raw_annotation"].astype(str).str.strip()
    calls["annotation_key"] = calls["raw_annotation"].map(normalize_annotation)
    vocabulary = params["vocabulary"]
    accession = re.compile(r"^[a-z]{1,5}\d+(?:\s+\d+){2,3}$")
    calls = calls.loc[
        calls["annotation_key"]
        .str.len()
        .between(
            int(vocabulary["minimum_normalized_characters"]),
            int(vocabulary["maximum_normalized_characters"]),
        )
        & calls["annotation_key"].str.split().str.len().le(int(vocabulary["maximum_words"]))
        & ~calls["annotation_key"].str.startswith("prodigal orf ")
        & ~calls["annotation_key"].map(lambda value: bool(accession.fullmatch(value)))
    ]
    calls = calls.sort_values(
        ["sequence_id", "annotation_key", "raw_annotation"], kind="stable"
    ).drop_duplicates(["sequence_id", "annotation_key"])
    if calls.empty:
        raise ValueError("annotation filtering removed every call")
    return train, validation, calls


def _annotation_support(train: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    train_support = train.groupby("annotation_key").agg(
        train_positive_rows=("sequence_id", "nunique"),
        train_positive_components=("leakage_component_v2", "nunique"),
    )
    validation_support = validation.groupby("annotation_key").agg(
        validation_positive_rows=("sequence_id", "nunique"),
        validation_positive_components=("leakage_component_v2", "nunique"),
    )
    return train_support.join(validation_support, how="left").fillna(0).reset_index()


def _select_atoms(
    train_calls: pd.DataFrame,
    support: pd.DataFrame,
    params: dict[str, Any],
) -> list[str]:
    vocabulary = params["vocabulary"]
    eligible = support.loc[
        support["train_positive_rows"].between(
            int(vocabulary["minimum_train_rows"]), int(vocabulary["maximum_train_rows"])
        )
        & support["train_positive_components"].ge(int(vocabulary["minimum_train_components"]))
        & support["validation_positive_rows"].ge(int(vocabulary["minimum_validation_rows"]))
        & support["validation_positive_components"].ge(
            int(vocabulary["minimum_validation_components"])
        )
    ].sort_values(
        ["train_positive_rows", "annotation_key"],
        ascending=[False, True],
        kind="stable",
    )
    train_sets = _call_sets(train_calls, eligible["annotation_key"].astype(str).tolist())
    selected: list[str] = []
    maximum_jaccard = float(vocabulary["maximum_atom_jaccard"])
    for key in eligible["annotation_key"].astype(str):
        if all(
            _jaccard(train_sets[key], train_sets[other]) < maximum_jaccard for other in selected
        ):
            selected.append(key)
        if len(selected) == int(vocabulary["atoms"]):
            break
    if len(selected) != int(vocabulary["atoms"]):
        raise RuntimeError("not enough eligible weak-label annotation atoms")
    return selected


def _display_names(calls: pd.DataFrame, atoms: list[str]) -> dict[str, str]:
    counts = (
        calls.loc[calls["annotation_key"].isin(atoms)]
        .groupby(["annotation_key", "raw_annotation"])["sequence_id"]
        .nunique()
        .rename("rows")
        .reset_index()
        .sort_values(
            ["annotation_key", "rows", "raw_annotation"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("annotation_key")
    )
    result = dict(
        zip(
            counts["annotation_key"].astype(str),
            counts["raw_annotation"].astype(str),
            strict=False,
        )
    )
    if set(result) != set(atoms):
        raise RuntimeError("an annotation atom lacks a display name")
    return result


def _atomic_query_rows(
    atoms: list[str],
    support: pd.DataFrame,
    display_names: dict[str, str],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    by_key = support.set_index("annotation_key").to_dict("index")
    protocol = str(params["protocol_version"])
    rows = []
    for key in atoms:
        query_id = _sha256(f"{protocol}|atomic|{key}")
        rows.append(
            {
                "query_id": query_id,
                "query_kind": "atomic",
                "annotation_keys_json": stable_json([key]),
                "canonical_query_text": f"plasmid annotated with {display_names[key]}",
                "train_positive_rows": int(by_key[key]["train_positive_rows"]),
                "train_positive_components": int(by_key[key]["train_positive_components"]),
                "validation_positive_rows": int(by_key[key]["validation_positive_rows"]),
                "validation_positive_components": int(
                    by_key[key]["validation_positive_components"]
                ),
                "selection_sha256": query_id,
                "controlled_split": "atomic_training_query",
            }
        )
    return rows


def _pair_query_rows(
    atoms: list[str],
    train_sets: dict[str, set[str]],
    validation_sets: dict[str, set[str]],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    display_names: dict[str, str],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    conjunctions = params["conjunctions"]
    protocol = str(params["protocol_version"])
    train_components = train.set_index("sequence_id")["leakage_component_v2"].astype(str)
    validation_components = validation.set_index("sequence_id")["leakage_component_v2"].astype(str)
    candidates = []
    ordered_atoms = sorted(atoms)
    for position, left in enumerate(ordered_atoms):
        for right in ordered_atoms[position + 1 :]:
            train_positive = train_sets[left] & train_sets[right]
            validation_positive = validation_sets[left] & validation_sets[right]
            jaccard = _jaccard(train_sets[left], train_sets[right])
            if (
                len(train_positive) < int(conjunctions["minimum_train_rows"])
                or len(validation_positive) < int(conjunctions["minimum_validation_rows"])
                or train_components.loc[list(train_positive)].nunique()
                < int(conjunctions["minimum_train_components"])
                or validation_components.loc[list(validation_positive)].nunique()
                < int(conjunctions["minimum_validation_components"])
                or jaccard > float(conjunctions["maximum_atom_jaccard"])
            ):
                continue
            selection_hash = _sha256(f"{protocol}|pair-selection|{left}|{right}")
            candidates.append(
                (
                    selection_hash,
                    left,
                    right,
                    len(train_positive),
                    int(train_components.loc[list(train_positive)].nunique()),
                    len(validation_positive),
                    int(validation_components.loc[list(validation_positive)].nunique()),
                )
            )
    candidates.sort()
    selected: list[tuple[Any, ...]] = []
    selected_hashes: set[str] = set()
    degree = {atom: 0 for atom in atoms}
    maximum_degree = int(conjunctions["maximum_pairs_per_atom"])

    # Give every selected atom at least one measured conjunction before filling by hash order.
    for atom in sorted(atoms):
        if degree[atom]:
            continue
        for candidate in candidates:
            selection_hash, left, right, *_ = candidate
            if atom not in {left, right} or selection_hash in selected_hashes:
                continue
            if degree[left] < maximum_degree and degree[right] < maximum_degree:
                selected.append(candidate)
                selected_hashes.add(selection_hash)
                degree[left] += 1
                degree[right] += 1
                break
    for candidate in candidates:
        if len(selected) == int(conjunctions["pairs"]):
            break
        selection_hash, left, right, *_ = candidate
        if (
            selection_hash not in selected_hashes
            and degree[left] < maximum_degree
            and degree[right] < maximum_degree
        ):
            selected.append(candidate)
            selected_hashes.add(selection_hash)
            degree[left] += 1
            degree[right] += 1
    if len(selected) != int(conjunctions["pairs"]) or min(degree.values()) < 1:
        raise RuntimeError("not enough eligible conjunctions to cover every annotation atom")

    rows = []
    for (
        selection_hash,
        left,
        right,
        train_rows,
        train_component_count,
        validation_rows,
        validation_component_count,
    ) in selected:
        query_id = _sha256(f"{protocol}|pair|{left}|{right}")
        rows.append(
            {
                "query_id": query_id,
                "query_kind": "pair_conjunction",
                "annotation_keys_json": stable_json([left, right]),
                "canonical_query_text": (
                    f"plasmid annotated with both {display_names[left]} and {display_names[right]}"
                ),
                "train_positive_rows": train_rows,
                "train_positive_components": train_component_count,
                "validation_positive_rows": validation_rows,
                "validation_positive_components": validation_component_count,
                "selection_sha256": selection_hash,
                "controlled_split": "atoms_seen_conjunction_unseen",
            }
        )
    return rows


def _training_masks(
    train: pd.DataFrame,
    atomic_queries: pd.DataFrame,
    train_sets: dict[str, set[str]],
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    sequence_ids = train["sequence_id"].astype(str).to_numpy()
    position = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    verified = np.zeros((len(atomic_queries), len(train)), dtype=bool)
    known = np.zeros_like(verified)
    sampling = params["weak_negatives"]
    ratio = float(sampling["maximum_ratio_per_positive"])
    maximum = int(sampling["maximum_rows_per_atom"])
    salt = str(sampling["selection_salt"])
    for query_position, query in enumerate(atomic_queries.itertuples(index=False)):
        key = _query_keys(query.annotation_keys_json)[0]
        positive_ids = train_sets[key]
        positive_positions = [position[sequence_id] for sequence_id in positive_ids]
        verified[query_position, positive_positions] = True
        known[query_position, positive_positions] = True
        negative_rows = min(maximum, int(math.ceil(len(positive_ids) * ratio)))
        negative_ids = [
            sequence_id for sequence_id in sequence_ids if sequence_id not in positive_ids
        ]
        negative_ids.sort(key=lambda sequence_id: _sha256(f"{salt}|{key}|{sequence_id}"))
        known[query_position, [position[value] for value in negative_ids[:negative_rows]]] = True
    if not verified.any(axis=1).all() or not (known & ~verified).any(axis=1).all():
        raise RuntimeError("every annotation atom needs positive and sampled weak-negative rows")
    return verified, known


def _validation_masks(
    validation: pd.DataFrame,
    queries: pd.DataFrame,
    validation_sets: dict[str, set[str]],
) -> np.ndarray:
    sequence_ids = validation["sequence_id"].astype(str).to_numpy()
    position = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}
    verified = np.zeros((len(queries), len(validation)), dtype=bool)
    for query_position, query in enumerate(queries.itertuples(index=False)):
        keys = _query_keys(query.annotation_keys_json)
        positive_ids = set.intersection(*(validation_sets[key] for key in keys))
        verified[query_position, [position[value] for value in positive_ids]] = True
    return verified


def _call_sets(calls: pd.DataFrame, atoms: list[str]) -> dict[str, set[str]]:
    selected = calls.loc[calls["annotation_key"].isin(atoms)]
    result = {
        str(key): set(group["sequence_id"].astype(str))
        for key, group in selected.groupby("annotation_key", sort=True)
    }
    missing = set(atoms).difference(result)
    if missing:
        raise RuntimeError(f"selected annotation atoms lack calls: {sorted(missing)}")
    return result


def _query_keys(value: str) -> list[str]:
    import json

    keys = list(map(str, json.loads(str(value))))
    if len(keys) not in {1, 2}:
        raise ValueError("weak-annotation query must contain one or two atoms")
    return keys


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right)


def _interval(values: np.ndarray) -> list[float]:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
