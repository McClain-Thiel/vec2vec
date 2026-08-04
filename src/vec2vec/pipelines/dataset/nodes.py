"""Assemble the paired retrieval dataset and its leakage-aware splits."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import qc, relevance, splits
from vec2vec.lib.sequences import sequence_sha256
from vec2vec.lib.text import as_list, normalize_values

logger = logging.getLogger(__name__)

#: Fixed namespace, so the same (prompt hash, sequence) always yields the same uuid.
_UUID_NAMESPACE = uuid.UUID("6f9b2c1e-0d3a-4a5b-8c7d-9e1f2a3b4c5d")

#: Record columns carried into the paired dataset alongside sequence and text.
#:
#: Everything else stays in ``addgene_records``, joinable on ``sequence_id``,
#: rather than being duplicated into this table as an opaque JSON blob.
CARRIED_FIELDS = (
    "addgene_id",
    "name",
    "backbone",
    "length_bp",
    "url",
    "article_doi",
    "article_pubmed_id",
    *relevance.FUNCTIONAL_FIELDS,
    *relevance.PAYLOAD_FIELDS,
)


def assemble_pairs(
    records: pd.DataFrame,
    descriptions: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Join descriptions to their source records and derive per-row identity.

    Only complete plasmid sequences are eligible: a description written from a
    whole plasmid's metadata does not describe a fragment.
    """
    full = records.loc[records["sequence_kind"] == "full"]
    dropped = len(records) - len(full)
    if dropped:
        logger.info("Excluded %s non-full sequences from the paired dataset", f"{dropped:,}")

    pairs = (
        descriptions.merge(full, on="sequence_id", how="inner", suffixes=("", "_record"))
        .merge(features, on="sequence_id", how="left")
        .reset_index(drop=True)
    )
    if pairs.empty:
        raise ValueError("no (description, sequence) pairs were produced")
    if pairs["sequence_id"].duplicated().any():
        raise ValueError("the joined dataset contains duplicate sequence_id values")

    # The join suffixes the record's own Addgene blurb, which fed the prompt and
    # is worth keeping beside the text it produced.
    pairs = pairs.rename(columns={"description_record": "source_description"})
    pairs["annotations"] = pairs["annotation_features"].map(as_list)
    pairs["sequence_sha256"] = pairs["sequence"].map(sequence_sha256)
    pairs["uuid"] = [
        str(uuid.uuid5(_UUID_NAMESPACE, f"{prompt_hash}:{sequence_id}"))
        for prompt_hash, sequence_id in zip(
            pairs["prompt_hash"].fillna(""), pairs["sequence_id"], strict=True
        )
    ]
    pairs["family_key"] = [
        splits.family_key(backbone, name, sequence_id)
        for backbone, name, sequence_id in zip(
            pairs["backbone"], pairs["name"], pairs["sequence_id"], strict=True
        )
    ]

    columns = [
        "uuid",
        "sequence_id",
        "sequence",
        "sequence_sha256",
        "description",
        "source_description",
        "annotations",
        "family_key",
        *CARRIED_FIELDS,
        "generation_model",
        "prompt_version",
        "prompt_hash",
        "input_hash",
        "cost_usd",
    ]
    logger.info(
        "Assembled %s pairs across %s families and %s distinct sequences",
        f"{len(pairs):,}",
        f"{pairs['family_key'].nunique():,}",
        f"{pairs['sequence_sha256'].nunique():,}",
    )
    return pairs.loc[:, columns]


def _constraint_columns(
    pairs: pd.DataFrame,
    field_sets: dict[str, tuple[str, ...]],
) -> dict[str, tuple[list[str], np.ndarray]]:
    """Serialize the constraints each description surfaces, per named field set.

    Every field set is resolved in one pass. The field sets overlap — the
    structured set is a superset of the functional one — so normalizing each
    column once up front and walking the rows once avoids re-normalizing the
    same values, and the same annotation lists, for each set in turn.
    """
    needed = sorted({field for fields in field_sets.values() for field in fields})
    normalized = {field: pairs[field].map(normalize_values).tolist() for field in needed}
    features = pairs["annotations"].map(normalize_values).tolist()
    descriptions = pairs["description"].astype(str).tolist()
    lengths = pairs["length_bp"].astype(int).tolist()

    results: dict[str, tuple[list[str], np.ndarray]] = {
        name: ([], [])
        for name in field_sets  # type: ignore[misc]
    }
    for index in range(len(pairs)):
        for name, fields in field_sets.items():
            constraints = relevance.extract_surface_constraints(
                descriptions[index],
                {field: normalized[field][index] for field in fields},
                features[index],
                lengths[index],
            )
            serialized, counts = results[name]
            serialized.append(
                json.dumps(constraints.to_dict(), sort_keys=True, separators=(",", ":"))
            )
            counts.append(constraints.group_count)
    return {
        name: (serialized, np.asarray(counts, dtype=np.int16))
        for name, (serialized, counts) in results.items()
    }


def add_splits_and_constraints(
    pairs: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign leakage-aware splits and attach surfaced retrieval constraints.

    Splits are computed once over components that union rows sharing a family
    *or* an exact sequence, so no family and no duplicate sequence can straddle
    a split by construction.

    Returns:
        The retrieval dataset, and an audit of how it was built.
    """
    fractions = splits.SplitFractions(
        train=float(params["train_fraction"]),
        val=float(params["val_fraction"]),
    )
    seed = int(params["seed"])

    dataset = pairs.reset_index(drop=True).copy()
    components = splits.leakage_components(
        dataset["family_key"].astype(str).tolist(),
        dataset["sequence_sha256"].astype(str).tolist(),
    )
    dataset["leakage_component"] = components
    dataset["split_grouped"] = splits.assign_grouped_split(components, fractions, seed)
    dataset["split_random"] = splits.assign_random_split(len(dataset), fractions, seed)

    impure = splits.split_purity(components, dataset["split_grouped"].to_numpy())
    if impure:
        raise RuntimeError(f"{impure} leakage components straddle a grouped split")

    constraints = _constraint_columns(
        dataset,
        {
            "surfaced": relevance.FUNCTIONAL_FIELDS,
            "structured": relevance.STRUCTURED_FIELDS,
        },
    )
    for prefix, (serialized, counts) in constraints.items():
        dataset[f"{prefix}_constraints_json"] = serialized
        dataset[f"{prefix}_constraint_groups"] = counts

    audit = {
        "rows": len(dataset),
        "families": int(dataset["family_key"].nunique()),
        "exact_sequence_groups": int(dataset["sequence_sha256"].nunique()),
        "leakage_components": int(len(np.unique(components))),
        "split_seed": seed,
        "split_fractions": {
            "train": fractions.train,
            "val": fractions.val,
            "test": round(fractions.test, 6),
        },
        "split_grouped": dataset["split_grouped"].value_counts().to_dict(),
        "split_random": dataset["split_random"].value_counts().to_dict(),
        "components_straddling_grouped_split": impure,
        "sequence_length_bp": qc.sequence_length_summary(dataset["length_bp"]),
        "surfaced_constraint_groups": _histogram(dataset["surfaced_constraint_groups"]),
        "structured_constraint_groups": _histogram(dataset["structured_constraint_groups"]),
    }
    logger.info("Retrieval dataset: %s", json.dumps(audit["split_grouped"]))
    return dataset, audit


def _histogram(column: pd.Series) -> dict[str, int]:
    """Value counts as a JSON-friendly, key-sorted mapping."""
    return {str(key): int(value) for key, value in column.value_counts().sort_index().items()}
