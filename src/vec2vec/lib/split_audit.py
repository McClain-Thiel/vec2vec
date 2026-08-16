"""Deterministic checks for grouped-split concentration and sequence leakage."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib.constraint_state import retrieval_population_sha256
from vec2vec.lib.sequences import DNA_ALPHABET, sequence_sha256

SPLIT_LABELS = ("train", "val", "test")

BLAST_COLUMNS = (
    "qseqid",
    "sseqid",
    "qlen",
    "slen",
    "pident",
    "alignment_length",
    "nident",
    "mismatch",
    "gaps",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
)

BLAST_OUTFMT_FIELDS = (
    "qseqid",
    "sseqid",
    "qlen",
    "slen",
    "pident",
    "length",
    "nident",
    "mismatch",
    "gaps",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
)

EDGE_COLUMNS = (
    "search_pair",
    "query_sequence_id",
    "subject_sequence_id",
    "query_split",
    "subject_split",
    "query_length_bp",
    "subject_length_bp",
    "length_ratio",
    "identity",
    "query_coverage",
    "subject_coverage",
    "alignment_length",
    "identical_bases",
    "mismatches",
    "gaps",
    "orientation",
    "evalue",
    "bitscore",
    "primary_near_duplicate",
    "sensitivity_near_duplicate",
    "similarity_class",
)

PAF_COLUMNS = (
    "qname",
    "qlen",
    "qstart",
    "qend",
    "strand",
    "tname",
    "tlen",
    "tstart",
    "tend",
    "matching_bases",
    "alignment_block_length",
    "mapq",
)

MINIMAP_EDGE_COLUMNS = (
    "search_pair",
    "query_sequence_id",
    "subject_sequence_id",
    "query_split",
    "subject_split",
    "query_length_bp",
    "subject_length_bp",
    "length_ratio",
    "identity",
    "query_coverage",
    "subject_coverage",
    "alignment_block_length",
    "matching_bases",
    "mapq",
    "orientation",
    "primary_near_duplicate",
    "sensitivity_near_duplicate",
    "similarity_class",
)

_INVALID_DNA = re.compile(f"[^{''.join(sorted(DNA_ALPHABET))}]")


@dataclass(frozen=True)
class SimilarityRule:
    """Inclusive whole-plasmid similarity thresholds, expressed as fractions."""

    minimum_identity: float
    minimum_query_coverage: float
    minimum_subject_coverage: float
    minimum_length_ratio: float

    def __post_init__(self) -> None:
        values = {
            "minimum_identity": self.minimum_identity,
            "minimum_query_coverage": self.minimum_query_coverage,
            "minimum_subject_coverage": self.minimum_subject_coverage,
            "minimum_length_ratio": self.minimum_length_ratio,
        }
        invalid = {name: value for name, value in values.items() if not 0.0 < value <= 1.0}
        if invalid:
            raise ValueError(f"similarity thresholds must be in (0, 1]: {invalid}")


def similarity_rule(config: Mapping[str, Any]) -> SimilarityRule:
    """Build one validated rule from a resolved configuration mapping."""
    return SimilarityRule(
        minimum_identity=float(config["minimum_identity"]),
        minimum_query_coverage=float(config["minimum_query_coverage"]),
        minimum_subject_coverage=float(config["minimum_subject_coverage"]),
        minimum_length_ratio=float(config["minimum_length_ratio"]),
    )


def validate_retrieval(
    frame: pd.DataFrame,
    *,
    expected_population_sha256: str,
    allowed_splits: Sequence[str] = SPLIT_LABELS,
) -> dict[str, Any]:
    """Validate sequence, identity, and existing split invariants at the audit boundary."""
    required = {
        "sequence_id",
        "sequence",
        "sequence_sha256",
        "family_key",
        "leakage_component",
        "split_grouped",
        "length_bp",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"retrieval dataset is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("retrieval dataset is empty")
    if frame[list(required)].isna().any().any():
        counts = frame[list(required)].isna().sum()
        bad = {column: int(count) for column, count in counts.items() if count}
        raise ValueError(f"retrieval dataset contains missing required values: {bad}")
    if frame["sequence_id"].duplicated().any():
        examples = frame.loc[frame["sequence_id"].duplicated(False), "sequence_id"].head(5)
        raise ValueError(
            f"retrieval dataset contains duplicate sequence_id values: {examples.tolist()}"
        )

    observed_splits = set(frame["split_grouped"].astype(str))
    unexpected = observed_splits.difference(str(value) for value in allowed_splits)
    if unexpected:
        raise ValueError(f"retrieval dataset contains unexpected splits: {sorted(unexpected)}")

    sequences = frame["sequence"].astype(str)
    lengths = frame["length_bp"].astype("int64")
    invalid_lengths = lengths.ne(sequences.str.len()) | lengths.le(0)
    if invalid_lengths.any():
        examples = frame.loc[invalid_lengths, ["sequence_id", "length_bp"]].head(5)
        raise ValueError(
            f"{int(invalid_lengths.sum())} sequences have invalid length_bp values: "
            f"{examples.to_dict('records')}"
        )
    invalid_alphabet = sequences.str.contains(_INVALID_DNA, regex=True)
    if invalid_alphabet.any():
        examples = frame.loc[invalid_alphabet, "sequence_id"].head(5).tolist()
        raise ValueError(
            f"{int(invalid_alphabet.sum())} sequences contain non-IUPAC characters: {examples}"
        )

    observed_hashes = sequences.map(sequence_sha256)
    invalid_hashes = observed_hashes.ne(frame["sequence_sha256"].astype(str))
    if invalid_hashes.any():
        examples = frame.loc[invalid_hashes, "sequence_id"].head(5).tolist()
        raise ValueError(
            f"{int(invalid_hashes.sum())} sequence_sha256 values do not match sequence content: "
            f"{examples}"
        )

    population_hash = retrieval_population_sha256(frame)
    if population_hash != expected_population_sha256:
        raise ValueError(
            "retrieval population changed: "
            f"expected {expected_population_sha256}, observed {population_hash}"
        )

    purity: dict[str, int] = {}
    for key in ("family_key", "sequence_sha256", "leakage_component"):
        split_counts = frame.groupby(key, sort=False)["split_grouped"].nunique()
        purity[key] = int(split_counts.gt(1).sum())
        if purity[key]:
            raise ValueError(f"{purity[key]} {key} groups cross the existing grouped split")

    return {
        "rows": int(len(frame)),
        "input_population_sha256": population_hash,
        "rows_by_split": {
            str(split): int(count)
            for split, count in frame["split_grouped"].value_counts().sort_index().items()
        },
        "sequence_ids": int(frame["sequence_id"].nunique()),
        "exact_sequence_groups": int(frame["sequence_sha256"].nunique()),
        "families": int(frame["family_key"].nunique()),
        "leakage_components": int(frame["leakage_component"].nunique()),
        "groups_crossing_existing_split": purity,
        "sequence_lengths_match": True,
        "sequence_hashes_match": True,
        "sequence_alphabet": "IUPAC_DNA",
    }


def build_sequence_tokens(frame: pd.DataFrame) -> pd.DataFrame:
    """Return stable BLAST-safe identifiers without exposing source identifiers to parsing."""
    ordered = frame.sort_values("sequence_id", kind="stable").reset_index(drop=True)
    tokens = [f"v2v_{index:09d}" for index in range(len(ordered))]
    return pd.DataFrame(
        {
            "token": tokens,
            "sequence_id": ordered["sequence_id"].astype(str),
            "split_grouped": ordered["split_grouped"].astype(str),
            "length_bp": ordered["length_bp"].astype("int64"),
            "sequence": ordered["sequence"].astype(str),
            "leakage_component": ordered["leakage_component"].astype(str),
        }
    )


def empty_blast_frame() -> pd.DataFrame:
    """Return an empty BLAST table with the stable parser contract."""
    return pd.DataFrame(columns=BLAST_COLUMNS)


def validate_candidate_cap(hsps: pd.DataFrame, *, maximum_targets: int) -> None:
    """Fail conservatively when BLAST may have truncated a query's target list."""
    if maximum_targets < 1:
        raise ValueError("maximum_targets must be positive")
    if hsps.empty:
        return
    missing = set(BLAST_COLUMNS).difference(hsps.columns)
    if missing:
        raise ValueError(f"BLAST table is missing columns: {sorted(missing)}")
    counts = hsps.groupby("qseqid", sort=True)["sseqid"].nunique()
    saturated = counts[counts.ge(maximum_targets)]
    if not saturated.empty:
        examples = {str(key): int(value) for key, value in saturated.head(5).items()}
        raise RuntimeError(
            f"{len(saturated)} queries reached the BLAST target cap {maximum_targets}: {examples}"
        )


def classify_blast_hsps(
    hsps: pd.DataFrame,
    *,
    query_tokens: pd.DataFrame,
    subject_tokens: pd.DataFrame,
    search_pair: str,
    query_repeat: int,
    primary_rule: SimilarityRule,
    sensitivity_rule: SimilarityRule,
) -> pd.DataFrame:
    """Select and classify the best whole-plasmid HSP for each candidate pair."""
    if query_repeat < 1:
        raise ValueError("query_repeat must be positive")
    if hsps.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    missing = set(BLAST_COLUMNS).difference(hsps.columns)
    if missing:
        raise ValueError(f"BLAST table is missing columns: {sorted(missing)}")

    query_lookup = _token_lookup(query_tokens, "query")
    subject_lookup = _token_lookup(subject_tokens, "subject")
    unknown_queries = set(hsps["qseqid"].astype(str)).difference(query_lookup)
    unknown_subjects = set(hsps["sseqid"].astype(str)).difference(subject_lookup)
    if unknown_queries or unknown_subjects:
        raise ValueError(
            "BLAST output contains unknown identifiers: "
            f"queries={sorted(unknown_queries)[:5]}, subjects={sorted(unknown_subjects)[:5]}"
        )

    records: list[dict[str, Any]] = []
    for row in hsps.itertuples(index=False):
        query = query_lookup[str(row.qseqid)]
        subject = subject_lookup[str(row.sseqid)]
        query_length = int(query["length_bp"])
        subject_length = int(subject["length_bp"])
        expected_query_length = query_length * query_repeat
        if int(row.qlen) != expected_query_length or int(row.slen) != subject_length:
            raise ValueError(
                "BLAST length does not match source data for "
                f"{row.qseqid}/{row.sseqid}: "
                f"qlen={row.qlen} expected={expected_query_length}, "
                f"slen={row.slen} expected={subject_length}"
            )
        query_coordinates = (int(row.qstart), int(row.qend))
        subject_coordinates = (int(row.sstart), int(row.send))
        if any(value < 1 or value > expected_query_length for value in query_coordinates):
            raise ValueError(f"BLAST query coordinates are out of bounds: {query_coordinates}")
        if any(value < 1 or value > subject_length for value in subject_coordinates):
            raise ValueError(f"BLAST subject coordinates are out of bounds: {subject_coordinates}")
        alignment_length = int(row.alignment_length)
        identical_bases = int(row.nident)
        mismatch = int(row.mismatch)
        gaps = int(row.gaps)
        if (
            alignment_length < 1
            or not 0 <= identical_bases <= alignment_length
            or mismatch < 0
            or gaps < 0
        ):
            raise ValueError("BLAST output contains invalid alignment or identity counts")
        identity = identical_bases / alignment_length
        if abs(identity * 100.0 - float(row.pident)) > 0.011:
            raise ValueError("BLAST pident disagrees with nident/alignment_length")

        query_span = abs(query_coordinates[1] - query_coordinates[0]) + 1
        subject_span = abs(subject_coordinates[1] - subject_coordinates[0]) + 1
        query_coverage = min(query_span, query_length) / query_length
        subject_coverage = min(subject_span, subject_length) / subject_length
        length_ratio = min(query_length, subject_length) / max(query_length, subject_length)
        orientation = (
            "same"
            if (int(row.qend) - int(row.qstart)) * (int(row.send) - int(row.sstart)) >= 0
            else "reverse_complement"
        )
        records.append(
            {
                "search_pair": search_pair,
                "qseqid": str(row.qseqid),
                "sseqid": str(row.sseqid),
                "query_sequence_id": str(query["sequence_id"]),
                "subject_sequence_id": str(subject["sequence_id"]),
                "query_split": str(query["split_grouped"]),
                "subject_split": str(subject["split_grouped"]),
                "query_length_bp": query_length,
                "subject_length_bp": subject_length,
                "length_ratio": length_ratio,
                "identity": identity,
                "query_coverage": query_coverage,
                "subject_coverage": subject_coverage,
                "minimum_coverage": min(query_coverage, subject_coverage),
                "alignment_length": alignment_length,
                "identical_bases": identical_bases,
                "mismatches": mismatch,
                "gaps": gaps,
                "orientation": orientation,
                "evalue": float(row.evalue),
                "bitscore": float(row.bitscore),
            }
        )

    candidates = pd.DataFrame.from_records(records)
    candidates = candidates.sort_values(
        [
            "qseqid",
            "sseqid",
            "minimum_coverage",
            "identity",
            "bitscore",
            "alignment_length",
        ],
        ascending=[True, True, False, False, False, False],
        kind="stable",
    ).drop_duplicates(["qseqid", "sseqid"], keep="first")

    candidates["primary_near_duplicate"] = _meets_rule(candidates, primary_rule)
    candidates["sensitivity_near_duplicate"] = _meets_rule(candidates, sensitivity_rule)
    if (candidates["primary_near_duplicate"] & ~candidates["sensitivity_near_duplicate"]).any():
        raise ValueError("the primary rule is not a subset of the sensitivity rule")
    candidates["similarity_class"] = np.select(
        [
            candidates["primary_near_duplicate"],
            candidates["sensitivity_near_duplicate"],
        ],
        ["primary", "sensitivity_only"],
        default="candidate_only",
    )
    candidates = candidates.sort_values(
        ["similarity_class", "search_pair", "query_sequence_id", "subject_sequence_id"],
        kind="stable",
    ).reset_index(drop=True)
    return candidates.loc[:, EDGE_COLUMNS]


def classify_minimap_alignments(
    alignments: pd.DataFrame,
    *,
    query_tokens: pd.DataFrame,
    subject_tokens: pd.DataFrame,
    search_pair: str,
    query_repeat: int,
    primary_rule: SimilarityRule,
    sensitivity_rule: SimilarityRule,
) -> pd.DataFrame:
    """Select and classify the best minimap2 PAF alignment per sequence pair."""
    if query_repeat < 1:
        raise ValueError("query_repeat must be positive")
    if alignments.empty:
        return pd.DataFrame(columns=MINIMAP_EDGE_COLUMNS)
    missing = set(PAF_COLUMNS).difference(alignments.columns)
    if missing:
        raise ValueError(f"minimap2 PAF table is missing columns: {sorted(missing)}")

    query_lookup = _token_lookup(query_tokens, "query")
    subject_lookup = _token_lookup(subject_tokens, "subject")
    unknown_queries = set(alignments["qname"].astype(str)).difference(query_lookup)
    unknown_subjects = set(alignments["tname"].astype(str)).difference(subject_lookup)
    if unknown_queries or unknown_subjects:
        raise ValueError(
            "minimap2 output contains unknown identifiers: "
            f"queries={sorted(unknown_queries)[:5]}, subjects={sorted(unknown_subjects)[:5]}"
        )

    records: list[dict[str, Any]] = []
    for row in alignments.itertuples(index=False):
        query = query_lookup[str(row.qname)]
        subject = subject_lookup[str(row.tname)]
        query_length = int(query["length_bp"])
        subject_length = int(subject["length_bp"])
        expected_query_length = query_length * query_repeat
        if int(row.qlen) != expected_query_length or int(row.tlen) != subject_length:
            raise ValueError(
                "minimap2 length does not match source data for "
                f"{row.qname}/{row.tname}: qlen={row.qlen} expected={expected_query_length}, "
                f"tlen={row.tlen} expected={subject_length}"
            )
        query_coordinates = (int(row.qstart), int(row.qend))
        subject_coordinates = (int(row.tstart), int(row.tend))
        if not 0 <= query_coordinates[0] < query_coordinates[1] <= expected_query_length:
            raise ValueError(f"minimap2 query coordinates are out of bounds: {query_coordinates}")
        if not 0 <= subject_coordinates[0] < subject_coordinates[1] <= subject_length:
            raise ValueError(
                f"minimap2 subject coordinates are out of bounds: {subject_coordinates}"
            )
        alignment_length = int(row.alignment_block_length)
        matching_bases = int(row.matching_bases)
        mapq = int(row.mapq)
        if (
            alignment_length < 1
            or not 0 <= matching_bases <= alignment_length
            or not 0 <= mapq <= 255
        ):
            raise ValueError("minimap2 output contains invalid alignment counts or mapq")
        strand = str(row.strand)
        if strand not in {"+", "-"}:
            raise ValueError(f"minimap2 output contains an invalid strand: {strand}")

        query_span = query_coordinates[1] - query_coordinates[0]
        subject_span = subject_coordinates[1] - subject_coordinates[0]
        query_coverage = min(query_span, query_length) / query_length
        subject_coverage = min(subject_span, subject_length) / subject_length
        identity = matching_bases / alignment_length
        length_ratio = min(query_length, subject_length) / max(query_length, subject_length)
        records.append(
            {
                "search_pair": search_pair,
                "qname": str(row.qname),
                "tname": str(row.tname),
                "query_sequence_id": str(query["sequence_id"]),
                "subject_sequence_id": str(subject["sequence_id"]),
                "query_split": str(query["split_grouped"]),
                "subject_split": str(subject["split_grouped"]),
                "query_length_bp": query_length,
                "subject_length_bp": subject_length,
                "length_ratio": length_ratio,
                "identity": identity,
                "query_coverage": query_coverage,
                "subject_coverage": subject_coverage,
                "minimum_coverage": min(query_coverage, subject_coverage),
                "alignment_block_length": alignment_length,
                "matching_bases": matching_bases,
                "mapq": mapq,
                "orientation": "same" if strand == "+" else "reverse_complement",
            }
        )

    candidates = pd.DataFrame.from_records(records)
    candidates = candidates.sort_values(
        ["qname", "tname", "minimum_coverage", "identity", "mapq", "alignment_block_length"],
        ascending=[True, True, False, False, False, False],
        kind="stable",
    ).drop_duplicates(["qname", "tname"], keep="first")
    candidates["primary_near_duplicate"] = _meets_rule(candidates, primary_rule)
    candidates["sensitivity_near_duplicate"] = _meets_rule(candidates, sensitivity_rule)
    if (candidates["primary_near_duplicate"] & ~candidates["sensitivity_near_duplicate"]).any():
        raise ValueError("the primary rule is not a subset of the sensitivity rule")
    candidates["similarity_class"] = np.select(
        [candidates["primary_near_duplicate"], candidates["sensitivity_near_duplicate"]],
        ["primary", "sensitivity_only"],
        default="candidate_only",
    )
    candidates = candidates.sort_values(
        ["similarity_class", "search_pair", "query_sequence_id", "subject_sequence_id"],
        kind="stable",
    ).reset_index(drop=True)
    return candidates.loc[:, MINIMAP_EDGE_COLUMNS]


def profile_split_concentration(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Measure existing component concentration without changing the split."""
    grouped = frame.groupby("leakage_component", sort=True, dropna=False)
    split_counts = grouped["split_grouped"].nunique()
    if split_counts.gt(1).any():
        raise ValueError(f"{int(split_counts.gt(1).sum())} leakage components cross splits")
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

    summaries: dict[str, Any] = {}
    for split, split_frame in profile.groupby("split_grouped", sort=True):
        sizes = split_frame["rows"].to_numpy(dtype=np.int64)
        row_count = int(sizes.sum())
        weights = sizes / row_count
        largest_fraction = float(sizes.max() / row_count)
        summaries[str(split)] = {
            "rows": row_count,
            "components": int(len(sizes)),
            "singleton_components": int(np.count_nonzero(sizes == 1)),
            "median_component_rows": float(np.quantile(sizes, 0.5, method="linear")),
            "p90_component_rows": float(np.quantile(sizes, 0.9, method="linear")),
            "largest_component_rows": int(sizes.max()),
            "largest_component_row_fraction": largest_fraction,
            "ten_largest_component_rows": int(np.sort(sizes)[-10:].sum()),
            "ten_largest_component_row_fraction": float(np.sort(sizes)[-10:].sum() / row_count),
            "effective_component_count": float(1.0 / np.square(weights).sum()),
            "component_macro_required": largest_fraction >= 0.25,
        }
    return profile, summaries


def augmented_component_summary(
    frame: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    edge_flag: str,
) -> dict[str, Any]:
    """Summarize how qualifying edges connect current leakage components."""
    if edge_flag not in edges.columns:
        raise ValueError(f"edge table is missing {edge_flag}")
    selected = edges.loc[edges[edge_flag].astype(bool)].copy()
    component_rows = frame.groupby("leakage_component", sort=True).size().astype("int64")
    component_splits = frame.groupby("leakage_component", sort=True)["split_grouped"].first()
    sequence_components = frame.set_index("sequence_id")["leakage_component"].to_dict()
    unknown = (
        set(selected["query_sequence_id"])
        .union(selected["subject_sequence_id"])
        .difference(sequence_components)
    )
    if unknown:
        raise ValueError(f"edge table contains unknown sequence IDs: {sorted(unknown)[:5]}")

    components = component_rows.index.tolist()
    positions = {component: position for position, component in enumerate(components)}
    parent = list(range(len(components)))

    def find(position: int) -> int:
        root = position
        while parent[root] != root:
            root = parent[root]
        while parent[position] != root:
            parent[position], position = root, parent[position]
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in selected.itertuples(index=False):
        left = sequence_components[str(row.query_sequence_id)]
        right = sequence_components[str(row.subject_sequence_id)]
        union(positions[left], positions[right])

    by_root: dict[int, list[Any]] = {}
    for component in components:
        by_root.setdefault(find(positions[component]), []).append(component)
    merged = [members for members in by_root.values() if len(members) > 1]
    crossing = [
        members
        for members in merged
        if len({str(component_splits.loc[component]) for component in members}) > 1
    ]
    augmented_sizes = [int(component_rows.loc[members].sum()) for members in by_root.values()]
    return {
        "edge_flag": edge_flag,
        "qualifying_edges": int(len(selected)),
        "involved_sequences": int(
            len(set(selected["query_sequence_id"]).union(selected["subject_sequence_id"]))
        ),
        "involved_current_components": int(
            len(
                {
                    sequence_components[sequence_id]
                    for sequence_id in set(selected["query_sequence_id"]).union(
                        selected["subject_sequence_id"]
                    )
                }
            )
        ),
        "augmented_components": int(len(by_root)),
        "augmented_components_merging_current_components": int(len(merged)),
        "augmented_components_crossing_original_splits": int(len(crossing)),
        "largest_augmented_component_rows": int(max(augmented_sizes)),
    }


def summarize_edges(edges: pd.DataFrame) -> dict[str, Any]:
    """Return transparent edge counts by search pair and threshold class."""
    summary: dict[str, Any] = {
        "candidate_edges": int(len(edges)),
        "primary_edges": int(edges["primary_near_duplicate"].sum()) if len(edges) else 0,
        "sensitivity_edges": int(edges["sensitivity_near_duplicate"].sum()) if len(edges) else 0,
        "sensitivity_only_edges": int(
            (edges["sensitivity_near_duplicate"] & ~edges["primary_near_duplicate"]).sum()
        )
        if len(edges)
        else 0,
    }
    by_pair: dict[str, Any] = {}
    for search_pair, group in edges.groupby("search_pair", sort=True):
        by_pair[str(search_pair)] = {
            "candidate_edges": int(len(group)),
            "primary_edges": int(group["primary_near_duplicate"].sum()),
            "sensitivity_edges": int(group["sensitivity_near_duplicate"].sum()),
            "queries_with_candidates": int(group["query_sequence_id"].nunique()),
        }
    summary["by_search_pair"] = by_pair
    return summary


def _token_lookup(tokens: pd.DataFrame, label: str) -> dict[str, dict[str, Any]]:
    required = {"token", "sequence_id", "split_grouped", "length_bp"}
    missing = required.difference(tokens.columns)
    if missing:
        raise ValueError(f"{label} token table is missing columns: {sorted(missing)}")
    if tokens["token"].duplicated().any():
        raise ValueError(f"{label} token table contains duplicate tokens")
    return {
        str(record["token"]): record
        for record in tokens.loc[:, sorted(required)].to_dict("records")
    }


def _meets_rule(candidates: pd.DataFrame, rule: SimilarityRule) -> pd.Series:
    return (
        candidates["identity"].ge(rule.minimum_identity)
        & candidates["query_coverage"].ge(rule.minimum_query_coverage)
        & candidates["subject_coverage"].ge(rule.minimum_subject_coverage)
        & candidates["length_ratio"].ge(rule.minimum_length_ratio)
    )
