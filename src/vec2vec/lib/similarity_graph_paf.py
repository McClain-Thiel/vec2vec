"""Pure PAF parsing used by the global similarity graph."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from vec2vec.lib.split_audit import SimilarityRule


def parse_candidate_paf(
    path: Path,
    *,
    token_lengths: Mapping[str, int],
    query_tokens: set[str],
    query_repeat: int,
    cap: int,
    filters: Mapping[str, float],
) -> pd.DataFrame:
    """Summarize approximate PAF candidates without treating them as exact identities."""
    if query_repeat < 1 or cap < 1:
        raise ValueError("query_repeat and cap must be positive")
    raw_counts: dict[str, int] = defaultdict(int)
    targets: dict[str, set[str]] = defaultdict(set)
    filtered_targets: dict[str, set[str]] = defaultdict(set)
    unknown: set[str] = set()

    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                raise ValueError(f"PAF line {line_number} has fewer than 12 fields")
            query, target = fields[0], fields[5]
            if (
                query not in query_tokens
                or query not in token_lengths
                or target not in token_lengths
            ):
                unknown.update(value for value in (query, target) if value not in token_lengths)
                continue
            raw_counts[query] += 1
            if query == target:
                continue
            query_length = int(token_lengths[query])
            target_length = int(token_lengths[target])
            if int(fields[1]) != query_length * query_repeat or int(fields[6]) != target_length:
                raise ValueError(f"PAF lengths disagree with source data on line {line_number}")
            targets[query].add(target)
            query_coverage = min(int(fields[3]) - int(fields[2]), query_length) / query_length
            subject_coverage = (int(fields[8]) - int(fields[7])) / target_length
            length_ratio = min(query_length, target_length) / max(query_length, target_length)
            divergence = _paf_divergence(fields[12:], line_number=line_number)
            if (
                query_coverage >= float(filters["minimum_approximate_query_coverage"])
                and subject_coverage >= float(filters["minimum_approximate_subject_coverage"])
                and length_ratio >= float(filters["minimum_length_ratio"])
                and divergence <= float(filters["maximum_approximate_divergence"])
            ):
                filtered_targets[query].add(target)
    if unknown:
        raise ValueError(f"PAF contains unknown tokens: {sorted(unknown)[:5]}")

    records = []
    for query in sorted(query_tokens):
        raw = int(raw_counts.get(query, 0))
        unique = len(targets.get(query, set()))
        records.append(
            {
                "token": query,
                "raw_alignments": raw,
                "unique_nonself_targets": unique,
                "approximate_candidates": len(filtered_targets.get(query, set())),
                "potentially_saturated": raw >= cap or unique >= cap,
            }
        )
    return pd.DataFrame.from_records(records)


def parse_exact_paf(
    path: Path,
    *,
    token_records: Mapping[str, Mapping[str, Any]],
    query_tokens: set[str],
    query_repeat: int,
    cap: int,
    primary_rule: SimilarityRule,
    sensitivity_rule: SimilarityRule,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse base-aligned PAF and return exact per-query profiles and qualifying edges."""
    raw_counts: dict[str, int] = defaultdict(int)
    targets: dict[str, set[str]] = defaultdict(set)
    best: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="ascii") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                raise ValueError(f"PAF line {line_number} has fewer than 12 fields")
            if not any(tag.startswith("cg:Z:") for tag in fields[12:]):
                raise ValueError(
                    f"exact PAF line {line_number} has no CIGAR and is not base-aligned"
                )
            query, target = fields[0], fields[5]
            if (
                query not in query_tokens
                or query not in token_records
                or target not in token_records
            ):
                raise ValueError(f"PAF contains an unknown token on line {line_number}")
            raw_counts[query] += 1
            if query == target:
                continue
            query_record = token_records[query]
            target_record = token_records[target]
            query_length = int(query_record["length_bp"])
            target_length = int(target_record["length_bp"])
            if int(fields[1]) != query_length * query_repeat or int(fields[6]) != target_length:
                raise ValueError(f"PAF lengths disagree with source data on line {line_number}")
            targets[query].add(target)
            alignment_length = int(fields[10])
            matching_bases = int(fields[9])
            if alignment_length < 1 or not 0 <= matching_bases <= alignment_length:
                raise ValueError(f"PAF has invalid alignment counts on line {line_number}")
            query_coverage = min(int(fields[3]) - int(fields[2]), query_length) / query_length
            subject_coverage = (int(fields[8]) - int(fields[7])) / target_length
            identity = matching_bases / alignment_length
            length_ratio = min(query_length, target_length) / max(query_length, target_length)
            record = {
                "query_token": query,
                "subject_token": target,
                "query_sequence_id": str(query_record["sequence_id"]),
                "subject_sequence_id": str(target_record["sequence_id"]),
                "query_length_bp": query_length,
                "subject_length_bp": target_length,
                "identity": identity,
                "query_coverage": query_coverage,
                "subject_coverage": subject_coverage,
                "length_ratio": length_ratio,
                "orientation": "same" if fields[4] == "+" else "reverse_complement",
                "alignment_block_length": alignment_length,
                "matching_bases": matching_bases,
            }
            key = (query, target)
            score = (min(query_coverage, subject_coverage), identity, alignment_length)
            old = best.get(key)
            if old is None or score > old["_score"]:
                record["_score"] = score
                best[key] = record

    edge_records = []
    primary_by_query: dict[str, int] = defaultdict(int)
    sensitivity_by_query: dict[str, int] = defaultdict(int)
    for record in best.values():
        primary = _meets_rule(record, primary_rule)
        sensitivity = _meets_rule(record, sensitivity_rule)
        if primary and not sensitivity:
            raise ValueError("primary rule is not nested within the sensitivity rule")
        if not sensitivity:
            continue
        record.pop("_score")
        record["primary_near_duplicate"] = primary
        record["sensitivity_near_duplicate"] = sensitivity
        record["similarity_class"] = "primary" if primary else "sensitivity_only"
        edge_records.append(record)
        primary_by_query[record["query_token"]] += int(primary)
        sensitivity_by_query[record["query_token"]] += 1

    profiles = pd.DataFrame.from_records(
        [
            {
                "token": query,
                "raw_alignments": int(raw_counts.get(query, 0)),
                "unique_nonself_targets": len(targets.get(query, set())),
                "primary_edges": int(primary_by_query.get(query, 0)),
                "sensitivity_edges": int(sensitivity_by_query.get(query, 0)),
                "potentially_saturated": (
                    int(raw_counts.get(query, 0)) >= cap or len(targets.get(query, set())) >= cap
                ),
            }
            for query in sorted(query_tokens)
        ]
    )
    edges = pd.DataFrame.from_records(edge_records)
    return profiles, edges


def _paf_divergence(tags: list[str], *, line_number: int) -> float:
    values = [tag[5:] for tag in tags if tag.startswith("dv:f:")]
    if len(values) != 1:
        raise ValueError(f"PAF line {line_number} does not contain exactly one dv:f tag")
    value = float(values[0])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"PAF line {line_number} has invalid divergence {value}")
    return value


def _meets_rule(record: Mapping[str, Any], rule: SimilarityRule) -> bool:
    return bool(
        float(record["identity"]) >= rule.minimum_identity
        and float(record["query_coverage"]) >= rule.minimum_query_coverage
        and float(record["subject_coverage"]) >= rule.minimum_subject_coverage
        and float(record["length_ratio"]) >= rule.minimum_length_ratio
    )
