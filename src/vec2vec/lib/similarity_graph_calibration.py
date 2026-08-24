"""Pure selection, parsing, and projection logic for similarity-graph calibration."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from vec2vec.lib.split_audit import SimilarityRule


def select_calibration_queries(
    retrieval: pd.DataFrame,
    audit_edges: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Select a deterministic mix of representative and high-density queries."""
    required_retrieval = {"sequence_id", "length_bp", "leakage_component"}
    missing_retrieval = required_retrieval.difference(retrieval.columns)
    if missing_retrieval:
        raise ValueError(f"retrieval dataset is missing columns: {sorted(missing_retrieval)}")
    if retrieval["sequence_id"].duplicated().any():
        raise ValueError("retrieval dataset contains duplicate sequence_id values")
    required_edges = {"query_sequence_id", "subject_sequence_id"}
    missing_edges = required_edges.difference(audit_edges.columns)
    if missing_edges:
        raise ValueError(f"audit edge table is missing columns: {sorted(missing_edges)}")

    total = int(config["total_queries"])
    representative_count = int(config["representative_queries"])
    component_count = int(config["component_stress_queries"])
    edge_count = int(config["edge_stress_queries"])
    exact_count = int(config["exact_benchmark_queries"])
    strata = int(config["length_strata"])
    per_component = int(config["per_component_limit"])
    seed = int(config["seed"])
    if not 1 <= total <= len(retrieval):
        raise ValueError("total calibration queries must be within the retrieval population")
    if min(representative_count, component_count, edge_count, exact_count, strata) < 1:
        raise ValueError("calibration sample counts and strata must be positive")
    if representative_count + component_count + edge_count != total:
        raise ValueError("calibration cohort counts must sum to total_queries")
    if exact_count > total:
        raise ValueError("exact benchmark queries cannot exceed total calibration queries")
    if per_component < 1:
        raise ValueError("per_component_limit must be positive")

    frame = retrieval.loc[:, sorted(required_retrieval)].copy()
    frame["sequence_id"] = frame["sequence_id"].astype(str)
    frame["stable_order"] = frame["sequence_id"].map(lambda value: _stable_hash(value, seed=seed))
    ranked = frame.sort_values(["length_bp", "sequence_id"], kind="stable").index
    rank_by_index = pd.Series(range(len(frame)), index=ranked)
    frame["length_stratum"] = (
        rank_by_index.reindex(frame.index).mul(strata).floordiv(len(frame)).clip(upper=strata - 1)
    ).astype("int64")

    representative_ids = _stratified_ids(
        frame,
        count=representative_count,
        strata=strata,
    )
    component_ids = _component_stress_ids(
        frame,
        count=component_count,
        per_component=per_component,
    )
    edge_ids = _edge_stress_ids(
        frame,
        audit_edges,
        count=edge_count,
    )

    reasons: dict[str, set[str]] = defaultdict(set)
    for sequence_id in representative_ids:
        reasons[sequence_id].add("representative")
    for sequence_id in component_ids:
        reasons[sequence_id].add("component_stress")
    for sequence_id in edge_ids:
        reasons[sequence_id].add("edge_stress")

    selected = set(reasons)
    if len(selected) < total:
        fill = frame.loc[~frame["sequence_id"].isin(selected)].sort_values(
            ["stable_order", "sequence_id"], kind="stable"
        )
        for sequence_id in fill["sequence_id"].head(total - len(selected)):
            reasons[str(sequence_id)].add("deterministic_fill")
    if len(reasons) != total:
        raise RuntimeError(f"calibration selection produced {len(reasons)} rows, expected {total}")

    sample = frame.loc[frame["sequence_id"].isin(reasons)].copy()
    sample["representative"] = sample["sequence_id"].isin(representative_ids)
    sample["component_stress"] = sample["sequence_id"].isin(component_ids)
    sample["edge_stress"] = sample["sequence_id"].isin(edge_ids)
    sample["selection_reason"] = sample["sequence_id"].map(
        lambda value: "+".join(sorted(reasons[str(value)]))
    )
    sample["exact_benchmark"] = False

    stress = sample.loc[sample["component_stress"] | sample["edge_stress"]].sort_values(
        ["stable_order", "sequence_id"], kind="stable"
    )
    representative = sample.loc[
        sample["representative"] & ~(sample["component_stress"] | sample["edge_stress"])
    ].sort_values(["stable_order", "sequence_id"], kind="stable")
    stress_quota = exact_count // 2
    exact_ids = stress["sequence_id"].head(stress_quota).tolist()
    exact_ids.extend(representative["sequence_id"].head(exact_count - len(exact_ids)).tolist())
    if len(exact_ids) < exact_count:
        remaining = sample.loc[~sample["sequence_id"].isin(exact_ids)].sort_values(
            ["stable_order", "sequence_id"], kind="stable"
        )
        exact_ids.extend(remaining["sequence_id"].head(exact_count - len(exact_ids)).tolist())
    sample.loc[sample["sequence_id"].isin(exact_ids), "exact_benchmark"] = True

    if int(sample["exact_benchmark"].sum()) != exact_count:
        raise RuntimeError("exact benchmark selection did not produce its fixed row count")
    return sample.sort_values(["stable_order", "sequence_id"], kind="stable").reset_index(drop=True)


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


def project_full_run(
    runs: pd.DataFrame,
    query_profiles: pd.DataFrame,
    *,
    population_rows: int,
    maximum_cpu_hours: float,
    maximum_persisted_bytes: int,
) -> dict[str, Any]:
    """Scale observed per-query costs and report whether fixed limits are crossed."""
    required_runs = {"mode", "cap", "query_count", "wall_seconds", "cpu_seconds", "paf_bytes"}
    missing_runs = required_runs.difference(runs.columns)
    if missing_runs:
        raise ValueError(f"calibration runs are missing columns: {sorted(missing_runs)}")
    if population_rows < 1:
        raise ValueError("population_rows must be positive")

    projections: dict[str, Any] = {}
    for (mode, cap), group in runs.groupby(["mode", "cap"], sort=True):
        observed_queries = int(group["query_count"].sum())
        if observed_queries < 1:
            continue
        scale = population_rows / observed_queries
        key = f"{mode}_cap_{int(cap)}"
        profile = query_profiles.loc[
            query_profiles["mode"].eq(mode) & query_profiles["cap"].eq(cap)
        ]
        projected_cpu_hours = float(group["cpu_seconds"].sum() * scale / 3600.0)
        projected_bytes = int(math.ceil(group["paf_bytes"].sum() * scale))
        projections[key] = {
            "observed_queries": observed_queries,
            "observed_shards": int(len(group)),
            "observed_wall_seconds": float(group["wall_seconds"].sum()),
            "observed_cpu_seconds": float(group["cpu_seconds"].sum()),
            "observed_paf_bytes": int(group["paf_bytes"].sum()),
            "saturated_queries": int(profile["potentially_saturated"].sum()),
            "projected_population_cpu_hours": projected_cpu_hours,
            "projected_population_paf_bytes": projected_bytes,
            "within_cpu_limit": projected_cpu_hours <= maximum_cpu_hours,
            "within_persisted_byte_limit": projected_bytes <= maximum_persisted_bytes,
        }
    return projections


def _stratified_ids(frame: pd.DataFrame, *, count: int, strata: int) -> set[str]:
    base, remainder = divmod(count, strata)
    selected: list[str] = []
    for stratum in range(strata):
        quota = base + int(stratum < remainder)
        group = frame.loc[frame["length_stratum"].eq(stratum)].sort_values(
            ["stable_order", "sequence_id"], kind="stable"
        )
        selected.extend(group["sequence_id"].head(quota).tolist())
    return set(selected)


def _component_stress_ids(
    frame: pd.DataFrame,
    *,
    count: int,
    per_component: int,
) -> set[str]:
    component_sizes = frame.groupby("leakage_component", sort=False).size().rename("component_size")
    ordered_components = component_sizes.reset_index().sort_values(
        ["component_size", "leakage_component"], ascending=[False, True], kind="stable"
    )
    selected: list[str] = []
    for component in ordered_components["leakage_component"]:
        group = frame.loc[frame["leakage_component"].eq(component)].sort_values(
            ["stable_order", "sequence_id"], kind="stable"
        )
        selected.extend(group["sequence_id"].head(per_component).tolist())
        if len(selected) >= count:
            break
    return set(selected[:count])


def _edge_stress_ids(
    frame: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    count: int,
) -> set[str]:
    identifiers = pd.concat(
        [
            edges["query_sequence_id"].astype(str),
            edges["subject_sequence_id"].astype(str),
        ],
        ignore_index=True,
    )
    degrees = identifiers.value_counts().rename("audit_degree")
    candidates = frame.join(degrees, on="sequence_id")
    candidates["audit_degree"] = candidates["audit_degree"].fillna(0).astype("int64")
    candidates = candidates.sort_values(
        ["audit_degree", "stable_order", "sequence_id"],
        ascending=[False, True, True],
        kind="stable",
    )
    return set(candidates["sequence_id"].head(count).tolist())


def _stable_hash(value: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


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
