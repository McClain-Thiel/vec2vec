"""External-tool boundary for the E00 split audit."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import split_audit

logger = logging.getLogger(__name__)


def run_split_audit(
    retrieval: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the global lower-bound search and concentration measurements."""
    validation = split_audit.validate_retrieval(
        retrieval,
        expected_population_sha256=str(params["expected_input_population_sha256"]),
        allowed_splits=tuple(str(value) for value in params["split_labels"]),
    )
    primary_rule = split_audit.similarity_rule(params["primary_rule"])
    sensitivity_rule = split_audit.similarity_rule(params["sensitivity_rule"])
    _validate_rule_nesting(primary_rule, sensitivity_rule)
    search = _resolved_search_config(params["search"])

    tokens = split_audit.build_sequence_tokens(retrieval)
    component_profile, concentration = split_audit.profile_split_concentration(retrieval)
    tool_versions = {"minimap2": _tool_version(search["executable"])}

    with tempfile.TemporaryDirectory(prefix="vec2vec-split-audit-") as raw_directory:
        root = Path(raw_directory)
        synthetic = _run_synthetic_validation(
            root / "synthetic",
            search,
            primary_rule,
        )
        edges, search_records = _run_complete_searches(
            root / "complete",
            tokens,
            search,
            primary_rule,
            sensitivity_rule,
        )

    edge_summary = split_audit.summarize_edges(edges)
    edge_summary["counts_are_lower_bounds"] = True
    primary_augmented = split_audit.augmented_component_summary(
        retrieval,
        edges,
        edge_flag="primary_near_duplicate",
    )
    sensitivity_augmented = split_audit.augmented_component_summary(
        retrieval,
        edges,
        edge_flag="sensitivity_near_duplicate",
    )
    component_macro_splits = [
        split for split, record in concentration.items() if record["component_macro_required"]
    ]
    primary_edges = int(edge_summary["primary_edges"])
    strict_decision = (
        "fail_current_split_requires_v2"
        if primary_edges
        else "inconclusive_no_counterexample_in_lower_bound_search"
    )
    manifest = {
        "audit_version": str(params["audit_version"]),
        "protocol": (
            "studies/set_valued_compositional_embeddings/experiments/E00_split_similarity_audit.md"
        ),
        "input_retrieval_version": str(params["input_retrieval_version"]),
        "input_validation": validation,
        "resolved_configuration": params,
        "tool_versions": tool_versions,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "git": _git_provenance(),
        "synthetic_validation": synthetic,
        "searches": search_records,
        "edge_summary": edge_summary,
        "primary_augmented_components": primary_augmented,
        "sensitivity_augmented_components": sensitivity_augmented,
        "concentration_by_split": concentration,
        "decision": {
            "strict_near_duplicate_rule": strict_decision,
            "primary_cross_split_edges_allowed": 0,
            "component_macro_required_splits": component_macro_splits,
            "existing_split_overwritten": False,
            "model_outcomes_inspected": False,
            "all_query_sequences_searched_globally": True,
            "edge_enumeration_complete": False,
            "edge_counts_are_lower_bounds": True,
            "nonzero_primary_lower_bound_is_sufficient_to_reject_current_split": True,
        },
        "known_limitations": [
            "Minimap2 is a heuristic candidate search, not an all-pairs proof of absence.",
            "The secondary-alignment limit makes edge and augmented-component counts lower bounds.",
            "A zero primary lower bound would be inconclusive and could not pass the split.",
            "Sequence similarity does not establish functional independence.",
        ],
    }
    logger.info(
        "Split audit completed with lower bounds of %s primary and %s sensitivity edges",
        edge_summary["primary_edges"],
        edge_summary["sensitivity_edges"],
    )
    return edges, component_profile, manifest


def _run_complete_searches(
    root: Path,
    tokens: pd.DataFrame,
    search: dict[str, Any],
    primary_rule: split_audit.SimilarityRule,
    sensitivity_rule: split_audit.SimilarityRule,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    root.mkdir(parents=True, exist_ok=False)
    by_split = {
        split: tokens.loc[tokens["split_grouped"].eq(split)].reset_index(drop=True)
        for split in split_audit.SPLIT_LABELS
    }
    target_indexes: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for split in ("train", "val"):
        if by_split[split].empty:
            continue
        fasta = root / f"{split}_target.fasta"
        index = root / f"{split}.mmi"
        _write_fasta(by_split[split], fasta, repeat=1)
        duration, tool_log = _build_index(fasta, index, search)
        target_indexes[split] = index
        records.append(
            {
                "operation": "build_index",
                "subject_split": split,
                "subject_sequences": int(len(by_split[split])),
                "duration_seconds": duration,
                "tool_log": tool_log,
            }
        )

    query_paths: dict[str, Path] = {}
    for split in ("val", "test"):
        if by_split[split].empty:
            continue
        query_path = root / f"{split}_query_repeated.fasta"
        _write_fasta(by_split[split], query_path, repeat=search["query_repeat"])
        query_paths[split] = query_path

    edge_tables: list[pd.DataFrame] = []
    pair_definitions = (("val", "train"), ("test", "train"), ("test", "val"))
    for query_split, subject_split in pair_definitions:
        search_pair = f"{query_split}_vs_{subject_split}"
        if query_split not in query_paths or subject_split not in target_indexes:
            edge_tables.append(pd.DataFrame(columns=split_audit.MINIMAP_EDGE_COLUMNS))
            records.append(
                {
                    "operation": "map",
                    "search_pair": search_pair,
                    "query_sequences": int(len(by_split[query_split])),
                    "subject_sequences": int(len(by_split[subject_split])),
                    "raw_alignments": 0,
                    "candidate_pairs_lower_bound": 0,
                    "queries_potentially_truncated": 0,
                    "duration_seconds": 0.0,
                    "skipped_empty_split": True,
                    "tool_log": [],
                }
            )
            continue

        output_path = root / f"{search_pair}.paf"
        alignments, duration, tool_log = _map(
            query_paths[query_split],
            target_indexes[subject_split],
            output_path,
            search,
        )
        edges = split_audit.classify_minimap_alignments(
            alignments,
            query_tokens=by_split[query_split],
            subject_tokens=by_split[subject_split],
            search_pair=search_pair,
            query_repeat=search["query_repeat"],
            primary_rule=primary_rule,
            sensitivity_rule=sensitivity_rule,
        )
        raw_counts = alignments.groupby("qname", sort=False).size()
        potentially_truncated = raw_counts.ge(search["secondary_alignment_limit"]).sum()
        edge_tables.append(edges)
        records.append(
            {
                "operation": "map",
                "search_pair": search_pair,
                "query_sequences": int(len(by_split[query_split])),
                "subject_sequences": int(len(by_split[subject_split])),
                "raw_alignments": int(len(alignments)),
                "candidate_pairs_lower_bound": int(len(edges)),
                "queries_with_alignments": int(alignments["qname"].nunique())
                if len(alignments)
                else 0,
                "queries_potentially_truncated": int(potentially_truncated),
                "duration_seconds": duration,
                "skipped_empty_split": False,
                "tool_log": tool_log,
            }
        )

    combined = pd.concat(edge_tables, ignore_index=True)
    if combined.empty:
        combined = pd.DataFrame(columns=split_audit.MINIMAP_EDGE_COLUMNS)
    else:
        combined = combined.sort_values(
            ["search_pair", "query_sequence_id", "subject_sequence_id"], kind="stable"
        ).reset_index(drop=True)
    return combined, records


def _run_synthetic_validation(
    root: Path,
    search: dict[str, Any],
    primary_rule: split_audit.SimilarityRule,
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20260806)
    base = "".join(rng.choice(np.array(list("ACGT")), size=4_000).tolist())
    rotation = base[1_371:] + base[:1_371]
    reverse_complement = base.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    mutated = list(base)
    mutation_positions = np.linspace(17, len(base) - 19, num=39, dtype=int)
    alternatives = {"A": "C", "C": "G", "G": "T", "T": "A"}
    for position in mutation_positions:
        mutated[position] = alternatives[mutated[position]]
    shared_cassette_only = base[:1_000] + "".join(
        rng.choice(np.array(list("ACGT")), size=3_000).tolist()
    )

    query_tokens = _synthetic_tokens(
        {
            "synthetic_exact": base,
            "synthetic_rotation": rotation,
            "synthetic_reverse_complement": reverse_complement,
            "synthetic_one_percent_substitutions": "".join(mutated),
            "synthetic_shared_cassette_only": shared_cassette_only,
        },
        split="val",
        token_prefix="query",
    )
    subject_tokens = _synthetic_tokens(
        {"synthetic_reference": base},
        split="train",
        token_prefix="subject",
    )
    query_path = root / "query_repeated.fasta"
    subject_path = root / "subject.fasta"
    index_path = root / "subject.mmi"
    output_path = root / "synthetic.paf"
    _write_fasta(query_tokens, query_path, repeat=search["query_repeat"])
    _write_fasta(subject_tokens, subject_path, repeat=1)
    index_duration, index_log = _build_index(subject_path, index_path, search)
    alignments, search_duration, search_log = _map(
        query_path,
        index_path,
        output_path,
        search,
    )
    edges = split_audit.classify_minimap_alignments(
        alignments,
        query_tokens=query_tokens,
        subject_tokens=subject_tokens,
        search_pair="synthetic_val_vs_train",
        query_repeat=search["query_repeat"],
        primary_rule=primary_rule,
        sensitivity_rule=primary_rule,
    )
    expected_primary = {
        "synthetic_exact",
        "synthetic_rotation",
        "synthetic_reverse_complement",
        "synthetic_one_percent_substitutions",
    }
    observed_primary = set(
        edges.loc[edges["primary_near_duplicate"], "query_sequence_id"].astype(str)
    )
    if observed_primary != expected_primary:
        raise RuntimeError(
            "synthetic primary classifications changed: "
            f"expected={sorted(expected_primary)}, observed={sorted(observed_primary)}"
        )
    if "synthetic_shared_cassette_only" in observed_primary:
        raise RuntimeError(
            "the shared-cassette negative control was classified as a near duplicate"
        )
    return {
        "passed": True,
        "fixture_seed": 20260806,
        "sequence_length_bp": len(base),
        "substitution_count": int(len(mutation_positions)),
        "expected_primary_queries": sorted(expected_primary),
        "observed_primary_queries": sorted(observed_primary),
        "shared_cassette_query_detected_as_candidate": bool(
            edges["query_sequence_id"].eq("synthetic_shared_cassette_only").any()
        ),
        "shared_cassette_classified_as_primary": False,
        "index_duration_seconds": index_duration,
        "search_duration_seconds": search_duration,
        "tool_log": index_log + search_log,
    }


def _synthetic_tokens(
    sequences: dict[str, str],
    *,
    split: str,
    token_prefix: str,
) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "token": f"{token_prefix}_{index:03d}",
                "sequence_id": sequence_id,
                "split_grouped": split,
                "length_bp": len(sequence),
                "sequence": sequence,
                "leakage_component": f"{token_prefix}_{index:03d}",
            }
            for index, (sequence_id, sequence) in enumerate(sorted(sequences.items()))
        ]
    )


def _write_fasta(tokens: pd.DataFrame, path: Path, *, repeat: int) -> None:
    if repeat < 1:
        raise ValueError("FASTA repeat must be positive")
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in tokens.itertuples(index=False):
            sequence = str(row.sequence) * repeat
            handle.write(f">{row.token}\n")
            for start in range(0, len(sequence), 80):
                handle.write(f"{sequence[start : start + 80]}\n")


def _build_index(
    fasta: Path,
    index: Path,
    search: dict[str, Any],
) -> tuple[float, list[str]]:
    command = [
        search["executable"],
        "-x",
        search["preset"],
        "-t",
        str(search["threads"]),
        "-d",
        str(index),
        str(fasta),
    ]
    _, stderr, duration = _run_command(command)
    return duration, _tool_log(stderr)


def _map(
    query: Path,
    index: Path,
    output: Path,
    search: dict[str, Any],
) -> tuple[pd.DataFrame, float, list[str]]:
    command = [
        search["executable"],
        "-x",
        search["preset"],
        "-t",
        str(search["threads"]),
        "-N",
        str(search["secondary_alignment_limit"]),
        "-p",
        str(search["minimum_secondary_score_ratio"]),
        "-c",
        "-o",
        str(output),
        str(index),
        str(query),
    ]
    _, stderr, duration = _run_command(command)
    return _read_paf(output), duration, _tool_log(stderr)


def _read_paf(path: Path) -> pd.DataFrame:
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=split_audit.PAF_COLUMNS)
    frame = pd.read_csv(
        path,
        sep="\t",
        names=list(split_audit.PAF_COLUMNS),
        header=None,
        usecols=range(len(split_audit.PAF_COLUMNS)),
        dtype={"qname": "string", "tname": "string", "strand": "string"},
    )
    if frame[list(split_audit.PAF_COLUMNS)].isna().any().any():
        raise ValueError("minimap2 PAF output contains missing values")
    return frame


def _run_command(command: list[str]) -> tuple[str, str, float]:
    start = time.monotonic()
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"required executable was not found: {command[0]}") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"external command failed with exit code {error.returncode}: {command[0]}\n"
            f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        ) from error
    return result.stdout, result.stderr, time.monotonic() - start


def _tool_version(executable: str) -> str:
    stdout, stderr, _ = _run_command([executable, "--version"])
    lines = [line.strip() for line in (stdout + "\n" + stderr).splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{executable} did not report a version")
    return lines[0]


def _tool_log(stderr: str) -> list[str]:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-12:]


def _resolved_search_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "executable": str(config["executable"]),
        "preset": str(config["preset"]),
        "threads": int(config["threads"]),
        "query_repeat": int(config["query_repeat"]),
        "secondary_alignment_limit": int(config["secondary_alignment_limit"]),
        "minimum_secondary_score_ratio": float(config["minimum_secondary_score_ratio"]),
    }
    if not 1 <= resolved["threads"] <= 10:
        raise ValueError("split audit threads must be between 1 and 10")
    if resolved["query_repeat"] != 2:
        raise ValueError("split audit query_repeat must remain 2 for circular-origin handling")
    if resolved["preset"] != "asm20":
        raise ValueError("split audit preset must remain asm20 for the 95% sensitivity rule")
    if resolved["secondary_alignment_limit"] < 1:
        raise ValueError("secondary_alignment_limit must be positive")
    if not 0.0 < resolved["minimum_secondary_score_ratio"] <= 1.0:
        raise ValueError("minimum_secondary_score_ratio must be in (0, 1]")
    return resolved


def _validate_rule_nesting(
    primary: split_audit.SimilarityRule,
    sensitivity: split_audit.SimilarityRule,
) -> None:
    fields = (
        "minimum_identity",
        "minimum_query_coverage",
        "minimum_subject_coverage",
        "minimum_length_ratio",
    )
    invalid = [field for field in fields if getattr(primary, field) < getattr(sensitivity, field)]
    if invalid:
        raise ValueError(f"primary similarity rule is weaker than sensitivity rule: {invalid}")


def _git_provenance() -> dict[str, Any]:
    commit, _, _ = _run_command(["git", "rev-parse", "HEAD"])
    status, _, _ = _run_command(["git", "status", "--porcelain=v1"])
    status_lines = [line for line in status.splitlines() if line]
    return {
        "commit": commit.strip(),
        "worktree_dirty": bool(status_lines),
        "worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "changed_paths": [line[3:] for line in status_lines],
        "python_executable": sys.executable,
    }
