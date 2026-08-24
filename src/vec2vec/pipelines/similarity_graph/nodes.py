"""Adaptive Ray execution and validation for the global similarity graph."""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from vec2vec.lib import similarity_graph, split_audit
from vec2vec.pipelines.similarity_graph_calibration import nodes as calibration_nodes

logger = logging.getLogger(__name__)


def run_similarity_graph(
    retrieval: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    """Build a non-truncated adaptive graph under the fixed minimap2 protocol."""
    started = time.monotonic()
    validation = split_audit.validate_retrieval(
        retrieval,
        expected_population_sha256=str(params["expected_input_population_sha256"]),
    )
    search = _validate_search(params["candidate_search"], params["exact_search"])
    execution = _validate_execution(params["execution"])
    primary_rule = split_audit.similarity_rule(params["exact_search"]["primary_rule"])
    sensitivity_rule = split_audit.similarity_rule(params["exact_search"]["sensitivity_rule"])
    _validate_nested_rules(primary_rule, sensitivity_rule)
    _check_limits(started, execution, [])

    tokens = split_audit.build_sequence_tokens(retrieval)
    cache_root = (
        Path(execution["scratch_root"]) / str(params["expected_input_population_sha256"])
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    _, target_index, cache_manifest = calibration_nodes._prepare_target_cache(
        cache_root,
        tokens,
        search,
        expected_population_sha256=str(params["expected_input_population_sha256"]),
    )
    query_root = cache_root / "queries" / str(params["graph_version"])
    query_root.mkdir(parents=True, exist_ok=True)
    paf_root = query_root / "paf"
    if paf_root.exists():
        shutil.rmtree(paf_root)

    token_records = {
        str(row.token): {
            "sequence_id": str(row.sequence_id),
            "length_bp": int(row.length_bp),
        }
        for row in tokens.itertuples(index=False)
    }
    token_lengths = {token: int(record["length_bp"]) for token, record in token_records.items()}
    synthetic = calibration_nodes._run_synthetic_validation(
        cache_root / "synthetic" / str(params["graph_version"]),
        search=search,
        execution=execution,
        primary_rule=primary_rule,
        sensitivity_rule=sensitivity_rule,
    )

    try:
        import ray
    except ImportError as error:  # pragma: no cover - explicit environment boundary
        raise RuntimeError("Ray is required for the named similarity_graph pipeline") from error
    if ray.is_initialized():
        raise RuntimeError("an existing Ray runtime would make graph resources ambiguous")
    ray.init(
        num_cpus=execution["ray_workers"] * execution["threads_per_worker"],
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=1024 * 1024 * 1024,
    )
    run_records: list[dict[str, Any]] = []
    profile_tables: list[pd.DataFrame] = []
    directional_tables: list[pd.DataFrame] = []
    try:
        all_shards = calibration_nodes._write_query_shards(
            tokens,
            query_root / "candidate_normal",
            shard_queries=execution["shard_queries"],
            repeat=search["query_repeat"],
        )
        candidate_normal = calibration_nodes._run_ray_stage(
            ray,
            all_shards,
            target_index=target_index,
            mode="candidate",
            cap=search["normal_cap"],
            search=search,
            execution=execution,
            token_lengths=token_lengths,
            token_records=token_records,
            primary_rule=primary_rule,
            sensitivity_rule=sensitivity_rule,
            output_root=query_root / "paf",
        )
        _collect(candidate_normal, "candidate_normal", run_records, profile_tables, [])
        _check_limits(started, execution, run_records)
        normal_candidate_profile = pd.concat(profile_tables, ignore_index=True)
        adaptive_tokens = set(
            normal_candidate_profile.loc[
                normal_candidate_profile["stage"].eq("candidate_normal")
                & normal_candidate_profile["potentially_saturated"],
                "token",
            ].astype(str)
        )

        if adaptive_tokens:
            adaptive_frame = tokens.loc[tokens["token"].isin(adaptive_tokens)]
            adaptive_shards = calibration_nodes._write_query_shards(
                adaptive_frame,
                query_root / "candidate_adaptive",
                shard_queries=execution["adaptive_shard_queries"],
                repeat=search["query_repeat"],
            )
            candidate_adaptive = calibration_nodes._run_ray_stage(
                ray,
                adaptive_shards,
                target_index=target_index,
                mode="candidate",
                cap=search["adaptive_cap"],
                search=search,
                execution=execution,
                token_lengths=token_lengths,
                token_records=token_records,
                primary_rule=primary_rule,
                sensitivity_rule=sensitivity_rule,
                output_root=query_root / "paf",
            )
            _collect(
                candidate_adaptive,
                "candidate_adaptive",
                run_records,
                profile_tables,
                [],
            )
            adaptive_profile = pd.concat(
                [pd.DataFrame.from_records(result["profile"]) for result in candidate_adaptive],
                ignore_index=True,
            )
            if adaptive_profile["potentially_saturated"].any():
                count = int(adaptive_profile["potentially_saturated"].sum())
                raise RuntimeError(f"{count} candidate queries saturated at adaptive cap")
            _check_limits(started, execution, run_records)

        exact_normal_frame = tokens.loc[~tokens["token"].isin(adaptive_tokens)]
        exact_normal_shards = calibration_nodes._write_query_shards(
            exact_normal_frame,
            query_root / "exact_normal",
            shard_queries=execution["shard_queries"],
            repeat=search["query_repeat"],
        )
        exact_normal = calibration_nodes._run_ray_stage(
            ray,
            exact_normal_shards,
            target_index=target_index,
            mode="exact",
            cap=search["normal_cap"],
            search=search,
            execution=execution,
            token_lengths=token_lengths,
            token_records=token_records,
            primary_rule=primary_rule,
            sensitivity_rule=sensitivity_rule,
            output_root=query_root / "paf",
        )
        _collect(exact_normal, "exact_normal", run_records, profile_tables, directional_tables)
        _check_limits(started, execution, run_records)

        unexpected_tokens = {
            str(row["token"])
            for result in exact_normal
            for row in result["profile"]
            if bool(row["potentially_saturated"])
        }
        high_exact_tokens = adaptive_tokens.union(unexpected_tokens)
        if high_exact_tokens:
            high_exact_frame = tokens.loc[tokens["token"].isin(high_exact_tokens)]
            high_exact_shards = calibration_nodes._write_query_shards(
                high_exact_frame,
                query_root / "exact_adaptive",
                shard_queries=execution["adaptive_shard_queries"],
                repeat=search["query_repeat"],
            )
            exact_adaptive = calibration_nodes._run_ray_stage(
                ray,
                high_exact_shards,
                target_index=target_index,
                mode="exact",
                cap=search["adaptive_cap"],
                search=search,
                execution=execution,
                token_lengths=token_lengths,
                token_records=token_records,
                primary_rule=primary_rule,
                sensitivity_rule=sensitivity_rule,
                output_root=query_root / "paf",
            )
            adaptive_exact_profiles = pd.concat(
                [pd.DataFrame.from_records(result["profile"]) for result in exact_adaptive],
                ignore_index=True,
            )
            if adaptive_exact_profiles["potentially_saturated"].any():
                count = int(adaptive_exact_profiles["potentially_saturated"].sum())
                raise RuntimeError(f"{count} exact queries saturated at adaptive cap")
            _collect(
                exact_adaptive,
                "exact_adaptive",
                run_records,
                profile_tables,
                directional_tables,
            )
            _check_limits(started, execution, run_records)
    finally:
        ray.shutdown()

    profiles = pd.concat(profile_tables, ignore_index=True)
    if unexpected_tokens:
        profiles.loc[
            profiles["stage"].eq("exact_normal") & profiles["token"].isin(unexpected_tokens),
            "final_for_query",
        ] = False
    final_exact = profiles.loc[
        profiles["stage"].isin(["exact_normal", "exact_adaptive"]) & profiles["final_for_query"]
    ]
    final_counts = final_exact.groupby("token", sort=False).size()
    if len(final_counts) != len(tokens) or not final_counts.eq(1).all():
        raise RuntimeError("not every sequence has exactly one final exact-search record")
    if final_exact["potentially_saturated"].any():
        raise RuntimeError("a final exact query remains saturated")

    if directional_tables:
        directional = pd.concat(directional_tables, ignore_index=True)
        if unexpected_tokens:
            directional = directional.loc[
                ~(
                    directional["stage"].eq("exact_normal")
                    & directional["query_token"].isin(unexpected_tokens)
                )
            ]
    else:
        raise RuntimeError("full exact search returned no sensitivity edges")
    edges = similarity_graph.canonicalize_similarity_edges(
        directional,
        primary_rule=primary_rule,
        sensitivity_rule=sensitivity_rule,
    )
    nodes, components, graph_summary = similarity_graph.build_similarity_components(
        retrieval,
        edges,
    )
    token_metadata = tokens.loc[:, ["token", "sequence_id"]]
    profiles = (
        profiles.merge(
            token_metadata,
            on="token",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["stage", "cap", "token"], kind="stable")
        .reset_index(drop=True)
    )
    runs = pd.DataFrame.from_records(run_records)
    output_hashes = {
        "edges_sha256": similarity_graph.dataframe_content_sha256(
            edges, sort_columns=["sequence_a", "sequence_b"]
        ),
        "nodes_sha256": similarity_graph.dataframe_content_sha256(
            nodes, sort_columns=["sequence_id"]
        ),
        "components_sha256": similarity_graph.dataframe_content_sha256(
            components,
            sort_columns=["threshold", "similarity_component"],
        ),
        "query_profile_sha256": similarity_graph.dataframe_content_sha256(
            profiles,
            sort_columns=["stage", "cap", "token"],
        ),
        "runs_sha256": similarity_graph.dataframe_content_sha256(
            runs,
            sort_columns=["stage", "cap", "shard_id"],
        ),
    }
    observed_cpu_hours = float(runs["cpu_seconds"].sum() / 3600.0)
    observed_paf_bytes = int(runs["paf_bytes"].sum())
    manifest = {
        "graph_version": str(params["graph_version"]),
        "protocol": (
            "studies/set_valued_compositional_embeddings/experiments/E00_global_similarity_graph.md"
        ),
        "input_retrieval_version": str(params["input_retrieval_version"]),
        "calibration_artifact_version": str(params["calibration_artifact_version"]),
        "input_validation": validation,
        "resolved_configuration": params,
        "tools": {
            "minimap2": calibration_nodes._tool_version(search["executable"]),
            "ray": ray.__version__,
        },
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "observed_child_cpu_hours": observed_cpu_hours,
            "observed_raw_paf_bytes": observed_paf_bytes,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "git": _git_provenance(),
        "synthetic_validation": synthetic,
        "search_summary": {
            "queries": int(len(tokens)),
            "candidate_adaptive_queries": int(len(adaptive_tokens)),
            "unexpected_exact_adaptive_queries": int(len(unexpected_tokens)),
            "final_exact_queries": int(len(final_counts)),
            "final_saturated_queries": 0,
            "shards": int(len(runs)),
            "by_stage": {
                str(stage): {
                    "shards": int(len(group)),
                    "queries": int(group["query_count"].sum()),
                    "cpu_seconds": float(group["cpu_seconds"].sum()),
                    "paf_bytes": int(group["paf_bytes"].sum()),
                    "potentially_saturated_queries": int(
                        group["potentially_saturated_queries"].sum()
                    ),
                }
                for stage, group in runs.groupby("stage", sort=True)
            },
        },
        "graph_summary": graph_summary,
        "output_content_hashes": output_hashes,
        "decision": {
            "status": "complete_under_fixed_minimap_protocol",
            "all_queries_have_final_exact_search": True,
            "no_final_query_saturated": True,
            "edge_enumeration_complete_under_configured_caps": True,
            "mathematical_all_pairs_proof": False,
            "split_grouped_v2_assigned": False,
            "model_outcomes_inspected": False,
        },
        "known_limitations": [
            "Minimap2 candidate discovery remains heuristic.",
            "A single best local alignment can miss complex rearrangements.",
            "Sequence similarity does not establish functional independence.",
            "The graph needs an independent audit before use for split assignment.",
        ],
    }
    logger.info(
        "Global similarity graph completed with %s sensitivity and %s primary edges",
        graph_summary["sensitivity_edges"],
        graph_summary["primary_edges"],
    )
    return edges, nodes, components, profiles, runs, manifest


def _collect(
    results: list[dict[str, Any]],
    stage: str,
    run_records: list[dict[str, Any]],
    profile_tables: list[pd.DataFrame],
    edge_tables: list[pd.DataFrame],
) -> None:
    for result in results:
        run = dict(result["run"])
        run["stage"] = stage
        run_records.append(run)
        profile = pd.DataFrame.from_records(result["profile"])
        profile["stage"] = stage
        profile["final_for_query"] = stage not in {"candidate_normal", "candidate_adaptive"}
        profile_tables.append(profile)
        if result["edges"]:
            edges = pd.DataFrame.from_records(result["edges"])
            edges["stage"] = stage
            edge_tables.append(edges)


def _validate_search(candidate: dict[str, Any], exact: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "executable": str(candidate["executable"]),
        "preset": str(candidate["preset"]),
        "query_repeat": int(candidate["query_repeat"]),
        "minimum_secondary_score_ratio": float(candidate["minimum_secondary_score_ratio"]),
        "normal_cap": int(candidate["normal_cap"]),
        "adaptive_cap": int(candidate["adaptive_cap"]),
        "minimum_approximate_query_coverage": float(
            candidate["minimum_approximate_query_coverage"]
        ),
        "minimum_approximate_subject_coverage": float(
            candidate["minimum_approximate_subject_coverage"]
        ),
        "minimum_length_ratio": float(candidate["minimum_length_ratio"]),
        "maximum_approximate_divergence": float(candidate["maximum_approximate_divergence"]),
    }
    if resolved["preset"] != "asm20" or resolved["query_repeat"] != 2:
        raise ValueError("global graph must retain asm20 and doubled circular queries")
    if resolved["normal_cap"] != int(exact["normal_cap"]):
        raise ValueError("candidate and exact normal caps differ")
    if resolved["adaptive_cap"] != int(exact["adaptive_cap"]):
        raise ValueError("candidate and exact adaptive caps differ")
    if resolved["adaptive_cap"] <= resolved["normal_cap"]:
        raise ValueError("adaptive cap must exceed normal cap")
    return resolved


def _validate_execution(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "ray_workers": int(config["ray_workers"]),
        "threads_per_worker": int(config["threads_per_worker"]),
        "shard_queries": int(config["shard_queries"]),
        "adaptive_shard_queries": int(config["adaptive_shard_queries"]),
        "retry_uncheckpointed_exact_shard_queries": int(
            config["retry_uncheckpointed_exact_shard_queries"]
        ),
        "retry_uncheckpointed_adaptive_exact_shard_queries": int(
            config["retry_uncheckpointed_adaptive_exact_shard_queries"]
        ),
        "task_timeout_seconds": int(config["task_timeout_seconds"]),
        "full_run_wall_limit_seconds": int(config["full_run_wall_limit_seconds"]),
        "maximum_task_output_bytes": int(config["maximum_task_output_bytes"]),
        "minimum_free_disk_bytes": int(config["minimum_free_disk_bytes"]),
        "scratch_root": str(config["scratch_root"]),
        "maximum_cpu_hours": float(config["maximum_cpu_hours"]),
        "maximum_persisted_bytes": int(config["maximum_persisted_bytes"]),
    }
    numeric = [value for key, value in resolved.items() if key != "scratch_root"]
    if min(numeric) <= 0:
        raise ValueError("full graph execution limits must be positive")
    available = os.cpu_count() or 1
    if resolved["ray_workers"] * resolved["threads_per_worker"] > available:
        raise ValueError("configured Ray CPU demand exceeds this host; use an environment override")
    return resolved


def _validate_nested_rules(
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
        raise ValueError(f"primary rule is weaker than sensitivity rule: {invalid}")


def _check_limits(
    started: float,
    execution: dict[str, Any],
    run_records: list[dict[str, Any]],
) -> None:
    if time.monotonic() - started > execution["full_run_wall_limit_seconds"]:
        raise RuntimeError("global graph reached its fixed wall-time limit")
    scratch_root = calibration_nodes._existing_directory(Path(execution["scratch_root"]))
    if shutil.disk_usage(scratch_root).free < execution["minimum_free_disk_bytes"]:
        raise RuntimeError("global graph reached its fixed free-disk floor")
    cpu_hours = sum(float(record["cpu_seconds"]) for record in run_records) / 3600.0
    if cpu_hours > execution["maximum_cpu_hours"]:
        raise RuntimeError("global graph reached its fixed CPU-hour limit")
    raw_bytes = sum(int(record["paf_bytes"]) for record in run_records)
    if raw_bytes > execution["maximum_persisted_bytes"]:
        raise RuntimeError("global graph crossed its conservative raw-output byte limit")


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"], check=True, capture_output=True, text=True
    ).stdout
    status_lines = [line for line in status.splitlines() if line]
    return {
        "commit": commit,
        "worktree_dirty": bool(status_lines),
        "worktree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "changed_paths": [line[3:] for line in status_lines],
        "python_executable": sys.executable,
    }
