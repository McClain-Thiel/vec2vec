"""Ray and external-tool boundary for similarity-graph calibration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vec2vec.lib import similarity_graph_calibration as calibration
from vec2vec.lib import split_audit

logger = logging.getLogger(__name__)


def run_similarity_graph_calibration(
    retrieval: pd.DataFrame,
    audit_edges: pd.DataFrame,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run the fixed local calibration and return versioned evidence tables."""
    started = time.monotonic()
    validation = split_audit.validate_retrieval(
        retrieval,
        expected_population_sha256=str(params["expected_input_population_sha256"]),
    )
    _validate_audit_edges(audit_edges, params)
    execution = _validate_execution(params["execution"])
    _check_free_disk(execution)

    tokens = split_audit.build_sequence_tokens(retrieval)
    sample = calibration.select_calibration_queries(retrieval, audit_edges, params["sample"])
    sample_tokens = tokens.merge(
        sample.drop(columns=["length_bp", "leakage_component"]),
        on="sequence_id",
        how="inner",
        validate="one_to_one",
    )
    if len(sample_tokens) != int(params["sample"]["total_queries"]):
        raise RuntimeError("calibration sample did not join one-to-one with sequence tokens")

    cache_root = (
        Path(execution["scratch_root"]) / str(params["expected_input_population_sha256"])
    ).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    _, target_index, cache_manifest = _prepare_target_cache(
        cache_root,
        tokens,
        params["candidate_search"],
        expected_population_sha256=str(params["expected_input_population_sha256"]),
    )
    query_root = cache_root / "calibration_queries" / str(params["calibration_version"])
    if query_root.exists():
        shutil.rmtree(query_root)
    query_root.mkdir(parents=True)

    token_records = {
        str(row.token): {
            "sequence_id": str(row.sequence_id),
            "length_bp": int(row.length_bp),
        }
        for row in tokens.itertuples(index=False)
    }
    token_lengths = {token: int(record["length_bp"]) for token, record in token_records.items()}
    search = _validate_search(params["candidate_search"])
    primary_rule = split_audit.similarity_rule(params["exact_benchmark"]["primary_rule"])
    sensitivity_rule = split_audit.similarity_rule(params["exact_benchmark"]["sensitivity_rule"])
    _validate_nested_rules(primary_rule, sensitivity_rule)
    synthetic_validation = _run_synthetic_validation(
        cache_root / "synthetic" / str(params["calibration_version"]),
        search=search,
        execution=execution,
        primary_rule=primary_rule,
        sensitivity_rule=sensitivity_rule,
    )

    try:
        import ray
    except ImportError as error:  # pragma: no cover - explicit environment boundary
        raise RuntimeError(
            "Ray is required for this named pipeline; install the similarity-graph extra"
        ) from error

    if ray.is_initialized():
        raise RuntimeError("an existing Ray runtime would make calibration resources ambiguous")
    ray.init(
        num_cpus=execution["ray_workers"] * execution["threads_per_worker"],
        include_dashboard=False,
        log_to_driver=False,
        object_store_memory=512 * 1024 * 1024,
    )
    run_records: list[dict[str, Any]] = []
    profile_tables: list[pd.DataFrame] = []
    exact_edge_tables: list[pd.DataFrame] = []
    try:
        normal_shards = _write_query_shards(
            sample_tokens,
            query_root / "normal",
            shard_queries=execution["shard_queries"],
            repeat=search["query_repeat"],
        )
        for cap in search["caps"]:
            _check_wall_and_disk(started, execution)
            results = _run_ray_stage(
                ray,
                normal_shards,
                target_index=target_index,
                mode="candidate",
                cap=cap,
                search=search,
                execution=execution,
                token_lengths=token_lengths,
                token_records=token_records,
                primary_rule=primary_rule,
                sensitivity_rule=sensitivity_rule,
                output_root=query_root / "paf",
            )
            _collect_stage(results, run_records, profile_tables, exact_edge_tables)

        profiles_so_far = pd.concat(profile_tables, ignore_index=True)
        maximum_normal_cap = max(search["caps"])
        saturated = profiles_so_far.loc[
            profiles_so_far["mode"].eq("candidate")
            & profiles_so_far["cap"].eq(maximum_normal_cap)
            & profiles_so_far["potentially_saturated"]
        ].copy()
        tail_count = min(int(params["sample"]["adaptive_tail_queries"]), len(saturated))
        tail_tokens = saturated.sort_values(["token"], kind="stable")["token"].head(tail_count)
        if tail_count:
            tail_frame = sample_tokens.loc[sample_tokens["token"].isin(tail_tokens)]
            tail_shards = _write_query_shards(
                tail_frame,
                query_root / "adaptive_tail",
                shard_queries=execution["shard_queries"],
                repeat=search["query_repeat"],
            )
            _check_wall_and_disk(started, execution)
            tail_results = _run_ray_stage(
                ray,
                tail_shards,
                target_index=target_index,
                mode="candidate_tail",
                cap=search["adaptive_tail_cap"],
                search=search,
                execution=execution,
                token_lengths=token_lengths,
                token_records=token_records,
                primary_rule=primary_rule,
                sensitivity_rule=sensitivity_rule,
                output_root=query_root / "paf",
            )
            _collect_stage(tail_results, run_records, profile_tables, exact_edge_tables)

        exact_frame = sample_tokens.loc[sample_tokens["exact_benchmark"]].copy()
        exact_shards = _write_query_shards(
            exact_frame,
            query_root / "exact",
            shard_queries=execution["shard_queries"],
            repeat=search["query_repeat"],
        )
        for cap in (int(value) for value in params["exact_benchmark"]["caps"]):
            _check_wall_and_disk(started, execution)
            exact_results = _run_ray_stage(
                ray,
                exact_shards,
                target_index=target_index,
                mode="exact",
                cap=cap,
                search=search,
                execution=execution,
                token_lengths=token_lengths,
                token_records=token_records,
                primary_rule=primary_rule,
                sensitivity_rule=sensitivity_rule,
                output_root=query_root / "paf",
            )
            _collect_stage(exact_results, run_records, profile_tables, exact_edge_tables)
    finally:
        ray.shutdown()

    runs = (
        pd.DataFrame.from_records(run_records)
        .sort_values(["mode", "cap", "shard_id"], kind="stable")
        .reset_index(drop=True)
    )
    profiles = pd.concat(profile_tables, ignore_index=True)
    sample_metadata = sample_tokens.drop(columns=["sequence"])
    profiles = (
        profiles.merge(
            sample_metadata,
            on="token",
            how="left",
            validate="many_to_one",
        )
        .sort_values(["mode", "cap", "token"], kind="stable")
        .reset_index(drop=True)
    )
    if exact_edge_tables:
        exact_edges = pd.concat(exact_edge_tables, ignore_index=True)
        exact_edges = exact_edges.sort_values(
            ["cap", "query_sequence_id", "subject_sequence_id"], kind="stable"
        ).reset_index(drop=True)
    else:
        exact_edges = _empty_exact_edges()

    full_limits = params["proposed_full_run_limits"]
    projections = calibration.project_full_run(
        runs,
        profiles,
        population_rows=len(retrieval),
        maximum_cpu_hours=float(full_limits["maximum_cpu_hours"]),
        maximum_persisted_bytes=int(full_limits["maximum_persisted_bytes"]),
    )
    tail_profile = profiles.loc[profiles["mode"].eq("candidate_tail")]
    tail_saturated = int(tail_profile["potentially_saturated"].sum())
    within_limits = all(
        record["within_cpu_limit"] and record["within_persisted_byte_limit"]
        for record in projections.values()
    )
    decision = (
        "proceed_to_bounded_full_design"
        if within_limits and tail_saturated == 0
        else "redesign_before_full_run"
    )
    manifest = {
        "calibration_version": str(params["calibration_version"]),
        "protocol": (
            "studies/set_valued_compositional_embeddings/experiments/"
            "E00_similarity_graph_calibration.md"
        ),
        "input_retrieval_version": str(params["input_retrieval_version"]),
        "input_split_audit_edges_version": str(params["input_split_audit_edges_version"]),
        "input_validation": validation,
        "resolved_configuration": params,
        "sample": {
            "rows": int(len(sample)),
            "representative_rows": int(sample["representative"].sum()),
            "component_stress_rows": int(sample["component_stress"].sum()),
            "edge_stress_rows": int(sample["edge_stress"].sum()),
            "exact_benchmark_rows": int(sample["exact_benchmark"].sum()),
            "adaptive_tail_rows": int(len(tail_profile)),
        },
        "cache": cache_manifest,
        "synthetic_validation": synthetic_validation,
        "tools": {
            "minimap2": _tool_version(search["executable"]),
            "ray": ray.__version__,
        },
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
        },
        "git": _git_provenance(),
        "projections": projections,
        "decision": {
            "status": decision,
            "within_fixed_compute_and_storage_limits": within_limits,
            "adaptive_tail_saturated_queries": tail_saturated,
            "candidate_identity_is_approximate": True,
            "exact_identity_uses_base_level_alignment": True,
            "full_graph_started": False,
            "model_outcomes_inspected": False,
        },
        "known_limitations": [
            "Minimap2 candidate generation is heuristic.",
            "The calibration sample can miss an unusually dense sequence family.",
            "Approximate PAF output is not measured identity evidence.",
            "A full run needs adaptive handling for any saturated query.",
        ],
    }
    logger.info("Similarity calibration completed with decision %s", decision)
    return runs, profiles, exact_edges, manifest


def _run_ray_stage(
    ray: Any,
    shards: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    remote = ray.remote(_run_shard_to_checkpoint)
    threads = int(kwargs["execution"]["threads_per_worker"])
    shared_keys = {
        "search",
        "execution",
        "token_lengths",
        "token_records",
        "primary_rule",
        "sensitivity_rule",
    }
    shared = {key: ray.put(value) if key in shared_keys else value for key, value in kwargs.items()}
    checkpoint_root = Path(kwargs["output_root"]).parent / "checkpoints"
    stage_identity = _stage_checkpoint_identity(**kwargs)
    retry_maximum_queries = int(
        kwargs["execution"].get(
            "retry_uncheckpointed_exact_shard_queries",
            kwargs["execution"]["shard_queries"],
        )
    )
    if str(kwargs["mode"]) == "exact" and int(kwargs["cap"]) == int(
        kwargs["search"]["adaptive_cap"]
    ):
        retry_maximum_queries = int(
            kwargs["execution"].get(
                "retry_uncheckpointed_adaptive_exact_shard_queries",
                kwargs["execution"]["adaptive_shard_queries"],
            )
        )
    shards = _split_uncheckpointed_exact_shards(
        shards,
        checkpoint_root=checkpoint_root,
        stage_identity=stage_identity,
        mode=str(kwargs["mode"]),
        cap=int(kwargs["cap"]),
        maximum_queries=retry_maximum_queries,
    )
    descriptors: list[dict[str, Any] | None] = [None] * len(shards)
    pending_indices: list[int] = []
    references = []
    for index, shard in enumerate(shards):
        identity = _shard_checkpoint_identity(shard=shard, stage_identity=stage_identity)
        checkpoint_dir = _shard_checkpoint_directory(
            checkpoint_root,
            mode=str(kwargs["mode"]),
            cap=int(kwargs["cap"]),
            shard_id=int(shard["shard_id"]),
            identity=identity,
        )
        descriptor = _validated_checkpoint_descriptor(checkpoint_dir, identity)
        if descriptor is not None:
            descriptors[index] = descriptor
            continue
        pending_indices.append(index)
        references.append(
            remote.options(num_cpus=threads).remote(
                shard=shard,
                checkpoint_dir=checkpoint_dir,
                checkpoint_identity=identity,
                **shared,
            )
        )
    if references:
        completed = _get_ray_descriptors_with_disk_guard(
            ray,
            references,
            execution=kwargs["execution"],
        )
        for index, descriptor in zip(pending_indices, completed, strict=True):
            descriptors[index] = descriptor
    if any(descriptor is None for descriptor in descriptors):
        raise RuntimeError("Ray stage did not produce one checkpoint descriptor per shard")
    return [
        _load_shard_checkpoint(descriptor, reused=index not in pending_indices)
        for index, descriptor in enumerate(descriptors)
        if descriptor is not None
    ]


def _get_ray_descriptors_with_disk_guard(
    ray: Any,
    references: list[Any],
    *,
    execution: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect Ray results while enforcing the fixed free-disk floor."""
    pending = set(references)
    completed: dict[Any, dict[str, Any]] = {}
    try:
        while pending:
            _check_free_disk(execution)
            ready, _ = ray.wait(list(pending), num_returns=1, timeout=5.0)
            for reference in ready:
                completed[reference] = ray.get(reference)
                pending.remove(reference)
    except BaseException:
        for reference in pending:
            ray.cancel(reference, force=True)
        raise
    return [completed[reference] for reference in references]


def _split_uncheckpointed_exact_shards(
    shards: list[dict[str, Any]],
    *,
    checkpoint_root: Path,
    stage_identity: dict[str, Any],
    mode: str,
    cap: int,
    maximum_queries: int,
) -> list[dict[str, Any]]:
    """Keep completed exact shards and deterministically divide only unfinished shards."""
    if mode != "exact":
        return shards
    if maximum_queries < 1:
        raise ValueError("retry exact shard query limit must be positive")

    expanded: list[dict[str, Any]] = []
    reused_parent_shards = 0
    split_parent_shards = 0
    for shard in shards:
        identity = _shard_checkpoint_identity(shard=shard, stage_identity=stage_identity)
        checkpoint_dir = _shard_checkpoint_directory(
            checkpoint_root,
            mode=mode,
            cap=cap,
            shard_id=int(shard["shard_id"]),
            identity=identity,
        )
        if _validated_checkpoint_descriptor(checkpoint_dir, identity) is not None:
            expanded.append(shard)
            reused_parent_shards += 1
            continue
        if len(shard["tokens"]) <= maximum_queries:
            expanded.append(shard)
            continue
        expanded.extend(_split_query_shard(shard, maximum_queries=maximum_queries))
        split_parent_shards += 1

    if split_parent_shards:
        logger.info(
            "Kept %d completed exact parent shards and split %d unfinished parent shards "
            "into deterministic shards of at most %d queries",
            reused_parent_shards,
            split_parent_shards,
            maximum_queries,
        )
    return expanded


def _split_query_shard(
    shard: dict[str, Any],
    *,
    maximum_queries: int,
) -> list[dict[str, Any]]:
    """Split one existing FASTA shard without changing any query sequence bytes."""
    parent_id = int(shard["shard_id"])
    tokens = [str(value) for value in shard["tokens"]]
    records = _read_fasta_records(Path(shard["path"]))
    record_tokens = [record.splitlines()[0][1:] for record in records]
    if record_tokens != tokens:
        raise RuntimeError(f"FASTA tokens changed before splitting shard {parent_id}")

    part_count = (len(tokens) + maximum_queries - 1) // maximum_queries
    if part_count >= 1_000:
        raise ValueError(f"shard {parent_id} needs too many deterministic retry parts")
    retry_root = Path(shard["path"]).parent / "retry_subshards"
    retry_root.mkdir(parents=True, exist_ok=True)
    split_shards = []
    for part, start in enumerate(range(0, len(tokens), maximum_queries)):
        child_id = 1_000_000 + parent_id * 1_000 + part
        path = retry_root / f"parent-{parent_id:04d}-part-{part:03d}.fasta"
        path.write_text("".join(records[start : start + maximum_queries]), encoding="ascii")
        split_shards.append(
            {
                "shard_id": child_id,
                "path": path,
                "tokens": tokens[start : start + maximum_queries],
            }
        )
    return split_shards


def _read_fasta_records(path: Path) -> list[str]:
    text = path.read_text(encoding="ascii")
    if not text.startswith(">"):
        raise RuntimeError(f"query FASTA does not start with a header: {path}")
    records = []
    current = []
    for line in text.splitlines(keepends=True):
        if line.startswith(">") and current:
            records.append("".join(current))
            current = []
        current.append(line)
    if current:
        records.append("".join(current))
    if any(not record.endswith("\n") for record in records):
        raise RuntimeError(f"query FASTA has an unterminated record: {path}")
    return records


def _run_shard_to_checkpoint(
    *,
    checkpoint_dir: Path,
    checkpoint_identity: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one shard and persist its parsed evidence before returning to Ray."""
    result = _run_shard(**kwargs)
    return _write_shard_checkpoint(
        result,
        checkpoint_dir=Path(checkpoint_dir),
        identity=checkpoint_identity,
    )


def _stage_checkpoint_identity(
    *,
    target_index: Path,
    mode: str,
    cap: int,
    search: dict[str, Any],
    execution: dict[str, Any],
    token_lengths: dict[str, int],
    token_records: dict[str, dict[str, Any]],
    primary_rule: split_audit.SimilarityRule,
    sensitivity_rule: split_audit.SimilarityRule,
    output_root: Path,
) -> dict[str, Any]:
    del output_root
    target_stat = Path(target_index).stat()
    return {
        "checkpoint_schema_version": 1,
        "mode": str(mode),
        "cap": int(cap),
        "target_index": {
            "path": str(Path(target_index).resolve()),
            "bytes": int(target_stat.st_size),
            "mtime_ns": int(target_stat.st_mtime_ns),
        },
        "search": search,
        "threads_per_worker": int(execution["threads_per_worker"]),
        "token_lengths_sha256": _json_sha256(token_lengths),
        "token_records_sha256": _json_sha256(token_records),
        "primary_rule": _rule_record(primary_rule),
        "sensitivity_rule": _rule_record(sensitivity_rule),
    }


def _shard_checkpoint_identity(
    *,
    shard: dict[str, Any],
    stage_identity: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "stage": stage_identity,
        "shard_id": int(shard["shard_id"]),
        "query_tokens": [str(value) for value in shard["tokens"]],
        "query_fasta_sha256": _file_sha256(Path(shard["path"])),
    }
    return {
        "sha256": _json_sha256(payload),
        "payload": payload,
    }


def _shard_checkpoint_directory(
    root: Path,
    *,
    mode: str,
    cap: int,
    shard_id: int,
    identity: dict[str, Any],
) -> Path:
    return Path(root) / f"{mode}-cap{cap}" / f"shard-{shard_id:04d}" / str(identity["sha256"])


def _validated_checkpoint_descriptor(
    checkpoint_dir: Path,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = Path(checkpoint_dir) / "manifest.json"
    if not manifest_path.exists():
        if Path(checkpoint_dir).exists():
            logger.warning("Ignoring incomplete shard checkpoint %s", checkpoint_dir)
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity") != identity:
        raise RuntimeError(f"shard checkpoint identity mismatch: {checkpoint_dir}")
    for name, expected_sha256 in manifest["files_sha256"].items():
        path = Path(checkpoint_dir) / name
        if not path.exists() or _file_sha256(path) != expected_sha256:
            raise RuntimeError(f"shard checkpoint file failed validation: {path}")
    return {
        "checkpoint_dir": str(Path(checkpoint_dir)),
        "identity": identity,
    }


def _write_shard_checkpoint(
    result: dict[str, Any],
    *,
    checkpoint_dir: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp-{os.getpid()}-{time.time_ns()}"
    run_path = checkpoint_dir / "run.json"
    profile_path = checkpoint_dir / "profile.parquet"
    edges_path = checkpoint_dir / "edges.parquet"
    manifest_path = checkpoint_dir / "manifest.json"
    temporary = {
        run_path: run_path.with_name(run_path.name + suffix),
        profile_path: profile_path.with_name(profile_path.name + suffix),
        edges_path: edges_path.with_name(edges_path.name + suffix),
        manifest_path: manifest_path.with_name(manifest_path.name + suffix),
    }
    run_record = dict(result["run"])
    run_record["checkpoint_reused"] = False
    temporary[run_path].write_text(
        json.dumps(run_record, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame.from_records(result["profile"]).to_parquet(
        temporary[profile_path],
        index=False,
    )
    pd.DataFrame.from_records(result["edges"]).to_parquet(
        temporary[edges_path],
        index=False,
    )
    for final_path in (run_path, profile_path, edges_path):
        os.replace(temporary[final_path], final_path)
    manifest = {
        "identity": identity,
        "profile_rows": int(len(result["profile"])),
        "edge_rows": int(len(result["edges"])),
        "files_sha256": {
            path.name: _file_sha256(path) for path in (run_path, profile_path, edges_path)
        },
    }
    temporary[manifest_path].write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary[manifest_path], manifest_path)
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "identity": identity,
    }


def _load_shard_checkpoint(
    descriptor: dict[str, Any],
    *,
    reused: bool,
) -> dict[str, Any]:
    checkpoint_dir = Path(descriptor["checkpoint_dir"])
    validated = _validated_checkpoint_descriptor(checkpoint_dir, descriptor["identity"])
    if validated is None:
        raise RuntimeError(f"completed shard checkpoint is missing: {checkpoint_dir}")
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    run_record = json.loads((checkpoint_dir / "run.json").read_text(encoding="utf-8"))
    run_record["checkpoint_reused"] = bool(reused)
    profile = pd.read_parquet(checkpoint_dir / "profile.parquet")
    edges = pd.read_parquet(checkpoint_dir / "edges.parquet")
    if len(profile) != int(manifest["profile_rows"]):
        raise RuntimeError(f"shard checkpoint profile row count changed: {checkpoint_dir}")
    if len(edges) != int(manifest["edge_rows"]):
        raise RuntimeError(f"shard checkpoint edge row count changed: {checkpoint_dir}")
    return {
        "run": run_record,
        "profile": profile.to_dict("records"),
        "edges": edges.to_dict("records"),
    }


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rule_record(rule: split_audit.SimilarityRule) -> dict[str, float]:
    return {
        "minimum_identity": float(rule.minimum_identity),
        "minimum_query_coverage": float(rule.minimum_query_coverage),
        "minimum_subject_coverage": float(rule.minimum_subject_coverage),
        "minimum_length_ratio": float(rule.minimum_length_ratio),
    }


def _run_shard(
    *,
    shard: dict[str, Any],
    target_index: Path,
    output_root: Path,
    mode: str,
    cap: int,
    search: dict[str, Any],
    execution: dict[str, Any],
    token_lengths: dict[str, int],
    token_records: dict[str, dict[str, Any]],
    primary_rule: split_audit.SimilarityRule,
    sensitivity_rule: split_audit.SimilarityRule,
) -> dict[str, Any]:
    """Run and parse one isolated minimap2 shard inside a Ray worker."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    shard_id = int(shard["shard_id"])
    output = output_root / f"{mode}-cap{cap}-shard{shard_id:04d}.paf"
    stderr_path = output.with_suffix(".stderr")
    command = [
        str(search["executable"]),
        "-x",
        str(search["preset"]),
        "-t",
        str(execution["threads_per_worker"]),
        "-N",
        str(cap),
        "-p",
        str(search["minimum_secondary_score_ratio"]),
    ]
    if mode == "exact":
        command.append("-c")
    command.extend(["-o", str(output), str(target_index), str(shard["path"])])

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.monotonic()
    with stderr_path.open("w", encoding="utf-8") as stderr_handle:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr_handle)
        while process.poll() is None:
            elapsed = time.monotonic() - start
            size = output.stat().st_size if output.exists() else 0
            if elapsed > int(execution["task_timeout_seconds"]):
                _terminate_process(process)
                raise RuntimeError(f"{mode} cap {cap} shard {shard_id} reached its time limit")
            if size > int(execution["maximum_task_output_bytes"]):
                _terminate_process(process)
                raise RuntimeError(f"{mode} cap {cap} shard {shard_id} reached its output limit")
            time.sleep(0.25)
    if process.returncode != 0:
        stderr = stderr_path.read_text(encoding="utf-8")
        raise RuntimeError(
            f"minimap2 failed for {mode} cap {cap} shard {shard_id}:\n{stderr[-4000:]}"
        )
    wall_seconds = time.monotonic() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_seconds = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    paf_bytes = output.stat().st_size
    query_tokens = set(str(value) for value in shard["tokens"])
    if mode == "exact":
        profile, edges = calibration.parse_exact_paf(
            output,
            token_records=token_records,
            query_tokens=query_tokens,
            query_repeat=int(search["query_repeat"]),
            cap=cap,
            primary_rule=primary_rule,
            sensitivity_rule=sensitivity_rule,
        )
    else:
        profile = calibration.parse_candidate_paf(
            output,
            token_lengths=token_lengths,
            query_tokens=query_tokens,
            query_repeat=int(search["query_repeat"]),
            cap=cap,
            filters=search,
        )
        edges = pd.DataFrame()
    profile["mode"] = mode
    profile["cap"] = cap
    profile["shard_id"] = shard_id
    if not edges.empty:
        edges["cap"] = cap
        edges["shard_id"] = shard_id
    stderr_lines = [
        line.strip()
        for line in stderr_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output.unlink()
    stderr_path.unlink()
    return {
        "run": {
            "mode": mode,
            "cap": cap,
            "shard_id": shard_id,
            "query_count": len(query_tokens),
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "paf_bytes": paf_bytes,
            "raw_alignments": int(profile["raw_alignments"].sum()),
            "unique_nonself_targets": int(profile["unique_nonself_targets"].sum()),
            "filtered_or_sensitivity_candidates": int(
                profile.get("approximate_candidates", profile.get("sensitivity_edges")).sum()
            ),
            "potentially_saturated_queries": int(profile["potentially_saturated"].sum()),
            "tool_log": stderr_lines[-12:],
        },
        "profile": profile.to_dict("records"),
        "edges": edges.to_dict("records"),
    }


def _collect_stage(
    results: list[dict[str, Any]],
    run_records: list[dict[str, Any]],
    profile_tables: list[pd.DataFrame],
    exact_edge_tables: list[pd.DataFrame],
) -> None:
    for result in results:
        run_records.append(result["run"])
        profile_tables.append(pd.DataFrame.from_records(result["profile"]))
        if result["edges"]:
            exact_edge_tables.append(pd.DataFrame.from_records(result["edges"]))


def _run_synthetic_validation(
    root: Path,
    *,
    search: dict[str, Any],
    execution: dict[str, Any],
    primary_rule: split_audit.SimilarityRule,
    sensitivity_rule: split_audit.SimilarityRule,
) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    rng = np.random.default_rng(20260810)
    base = "".join(rng.choice(np.array(list("ACGT")), size=4_000).tolist())
    rotation = base[1_223:] + base[:1_223]
    reverse_complement = base.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    alternatives = {"A": "C", "C": "G", "G": "T", "T": "A"}
    mutated = list(base)
    for position in np.linspace(23, len(base) - 29, num=39, dtype=int):
        mutated[position] = alternatives[mutated[position]]
    shared_cassette = base[:1_000] + "".join(
        rng.choice(np.array(list("ACGT")), size=3_000).tolist()
    )
    query_sequences = {
        "query_exact": base,
        "query_rotation": rotation,
        "query_reverse": reverse_complement,
        "query_mutated": "".join(mutated),
        "query_cassette": shared_cassette,
    }
    query_frame = pd.DataFrame.from_records(
        [
            {"token": token, "sequence": sequence}
            for token, sequence in sorted(query_sequences.items())
        ]
    )
    target_frame = pd.DataFrame.from_records([{"token": "target", "sequence": base}])
    query_path = root / "queries.fasta"
    target_path = root / "target.fasta"
    index_path = root / "target.mmi"
    _write_fasta(query_frame, query_path, repeat=int(search["query_repeat"]))
    _write_fasta(target_frame, target_path, repeat=1)
    result = subprocess.run(
        [
            str(search["executable"]),
            "-x",
            str(search["preset"]),
            "-t",
            str(execution["threads_per_worker"]),
            "-d",
            str(index_path),
            str(target_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"synthetic minimap2 index failed:\n{result.stderr[-4000:]}")
    token_records = {
        token: {"sequence_id": token, "length_bp": len(sequence)}
        for token, sequence in query_sequences.items()
    }
    token_records["target"] = {"sequence_id": "target", "length_bp": len(base)}
    shard = {
        "shard_id": 0,
        "path": query_path,
        "tokens": sorted(query_sequences),
    }
    candidate = _run_shard(
        shard=shard,
        target_index=index_path,
        output_root=root / "candidate",
        mode="candidate",
        cap=10,
        search=search,
        execution=execution,
        token_lengths={key: int(value["length_bp"]) for key, value in token_records.items()},
        token_records=token_records,
        primary_rule=primary_rule,
        sensitivity_rule=sensitivity_rule,
    )
    exact = _run_shard(
        shard=shard,
        target_index=index_path,
        output_root=root / "exact",
        mode="exact",
        cap=10,
        search=search,
        execution=execution,
        token_lengths={key: int(value["length_bp"]) for key, value in token_records.items()},
        token_records=token_records,
        primary_rule=primary_rule,
        sensitivity_rule=sensitivity_rule,
    )
    candidate_profile = pd.DataFrame.from_records(candidate["profile"]).set_index("token")
    exact_edges = pd.DataFrame.from_records(exact["edges"])
    expected = {"query_exact", "query_rotation", "query_reverse", "query_mutated"}
    candidate_positive = set(
        candidate_profile.index[candidate_profile["approximate_candidates"].gt(0)]
    )
    exact_primary = set(
        exact_edges.loc[exact_edges["primary_near_duplicate"], "query_token"]
        if len(exact_edges)
        else []
    )
    if candidate_positive != expected or exact_primary != expected:
        raise RuntimeError(
            "similarity calibration synthetic classifications changed: "
            f"candidate={sorted(candidate_positive)}, exact={sorted(exact_primary)}"
        )
    return {
        "passed": True,
        "fixture_seed": 20260810,
        "sequence_length_bp": len(base),
        "substitution_count": 39,
        "candidate_positive_queries": sorted(candidate_positive),
        "exact_primary_queries": sorted(exact_primary),
        "shared_cassette_candidate": bool(
            candidate_profile.loc["query_cassette", "approximate_candidates"]
        ),
        "shared_cassette_exact": bool(
            len(exact_edges) and exact_edges["query_token"].eq("query_cassette").any()
        ),
    }


def _write_query_shards(
    frame: pd.DataFrame,
    root: Path,
    *,
    shard_queries: int,
    repeat: int,
) -> list[dict[str, Any]]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    ordered = frame.sort_values("token", kind="stable").reset_index(drop=True)
    shards = []
    for shard_id, start in enumerate(range(0, len(ordered), shard_queries)):
        shard = ordered.iloc[start : start + shard_queries]
        path = root / f"queries-{shard_id:04d}.fasta"
        _write_fasta(shard, path, repeat=repeat)
        shards.append(
            {
                "shard_id": shard_id,
                "path": path,
                "tokens": shard["token"].astype(str).tolist(),
            }
        )
    return shards


def _prepare_target_cache(
    root: Path,
    tokens: pd.DataFrame,
    search: dict[str, Any],
    *,
    expected_population_sha256: str,
) -> tuple[Path, Path, dict[str, Any]]:
    fasta = root / "all_targets.fasta"
    index = root / "all_targets.mmi"
    manifest_path = root / "target_cache_manifest.json"
    if manifest_path.exists() and fasta.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("population_sha256") != expected_population_sha256:
            raise RuntimeError("target FASTA cache belongs to a different retrieval population")
        observed_hash = _file_sha256(fasta)
        if observed_hash != manifest.get("fasta_sha256"):
            raise RuntimeError("target FASTA cache hash does not match its manifest")
    else:
        _write_fasta(tokens, fasta, repeat=1)
        manifest = {
            "population_sha256": expected_population_sha256,
            "rows": int(len(tokens)),
            "fasta_bytes": fasta.stat().st_size,
            "fasta_sha256": _file_sha256(fasta),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if not index.exists():
        command = [
            str(search["executable"]),
            "-x",
            str(search["preset"]),
            "-t",
            "8",
            "-d",
            str(index),
            str(fasta),
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"minimap2 index build failed:\n{result.stderr[-4000:]}")
    manifest["index_bytes"] = index.stat().st_size
    return fasta, index, manifest


def _write_fasta(frame: pd.DataFrame, path: Path, *, repeat: int) -> None:
    if repeat < 1:
        raise ValueError("FASTA repeat must be positive")
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for row in frame.itertuples(index=False):
            sequence = str(row.sequence) * repeat
            handle.write(f">{row.token}\n")
            for start in range(0, len(sequence), 80):
                handle.write(f"{sequence[start : start + 80]}\n")


def _validate_audit_edges(edges: pd.DataFrame, params: dict[str, Any]) -> None:
    required = {
        "query_sequence_id",
        "subject_sequence_id",
        "primary_near_duplicate",
        "sensitivity_near_duplicate",
    }
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(f"split-audit edge table is missing columns: {sorted(missing)}")
    if len(edges) != 333_686:
        raise ValueError(
            f"split-audit edge row count changed: observed {len(edges)}, expected 333686"
        )
    primary = int(edges["primary_near_duplicate"].sum())
    sensitivity = int(edges["sensitivity_near_duplicate"].sum())
    if (primary, sensitivity) != (7_624, 13_751):
        raise ValueError(
            "split-audit edge classifications changed: "
            f"observed primary={primary}, sensitivity={sensitivity}"
        )


def _validate_search(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "executable": str(config["executable"]),
        "preset": str(config["preset"]),
        "query_repeat": int(config["query_repeat"]),
        "minimum_secondary_score_ratio": float(config["minimum_secondary_score_ratio"]),
        "caps": [int(value) for value in config["caps"]],
        "adaptive_tail_cap": int(config["adaptive_tail_cap"]),
        "minimum_approximate_query_coverage": float(config["minimum_approximate_query_coverage"]),
        "minimum_approximate_subject_coverage": float(
            config["minimum_approximate_subject_coverage"]
        ),
        "minimum_length_ratio": float(config["minimum_length_ratio"]),
        "maximum_approximate_divergence": float(config["maximum_approximate_divergence"]),
    }
    if resolved["preset"] != "asm20" or resolved["query_repeat"] != 2:
        raise ValueError("calibration must retain asm20 and doubled circular queries")
    if sorted(resolved["caps"]) != resolved["caps"] or min(resolved["caps"]) < 1:
        raise ValueError("candidate caps must be positive and increasing")
    if resolved["adaptive_tail_cap"] <= max(resolved["caps"]):
        raise ValueError("adaptive tail cap must exceed normal candidate caps")
    for field in (
        "minimum_secondary_score_ratio",
        "minimum_approximate_query_coverage",
        "minimum_approximate_subject_coverage",
        "minimum_length_ratio",
    ):
        if not 0.0 < float(resolved[field]) <= 1.0:
            raise ValueError(f"{field} must be in (0, 1]")
    if not 0.0 <= resolved["maximum_approximate_divergence"] <= 1.0:
        raise ValueError("maximum approximate divergence must be in [0, 1]")
    return resolved


def _validate_execution(config: dict[str, Any]) -> dict[str, Any]:
    resolved = {
        "ray_workers": int(config["ray_workers"]),
        "threads_per_worker": int(config["threads_per_worker"]),
        "shard_queries": int(config["shard_queries"]),
        "task_timeout_seconds": int(config["task_timeout_seconds"]),
        "calibration_wall_limit_seconds": int(config["calibration_wall_limit_seconds"]),
        "maximum_task_output_bytes": int(config["maximum_task_output_bytes"]),
        "minimum_free_disk_bytes": int(config["minimum_free_disk_bytes"]),
        "scratch_root": str(config["scratch_root"]),
    }
    if min(value for key, value in resolved.items() if key != "scratch_root") < 1:
        raise ValueError("calibration execution limits must be positive")
    if resolved["ray_workers"] * resolved["threads_per_worker"] > (os.cpu_count() or 1):
        raise ValueError("Ray worker CPUs exceed the local logical CPU count")
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


def _check_free_disk(execution: dict[str, Any]) -> None:
    free = shutil.disk_usage(Path(execution["scratch_root"]).resolve().parent).free
    if free < execution["minimum_free_disk_bytes"]:
        raise RuntimeError(
            f"free disk {free} is below calibration minimum {execution['minimum_free_disk_bytes']}"
        )


def _check_wall_and_disk(started: float, execution: dict[str, Any]) -> None:
    elapsed = time.monotonic() - started
    if elapsed > execution["calibration_wall_limit_seconds"]:
        raise RuntimeError("calibration reached its fixed wall-time limit")
    _check_free_disk(execution)


def _terminate_process(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _tool_version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], check=True, capture_output=True, text=True)
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"{executable} did not report a version")
    return lines[0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _empty_exact_edges() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "query_token",
            "subject_token",
            "query_sequence_id",
            "subject_sequence_id",
            "query_length_bp",
            "subject_length_bp",
            "identity",
            "query_coverage",
            "subject_coverage",
            "length_ratio",
            "orientation",
            "alignment_block_length",
            "matching_bases",
            "primary_near_duplicate",
            "sensitivity_near_duplicate",
            "similarity_class",
            "cap",
            "shard_id",
        ]
    )
