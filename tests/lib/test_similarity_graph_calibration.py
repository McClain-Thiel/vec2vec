from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from vec2vec.lib.similarity_graph_paf import (
    parse_candidate_paf,
    parse_exact_paf,
)
from vec2vec.lib.split_audit import SimilarityRule
from vec2vec.pipelines.modeling_data.graph_runtime import (
    _check_free_disk,
    _get_ray_descriptors_with_disk_guard,
    _load_shard_checkpoint,
    _prepare_target_cache,
    _shard_checkpoint_directory,
    _shard_checkpoint_identity,
    _split_uncheckpointed_exact_shards,
    _validated_checkpoint_descriptor,
    _validated_tool_version,
    _write_shard_checkpoint,
)


def test_target_cache_rebuilds_an_unmanifested_partial_index_atomically(tmp_path, monkeypatch):
    root = tmp_path / "missing" / "cache"
    root.mkdir(parents=True)
    partial_index = root / "all_targets.mmi"
    partial_index.write_bytes(b"partial")
    calls = []

    def build_index(command, **kwargs):
        calls.append(command)
        target = Path(command[command.index("-d") + 1])
        target.write_bytes(b"complete-index")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr("vec2vec.pipelines.modeling_data.graph_runtime.subprocess.run", build_index)
    tokens = pd.DataFrame({"token": ["one"], "sequence": ["ACGT"]})
    _, index, manifest = _prepare_target_cache(
        root,
        tokens,
        {"executable": "minimap2", "preset": "asm20"},
        expected_population_sha256="population",
    )

    assert len(calls) == 1
    assert index.read_bytes() == b"complete-index"
    assert manifest["index_bytes"] == len(b"complete-index")
    assert json.loads((root / "target_cache_manifest.json").read_text()) == manifest
    assert list(root.glob("*.tmp")) == []

    _prepare_target_cache(
        root,
        tokens,
        {"executable": "minimap2", "preset": "asm20"},
        expected_population_sha256="population",
    )
    assert len(calls) == 1


def test_graph_rejects_a_different_minimap_version(monkeypatch):
    monkeypatch.setattr(
        "vec2vec.pipelines.modeling_data.graph_runtime._tool_version",
        lambda executable: "2.30-r1",
    )

    with pytest.raises(RuntimeError, match="minimap2 version changed"):
        _validated_tool_version({"executable": "minimap2", "expected_tool_version": "2.31-r1302"})


def test_free_disk_preflight_uses_nearest_existing_directory(tmp_path, monkeypatch):
    observed = []

    def disk_usage(path):
        observed.append(Path(path))
        return SimpleNamespace(free=100)

    monkeypatch.setattr(
        "vec2vec.pipelines.modeling_data.graph_runtime.shutil.disk_usage", disk_usage
    )
    _check_free_disk(
        {
            "scratch_root": str(tmp_path / "not-yet-created" / "nested"),
            "minimum_free_disk_bytes": 1,
        }
    )

    assert observed == [tmp_path.resolve()]


def test_candidate_parser_uses_divergence_only_as_an_approximate_filter(tmp_path: Path):
    path = tmp_path / "candidate.paf"
    path.write_text(
        "\n".join(
            [
                _paf_line("q", "q", divergence=0.0),
                _paf_line("q", "pass", divergence=0.02),
                _paf_line("q", "divergent", divergence=0.20),
            ]
        )
        + "\n",
        encoding="ascii",
    )

    profile = parse_candidate_paf(
        path,
        token_lengths={"q": 100, "pass": 100, "divergent": 100},
        query_tokens={"q"},
        query_repeat=2,
        cap=3,
        filters={
            "minimum_approximate_query_coverage": 0.80,
            "minimum_approximate_subject_coverage": 0.80,
            "minimum_length_ratio": 0.90,
            "maximum_approximate_divergence": 0.10,
        },
    )

    assert profile.loc[0, "raw_alignments"] == 3
    assert profile.loc[0, "unique_nonself_targets"] == 2
    assert profile.loc[0, "approximate_candidates"] == 1
    assert bool(profile.loc[0, "potentially_saturated"])


def test_exact_parser_keeps_best_alignment_and_applies_fixed_rules(tmp_path: Path):
    path = tmp_path / "exact.paf"
    path.write_text(
        "\n".join(
            [
                _paf_line("q", "subject", matches=80, block=80, query_end=80, target_end=80),
                _paf_line("q", "subject", matches=99, block=100),
            ]
        )
        + "\n",
        encoding="ascii",
    )
    primary = SimilarityRule(0.99, 0.95, 0.95, 0.95)
    sensitivity = SimilarityRule(0.95, 0.90, 0.90, 0.90)

    profile, edges = parse_exact_paf(
        path,
        token_records={
            "q": {"sequence_id": "query-id", "length_bp": 100},
            "subject": {"sequence_id": "subject-id", "length_bp": 100},
        },
        query_tokens={"q"},
        query_repeat=2,
        cap=10,
        primary_rule=primary,
        sensitivity_rule=sensitivity,
    )

    assert len(edges) == 1
    assert edges.loc[0, "identity"] == 0.99
    assert bool(edges.loc[0, "primary_near_duplicate"])
    assert profile.loc[0, "primary_edges"] == 1
    assert not bool(profile.loc[0, "potentially_saturated"])


def test_shard_checkpoint_round_trip_and_content_validation(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint"
    identity = {"sha256": "identity", "payload": {"mode": "exact", "cap": 1_000}}
    result = {
        "run": {"mode": "exact", "cap": 1_000, "shard_id": 4, "tool_log": []},
        "profile": [{"token": "query", "potentially_saturated": False}],
        "edges": [{"query_token": "query", "subject_token": "subject"}],
    }

    descriptor = _write_shard_checkpoint(
        result,
        checkpoint_dir=checkpoint_dir,
        identity=identity,
    )
    loaded = _load_shard_checkpoint(descriptor, reused=True)

    assert loaded["run"]["checkpoint_reused"] is True
    assert loaded["profile"] == result["profile"]
    assert loaded["edges"] == result["edges"]

    empty_result = {
        "run": {"mode": "candidate", "cap": 10, "shard_id": 0, "tool_log": []},
        "profile": [{"token": "query", "potentially_saturated": False}],
        "edges": [],
    }
    empty_descriptor = _write_shard_checkpoint(
        empty_result,
        checkpoint_dir=tmp_path / "empty-checkpoint",
        identity={"sha256": "empty", "payload": {"mode": "candidate", "cap": 10}},
    )
    assert _load_shard_checkpoint(empty_descriptor, reused=False)["edges"] == []

    (checkpoint_dir / "run.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed validation"):
        _validated_checkpoint_descriptor(checkpoint_dir, identity)


def test_retry_split_keeps_completed_parent_and_splits_only_unfinished_parent(tmp_path: Path):
    first_path = tmp_path / "queries-0000.fasta"
    second_path = tmp_path / "queries-0001.fasta"
    first_path.write_text(">a\nAAAA\n>b\nCCCC\n", encoding="ascii")
    second_path.write_text(">c\nGGGG\n>d\nTTTT\n>e\nACGT\n", encoding="ascii")
    shards = [
        {"shard_id": 0, "path": first_path, "tokens": ["a", "b"]},
        {"shard_id": 1, "path": second_path, "tokens": ["c", "d", "e"]},
    ]
    stage_identity = {"checkpoint_schema_version": 1, "mode": "exact", "cap": 1_000}
    first_identity = _shard_checkpoint_identity(shard=shards[0], stage_identity=stage_identity)
    checkpoint_root = tmp_path / "checkpoints"
    _write_shard_checkpoint(
        {
            "run": {"mode": "exact", "cap": 1_000, "shard_id": 0},
            "profile": [{"token": "a"}, {"token": "b"}],
            "edges": [],
        },
        checkpoint_dir=_shard_checkpoint_directory(
            checkpoint_root,
            mode="exact",
            cap=1_000,
            shard_id=0,
            identity=first_identity,
        ),
        identity=first_identity,
    )

    expanded = _split_uncheckpointed_exact_shards(
        shards,
        checkpoint_root=checkpoint_root,
        stage_identity=stage_identity,
        mode="exact",
        cap=1_000,
        maximum_queries=2,
    )

    assert expanded[0] == shards[0]
    assert [shard["tokens"] for shard in expanded[1:]] == [["c", "d"], ["e"]]
    assert [shard["shard_id"] for shard in expanded[1:]] == [1_001_000, 1_001_001]
    assert expanded[1]["path"].read_text(encoding="ascii") == ">c\nGGGG\n>d\nTTTT\n"
    assert expanded[2]["path"].read_text(encoding="ascii") == ">e\nACGT\n"


def test_retry_split_can_isolate_each_unfinished_dense_query(tmp_path: Path):
    path = tmp_path / "queries-0034.fasta"
    path.write_text(">a\nAAAA\n>b\nCCCC\n>c\nGGGG\n", encoding="ascii")

    expanded = _split_uncheckpointed_exact_shards(
        [{"shard_id": 34, "path": path, "tokens": ["a", "b", "c"]}],
        checkpoint_root=tmp_path / "checkpoints",
        stage_identity={"checkpoint_schema_version": 1, "mode": "exact", "cap": 10_000},
        mode="exact",
        cap=10_000,
        maximum_queries=1,
    )

    assert [shard["tokens"] for shard in expanded] == [["a"], ["b"], ["c"]]
    assert [shard["shard_id"] for shard in expanded] == [1_034_000, 1_034_001, 1_034_002]


def test_ray_collection_cancels_unfinished_tasks_below_disk_floor(monkeypatch):
    class FakeRay:
        def __init__(self):
            self.cancelled = []

        def wait(self, references, *, num_returns, timeout):
            raise AssertionError("disk guard must run before waiting")

        def cancel(self, reference, *, force):
            self.cancelled.append((reference, force))

    fake_ray = FakeRay()

    def fail_disk_check(execution):
        raise RuntimeError("free disk is below calibration minimum")

    monkeypatch.setattr(
        "vec2vec.pipelines.modeling_data.graph_runtime._check_free_disk",
        fail_disk_check,
    )

    with pytest.raises(RuntimeError, match="free disk is below calibration minimum"):
        _get_ray_descriptors_with_disk_guard(
            fake_ray,
            ["first", "second"],
            execution={"minimum_free_disk_bytes": 40_000_000_000},
        )

    assert sorted(fake_ray.cancelled) == [("first", True), ("second", True)]


def _paf_line(
    query: str,
    target: str,
    *,
    divergence: float = 0.01,
    matches: int = 99,
    block: int = 100,
    query_end: int = 100,
    target_end: int = 100,
) -> str:
    fields = [
        query,
        "200",
        "0",
        str(query_end),
        "+",
        target,
        "100",
        "0",
        str(target_end),
        str(matches),
        str(block),
        "60",
        f"dv:f:{divergence}",
        f"cg:Z:{block}M",
    ]
    return "\t".join(fields)
