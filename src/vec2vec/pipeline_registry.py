"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from vec2vec.pipelines import (
    agent_judge,
    audit,
    constraint_evidence,
    constraint_semantics,
    constraint_state,
    dataset,
    descriptions,
    facet_audit,
    fixed_representation_invariance,
    fixed_representation_smoke,
    processing,
    query_benchmark,
    similarity_graph,
    similarity_graph_calibration,
    similarity_split,
    split_audit,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    ``__default__`` covers everything that can be re-run for free from data
    already in the lake. Run the rest by name:

    - ``descriptions`` calls a paid API once per plasmid.
    - ``agent_judge_targeted_packets`` prepares the v0.2 check without a paid call.
    - ``agent_judge_targeted`` calls a paid strong model on those inspected packets.
    - ``agent_judge_targeted_smoke`` calls it once to check the new response contract.
    - ``constraint_benchmark_packets`` prepares the fixed accuracy check without a paid call.
    - ``constraint_benchmark_smoke`` calls the strong model once per facet.
    - ``constraint_benchmark_judge`` calls it on the fixed 240-application sample.
    - ``fixed_representation_invariance`` calls one pinned DNA encoder on 512 rows.
    - ``import_descriptions`` adopts already-published descriptions instead.

    Historical v3 pilot, validator, and comparison outputs remain catalog-readable, but their
    superseded one-axis pipelines are not registered.
    """
    pipelines = {
        "processing": processing.create_pipeline(),
        "descriptions": descriptions.create_pipeline(),
        "agent_judge_targeted_packets": agent_judge.create_targeted_packet_pipeline(),
        "agent_judge_targeted": agent_judge.create_targeted_pipeline(),
        "agent_judge_targeted_smoke": agent_judge.create_targeted_smoke_pipeline(),
        "constraint_benchmark_packets": (agent_judge.create_constraint_benchmark_packet_pipeline()),
        "constraint_benchmark_smoke": agent_judge.create_constraint_benchmark_smoke_pipeline(),
        "constraint_benchmark_judge": agent_judge.create_constraint_benchmark_pipeline(),
        "import_descriptions": descriptions.create_import_pipeline(),
        "dataset": dataset.create_pipeline(),
        "audit": audit.create_pipeline(),
        "constraint_semantics": constraint_semantics.create_pipeline(),
        "constraint_evidence": constraint_evidence.create_pipeline(),
        "constraint_state": constraint_state.create_pipeline(),
        "query_benchmark": query_benchmark.create_pipeline(),
        "split_audit": split_audit.create_pipeline(),
        "similarity_graph_calibration": similarity_graph_calibration.create_pipeline(),
        "similarity_graph": similarity_graph.create_pipeline(),
        "similarity_split": similarity_split.create_pipeline(),
        "fixed_representation_invariance": fixed_representation_invariance.create_pipeline(),
        "fixed_representation_smoke": fixed_representation_smoke.create_pipeline(),
        "facet_audit_sample": facet_audit.create_pipeline(),
        "facet_audit_review_export": facet_audit.create_review_export_pipeline(),
    }
    pipelines["__default__"] = pipelines["processing"] + pipelines["dataset"] + pipelines["audit"]
    return pipelines
