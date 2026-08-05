"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from vec2vec.pipelines import (
    agent_judge,
    audit,
    constraint_semantics,
    dataset,
    descriptions,
    facet_audit,
    processing,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    ``__default__`` covers everything that can be re-run for free from data
    already in the lake. Run the rest by name:

    - ``descriptions`` calls a paid API once per plasmid.
    - ``agent_judge_pilot`` calls a paid API once per review packet.
    - ``agent_judge_smoke`` calls it for a fixed six-packet subset.
    - ``agent_judge_validator`` calls an independent paid model on the fixed pilot packets.
    - ``agent_judge_targeted_packets`` prepares the v0.2 check without a paid call.
    - ``agent_judge_targeted`` calls a paid strong model on those inspected packets.
    - ``agent_judge_targeted_smoke`` calls it once to check the new response contract.
    - ``import_descriptions`` adopts already-published descriptions instead.
    """
    pipelines = {
        "processing": processing.create_pipeline(),
        "descriptions": descriptions.create_pipeline(),
        "agent_judge_pilot": agent_judge.create_pipeline(),
        "agent_judge_smoke": agent_judge.create_smoke_pipeline(),
        "agent_judge_validator": agent_judge.create_validator_pipeline(),
        "agent_judge_targeted_packets": agent_judge.create_targeted_packet_pipeline(),
        "agent_judge_targeted": agent_judge.create_targeted_pipeline(),
        "agent_judge_targeted_smoke": agent_judge.create_targeted_smoke_pipeline(),
        "agent_judge_comparison": agent_judge.create_comparison_pipeline(),
        "import_descriptions": descriptions.create_import_pipeline(),
        "dataset": dataset.create_pipeline(),
        "audit": audit.create_pipeline(),
        "constraint_semantics": constraint_semantics.create_pipeline(),
        "facet_audit_sample": facet_audit.create_pipeline(),
        "facet_audit_review_export": facet_audit.create_review_export_pipeline(),
    }
    pipelines["__default__"] = pipelines["processing"] + pipelines["dataset"] + pipelines["audit"]
    return pipelines
