"""Pipeline for the paid, agent-assisted review pilot."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.agent_judge import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Build evidence packets, request judgments, and report all outcomes."""
    return Pipeline(
        [
            node(
                func=nodes.build_packets,
                inputs=["e00_facet_audit_sample", "params:agent_judge_pilot"],
                outputs="e00_agent_judge_pilot_packets",
                name="build_e00_agent_judge_pilot_packets",
            ),
            node(
                func=nodes.judge_packets,
                inputs=[
                    "e00_agent_judge_pilot_packets",
                    "params:agent_judge_pilot",
                    "params:openrouter",
                ],
                outputs="e00_agent_judge_pilot_decisions",
                name="judge_e00_agent_judge_pilot_packets",
            ),
            node(
                func=nodes.summarize,
                inputs=[
                    "e00_agent_judge_pilot_packets",
                    "e00_agent_judge_pilot_decisions",
                    "params:agent_judge_pilot",
                ],
                outputs="e00_agent_judge_pilot_summary",
                name="summarize_e00_agent_judge_pilot",
            ),
        ]
    )


def create_smoke_pipeline(**kwargs) -> Pipeline:
    """Run a few paid calls on fixed prepared packets before the full pilot."""
    return Pipeline(
        [
            node(
                func=nodes.select_smoke_packets,
                inputs=["e00_agent_judge_pilot_packets", "params:agent_judge_smoke"],
                outputs="e00_agent_judge_smoke_packets",
                name="select_e00_agent_judge_smoke_packets",
            ),
            node(
                func=nodes.judge_packets,
                inputs=[
                    "e00_agent_judge_smoke_packets",
                    "params:agent_judge_smoke",
                    "params:openrouter",
                ],
                outputs="e00_agent_judge_smoke_decisions",
                name="judge_e00_agent_judge_smoke_packets",
            ),
            node(
                func=nodes.summarize,
                inputs=[
                    "e00_agent_judge_smoke_packets",
                    "e00_agent_judge_smoke_decisions",
                    "params:agent_judge_smoke",
                ],
                outputs="e00_agent_judge_smoke_summary",
                name="summarize_e00_agent_judge_smoke",
            ),
        ]
    )


def create_validator_pipeline(**kwargs) -> Pipeline:
    """Run an independent stronger model on the exact frozen pilot packets."""
    return Pipeline(
        [
            node(
                func=nodes.judge_packets,
                inputs=[
                    "e00_agent_judge_pilot_packets",
                    "params:agent_judge_validator",
                    "params:openrouter",
                ],
                outputs="e00_agent_validator_decisions",
                name="validate_e00_agent_judge_pilot_packets",
            ),
            node(
                func=nodes.summarize,
                inputs=[
                    "e00_agent_judge_pilot_packets",
                    "e00_agent_validator_decisions",
                    "params:agent_judge_validator",
                ],
                outputs="e00_agent_validator_summary",
                name="summarize_e00_agent_validator",
            ),
        ]
    )


def create_targeted_packet_pipeline(**kwargs) -> Pipeline:
    """Prepare and persist the v0.2 targeted packets without a paid request."""
    return Pipeline(
        [
            node(
                func=nodes.build_targeted_packets,
                inputs=["e00_facet_audit_sample", "params:agent_judge_targeted"],
                outputs="e00_agent_judge_targeted_packets",
                name="build_e00_agent_judge_targeted_packets",
            )
        ]
    )


def create_targeted_pipeline(**kwargs) -> Pipeline:
    """Run the strong model on the inspected v0.2 targeted packets."""
    return Pipeline(
        [
            node(
                func=nodes.judge_packets,
                inputs=[
                    "e00_agent_judge_targeted_packets",
                    "params:agent_judge_targeted",
                    "params:openrouter",
                ],
                outputs="e00_agent_judge_targeted_decisions",
                name="judge_e00_agent_judge_targeted_packets",
            ),
            node(
                func=nodes.summarize,
                inputs=[
                    "e00_agent_judge_targeted_packets",
                    "e00_agent_judge_targeted_decisions",
                    "params:agent_judge_targeted",
                ],
                outputs="e00_agent_judge_targeted_summary",
                name="summarize_e00_agent_judge_targeted",
            ),
        ]
    )


def create_targeted_smoke_pipeline(**kwargs) -> Pipeline:
    """Run one paid response-contract diagnostic on the frozen v0.2 packets."""
    return Pipeline(
        [
            node(
                func=nodes.select_smoke_packets,
                inputs=[
                    "e00_agent_judge_targeted_packets",
                    "params:agent_judge_targeted_smoke",
                ],
                outputs="e00_agent_judge_targeted_smoke_packets",
                name="select_e00_agent_judge_targeted_smoke_packet",
            ),
            node(
                func=nodes.judge_packets,
                inputs=[
                    "e00_agent_judge_targeted_smoke_packets",
                    "params:agent_judge_targeted_smoke",
                    "params:openrouter",
                ],
                outputs="e00_agent_judge_targeted_smoke_decisions",
                name="judge_e00_agent_judge_targeted_smoke_packet",
            ),
            node(
                func=nodes.summarize,
                inputs=[
                    "e00_agent_judge_targeted_smoke_packets",
                    "e00_agent_judge_targeted_smoke_decisions",
                    "params:agent_judge_targeted_smoke",
                ],
                outputs="e00_agent_judge_targeted_smoke_summary",
                name="summarize_e00_agent_judge_targeted_smoke",
            ),
        ]
    )


def create_comparison_pipeline(**kwargs) -> Pipeline:
    """Compare the independent model decisions without accepting labels."""
    return Pipeline(
        [
            node(
                func=nodes.compare_decisions,
                inputs=[
                    "e00_agent_judge_pilot_packets",
                    "e00_agent_judge_pilot_decisions",
                    "e00_agent_validator_decisions",
                ],
                outputs=["e00_agent_judge_comparison", "e00_agent_judge_comparison_summary"],
                name="compare_e00_agent_judges",
            )
        ]
    )


def create_constraint_benchmark_packet_pipeline(**kwargs) -> Pipeline:
    """Prepare the complete accuracy benchmark without a paid request."""
    return Pipeline(
        [
            node(
                func=nodes.build_constraint_benchmark_packets,
                inputs=[
                    "e00_constraint_benchmark_sample",
                    "params:constraint_benchmark_judge",
                ],
                outputs="e00_constraint_benchmark_packets",
                name="build_e00_constraint_benchmark_packets",
            )
        ]
    )


def create_constraint_benchmark_smoke_pipeline(**kwargs) -> Pipeline:
    """Run one paid packet per facet before the complete benchmark."""
    return Pipeline(
        [
            node(
                func=nodes.select_smoke_packets,
                inputs=[
                    "e00_constraint_benchmark_packets",
                    "params:constraint_benchmark_smoke",
                ],
                outputs="e00_constraint_benchmark_smoke_packets",
                name="select_e00_constraint_benchmark_smoke_packets",
            ),
            node(
                func=nodes.judge_packets,
                inputs=[
                    "e00_constraint_benchmark_smoke_packets",
                    "params:constraint_benchmark_smoke",
                    "params:openrouter",
                ],
                outputs="e00_constraint_benchmark_smoke_decisions",
                name="judge_e00_constraint_benchmark_smoke_packets",
            ),
            node(
                func=nodes.summarize,
                inputs=[
                    "e00_constraint_benchmark_smoke_packets",
                    "e00_constraint_benchmark_smoke_decisions",
                    "params:constraint_benchmark_smoke",
                ],
                outputs="e00_constraint_benchmark_smoke_summary",
                name="summarize_e00_constraint_benchmark_smoke",
            ),
        ]
    )


def create_constraint_benchmark_pipeline(**kwargs) -> Pipeline:
    """Run the strong model on the fixed 240-application benchmark."""
    return Pipeline(
        [
            node(
                func=nodes.judge_packets,
                inputs=[
                    "e00_constraint_benchmark_packets",
                    "params:constraint_benchmark_judge",
                    "params:openrouter",
                ],
                outputs="e00_constraint_benchmark_decisions",
                name="judge_e00_constraint_benchmark_packets",
            ),
            node(
                func=nodes.summarize,
                inputs=[
                    "e00_constraint_benchmark_packets",
                    "e00_constraint_benchmark_decisions",
                    "params:constraint_benchmark_judge",
                ],
                outputs="e00_constraint_benchmark_summary",
                name="summarize_e00_constraint_benchmark",
            ),
        ]
    )
