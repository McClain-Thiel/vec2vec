"""Pipeline for frozen rule-derived plasmid-constraint states."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.constraint_state import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Apply the frozen evidence and conflict rules to the complete population."""
    return Pipeline(
        [
            node(
                func=nodes.build_states,
                inputs=[
                    "retrieval_dataset@constraint_state",
                    "params:constraint_state",
                    "params:constraint_evidence",
                    "params:facet_audit",
                ],
                outputs=[
                    "e00_constraint_vocabulary",
                    "e00_plasmid_constraint_state",
                    "e00_constraint_state_manifest",
                ],
                name="build_e00_constraint_states",
            )
        ]
    )
