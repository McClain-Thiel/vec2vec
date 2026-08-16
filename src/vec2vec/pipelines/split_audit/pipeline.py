"""Pipeline for the E00 split similarity and concentration audit."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.split_audit import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Audit the preserved grouped split without inspecting model outcomes."""
    return Pipeline(
        [
            node(
                func=nodes.run_split_audit,
                inputs=["retrieval_dataset@split_audit", "params:split_audit"],
                outputs=[
                    "e00_split_audit_edges",
                    "e00_split_audit_component_profile",
                    "e00_split_audit_manifest",
                ],
                name="audit_e00_split_similarity_and_concentration",
            )
        ]
    )
