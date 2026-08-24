"""Pipeline for deterministic training evidence and benchmark sampling."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.constraint_evidence import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Apply enabled exact rules without inspecting or labeling test rows."""
    return Pipeline(
        [
            node(
                func=nodes.build_evidence,
                inputs=[
                    "retrieval_dataset@constraint_evidence",
                    "addgene_annotations@plannotate",
                    "params:constraint_evidence",
                    "params:facet_audit",
                ],
                outputs=[
                    "e00_training_constraint_evidence",
                    "e00_constraint_benchmark_sample",
                    "e00_constraint_evidence_manifest",
                ],
                name="build_e00_constraint_evidence",
            )
        ]
    )
