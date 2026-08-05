"""Pipeline for the frozen E00 manual facet-audit sample."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.facet_audit import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Sample metadata for human review without accepting constraint labels."""
    return Pipeline(
        [
            node(
                func=nodes.build_sample,
                inputs=["retrieval_dataset@facet_audit", "params:facet_audit"],
                outputs=[
                    "e00_facet_audit_sample",
                    "e00_facet_audit_vocabulary",
                    "e00_facet_audit_manifest",
                ],
                name="build_e00_facet_audit_sample",
            )
        ]
    )


def create_review_export_pipeline(**kwargs) -> Pipeline:
    """Export the fixed audit sample without showing any model conclusion."""
    return Pipeline(
        [
            node(
                func=nodes.build_review_export,
                inputs="e00_facet_audit_sample",
                outputs="e00_facet_audit_decision_template",
                name="build_e00_facet_audit_decision_template",
            )
        ]
    )
