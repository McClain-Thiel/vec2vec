"""Supervision audits over the assembled retrieval dataset."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.audit import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Measure structured-query coverage and same-backbone hard-negative yield."""
    return Pipeline(
        [
            node(
                func=nodes.audit_hard_negatives,
                inputs=["retrieval_dataset@audit", "params:audit"],
                outputs=[
                    "hard_negative_yield",
                    "hard_negative_examples",
                    "hard_negative_summary",
                ],
                name="audit_hard_negatives",
            )
        ]
    )
