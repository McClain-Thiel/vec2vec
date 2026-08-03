"""Descriptions and records to the paired retrieval dataset."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.dataset import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Join descriptions to records, then split and annotate them for retrieval."""
    return Pipeline(
        [
            node(
                func=nodes.assemble_pairs,
                inputs=[
                    "addgene_records@full",
                    "plasmid_descriptions",
                    "addgene_annotation_features",
                ],
                outputs="plasmid_pairs",
                name="assemble_pairs",
            ),
            node(
                func=nodes.add_splits_and_constraints,
                inputs=["plasmid_pairs", "params:dataset"],
                outputs=["retrieval_dataset", "retrieval_dataset_audit"],
                name="add_splits_and_constraints",
            ),
        ]
    )
