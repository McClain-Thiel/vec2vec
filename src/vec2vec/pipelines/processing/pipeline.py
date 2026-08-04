"""Raw releases to canonical record tables."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.processing import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Flatten the Addgene release into records, annotations and feature lists."""
    return Pipeline(
        [
            node(
                func=nodes.process_addgene_records,
                inputs=["addgene_raw", "params:addgene"],
                outputs="addgene_records@full",
                name="process_addgene_records",
            ),
            node(
                func=nodes.normalize_annotations,
                inputs=["plannotate_raw", "plasmidkit_raw"],
                outputs="addgene_annotations",
                name="normalize_addgene_annotations",
            ),
            node(
                func=nodes.build_annotation_features,
                inputs=["addgene_annotations", "params:annotations"],
                outputs="addgene_annotation_features",
                name="build_addgene_annotation_features",
            ),
            node(
                func=nodes.summarize_records,
                inputs=[
                    "addgene_records@metadata",
                    "addgene_annotations",
                    "addgene_annotation_features",
                ],
                outputs="addgene_processing_report",
                name="summarize_addgene_processing",
            ),
        ]
    )
