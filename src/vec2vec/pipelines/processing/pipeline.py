"""Raw releases to canonical record tables."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline, node

from vec2vec.pipelines.processing import nodes


def _addgene_nodes() -> list[Node]:
    return [
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


def create_pipeline(**kwargs) -> Pipeline:
    """Flatten the Addgene release into records, annotations and feature lists."""
    return Pipeline(_addgene_nodes())


def create_plsdb_pipeline(**kwargs) -> Pipeline:
    """Flatten the PLSDB release into records.

    Registered separately: PLSDB backs encoder benchmarking rather than the
    paired retrieval dataset, and its FASTA is large enough that you rarely want
    it in the same run.
    """
    return Pipeline(
        [
            node(
                func=nodes.process_plsdb_records,
                inputs=["plsdb_sequences", "plsdb_nuccore", "plsdb_taxonomy", "params:plsdb"],
                outputs="plsdb_records",
                name="process_plsdb_records",
            )
        ]
    )
