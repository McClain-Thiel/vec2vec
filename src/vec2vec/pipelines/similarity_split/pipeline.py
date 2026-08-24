"""Pipeline definition for split_grouped_v2 construction and independent audit."""

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.similarity_split import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Return the explicit similarity-closed grouped-split pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.build_similarity_split,
                inputs=[
                    "retrieval_dataset@split_audit",
                    "e00_similarity_graph_nodes",
                    "params:similarity_split",
                ],
                outputs=[
                    "e00_split_grouped_v2",
                    "e00_split_grouped_v2_components",
                    "e00_split_grouped_v2_build_summary",
                ],
                name="build_e00_similarity_closed_split_v2",
            ),
            node(
                func=nodes.audit_similarity_split,
                inputs=[
                    "retrieval_dataset@split_audit",
                    "e00_similarity_graph_nodes",
                    "e00_similarity_graph_edges",
                    "e00_similarity_graph_manifest",
                    "e00_split_grouped_v2",
                    "e00_split_grouped_v2_build_summary",
                    "params:similarity_split",
                ],
                outputs=[
                    "e00_split_grouped_v2_cross_edges",
                    "e00_split_grouped_v2_manifest",
                ],
                name="audit_e00_similarity_closed_split_v2",
            ),
        ]
    )
