"""Pipeline definition for the explicit full similarity graph."""

from kedro.pipeline import Pipeline, node, pipeline

from vec2vec.pipelines.similarity_graph.nodes import run_similarity_graph


def create_pipeline(**kwargs) -> Pipeline:
    """Return the named, costly graph pipeline; it is not in ``__default__``."""
    return pipeline(
        [
            node(
                func=run_similarity_graph,
                inputs=["retrieval_dataset@split_audit", "params:similarity_graph"],
                outputs=[
                    "e00_similarity_graph_edges",
                    "e00_similarity_graph_nodes",
                    "e00_similarity_graph_components",
                    "e00_similarity_graph_query_profile",
                    "e00_similarity_graph_runs",
                    "e00_similarity_graph_manifest",
                ],
                name="run_global_similarity_graph",
            )
        ]
    )
