"""Pipeline definition for the explicit local similarity-graph calibration."""

from kedro.pipeline import Pipeline, node, pipeline

from vec2vec.pipelines.similarity_graph_calibration.nodes import run_similarity_graph_calibration


def create_pipeline(**kwargs) -> Pipeline:
    """Return the named calibration pipeline; it is deliberately not in ``__default__``."""
    return pipeline(
        [
            node(
                func=run_similarity_graph_calibration,
                inputs=[
                    "retrieval_dataset@split_audit",
                    "e00_split_audit_edges@calibration",
                    "params:similarity_graph_calibration",
                ],
                outputs=[
                    "e00_similarity_calibration_runs",
                    "e00_similarity_calibration_query_profile",
                    "e00_similarity_calibration_exact_edges",
                    "e00_similarity_calibration_manifest",
                ],
                name="run_similarity_graph_calibration",
            )
        ]
    )
