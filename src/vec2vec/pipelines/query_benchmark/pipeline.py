"""Pipeline definition for the frozen query-benchmark data product."""

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.query_benchmark import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Return the explicit, version-pinned query-benchmark pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.build_benchmark,
                inputs=[
                    "retrieval_dataset@query_benchmark",
                    "e00_split_grouped_v2",
                    "e00_similarity_graph_edges",
                    "e00_similarity_graph_manifest",
                    "e00_split_grouped_v2_manifest",
                    "e00_constraint_vocabulary",
                    "e00_plasmid_constraint_state",
                    "e00_constraint_state_manifest",
                    "params:query_benchmark",
                ],
                outputs=[
                    "e00_query_catalog",
                    "e00_candidate_galleries",
                    "e00_query_candidate_state",
                    "e00_candidate_base_mass",
                    "e00_benchmark_control_rankings",
                    "e00_benchmark_control_metrics",
                    "e00_query_benchmark_manifest",
                ],
                name="build_e00_frozen_query_benchmark_v0_1",
            )
        ]
    )
