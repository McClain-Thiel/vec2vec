"""The complete DAG for data consumed by the selected model."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node, pipeline

from vec2vec.pipelines import dataset, descriptions, processing
from vec2vec.pipelines.modeling_data import nodes
from vec2vec.pipelines.query_benchmark.nodes import build_benchmark
from vec2vec.pipelines.similarity_graph.nodes import run_similarity_graph
from vec2vec.pipelines.similarity_split.nodes import (
    audit_similarity_split,
    build_similarity_split,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Build raw inputs through the selected frozen feature tables.

    Tags allow costly stages to run independently while Kedro retains one
    dependency graph. The graph and Qwen nodes are never in ``__default__``.
    """
    source_data = pipeline(
        processing.create_pipeline()
        + descriptions.create_import_pipeline()
        + dataset.create_pipeline(),
        tags="source-data",
    )
    selected_data = Pipeline(
        [
            node(
                nodes.build_constraint_states,
                inputs=[
                    "retrieval_dataset@constraint_state",
                    "params:constraint_state",
                    "params:constraint_evidence",
                    "params:constraint_mappings",
                ],
                outputs=[
                    "e00_constraint_vocabulary",
                    "e00_plasmid_constraint_state",
                    "e00_constraint_state_manifest",
                ],
                name="build_constraint_states",
                tags="constraints",
            ),
            node(
                run_similarity_graph,
                inputs=["retrieval_dataset@split_audit", "params:similarity_graph"],
                outputs=[
                    "e00_similarity_graph_edges",
                    "e00_similarity_graph_nodes",
                    "e00_similarity_graph_components",
                    "e00_similarity_graph_query_profile",
                    "e00_similarity_graph_runs",
                    "e00_similarity_graph_manifest",
                ],
                name="build_similarity_graph",
                tags="similarity-graph",
            ),
            node(
                build_similarity_split,
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
                name="build_similarity_split",
                tags="similarity-split",
            ),
            node(
                audit_similarity_split,
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
                name="audit_similarity_split",
                tags="similarity-split",
            ),
            node(
                build_benchmark,
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
                name="build_query_benchmark",
                tags="queries",
            ),
            node(
                nodes.build_model_inputs,
                inputs=[
                    "retrieval_dataset@fixed_representation_bakeoff",
                    "e00_split_grouped_v2",
                    "e00_split_grouped_v2_manifest",
                    "e00_query_catalog",
                    "e00_query_candidate_state",
                    "e00_query_benchmark_manifest",
                    "params:modeling_features",
                ],
                outputs=[
                    "e02b_pairs",
                    "e02b_exclusions",
                    "e02b_queries",
                    "e02b_query_states",
                    "e02b_input_manifest",
                ],
                name="build_model_panels",
                tags="model-panels",
            ),
            node(
                nodes.fit_selected_dna_features,
                inputs=["e02b_pairs", "e02b_input_manifest", "params:modeling_features"],
                outputs=[
                    "e02b_dna_features_tfidf_6mer_svd_512",
                    "e02b_tfidf_vocabulary",
                    "e02b_tfidf_svd_state",
                    "e02b_dna_manifest_tfidf_6mer_svd_512",
                ],
                name="fit_selected_dna_features",
                tags="tfidf",
            ),
            node(
                nodes.extract_selected_text_features,
                inputs=[
                    "e02b_pairs",
                    "e02b_queries",
                    "e02b_input_manifest",
                    "params:modeling_features",
                ],
                outputs=[
                    "e02b_text_features_qwen3_embedding_0_6b",
                    "e02b_text_manifest_qwen3_embedding_0_6b",
                ],
                name="extract_selected_text_features",
                tags=["qwen", "gpu"],
            ),
            node(
                nodes.build_model_inputs,
                inputs=[
                    "retrieval_dataset@fixed_representation_bakeoff",
                    "e00_split_grouped_v2",
                    "e00_split_grouped_v2_manifest",
                    "e00_query_catalog",
                    "e00_query_candidate_state",
                    "e00_query_benchmark_manifest",
                    "params:e06_modeling_features",
                ],
                outputs=[
                    "e06_pairs",
                    "e06_exclusions",
                    "e06_queries",
                    "e06_query_states",
                    "e06_input_manifest",
                ],
                name="build_population_model_panels",
                tags=["e06", "e06-inputs"],
            ),
            node(
                nodes.fit_selected_dna_features,
                inputs=["e06_pairs", "e06_input_manifest", "params:e06_modeling_features"],
                outputs=[
                    "e06_dna_features_tfidf_6mer_svd_512",
                    "e06_tfidf_vocabulary",
                    "e06_tfidf_svd_state",
                    "e06_dna_manifest_tfidf_6mer_svd_512",
                ],
                name="fit_population_dna_features",
                tags=["e06", "e06-tfidf"],
            ),
            node(
                nodes.extract_selected_text_features,
                inputs=[
                    "e06_pairs",
                    "e06_queries",
                    "e06_input_manifest",
                    "params:e06_modeling_features",
                ],
                outputs=[
                    "e06_text_features_qwen3_embedding_0_6b",
                    "e06_text_manifest_qwen3_embedding_0_6b",
                ],
                name="extract_population_text_features",
                tags=["e06", "e06-qwen", "gpu"],
            ),
        ]
    )
    return source_data + selected_data
