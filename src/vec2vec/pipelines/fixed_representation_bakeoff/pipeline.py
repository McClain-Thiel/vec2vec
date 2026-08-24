"""Pipeline definitions for the reduced-population Gate 1 bake-off."""

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.fixed_representation_bakeoff import nodes


def create_input_pipeline(**kwargs) -> Pipeline:
    """Return the deterministic, network-free E02b input pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.build_inputs,
                inputs=[
                    "retrieval_dataset@fixed_representation_bakeoff",
                    "e00_split_grouped_v2",
                    "e00_split_grouped_v2_manifest",
                    "e00_query_catalog",
                    "e00_query_candidate_state",
                    "e00_query_benchmark_manifest",
                    "params:fixed_representation_bakeoff",
                ],
                outputs=[
                    "e02b_pairs",
                    "e02b_exclusions",
                    "e02b_queries",
                    "e02b_query_states",
                    "e02b_input_manifest",
                ],
                name="build_e02b_inputs",
            )
        ]
    )


def create_dna_feature_pipeline(**kwargs) -> Pipeline:
    """Return one explicitly selected neural-DNA extraction pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.extract_dna_features,
                inputs=[
                    "e02b_pairs",
                    "e02b_input_manifest",
                    "e02_fixed_representation_invariance_manifest",
                    "params:fixed_representation_bakeoff",
                    "params:fixed_representation_bakeoff_feature_candidate",
                ],
                outputs=["e02b_dna_features", "e02b_dna_coverage", "e02b_dna_manifest"],
                name="extract_e02b_neural_dna_features",
            )
        ]
    )


def create_text_feature_pipeline(**kwargs) -> Pipeline:
    """Return one explicitly selected frozen-text extraction pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.extract_text_features,
                inputs=[
                    "e02b_pairs",
                    "e02b_queries",
                    "e02b_input_manifest",
                    "params:fixed_representation_bakeoff",
                    "params:fixed_representation_bakeoff_feature_candidate",
                ],
                outputs=["e02b_text_features", "e02b_text_manifest"],
                name="extract_e02b_text_features",
            )
        ]
    )


def create_tfidf_feature_pipeline(**kwargs) -> Pipeline:
    """Return the train-fitted deterministic DNA baseline pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.fit_tfidf_features,
                inputs=[
                    "e02b_pairs",
                    "e02b_input_manifest",
                    "params:fixed_representation_bakeoff",
                ],
                outputs=[
                    "e02b_dna_features",
                    "e02b_tfidf_vocabulary",
                    "e02b_tfidf_svd_state",
                    "e02b_dna_manifest",
                ],
                name="fit_e02b_tfidf_dna_features",
            )
        ]
    )


def create_alignment_pipeline(**kwargs) -> Pipeline:
    """Return the final validation-only factorial alignment pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.run_alignment,
                inputs=[
                    "e02b_pairs",
                    "e02b_queries",
                    "e02b_query_states",
                    "e02b_input_manifest",
                    "e02b_dna_features_tfidf_6mer_svd_512",
                    "e02b_dna_features_carbon_500m",
                    "e02b_dna_features_generanno_prokaryote_500m",
                    "e02b_dna_features_generator_v2_prokaryote_1_2b",
                    "e02b_dna_manifest_tfidf_6mer_svd_512",
                    "e02b_dna_manifest_carbon_500m",
                    "e02b_dna_manifest_generanno_prokaryote_500m",
                    "e02b_dna_manifest_generator_v2_prokaryote_1_2b",
                    "e02b_text_features_bge_base_en_v1_5",
                    "e02b_text_features_gte_modernbert_base",
                    "e02b_text_features_qwen3_embedding_0_6b",
                    "e02b_text_manifest_bge_base_en_v1_5",
                    "e02b_text_manifest_gte_modernbert_base",
                    "e02b_text_manifest_qwen3_embedding_0_6b",
                    "params:fixed_representation_bakeoff",
                ],
                outputs=[
                    "e02b_whitening_state",
                    "e02b_probe_checkpoints",
                    "e02b_training_history",
                    "e02b_paired_metrics",
                    "e02b_query_rankings",
                    "e02b_query_metrics",
                    "e02b_query_summaries",
                    "e02b_bootstrap_draws",
                    "e02b_selection_report",
                ],
                name="run_e02b_factorial_alignment",
            )
        ]
    )
