"""Gate 2 set-supervision pipeline definition."""

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.set_supervision.nodes import run_comparison


def create_pipeline(**kwargs) -> Pipeline:
    """Return the explicit paid Gate 2 comparison pipeline."""
    return Pipeline(
        [
            node(
                func=run_comparison,
                inputs=[
                    "e02b_pairs",
                    "e02b_queries",
                    "e02b_query_states",
                    "e00_query_candidate_state",
                    "e00_query_benchmark_manifest",
                    "e02b_input_manifest",
                    "e02b_dna_features_tfidf_6mer_svd_512",
                    "e02b_dna_manifest_tfidf_6mer_svd_512",
                    "e02b_text_features_qwen3_embedding_0_6b",
                    "e02b_text_manifest_qwen3_embedding_0_6b",
                    "params:set_supervision",
                ],
                outputs=[
                    "e03e04_whitening_state",
                    "e03e04_checkpoints",
                    "e03e04_training_history",
                    "e03e04_query_rankings",
                    "e03e04_query_metrics",
                    "e03e04_query_summaries",
                    "e03e04_bootstrap_draws",
                    "e03e04_report",
                ],
                name="compare_paired_identity_and_verified_set_supervision",
            )
        ]
    )
