from __future__ import annotations

from vec2vec.pipelines.fixed_representation_bakeoff import pipeline


def test_alignment_pipeline_declares_complete_factorial_artifacts() -> None:
    graph = pipeline.create_alignment_pipeline()

    assert len(graph.nodes) == 1
    assert graph.inputs() == {
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
    }
    assert graph.outputs() == {
        "e02b_whitening_state",
        "e02b_probe_checkpoints",
        "e02b_training_history",
        "e02b_paired_metrics",
        "e02b_query_rankings",
        "e02b_query_metrics",
        "e02b_query_summaries",
        "e02b_bootstrap_draws",
        "e02b_selection_report",
    }
