"""Reduced-population Gate 1 fixed-representation bake-off pipeline."""

from vec2vec.pipelines.fixed_representation_bakeoff.pipeline import (
    create_alignment_pipeline,
    create_dna_feature_pipeline,
    create_input_pipeline,
    create_text_feature_pipeline,
    create_tfidf_feature_pipeline,
)

__all__ = [
    "create_alignment_pipeline",
    "create_dna_feature_pipeline",
    "create_input_pipeline",
    "create_text_feature_pipeline",
    "create_tfidf_feature_pipeline",
]
