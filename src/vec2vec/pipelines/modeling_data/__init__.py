"""One DAG for the data consumed by the selected vec2vec model."""

from vec2vec.pipelines.modeling_data.pipeline import create_pipeline

__all__ = ["create_pipeline"]
