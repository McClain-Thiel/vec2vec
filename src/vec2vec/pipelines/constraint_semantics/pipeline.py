"""Pipeline for reproducible E00 data profiling."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.constraint_semantics import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Profile constraint fields, split components, and pLannotate coverage."""
    return Pipeline(
        [
            node(
                func=nodes.profile_constraint_values,
                inputs=["retrieval_dataset@semantics", "params:constraint_semantics"],
                outputs=["e00_constraint_field_profile", "e00_constraint_value_profile"],
                name="profile_e00_constraint_values",
            ),
            node(
                func=nodes.profile_components,
                inputs="retrieval_dataset@semantics",
                outputs=["e00_split_component_profile", "e00_split_profile"],
                name="profile_e00_split_components",
            ),
            node(
                func=nodes.profile_primary_annotations,
                inputs=[
                    "retrieval_dataset@semantics",
                    "addgene_annotations@plannotate",
                    "params:constraint_semantics",
                ],
                outputs="e00_plannotate_profile",
                name="profile_e00_plannotate",
            ),
        ]
    )
