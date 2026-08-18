"""Pipeline definition for the Gate 1 DNA numerical smoke check."""

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.fixed_representation_smoke import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Return the paid, explicitly named Gate 1 smoke pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.build_smoke_panel,
                inputs=[
                    "retrieval_dataset@fixed_representation_smoke",
                    "e00_split_grouped_v2",
                    "params:fixed_representation_smoke",
                ],
                outputs=[
                    "e02_fixed_representation_smoke_panel",
                    "e02_fixed_representation_smoke_panel_manifest",
                ],
                name="build_e02_fixed_representation_smoke_panel",
            ),
            node(
                func=nodes.run_numerical_smoke,
                inputs=[
                    "e02_fixed_representation_smoke_panel",
                    "e02_fixed_representation_smoke_panel_manifest",
                    "params:fixed_representation_smoke",
                    "params:fixed_representation_smoke_candidate",
                ],
                outputs=[
                    "e02_fixed_representation_smoke_features",
                    "e02_fixed_representation_smoke_coverage",
                    "e02_fixed_representation_smoke_diagnostics",
                    "e02_fixed_representation_smoke_manifest",
                ],
                name="run_e02_fixed_representation_numerical_smoke",
            ),
        ]
    )
