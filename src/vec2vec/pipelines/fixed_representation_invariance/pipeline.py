"""Pipeline definition for the Gate 1 full-panel DNA invariance check."""

from kedro.pipeline import Pipeline, node

from vec2vec.pipelines.fixed_representation_invariance import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Return the paid, explicitly named Gate 1 invariance pipeline."""
    return Pipeline(
        [
            node(
                func=nodes.run_invariance_check,
                inputs=[
                    "e02_fixed_representation_smoke_panel",
                    "e02_fixed_representation_smoke_panel_manifest",
                    "e02_fixed_representation_smoke_manifest",
                    "params:fixed_representation_smoke",
                    "params:fixed_representation_invariance",
                    "params:fixed_representation_invariance_candidate",
                ],
                outputs=[
                    "e02_fixed_representation_invariance_features",
                    "e02_fixed_representation_invariance_coverage",
                    "e02_fixed_representation_invariance_similarities",
                    "e02_fixed_representation_invariance_diagnostics",
                    "e02_fixed_representation_invariance_manifest",
                ],
                name="run_e02_fixed_representation_invariance_check",
            )
        ]
    )
