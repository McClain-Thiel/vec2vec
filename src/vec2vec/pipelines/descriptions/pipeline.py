"""Records to quality-checked natural-language descriptions."""

from __future__ import annotations

from kedro.pipeline import Node, Pipeline, node

from vec2vec.pipelines.descriptions import nodes


def _qc_node() -> Node:
    return node(
        func=nodes.check_descriptions,
        inputs=[
            "plasmid_descriptions",
            "addgene_records@metadata",
            "addgene_annotation_features",
            "params:description_qc",
        ],
        outputs=["description_qc_report", "description_qc_flagged"],
        name="check_descriptions",
    )


def create_pipeline(**kwargs) -> Pipeline:
    """Generate descriptions, merge the partitions, and quality-check the result.

    Registered outside the default pipeline: generation calls a paid API once per
    plasmid. Run it explicitly, with ``params:descriptions.cost_cap_usd`` set.
    """
    return Pipeline(
        [
            node(
                func=nodes.generate_descriptions,
                inputs=[
                    "addgene_records@metadata",
                    "addgene_annotation_features",
                    "description_partitions_completed",
                    "params:descriptions",
                ],
                outputs="description_partitions",
                name="generate_descriptions",
            ),
            node(
                # Reads through the write-side catalog entry, which is what makes
                # the merge depend on generation having finished.
                func=nodes.merge_descriptions,
                inputs="description_partitions",
                outputs="plasmid_descriptions",
                name="merge_descriptions",
            ),
            _qc_node(),
        ]
    )


def create_import_pipeline(**kwargs) -> Pipeline:
    """Adopt already-published descriptions instead of paying to regenerate them.

    Use this to rebuild the retrieval dataset for free; run the ``descriptions``
    pipeline afterwards only for the plasmids the import did not cover.
    """
    return Pipeline(
        [
            node(
                func=nodes.import_published_descriptions,
                inputs=[
                    "published_paired_dataset",
                    "addgene_records@metadata",
                    "params:import_descriptions",
                ],
                outputs="plasmid_descriptions",
                name="import_published_descriptions",
            ),
            _qc_node(),
        ]
    )
