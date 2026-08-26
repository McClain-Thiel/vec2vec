"""Project pipelines."""

from __future__ import annotations

from kedro.pipeline import Pipeline

from vec2vec.pipelines import audit, dataset, descriptions, modeling_data, processing


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    ``__default__`` covers everything that can be re-run for free from data
    already in the lake. Run the rest by name:

    - ``descriptions`` calls a paid API once per plasmid.
    - ``import_descriptions`` adopts already-published descriptions instead.
    - ``modeling_data`` is the complete, tagged DAG for the selected model inputs.
    """
    pipelines = {
        "processing": processing.create_pipeline(),
        "descriptions": descriptions.create_pipeline(),
        "import_descriptions": descriptions.create_import_pipeline(),
        "dataset": dataset.create_pipeline(),
        "audit": audit.create_pipeline(),
        "modeling_data": modeling_data.create_pipeline(),
    }
    pipelines["__default__"] = pipelines["processing"] + pipelines["dataset"] + pipelines["audit"]
    return pipelines
