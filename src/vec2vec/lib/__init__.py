"""Pure, framework-free implementations of the data pipeline's logic.

Nothing in this package imports Kedro or performs I/O: the pipelines under
``vec2vec.pipelines`` compose these functions, and the catalog decides where
their inputs and outputs live.
"""
