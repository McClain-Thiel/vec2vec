from __future__ import annotations

import numpy as np
import pandas as pd

from vec2vec.lib import set_supervision


def test_comparison_requires_practical_and_interval_improvement() -> None:
    summaries = pd.DataFrame(
        [
            {
                "objective": objective,
                "seed": seed,
                "query_kind": "pair_conjunction",
                "k": 10,
                "utility": utility,
            }
            for objective, utility in (("paired_identity", 0.1), ("verified_set", 0.15))
            for seed in (13, 42, 20260818)
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "objective": objective,
                "query_kind": "pair_conjunction",
                "draw": draw,
                "utility": base + (0.05 if objective == "verified_set" else 0.0),
            }
            for draw, base in enumerate((0.08, 0.09, 0.10, 0.11, 0.12) * 20)
            for objective in ("paired_identity", "verified_set")
        ]
    )

    result = set_supervision._comparison(
        summaries,
        bootstrap,
        {"primary_k": 10, "minimum_practical_improvement": 0.01},
    )

    assert np.isclose(result["verified_set_minus_paired_identity"], 0.05)
    assert np.allclose(result["paired_component_bootstrap_95_interval"], [0.05, 0.05])
    assert result["supports_set_supervision"] is True


def test_pipeline_is_not_in_default_registry() -> None:
    from vec2vec.pipeline_registry import register_pipelines

    pipelines = register_pipelines()
    assert "set_supervision" in pipelines
    assert not set(pipelines["set_supervision"].nodes).intersection(pipelines["__default__"].nodes)
