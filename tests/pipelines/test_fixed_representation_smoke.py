from __future__ import annotations

import numpy as np
import pandas as pd

from vec2vec.pipelines.fixed_representation_smoke import nodes


def test_numerical_diagnostics_reports_precision_and_coverage_failures() -> None:
    features = pd.DataFrame(
        [
            {
                "sequence_id": "s1",
                "candidate_id": "candidate",
                "precision": "bfloat16",
                "length_bp": 12,
                "length_decile": 0,
                "embedding_dimension": 2,
                "embedding": [1.0, 0.0],
            },
            {
                "sequence_id": "s1",
                "candidate_id": "candidate",
                "precision": "float32",
                "length_bp": 12,
                "length_decile": 0,
                "embedding_dimension": 2,
                "embedding": [0.98, np.sqrt(1.0 - 0.98**2)],
            },
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "sequence_id": "s1",
                "precision": precision,
                "newly_covered_base_count": 12,
                "sequence_length_bp": 12,
                "out_of_vocabulary_token_count": 0,
                "window_index": 0,
            }
            for precision in ("bfloat16", "float32")
        ]
    )

    diagnostics = nodes._numerical_diagnostics(features, coverage, minimum_cosine=0.99)

    assert diagnostics.loc[0, "bfloat16_float32_cosine"] == 0.98
    assert bool(diagnostics.loc[0, "coverage_pass"])
    assert not bool(diagnostics.loc[0, "passed_numerical_smoke"])
