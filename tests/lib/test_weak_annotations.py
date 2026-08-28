from __future__ import annotations

import numpy as np
import pandas as pd

from vec2vec.lib import weak_annotations


def _params() -> dict:
    return {
        "protocol_version": "test-v1",
        "inputs": {"training_rows": 8, "validation_rows": 6},
        "vocabulary": {
            "atoms": 3,
            "minimum_normalized_characters": 3,
            "maximum_normalized_characters": 64,
            "maximum_words": 8,
            "minimum_train_rows": 2,
            "maximum_train_rows": 7,
            "minimum_train_components": 1,
            "minimum_validation_rows": 1,
            "minimum_validation_components": 1,
            "maximum_atom_jaccard": 0.95,
        },
        "conjunctions": {
            "pairs": 2,
            "minimum_train_rows": 2,
            "minimum_train_components": 1,
            "minimum_validation_rows": 1,
            "minimum_validation_components": 1,
            "maximum_atom_jaccard": 0.80,
            "maximum_pairs_per_atom": 2,
        },
        "weak_negatives": {
            "maximum_ratio_per_positive": 1.0,
            "maximum_rows_per_atom": 4,
            "selection_salt": "test-negative-sample",
        },
    }


def test_build_weak_annotation_benchmark_is_complete_and_deterministic() -> None:
    sequence_ids = [
        *(f"train_{index}" for index in range(8)),
        *(f"val_{index}" for index in range(6)),
    ]
    roles = [*("alignment_train" for _ in range(8)), *("validation_gallery" for _ in range(6))]
    pairs = pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "sequence_sha256": [f"hash_{index}" for index in range(14)],
            "panel_role": roles,
            "leakage_component_v2": [f"component_{index}" for index in range(14)],
        }
    )
    positive_positions = {
        "Alpha-feature": {0, 1, 2, 3, 8, 9, 10},
        "Beta feature": {2, 3, 4, 5, 9, 10, 11},
        "Gamma_feature": {1, 3, 5, 6, 10, 11, 12},
    }
    annotations = []
    for position, sequence_id in enumerate(sequence_ids):
        calls = [name for name, positives in positive_positions.items() if position in positives]
        annotations.append(
            {
                "sequence_id": sequence_id,
                "sequence_sha256": f"hash_{position}",
                "annotations": calls,
            }
        )
    annotation_rows = pd.DataFrame(annotations)

    first = weak_annotations.build_weak_annotation_benchmark(pairs, annotation_rows, _params())
    second = weak_annotations.build_weak_annotation_benchmark(
        pairs.iloc[::-1], annotation_rows.iloc[::-1], _params()
    )

    pd.testing.assert_frame_equal(first.queries, second.queries)
    np.testing.assert_array_equal(first.train_verified, second.train_verified)
    np.testing.assert_array_equal(first.train_known, second.train_known)
    assert first.queries["query_kind"].value_counts().to_dict() == {
        "atomic": 3,
        "pair_conjunction": 2,
    }
    assert first.train_verified.shape == (3, 8)
    assert first.train_known.sum(axis=1).tolist() == [8, 8, 8]
    assert first.validation_verified.shape == (5, 6)


def test_retrieval_metrics_and_paired_bootstrap_use_weak_binary_labels() -> None:
    queries = pd.DataFrame(
        {
            "query_id": ["a", "b"],
            "query_kind": ["pair_conjunction", "pair_conjunction"],
        }
    )
    positives = np.asarray([[True, False, True], [False, True, False]])
    direct = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    additive = np.asarray([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]])
    metrics = []
    for seed in (13, 42):
        metrics.append(
            weak_annotations.retrieval_metrics(
                direct,
                queries,
                positives,
                seed=seed,
                representation="direct_text",
                cutoffs=(1, 2),
            )
        )
        metrics.append(
            weak_annotations.retrieval_metrics(
                additive,
                queries,
                positives,
                seed=seed,
                representation="atomic_sum",
                cutoffs=(1, 2),
            )
        )
    table = pd.concat(metrics, ignore_index=True)
    comparison = weak_annotations.paired_query_bootstrap(table, k=1, draws=100, seed=7)

    assert comparison["direct_text"] == 0.0
    assert comparison["atomic_sum"] == 1.0
    assert comparison["atomic_sum_minus_direct_text"] == 1.0
