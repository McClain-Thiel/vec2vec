from __future__ import annotations

import numpy as np
import pandas as pd

from vec2vec.lib import set_supervision


def _partition_queries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "semantic_query_id": "atomic-a",
                "query_kind": "atomic",
                "controlled_split": "atomic_seen",
                "constraint_ids_json": '["a"]',
            },
            {
                "semantic_query_id": "pair-ab",
                "query_kind": "pair_conjunction",
                "controlled_split": "atoms_seen_conjunction_unseen",
                "constraint_ids_json": '["a","b"]',
            },
            {
                "semantic_query_id": "atomic-b",
                "query_kind": "atomic",
                "controlled_split": "atomic_seen",
                "constraint_ids_json": '["b"]',
            },
        ]
    )


def _partition_params() -> dict[str, object]:
    return {
        "training_query_kind": "atomic",
        "evaluation_query_kind": "pair_conjunction",
        "expected_training_queries": 2,
        "expected_evaluation_queries": 1,
        "expected_evaluation_controlled_split": "atoms_seen_conjunction_unseen",
    }


def test_unseen_composition_partition_excludes_pairs_from_training() -> None:
    training, evaluation, training_positions, evaluation_positions = (
        set_supervision._query_partitions(_partition_queries(), _partition_params())
    )

    assert training["semantic_query_id"].tolist() == ["atomic-a", "atomic-b"]
    assert evaluation["semantic_query_id"].tolist() == ["pair-ab"]
    assert training_positions.tolist() == [0, 2]
    assert evaluation_positions.tolist() == [1]


def test_unseen_composition_requires_every_pair_atom_in_training() -> None:
    queries = _partition_queries()
    queries.loc[1, "constraint_ids_json"] = '["a","unseen"]'

    with np.testing.assert_raises_regex(ValueError, "unseen atomic constraint"):
        set_supervision._query_partitions(queries, _partition_params())


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


def test_sum_atomic_queries_uses_symbolic_constraint_ids() -> None:
    atomic = pd.DataFrame(
        [
            {"constraint_ids_json": '["b"]'},
            {"constraint_ids_json": '["a"]'},
        ]
    )
    conjunctions = pd.DataFrame(
        [
            {"constraint_ids_json": '["a","b"]'},
            {"constraint_ids_json": '["b","a"]'},
        ]
    )
    vectors = np.asarray([[0.0, 2.0], [1.0, 0.0]], dtype=np.float32)

    result = set_supervision._sum_atomic_queries(atomic, conjunctions, vectors)

    assert np.array_equal(result, np.asarray([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32))


def test_additive_comparison_uses_paired_component_draws() -> None:
    summaries = pd.DataFrame(
        [
            {
                "objective": "verified_set",
                "query_representation": representation,
                "seed": seed,
                "query_kind": "pair_conjunction",
                "k": 10,
                "utility": utility,
            }
            for representation, utility in (("direct_text", 0.1), ("atomic_sum", 0.2))
            for seed in (13, 42, 20260818)
        ]
    )
    bootstrap = pd.DataFrame(
        [
            {
                "objective": "verified_set",
                "query_representation": representation,
                "query_kind": "pair_conjunction",
                "draw": draw,
                "utility": base + (0.1 if representation == "atomic_sum" else 0.0),
            }
            for draw, base in enumerate((0.08, 0.09, 0.10, 0.11, 0.12) * 20)
            for representation in ("direct_text", "atomic_sum")
        ]
    )
    gaps = pd.DataFrame({"jensen_shannon_divergence": [0.02, 0.04]})

    result = set_supervision._additive_comparison(summaries, bootstrap, gaps, {"primary_k": 10})

    assert np.isclose(result["atomic_sum_minus_direct_text"], 0.1)
    assert np.allclose(result["paired_component_bootstrap_95_interval"], [0.1, 0.1])
    assert np.isclose(result["mean_jensen_shannon_divergence"], 0.03)


def test_jensen_shannon_rows_is_zero_only_for_equal_distributions() -> None:
    left = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    right = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    result = set_supervision._jensen_shannon_rows(left, right)

    assert np.isclose(result[0], 0.0)
    assert result[1] > 0.0
