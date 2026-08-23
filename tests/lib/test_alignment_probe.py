from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vec2vec.lib import alignment_probe


def test_whitening_is_train_fitted_and_finite() -> None:
    train = np.asarray(
        [[1.0, 2.0, 4.0], [2.0, 1.0, 3.0], [4.0, 3.0, 1.0], [3.0, 4.0, 2.0]],
        dtype=np.float32,
    )
    whitening = alignment_probe.Whitening.fit(train, epsilon=1e-6)
    transformed = whitening.transform(train)

    assert transformed.shape == train.shape
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-5)
    assert np.isfinite(transformed).all()


def test_paired_metrics_accept_duplicate_sequence_or_description_positives() -> None:
    sequence = np.eye(3, dtype=np.float32)
    text = sequence[[1, 0, 2]]

    metrics = alignment_probe.paired_retrieval_metrics(
        sequence,
        text,
        np.asarray(["same-sequence", "same-sequence", "third"]),
        np.asarray(["first", "second", "third"]),
    )

    assert metrics["sequence_to_description_r1"] == 1.0
    assert metrics["description_to_sequence_r1"] == 1.0


def test_query_metrics_keep_verified_contradicted_and_unknown_separate() -> None:
    queries = pd.DataFrame([{"query_id": "q1", "semantic_query_id": "s1", "query_kind": "atomic"}])
    gallery = pd.DataFrame(
        [
            {
                "sequence_id": f"p{index}",
                "similarity_component_primary": f"c{index}",
                "length_bp": 100 + index,
                "component_size": 1,
            }
            for index in range(4)
        ]
    )
    states = pd.DataFrame(
        [
            {"semantic_query_id": "s1", "sequence_id": "p0", "state": "verified"},
            {"semantic_query_id": "s1", "sequence_id": "p1", "state": "contradicted"},
        ]
    )
    query_vectors = np.asarray([[1.0, 0.0]], dtype=np.float32)
    gallery_vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)

    rankings, metrics, scores = alignment_probe.query_rankings_and_metrics(
        query_vectors,
        gallery_vectors,
        queries,
        gallery,
        states,
        cutoffs=(1, 2, 4),
    )

    at_four = metrics.query("k == 4").iloc[0]
    assert rankings["state"].tolist() == ["verified", "contradicted", "unknown", "unknown"]
    assert at_four["verified_fraction"] == 0.25
    assert at_four["contradicted_fraction"] == 0.25
    assert at_four["unknown_fraction"] == 0.5
    assert at_four["utility"] == 0.0
    assert scores.shape == (1, 4)


def test_component_bootstrap_resamples_complete_components() -> None:
    queries = pd.DataFrame([{"query_id": "q1", "semantic_query_id": "s1", "query_kind": "atomic"}])
    gallery = pd.DataFrame(
        {
            "sequence_id": ["p0", "p1", "p2", "p3"],
            "similarity_component_primary": ["a", "a", "b", "b"],
        }
    )
    states = pd.DataFrame(
        [
            {"semantic_query_id": "s1", "sequence_id": "p0", "state": "verified"},
            {"semantic_query_id": "s1", "sequence_id": "p1", "state": "verified"},
            {"semantic_query_id": "s1", "sequence_id": "p2", "state": "contradicted"},
            {"semantic_query_id": "s1", "sequence_id": "p3", "state": "contradicted"},
        ]
    )
    scores = [np.asarray([[4.0, 3.0, 2.0, 1.0]], dtype=np.float32)]

    lower, upper = alignment_probe.whole_component_bootstrap_utility(
        scores,
        queries,
        gallery,
        states,
        k=2,
        draws=200,
        seed=42,
    )

    assert lower == pytest.approx(-1.0)
    assert upper == pytest.approx(1.0)

    draws = alignment_probe.whole_component_bootstrap_draws(
        scores,
        queries,
        gallery,
        states,
        k=2,
        draws=200,
        seed=42,
    )
    assert len(draws) == 400
    assert set(draws["query_kind"]) == {"atomic", "combined"}
    assert draws.groupby("query_kind")["draw"].nunique().eq(200).all()


def test_paired_metrics_reject_missing_positive_group_identifiers() -> None:
    vectors = np.eye(2, dtype=np.float32)

    with pytest.raises(ValueError, match="must not be missing"):
        alignment_probe.paired_retrieval_metrics(
            vectors,
            vectors,
            np.asarray(["first", None], dtype=object),
            np.asarray(["first", "second"], dtype=object),
        )


def test_expired_authorized_deadline_fails_before_work() -> None:
    with pytest.raises(TimeoutError, match="authorized compute deadline"):
        alignment_probe._ensure_before_deadline(0.0, operation="test operation")


def test_symmetric_many_positive_loss_rewards_either_known_positive() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[3.0, 2.0], [2.0, 3.0]])
    identity = torch.eye(2, dtype=torch.bool)
    all_positive = torch.ones((2, 2), dtype=torch.bool)

    identity_loss = alignment_probe.symmetric_many_positive_loss(logits, identity)
    multi_positive_loss = alignment_probe.symmetric_many_positive_loss(logits, all_positive)

    assert float(multi_positive_loss) < float(identity_loss)
