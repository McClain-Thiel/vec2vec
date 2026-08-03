"""Tests for leakage-aware split assignment."""

from __future__ import annotations

import numpy as np
import pytest

from vec2vec.lib import splits


def test_family_key_prefers_backbone_then_name_then_id():
    assert splits.family_key("pUC19", "pABC_1", "addgene_1") == "backbone::pUC19"
    assert splits.family_key(None, "pLKO_1", "addgene_1") == "name::pLKO"
    assert splits.family_key("  ", "pLKO_1", "addgene_1") == "name::pLKO"
    assert splits.family_key("unknown", "pLKO_1", "addgene_1") == "name::pLKO"
    # Too short to be a family: the row stands alone.
    assert splits.family_key(None, "p1", "addgene_1") == "id::addgene_1"
    assert splits.family_key(None, None, "addgene_1") == "id::addgene_1"


def test_components_union_across_every_key_column():
    families = ["a", "a", "b", "c"]
    sequences = ["s1", "s2", "s2", "s3"]
    components = splits.leakage_components(families, sequences)
    # Rows 0-2 chain together: 0~1 by family, 1~2 by sequence. Row 3 is alone.
    assert components[0] == components[1] == components[2]
    assert components[3] != components[0]


def test_components_are_deterministic_and_order_stable():
    families = ["a", "b", "a", "c"]
    sequences = ["s1", "s2", "s3", "s4"]
    first = splits.leakage_components(families, sequences)
    second = splits.leakage_components(families, sequences)
    assert np.array_equal(first, second)


def test_grouped_split_never_straddles_a_component():
    rng = np.random.default_rng(0)
    families = [f"fam{index % 40}" for index in range(600)]
    sequences = [f"seq{int(value)}" for value in rng.integers(0, 500, size=600)]
    components = splits.leakage_components(families, sequences)
    labels = splits.assign_grouped_split(components, splits.SplitFractions(), seed=42)
    assert splits.split_purity(components, labels) == 0
    assert set(labels) <= set(splits.SPLIT_LABELS)


def test_grouped_split_is_reproducible_for_a_seed():
    components = splits.leakage_components([f"fam{index % 20}" for index in range(200)])
    first = splits.assign_grouped_split(components, splits.SplitFractions(), seed=7)
    second = splits.assign_grouped_split(components, splits.SplitFractions(), seed=7)
    other = splits.assign_grouped_split(components, splits.SplitFractions(), seed=8)
    assert list(first) == list(second)
    assert list(first) != list(other)


def test_random_split_hits_its_target_proportions():
    labels = splits.assign_random_split(1000, splits.SplitFractions(0.8, 0.1), seed=1)
    counts = {label: int((labels == label).sum()) for label in splits.SPLIT_LABELS}
    assert counts == {"train": 800, "val": 100, "test": 100}


def test_split_fractions_reject_impossible_targets():
    with pytest.raises(ValueError):
        splits.SplitFractions(train=0.95, val=0.1)
    with pytest.raises(ValueError):
        splits.SplitFractions(train=0.0, val=0.1)


def test_leakage_components_reject_ragged_input():
    with pytest.raises(ValueError):
        splits.leakage_components(["a", "b"], ["s1"])
