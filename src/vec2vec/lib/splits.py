"""Leakage-aware train/validation/test assignment.

Most Addgene plasmids share a backbone with many near-siblings, and some rows
share a byte-identical sequence. A naive random split therefore scatters nearly
identical scaffolds across train and test and inflates retrieval scores.

The grouped split fixes this in one pass: rows are unioned into components by
*both* family key and exact-sequence hash, and whole components are assigned to a
split together. Doing it once removes the assign-then-repair step the upstream
pipeline needed, and makes "no family or exact sequence crosses a split" true by
construction rather than by later verification.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

SPLIT_LABELS = ("train", "val", "test")

_NAME_PREFIX = re.compile(r"[_\d]")


@dataclass(frozen=True)
class SplitFractions:
    """Target proportion of rows in each split."""

    train: float = 0.8
    val: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.train < 1.0 or not 0.0 <= self.val < 1.0:
            raise ValueError("train must be in (0, 1) and val in [0, 1)")
        if self.train + self.val >= 1.0:
            raise ValueError("train + val must leave room for a test split")

    @property
    def test(self) -> float:
        return 1.0 - self.train - self.val


def _clean(value: Any) -> str:
    """Render a possibly-missing metadata value as trimmed text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def family_key(backbone: Any, name: Any, fallback_id: str) -> str:
    """Cluster key for grouped splitting.

    Prefers the declared backbone, falls back to the leading alphabetic run of
    the plasmid name (which captures vector families such as ``pLKO``), and
    finally isolates the row under its own identifier.
    """
    cleaned_backbone = _clean(backbone)
    if cleaned_backbone and cleaned_backbone.casefold() != "unknown":
        return f"backbone::{cleaned_backbone}"
    prefix = _NAME_PREFIX.split(_clean(name))[0]
    if len(prefix) >= 3:
        return f"name::{prefix}"
    return f"id::{fallback_id}"


class _UnionFind:
    """Disjoint-set forest with path compression and union by rank."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self._rank[left_root] < self._rank[right_root]:
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        if self._rank[left_root] == self._rank[right_root]:
            self._rank[left_root] += 1


def leakage_components(*key_columns: Sequence[str]) -> np.ndarray:
    """Union rows that share any key in any of *key_columns*.

    Returns:
        A component label per row. Labels are the index of each component's
        lowest-numbered member, so they are deterministic for a given input.
    """
    if not key_columns:
        raise ValueError("at least one key column is required")
    size = len(key_columns[0])
    if any(len(column) != size for column in key_columns):
        raise ValueError("key columns must all have the same length")

    union = _UnionFind(size)
    for column in key_columns:
        first_seen: dict[str, int] = {}
        for position, key in enumerate(column):
            union.union(position, first_seen.setdefault(key, position))
    return np.array([union.find(position) for position in range(size)], dtype=np.int64)


def assign_grouped_split(
    components: np.ndarray,
    fractions: SplitFractions,
    seed: int,
) -> np.ndarray:
    """Assign whole components to splits, balancing by cumulative row count."""
    size = len(components)
    members: dict[int, list[int]] = {}
    for position, component in enumerate(components.tolist()):
        members.setdefault(component, []).append(position)

    order = np.array(sorted(members), dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)

    train_target = int(size * fractions.train)
    val_target = train_target + int(size * fractions.val)
    labels = np.empty(size, dtype=object)
    placed = 0
    for component in order.tolist():
        rows = members[component]
        label = "train" if placed < train_target else ("val" if placed < val_target else "test")
        for position in rows:
            labels[position] = label
        placed += len(rows)
    return labels


def assign_random_split(size: int, fractions: SplitFractions, seed: int) -> np.ndarray:
    """Assign individual rows to splits uniformly at random.

    Provided as an explicit baseline: the gap between this and the grouped split
    quantifies how much of a retrieval score is scaffold memorization.
    """
    ranks = np.empty(size, dtype=np.int64)
    ranks[np.random.default_rng(seed).permutation(size)] = np.arange(size)
    train_target = int(size * fractions.train)
    val_target = train_target + int(size * fractions.val)
    return np.where(
        ranks < train_target, "train", np.where(ranks < val_target, "val", "test")
    ).astype(object)


def split_purity(components: np.ndarray, labels: np.ndarray) -> int:
    """Return the number of components whose rows landed in more than one split."""
    by_component: dict[int, set[str]] = {}
    for component, label in zip(components.tolist(), labels.tolist(), strict=True):
        by_component.setdefault(component, set()).add(str(label))
    return sum(1 for splits in by_component.values() if len(splits) > 1)
