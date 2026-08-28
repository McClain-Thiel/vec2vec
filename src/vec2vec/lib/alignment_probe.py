"""Fixed linear alignment probe and validation-only retrieval metrics."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Whitening:
    """A train-fitted transform ``(x - mean) @ matrix`` with no removed components."""

    mean: np.ndarray
    matrix: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, *, epsilon: float) -> Whitening:
        """Fit full-rank principal-component whitening on training rows only."""
        matrix = _finite_matrix(values, name="whitening inputs", minimum_rows=2)
        if epsilon <= 0.0:
            raise ValueError("whitening epsilon must be positive")
        mean = matrix.mean(axis=0)
        _, singular_values, directions = np.linalg.svd(
            matrix.astype(np.float64) - mean,
            full_matrices=False,
        )
        standard_deviations = singular_values / np.sqrt(len(matrix) - 1)
        transform = directions.T / (standard_deviations + epsilon)
        return cls(mean.astype(np.float32), transform.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the frozen transform and return finite float32 values."""
        matrix = _finite_matrix(values, name="whitening transform inputs")
        if matrix.shape[1] != len(self.mean):
            raise ValueError("whitening input dimension changed")
        result = (matrix - self.mean) @ self.matrix
        if not np.isfinite(result).all():
            raise ValueError("whitening produced a non-finite value")
        return result.astype(np.float32)


def train_alignment_probe(
    sequence_train: np.ndarray,
    text_train: np.ndarray,
    sequence_groups: np.ndarray,
    description_groups: np.ndarray,
    *,
    seed: int,
    projection_dimension: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    initial_temperature: float,
    maximum_logit_scale: float,
    device: str,
    deadline_monotonic: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the final-epoch symmetric many-positive linear probe."""
    import torch
    import torch.nn.functional as functional

    _ensure_before_deadline(deadline_monotonic, operation="alignment probe setup")
    sequence = _finite_matrix(sequence_train, name="sequence training features")
    text = _finite_matrix(text_train, name="text training features")
    if len(sequence) != len(text) or len(sequence) != len(sequence_groups):
        raise ValueError("training features and positive-group identifiers must align")
    if len(description_groups) != len(sequence):
        raise ValueError("description positive-group identifiers must align")
    if len(sequence) < batch_size:
        raise ValueError("training rows must contain one complete effective batch")
    if projection_dimension < 1 or epochs < 1 or batch_size < 2:
        raise ValueError("projection dimension, epochs, and batch size must be valid")
    if initial_temperature <= 0.0 or maximum_logit_scale <= 0.0:
        raise ValueError("temperature and logit-scale cap must be positive")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.device(device)
    sequence_tensor = torch.as_tensor(sequence, device=target)
    text_tensor = torch.as_tensor(text, device=target)
    sequence_group_tensor = torch.as_tensor(
        _factorize_groups(sequence_groups), dtype=torch.int64, device=target
    )
    description_group_tensor = torch.as_tensor(
        _factorize_groups(description_groups), dtype=torch.int64, device=target
    )
    sequence_head = torch.nn.Linear(sequence.shape[1], projection_dimension, bias=False).to(target)
    text_head = torch.nn.Linear(text.shape[1], projection_dimension, bias=False).to(target)
    logit_scale = torch.nn.Parameter(
        torch.tensor(float(np.log(1.0 / initial_temperature)), device=target)
    )
    optimizer = torch.optim.AdamW(
        [*sequence_head.parameters(), *text_head.parameters(), logit_scale],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float | int]] = []
    batches_per_epoch = int(np.ceil(len(sequence) / batch_size))
    for epoch in range(epochs):
        _ensure_before_deadline(deadline_monotonic, operation=f"alignment epoch {epoch + 1}")
        permutation = torch.randperm(len(sequence), device=target)
        epoch_loss = 0.0
        for batch_index in range(batches_per_epoch):
            _ensure_before_deadline(
                deadline_monotonic,
                operation=f"alignment epoch {epoch + 1} batch {batch_index + 1}",
            )
            indices = permutation[batch_index * batch_size : (batch_index + 1) * batch_size]
            projected_sequence = functional.normalize(
                sequence_head(sequence_tensor[indices]), dim=1
            )
            projected_text = functional.normalize(text_head(text_tensor[indices]), dim=1)
            scale = logit_scale.exp().clamp(max=maximum_logit_scale)
            logits = scale * projected_sequence @ projected_text.T
            sequence_ids = sequence_group_tensor[indices]
            description_ids = description_group_tensor[indices]
            positives = (sequence_ids[:, None] == sequence_ids[None, :]) | (
                description_ids[:, None] == description_ids[None, :]
            )
            loss = symmetric_many_positive_loss(logits, positives)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": epoch_loss / batches_per_epoch,
                "logit_scale": float(logit_scale.detach().exp().clamp(max=maximum_logit_scale)),
            }
        )
    state = {
        "seed": seed,
        "sequence_head": sequence_head.weight.detach().cpu().numpy().astype(np.float32),
        "text_head": text_head.weight.detach().cpu().numpy().astype(np.float32),
        "logit_scale": float(logit_scale.detach().cpu()),
        "batches_per_epoch": batches_per_epoch,
        "last_batch_rows": int(len(sequence) % batch_size or batch_size),
        "dropped_rows_per_epoch": 0,
    }
    return state, pd.DataFrame(history)


def train_controlled_query_probe(
    sequence_train: np.ndarray,
    query_train: np.ndarray,
    verified_mask: np.ndarray,
    *,
    objective: str,
    seed: int,
    projection_dimension: int,
    updates: int,
    learning_rate: float,
    weight_decay: float,
    initial_temperature: float,
    maximum_logit_scale: float,
    device: str,
    deadline_monotonic: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit one controlled-query probe on deterministic verified-candidate batches."""
    import torch
    import torch.nn.functional as functional

    if objective not in {"paired_identity", "verified_set"}:
        raise ValueError(f"unknown controlled-query objective: {objective}")
    sequence = _finite_matrix(sequence_train, name="controlled-query sequence features")
    queries = _finite_matrix(query_train, name="controlled-query text features")
    labels = np.asarray(verified_mask)
    if labels.dtype != np.bool_ or labels.shape != (len(queries), len(sequence)):
        raise ValueError("verified mask must be boolean with query-by-sequence shape")
    if not labels.any(axis=1).all():
        raise ValueError("every training query must have at least one verified sequence")
    if projection_dimension < 1 or updates < 1:
        raise ValueError("projection dimension and update budget must be positive")
    if initial_temperature <= 0.0 or maximum_logit_scale <= 0.0:
        raise ValueError("temperature and logit-scale cap must be positive")

    _ensure_before_deadline(deadline_monotonic, operation="controlled-query probe setup")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.device(device)
    sequence_tensor = torch.as_tensor(sequence, device=target)
    query_tensor = torch.as_tensor(queries, device=target)
    sequence_head = torch.nn.Linear(sequence.shape[1], projection_dimension, bias=False).to(target)
    text_head = torch.nn.Linear(queries.shape[1], projection_dimension, bias=False).to(target)
    logit_scale = torch.nn.Parameter(
        torch.tensor(float(np.log(1.0 / initial_temperature)), device=target)
    )
    initial_sequence_head_sha256 = _tensor_sha256(sequence_head.weight)
    initial_text_head_sha256 = _tensor_sha256(text_head.weight)
    optimizer = torch.optim.AdamW(
        [*sequence_head.parameters(), *text_head.parameters(), logit_scale],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    generator = np.random.default_rng(seed)
    positive_indices = [np.flatnonzero(row) for row in labels]
    identity_mask = torch.eye(len(queries), dtype=torch.bool, device=target)
    sampler_hash = hashlib.sha256()
    history: list[dict[str, float | int]] = []
    for update in range(1, updates + 1):
        _ensure_before_deadline(deadline_monotonic, operation=f"controlled-query update {update}")
        selected = np.asarray(
            [values[generator.integers(len(values))] for values in positive_indices],
            dtype=np.int64,
        )
        sampler_hash.update(selected.tobytes())
        true_positives = labels[:, selected]
        if not np.diag(true_positives).all():
            raise RuntimeError("controlled-query sampler selected a non-verified diagonal pair")
        selected_tensor = torch.as_tensor(selected, dtype=torch.int64, device=target)
        projected_sequence = functional.normalize(
            sequence_head(sequence_tensor[selected_tensor]), dim=1
        )
        projected_query = functional.normalize(text_head(query_tensor), dim=1)
        scale = logit_scale.exp().clamp(max=maximum_logit_scale)
        logits = scale * projected_query @ projected_sequence.T
        positive_mask = (
            identity_mask
            if objective == "paired_identity"
            else torch.as_tensor(true_positives, dtype=torch.bool, device=target)
        )
        loss = symmetric_many_positive_loss(logits, positive_mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(
            {
                "update": update,
                "loss": float(loss.detach().cpu()),
                "logit_scale": float(logit_scale.detach().exp().clamp(max=maximum_logit_scale)),
                "true_positive_pairs": int(true_positives.sum()),
                "unique_candidate_rows": int(len(np.unique(selected))),
            }
        )

    state = {
        "objective": objective,
        "seed": seed,
        "sequence_head": sequence_head.weight.detach().cpu().numpy().astype(np.float32),
        "text_head": text_head.weight.detach().cpu().numpy().astype(np.float32),
        "logit_scale": float(logit_scale.detach().cpu()),
        "updates": updates,
        "batch_rows": len(queries),
        "sampler_sha256": sampler_hash.hexdigest(),
        "initial_sequence_head_sha256": initial_sequence_head_sha256,
        "initial_text_head_sha256": initial_text_head_sha256,
    }
    return state, pd.DataFrame(history)


def train_maximum_entropy_probe(
    sequence_train: np.ndarray,
    query_train: np.ndarray,
    verified_mask: np.ndarray,
    known_mask: np.ndarray,
    log_base_mass: np.ndarray,
    *,
    base_measure: str,
    seed: int,
    projection_dimension: int,
    updates: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    device: str,
    record_initial: bool = False,
    deadline_monotonic: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit unnormalized projections against the exact known candidate universe."""
    import torch

    sequence = _finite_matrix(sequence_train, name="maximum-entropy sequence features")
    queries = _finite_matrix(query_train, name="maximum-entropy query features")
    verified = np.asarray(verified_mask)
    known = np.asarray(known_mask)
    if verified.dtype != np.bool_ or known.dtype != np.bool_:
        raise ValueError("maximum-entropy state masks must be boolean")
    if verified.shape != (len(queries), len(sequence)) or known.shape != verified.shape:
        raise ValueError("maximum-entropy state masks must align with queries and sequences")
    if np.any(verified & ~known):
        raise ValueError("verified candidates must be included in the known universe")
    contradicted = known & ~verified
    if not verified.any(axis=1).all() or not contradicted.any(axis=1).all():
        raise ValueError("every maximum-entropy query needs verified and contradicted candidates")
    log_mass = np.asarray(log_base_mass, dtype=np.float64)
    if log_mass.shape != (len(sequence),) or not np.isfinite(log_mass).all():
        raise ValueError("log base mass must be one finite value per training sequence")
    if not np.isclose(np.exp(log_mass).sum(), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("base measure must normalize over the active training population")
    if projection_dimension < 1 or updates < 1 or temperature <= 0.0:
        raise ValueError("projection dimension, updates, and temperature must be positive")

    _ensure_before_deadline(deadline_monotonic, operation="maximum-entropy probe setup")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.device(device)
    sequence_tensor = torch.as_tensor(sequence, device=target)
    query_tensor = torch.as_tensor(queries, device=target)
    verified_tensor = torch.as_tensor(verified, device=target)
    known_tensor = torch.as_tensor(known, device=target)
    log_mass_tensor = torch.as_tensor(log_mass, dtype=torch.float32, device=target)
    sequence_head = torch.nn.Linear(sequence.shape[1], projection_dimension, bias=False).to(target)
    text_head = torch.nn.Linear(queries.shape[1], projection_dimension, bias=False).to(target)
    torch.nn.init.normal_(
        sequence_head.weight,
        std=1.0 / np.sqrt(sequence.shape[1] * projection_dimension),
    )
    torch.nn.init.normal_(
        text_head.weight,
        std=1.0 / np.sqrt(queries.shape[1] * projection_dimension),
    )
    initial_sequence_head_sha256 = _tensor_sha256(sequence_head.weight)
    initial_text_head_sha256 = _tensor_sha256(text_head.weight)
    optimizer = torch.optim.AdamW(
        [*sequence_head.parameters(), *text_head.parameters()],
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    negative_infinity = torch.finfo(log_mass_tensor.dtype).min
    target_logits = log_mass_tensor.unsqueeze(0).expand_as(verified_tensor)
    target_probability = torch.softmax(
        target_logits.masked_fill(~verified_tensor, negative_infinity), dim=1
    )
    history: list[dict[str, float | int]] = []
    norm_sample = sequence_tensor[: min(len(sequence_tensor), 2048)]

    def measurements(update: int, loss: Any) -> dict[str, float | int]:
        with torch.no_grad():
            query_norms = torch.linalg.vector_norm(text_head(query_tensor), dim=1)
            sequence_norms = torch.linalg.vector_norm(sequence_head(norm_sample), dim=1)
        return {
            "update": update,
            "loss": float(loss.detach().cpu()),
            "query_norm_mean": float(query_norms.mean().cpu()),
            "query_norm_max": float(query_norms.max().cpu()),
            "sequence_norm_sample_mean": float(sequence_norms.mean().cpu()),
            "sequence_norm_sample_max": float(sequence_norms.max().cpu()),
        }

    if record_initial:
        with torch.no_grad():
            projected_query = text_head(query_tensor)
            query_in_sequence_space = projected_query @ sequence_head.weight
            initial_logits = (
                query_in_sequence_space @ sequence_tensor.T / temperature + log_mass_tensor
            )
            initial_known_logits = initial_logits.masked_fill(~known_tensor, negative_infinity)
            initial_loss = (
                torch.logsumexp(initial_known_logits, dim=1)
                - (target_probability * initial_logits).sum(dim=1)
            ).mean()
        history.append(measurements(0, initial_loss))

    for update in range(1, updates + 1):
        _ensure_before_deadline(deadline_monotonic, operation=f"maximum-entropy update {update}")
        projected_query = text_head(query_tensor)
        query_in_sequence_space = projected_query @ sequence_head.weight
        scores = query_in_sequence_space @ sequence_tensor.T / temperature
        logits = scores + log_mass_tensor
        known_logits = logits.masked_fill(~known_tensor, negative_infinity)
        loss = (
            torch.logsumexp(known_logits, dim=1) - (target_probability * logits).sum(dim=1)
        ).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"maximum-entropy loss became non-finite at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(measurements(update, loss))

    state = {
        "objective": "maximum_entropy",
        "base_measure": base_measure,
        "seed": seed,
        "sequence_head": sequence_head.weight.detach().cpu().numpy().astype(np.float32),
        "text_head": text_head.weight.detach().cpu().numpy().astype(np.float32),
        "temperature": temperature,
        "updates": updates,
        "known_pairs": int(known.sum()),
        "verified_pairs": int(verified.sum()),
        "contradicted_pairs": int(contradicted.sum()),
        "initial_sequence_head_sha256": initial_sequence_head_sha256,
        "initial_text_head_sha256": initial_text_head_sha256,
    }
    return state, pd.DataFrame(history)


def train_atomic_logistic_probe(
    sequence_train: np.ndarray,
    verified_mask: np.ndarray,
    *,
    updates: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
    deadline_monotonic: float | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit deterministic, class-balanced one-vs-rest atomic classifiers."""
    import torch

    sequence = _finite_matrix(sequence_train, name="atomic-classifier sequence features")
    verified = np.asarray(verified_mask)
    if verified.dtype != np.bool_ or verified.ndim != 2:
        raise ValueError("atomic-classifier verified mask must be a two-dimensional boolean array")
    if verified.shape[1] != len(sequence):
        raise ValueError("atomic-classifier labels must align with training sequences")
    if not verified.any(axis=1).all() or verified.all(axis=1).any():
        raise ValueError("every atomic classifier needs positive and weak-negative rows")
    if updates < 1 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("atomic-classifier optimizer settings are invalid")

    _ensure_before_deadline(deadline_monotonic, operation="atomic-classifier setup")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.device(device)
    features = torch.as_tensor(sequence, dtype=torch.float32, device=target)
    labels = torch.as_tensor(verified.T, dtype=torch.float32, device=target)
    positive_counts = labels.sum(dim=0)
    negative_counts = (1.0 - labels).sum(dim=0)
    head = torch.nn.Linear(sequence.shape[1], verified.shape[0], bias=True).to(target)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(
        [
            {"params": [head.weight], "weight_decay": weight_decay},
            {"params": [head.bias], "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )

    def loss_value() -> Any:
        logits = head(features)
        positive = (torch.nn.functional.softplus(-logits) * labels).sum(dim=0) / positive_counts
        negative = (torch.nn.functional.softplus(logits) * (1.0 - labels)).sum(
            dim=0
        ) / negative_counts
        return (0.5 * (positive + negative)).mean()

    def measurements(update: int, loss: Any) -> dict[str, float | int]:
        with torch.no_grad():
            weight_norms = torch.linalg.vector_norm(head.weight, dim=1)
        return {
            "update": update,
            "loss": float(loss.detach().cpu()),
            "weight_norm_mean": float(weight_norms.mean().cpu()),
            "weight_norm_max": float(weight_norms.max().cpu()),
            "bias_abs_max": float(head.bias.detach().abs().max().cpu()),
        }

    with torch.no_grad():
        history = [measurements(0, loss_value())]
    for update in range(1, updates + 1):
        _ensure_before_deadline(deadline_monotonic, operation=f"atomic-classifier update {update}")
        loss = loss_value()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"atomic-classifier loss became non-finite at update {update}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append(measurements(update, loss))

    with torch.no_grad():
        history[-1] = measurements(updates, loss_value())

    prevalence = verified.mean(axis=1, dtype=np.float64)
    log_prior_odds = np.log(prevalence) - np.log1p(-prevalence)
    state = {
        "objective": "class_balanced_full_weak_binary_cross_entropy",
        "weight": head.weight.detach().cpu().numpy().astype(np.float32),
        "bias": head.bias.detach().cpu().numpy().astype(np.float32),
        "log_prior_odds": log_prior_odds.astype(np.float32),
        "updates": updates,
        "training_rows": len(sequence),
        "atomic_queries": verified.shape[0],
        "positive_pairs": int(verified.sum()),
        "weak_negative_pairs": int(verified.size - verified.sum()),
    }
    return state, pd.DataFrame(history)


def atomic_logistic_scores(
    sequence_features: np.ndarray,
    state: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return balanced logits and empirical-prior-corrected logits per atom."""
    sequence = _finite_matrix(sequence_features, name="atomic-classifier scoring features")
    weight = _finite_matrix(state["weight"], name="atomic-classifier weights")
    bias = np.asarray(state["bias"], dtype=np.float32)
    log_prior_odds = np.asarray(state["log_prior_odds"], dtype=np.float32)
    if weight.shape[1] != sequence.shape[1]:
        raise ValueError("atomic-classifier feature and weight dimensions differ")
    if bias.shape != (len(weight),) or log_prior_odds.shape != bias.shape:
        raise ValueError("atomic-classifier bias or prior shape differs from its weights")
    raw = sequence @ weight.T + bias[None, :]
    calibrated = raw + log_prior_odds[None, :]
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise ValueError("atomic-classifier scoring produced non-finite values")
    return raw.astype(np.float32), calibrated.astype(np.float32)


def symmetric_many_positive_loss(logits: Any, positive_mask: Any) -> Any:
    """Return symmetric log-sum-exp contrastive loss for explicit positive sets."""
    import torch

    if logits.ndim != 2 or logits.shape[0] != logits.shape[1]:
        raise ValueError("contrastive logits must be square")
    if positive_mask.shape != logits.shape or positive_mask.dtype != torch.bool:
        raise ValueError("positive mask must be boolean and aligned with logits")
    if not bool(positive_mask.any(dim=1).all()) or not bool(positive_mask.any(dim=0).all()):
        raise ValueError("each row and column must have at least one positive")
    negative_infinity = torch.finfo(logits.dtype).min

    def direction(values: Any, mask: Any) -> Any:
        positive_mass = torch.logsumexp(values.masked_fill(~mask, negative_infinity), dim=1)
        return (torch.logsumexp(values, dim=1) - positive_mass).mean()

    return 0.5 * (direction(logits, positive_mask) + direction(logits.T, positive_mask.T))


def project(values: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Apply a learned head and L2-normalize each output row."""
    matrix = _finite_matrix(values, name="projection inputs")
    weights = _finite_matrix(head, name="projection head")
    if matrix.shape[1] != weights.shape[1]:
        raise ValueError("projection input and head dimensions differ")
    result = matrix @ weights.T
    norms = np.linalg.norm(result, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("projection produced a zero or non-finite vector")
    return (result / norms[:, None]).astype(np.float32)


def project_unnormalized(values: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Apply a learned head while preserving natural-parameter vector norms."""
    matrix = _finite_matrix(values, name="unnormalized projection inputs")
    weights = _finite_matrix(head, name="unnormalized projection head")
    if matrix.shape[1] != weights.shape[1]:
        raise ValueError("unnormalized projection input and head dimensions differ")
    result = matrix @ weights.T
    if not np.isfinite(result).all():
        raise ValueError("unnormalized projection produced a non-finite vector")
    return result.astype(np.float32)


def natural_parameter_scores(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    log_base_mass: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    """Score an exponential tilt of a normalized gallery base measure."""
    queries = _finite_matrix(query_vectors, name="natural-parameter query vectors")
    gallery = _finite_matrix(gallery_vectors, name="natural-parameter gallery vectors")
    log_mass = np.asarray(log_base_mass, dtype=np.float64)
    if queries.shape[1] != gallery.shape[1]:
        raise ValueError("natural-parameter query and gallery dimensions differ")
    if log_mass.shape != (len(gallery),) or not np.isfinite(log_mass).all():
        raise ValueError("gallery log base mass must be finite and aligned")
    if not np.isclose(np.exp(log_mass).sum(), 1.0, rtol=1e-10, atol=1e-10):
        raise ValueError("gallery base measure must normalize to one")
    if temperature <= 0.0:
        raise ValueError("natural-parameter temperature must be positive")
    return (
        queries.astype(np.float64) @ gallery.astype(np.float64).T / temperature + log_mass[None, :]
    )


def paired_retrieval_metrics(
    sequence_vectors: np.ndarray,
    text_vectors: np.ndarray,
    sequence_groups: np.ndarray,
    description_groups: np.ndarray,
    *,
    chunk_size: int = 512,
) -> dict[str, float | int]:
    """Compute best-positive bidirectional ranks without materializing all scores."""
    sequence = _finite_matrix(sequence_vectors, name="paired sequence vectors")
    text = _finite_matrix(text_vectors, name="paired text vectors")
    if sequence.shape != text.shape:
        raise ValueError("paired sequence and text matrices must have equal shapes")
    positives = _positive_indices(sequence_groups, description_groups)
    sequence_ranks = _best_positive_ranks(sequence, text, positives, chunk_size=chunk_size)
    text_ranks = _best_positive_ranks(text, sequence, positives, chunk_size=chunk_size)
    result: dict[str, float | int] = {"rows": int(len(sequence))}
    for direction, ranks in (
        ("sequence_to_description", sequence_ranks),
        ("description_to_sequence", text_ranks),
    ):
        result[f"{direction}_r1"] = float(np.mean(ranks <= 1))
        result[f"{direction}_r10"] = float(np.mean(ranks <= 10))
        result[f"{direction}_median_rank"] = float(np.median(ranks))
    return result


def query_rankings_and_metrics(
    query_vectors: np.ndarray,
    gallery_vectors: np.ndarray,
    queries: pd.DataFrame,
    gallery: pd.DataFrame,
    query_states: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Rank the reduced validation gallery and report three-state fractions."""
    query_matrix = _finite_matrix(query_vectors, name="query vectors")
    gallery_matrix = _finite_matrix(gallery_vectors, name="gallery vectors")
    if len(query_matrix) != len(queries) or len(gallery_matrix) != len(gallery):
        raise ValueError("query and gallery metadata must align with their vectors")
    if not cutoffs or min(cutoffs) < 1 or max(cutoffs) > len(gallery):
        raise ValueError("retrieval cutoffs must fit the validation gallery")
    scores = query_matrix @ gallery_matrix.T
    return query_rankings_and_metrics_from_scores(
        scores, queries, gallery, query_states, cutoffs=cutoffs
    )


def query_rankings_and_metrics_from_scores(
    scores: np.ndarray,
    queries: pd.DataFrame,
    gallery: pd.DataFrame,
    query_states: pd.DataFrame,
    *,
    cutoffs: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Rank a validated query-by-gallery score matrix and report state fractions."""
    score_matrix = _finite_matrix(scores, name="query scores")
    if score_matrix.shape != (len(queries), len(gallery)):
        raise ValueError("query scores must align with query and gallery metadata")
    if not cutoffs or min(cutoffs) < 1 or max(cutoffs) > len(gallery):
        raise ValueError("retrieval cutoffs must fit the validation gallery")
    top_count = max(cutoffs)
    order = np.argsort(-score_matrix, axis=1, kind="stable")[:, :top_count]
    top_indices = order
    top_scores = np.take_along_axis(score_matrix, order, axis=1)
    state_lookup = {
        (str(row.semantic_query_id), str(row.sequence_id)): str(row.state)
        for row in query_states.itertuples(index=False)
    }
    ranking_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for query_position, query in enumerate(queries.itertuples(index=False)):
        states: list[str] = []
        for rank, (gallery_position, score) in enumerate(
            zip(top_indices[query_position], top_scores[query_position], strict=True),
            start=1,
        ):
            candidate = gallery.iloc[int(gallery_position)]
            state = state_lookup.get(
                (str(query.semantic_query_id), str(candidate.sequence_id)), "unknown"
            )
            states.append(state)
            ranking_rows.append(
                {
                    "query_id": str(query.query_id),
                    "semantic_query_id": str(query.semantic_query_id),
                    "query_kind": str(query.query_kind),
                    "rank": rank,
                    "sequence_id": str(candidate.sequence_id),
                    "similarity_component_primary": str(candidate.similarity_component_primary),
                    "length_bp": int(candidate.length_bp),
                    "component_size": int(candidate.component_size),
                    "score": float(score),
                    "state": state,
                }
            )
        for cutoff in cutoffs:
            selected = states[:cutoff]
            verified = selected.count("verified") / cutoff
            contradicted = selected.count("contradicted") / cutoff
            unknown = selected.count("unknown") / cutoff
            metric_rows.append(
                {
                    "query_id": str(query.query_id),
                    "semantic_query_id": str(query.semantic_query_id),
                    "query_kind": str(query.query_kind),
                    "k": cutoff,
                    "verified_fraction": verified,
                    "contradicted_fraction": contradicted,
                    "unknown_fraction": unknown,
                    "known_fraction": verified + contradicted,
                    "utility": verified - contradicted,
                }
            )
    return pd.DataFrame(ranking_rows), pd.DataFrame(metric_rows), score_matrix


def whole_component_bootstrap_draws(
    scores_by_seed: list[np.ndarray],
    queries: pd.DataFrame,
    gallery: pd.DataFrame,
    query_states: pd.DataFrame,
    *,
    k: int,
    draws: int,
    seed: int,
    component_column: str = "similarity_component_primary",
    deadline_monotonic: float | None = None,
) -> pd.DataFrame:
    """Persistable utility draws from complete-component gallery resampling.

    One draw uses the same component multiplicities for every query and probe seed. The returned
    utility is macro-averaged over queries and then averaged over the supplied seeds. Separate
    rows retain the atomic, pair-conjunction, and combined query populations.
    """
    if draws < 2 or not scores_by_seed:
        raise ValueError("component bootstrap needs scores and at least two draws")
    if k < 1 or k > len(gallery):
        raise ValueError("component bootstrap cutoff must fit the validation gallery")
    if queries.empty or queries["query_id"].duplicated().any():
        raise ValueError("component bootstrap queries must be non-empty and unique")
    if gallery.empty or gallery["sequence_id"].duplicated().any():
        raise ValueError("component bootstrap gallery must be non-empty and unique")
    if component_column not in gallery:
        raise ValueError(f"bootstrap component column is missing: {component_column}")
    components, component_codes = np.unique(
        gallery[component_column].astype(str), return_inverse=True
    )
    generator = np.random.default_rng(seed)
    multiplicities = generator.multinomial(
        len(components),
        np.full(len(components), 1.0 / len(components)),
        size=draws,
    )
    state_lookup = {
        (str(row.semantic_query_id), str(row.sequence_id)): str(row.state)
        for row in query_states.itertuples(index=False)
    }
    gallery_ids = gallery["sequence_id"].astype(str).to_numpy()
    query_kinds = queries["query_kind"].astype(str).to_numpy()
    kind_labels = ["combined", *sorted(set(query_kinds))]
    draw_values = {label: np.zeros(draws, dtype=np.float64) for label in kind_labels}
    observations = {label: 0 for label in kind_labels}
    for seed_scores in scores_by_seed:
        _ensure_before_deadline(deadline_monotonic, operation="component bootstrap seed")
        if seed_scores.shape != (len(queries), len(gallery)):
            raise ValueError("bootstrap score matrix shape changed")
        for query_position, query in enumerate(queries.itertuples(index=False)):
            _ensure_before_deadline(deadline_monotonic, operation="component bootstrap query")
            order = np.argsort(-seed_scores[query_position], kind="stable")
            state_values = np.asarray(
                [
                    {"verified": 1.0, "contradicted": -1.0}.get(
                        state_lookup.get(
                            (str(query.semantic_query_id), gallery_ids[index]), "unknown"
                        ),
                        0.0,
                    )
                    for index in order
                ],
                dtype=np.float64,
            )
            ordered_components = component_codes[order]
            query_kind = str(query_kinds[query_position])
            for draw in range(draws):
                _ensure_before_deadline(deadline_monotonic, operation="component bootstrap draw")
                remaining = k
                utility = 0.0
                for state_value, component in zip(state_values, ordered_components, strict=True):
                    copies = min(remaining, int(multiplicities[draw, component]))
                    utility += copies * state_value
                    remaining -= copies
                    if remaining == 0:
                        break
                if remaining:
                    raise RuntimeError("component bootstrap could not fill the retrieval cutoff")
                query_utility = utility / k
                draw_values["combined"][draw] += query_utility
                draw_values[query_kind][draw] += query_utility
            observations["combined"] += 1
            observations[query_kind] += 1
    rows: list[dict[str, float | int | str]] = []
    for label in kind_labels:
        if observations[label] == 0:
            continue
        values = draw_values[label] / observations[label]
        rows.extend(
            {
                "draw": draw,
                "query_kind": label,
                "k": k,
                "utility": float(value),
                "probe_seeds": len(scores_by_seed),
                "query_seed_observations": observations[label],
            }
            for draw, value in enumerate(values)
        )
    return pd.DataFrame(rows)


def _best_positive_ranks(
    queries: np.ndarray,
    candidates: np.ndarray,
    positive_indices: list[np.ndarray],
    *,
    chunk_size: int,
) -> np.ndarray:
    ranks = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        scores = queries[start:stop] @ candidates.T
        for local_index, row_scores in enumerate(scores):
            position = start + local_index
            positive_score = float(row_scores[positive_indices[position]].max())
            ranks[position] = int(np.count_nonzero(row_scores > positive_score) + 1)
    return ranks


def _positive_indices(
    sequence_groups: np.ndarray, description_groups: np.ndarray
) -> list[np.ndarray]:
    raw_sequence = np.asarray(sequence_groups)
    raw_description = np.asarray(description_groups)
    if pd.isna(raw_sequence).any() or pd.isna(raw_description).any():
        raise ValueError("positive-group identifiers must not be missing")
    sequence = raw_sequence.astype(str)
    description = raw_description.astype(str)
    if len(sequence) != len(description):
        raise ValueError("positive-group arrays must align")
    return [
        np.flatnonzero((sequence == sequence[index]) | (description == description[index]))
        for index in range(len(sequence))
    ]


def _factorize_groups(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    if pd.isna(raw).any():
        raise ValueError("positive-group identifiers must not be missing")
    codes, _ = pd.factorize(raw.astype(str), sort=True)
    if np.any(codes < 0):
        raise ValueError("positive-group identifiers must not be missing")
    return codes


def _finite_matrix(
    values: np.ndarray,
    *,
    name: str,
    minimum_rows: int = 1,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < minimum_rows or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contain a non-finite value")
    return matrix


def _tensor_sha256(values: Any) -> str:
    array = values.detach().cpu().numpy().astype(np.float32)
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _ensure_before_deadline(deadline_monotonic: float | None, *, operation: str) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError(f"authorized compute deadline reached before {operation}")
