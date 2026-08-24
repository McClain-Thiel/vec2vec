"""Frozen Hugging Face text encoding for Gate 1 paired and query texts."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextEncoderRecipe(BaseModel):
    """Validated model, pooling, and role-specific prompt contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    revision: str
    transformers_version: str
    trust_remote_code: bool = False
    max_tokens: int = Field(gt=0)
    pooling: Literal["cls", "last_token"]
    document_prefix: str = ""
    query_prefix: str = ""
    normalize: bool = True
    attention_implementation: Literal["sdpa"] = "sdpa"
    batch_size: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_recipe(self) -> TextEncoderRecipe:
        """Reject an unpinned model contract."""
        invalid_revision = len(self.revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.revision
        )
        if invalid_revision:
            raise ValueError("revision must be a 40-character lowercase Git SHA")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        return self


@dataclass(frozen=True)
class EncodedTexts:
    """One batch-independent matrix with per-input token evidence."""

    vectors: np.ndarray
    token_counts: list[int]
    elapsed_seconds: float


class FrozenTextEncoder:
    """Load one pinned text model and apply its frozen official recipe."""

    def __init__(
        self,
        recipe: TextEncoderRecipe,
        *,
        precision: Literal["bfloat16", "float32"],
        device: str,
    ) -> None:
        self.recipe = recipe
        self.precision = precision
        self.device = device
        self.torch: Any | None = None
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load(self) -> None:
        """Load the exact tokenizer, model revision, and precision."""
        if self.model is not None:
            return
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        if transformers.__version__ != self.recipe.transformers_version:
            raise RuntimeError(
                f"{self.recipe.model_id} requires Transformers "
                f"{self.recipe.transformers_version}, but {transformers.__version__} is installed"
            )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.recipe.model_id,
            revision=self.recipe.revision,
            trust_remote_code=self.recipe.trust_remote_code,
        )
        if self.recipe.pooling == "last_token":
            tokenizer.padding_side = "left"
        model = transformers.AutoModel.from_pretrained(
            self.recipe.model_id,
            revision=self.recipe.revision,
            trust_remote_code=self.recipe.trust_remote_code,
            torch_dtype=getattr(torch, self.precision),
            attn_implementation=self.recipe.attention_implementation,
        )
        model.eval().to(self.device)
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model

    def close(self) -> None:
        """Release model references and cached GPU allocations."""
        torch = self.torch
        self.model = None
        self.tokenizer = None
        if torch is not None and str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode(
        self,
        texts: list[str],
        *,
        role: Literal["document", "query"],
        deadline_monotonic: float | None = None,
    ) -> EncodedTexts:
        """Encode all texts without truncation and return finite float32 vectors."""
        self._ensure_before_deadline(deadline_monotonic, operation="text model loading")
        self.load()
        self._ensure_before_deadline(deadline_monotonic, operation="text feature extraction")
        if self.torch is None or self.tokenizer is None or self.model is None:
            raise RuntimeError("text encoder did not load")
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("text inputs must be non-empty")
        prefix = self.recipe.document_prefix if role == "document" else self.recipe.query_prefix
        prompted = [f"{prefix}{text}" for text in texts]
        vectors: list[np.ndarray] = []
        token_counts: list[int] = []
        started = time.perf_counter()
        for start in range(0, len(prompted), self.recipe.batch_size):
            self._ensure_before_deadline(
                deadline_monotonic, operation=f"text batch starting at row {start}"
            )
            batch = prompted[start : start + self.recipe.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            counts = encoded["attention_mask"].sum(dim=1).tolist()
            oversized = [int(count) for count in counts if int(count) > self.recipe.max_tokens]
            if oversized:
                raise ValueError(
                    f"{len(oversized)} text inputs exceed the {self.recipe.max_tokens}-token limit"
                )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            pooled = self._pool(hidden, encoded["attention_mask"])
            if self.recipe.normalize:
                pooled = self.torch.nn.functional.normalize(pooled.float(), dim=1)
            matrix = pooled.float().cpu().numpy().astype(np.float32)
            if not np.isfinite(matrix).all():
                raise ValueError("text pooling produced a non-finite vector")
            vectors.append(matrix)
            token_counts.extend(int(count) for count in counts)
        return EncodedTexts(
            vectors=np.vstack(vectors),
            token_counts=token_counts,
            elapsed_seconds=time.perf_counter() - started,
        )

    def peak_device_memory_bytes(self) -> int | None:
        """Return peak allocated CUDA memory when CUDA is active."""
        if self.torch is None or not str(self.device).startswith("cuda"):
            return None
        if not self.torch.cuda.is_available():
            return None
        return int(self.torch.cuda.max_memory_allocated())

    def _pool(self, hidden: Any, attention_mask: Any) -> Any:
        if self.recipe.pooling == "cls":
            return hidden[:, 0]
        if bool(attention_mask[:, -1].all()):
            return hidden[:, -1]
        rows = self.torch.arange(hidden.shape[0], device=hidden.device)
        positions = attention_mask.sum(dim=1) - 1
        return hidden[rows, positions]

    @staticmethod
    def _ensure_before_deadline(deadline_monotonic: float | None, *, operation: str) -> None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError(f"authorized compute deadline reached before {operation}")
