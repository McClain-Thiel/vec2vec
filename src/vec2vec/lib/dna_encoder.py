"""Frozen Hugging Face DNA encoding for the Gate 1 numerical smoke check."""

from __future__ import annotations

import importlib
import inspect
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vec2vec.lib.fixed_representation import circular_subsequence, circular_window_plan


class EncoderRecipe(BaseModel):
    """Validated model and tokenization contract from Kedro configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    revision: str
    transformers_version: str
    model_class: Literal["causal_lm", "masked_lm"]
    trust_remote_code: bool
    model_max_tokens: int = Field(gt=0)
    tokenizer_unit_bp: int = Field(gt=0)
    sequence_prefix: str
    sequence_suffix: str
    excluded_content_tokens: tuple[str, ...]
    out_of_vocabulary_token: str
    pooling_layers: Literal[1, 4]
    attention_implementation: Literal["sdpa"] = "sdpa"

    @model_validator(mode="after")
    def validate_recipe(self) -> EncoderRecipe:
        """Reject an unpinned or ambiguous model contract."""
        invalid_revision = len(self.revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.revision
        )
        if invalid_revision:
            raise ValueError("revision must be a 40-character lowercase Git SHA")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not self.excluded_content_tokens:
            raise ValueError("excluded_content_tokens must name the model tags")
        return self


@dataclass(frozen=True)
class EncodedSequence:
    """One normalized plasmid vector and its per-window coverage evidence."""

    vector: np.ndarray
    coverage: list[dict[str, Any]]
    elapsed_seconds: float


class FrozenDnaEncoder:
    """Load one pinned DNA model and apply the frozen pooling contract."""

    def __init__(
        self,
        recipe: EncoderRecipe,
        *,
        precision: Literal["bfloat16", "float32"],
        device: str,
        overlap_fraction: float,
    ) -> None:
        self.recipe = recipe
        self.precision = precision
        self.device = device
        self.overlap_fraction = overlap_fraction
        self.torch: Any | None = None
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.base_model: Any | None = None
        self.maximum_content_bp: int | None = None
        self._excluded_token_ids: set[int] = set()
        self._out_of_vocabulary_token_id: int | None = None

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
        dtype = getattr(torch, self.precision)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.recipe.model_id,
            revision=self.recipe.revision,
            trust_remote_code=self.recipe.trust_remote_code,
        )
        model_loader = (
            transformers.AutoModelForCausalLM
            if self.recipe.model_class == "causal_lm"
            else transformers.AutoModelForMaskedLM
        )
        model = model_loader.from_pretrained(
            self.recipe.model_id,
            revision=self.recipe.revision,
            trust_remote_code=self.recipe.trust_remote_code,
            torch_dtype=dtype,
            attn_implementation=self.recipe.attention_implementation,
        )
        model.eval()
        model.to(self.device)
        base_model = getattr(model, "model", None)
        if base_model is None:
            raise TypeError(f"{self.recipe.model_id} does not expose a base model at .model")

        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model
        self.base_model = base_model
        self._excluded_token_ids = {
            int(tokenizer.convert_tokens_to_ids(token))
            for token in self.recipe.excluded_content_tokens
        }
        self._out_of_vocabulary_token_id = int(
            tokenizer.convert_tokens_to_ids(self.recipe.out_of_vocabulary_token)
        )
        self.maximum_content_bp = self._discover_maximum_content_bp()

    def close(self) -> None:
        """Release model references and cached GPU allocations."""
        torch = self.torch
        self.base_model = None
        self.model = None
        self.tokenizer = None
        if torch is not None and str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_sequence(self, sequence_id: str, sequence: str) -> EncodedSequence:
        """Encode one plasmid without dropping or truncating a base."""
        self.load()
        if not sequence:
            raise ValueError(f"sequence {sequence_id} is empty")
        invalid = sorted(set(sequence).difference("ACGT"))
        if invalid:
            raise ValueError(f"sequence {sequence_id} contains unsupported bases: {invalid}")
        if self.maximum_content_bp is None:
            raise RuntimeError("maximum content window is unresolved")

        windows = circular_window_plan(
            len(sequence),
            maximum_content_bp=self.maximum_content_bp,
            tokenizer_unit_bp=self.recipe.tokenizer_unit_bp,
            overlap_fraction=self.overlap_fraction,
        )
        vectors: list[np.ndarray] = []
        coverage: list[dict[str, Any]] = []
        started = time.perf_counter()
        for window in windows:
            content = circular_subsequence(sequence, window)
            vector, token_counts = self._encode_window(content)
            vectors.append(vector)
            coverage.append(
                {
                    "sequence_id": sequence_id,
                    "window_index": int(window.index),
                    "start_bp": int(window.start_bp),
                    "input_base_count": int(window.input_base_count),
                    "newly_covered_base_count": int(window.newly_covered_base_count),
                    "wrapped_input_base_count": int(window.wrapped_input_base_count),
                    **token_counts,
                }
            )
        weights = np.asarray(
            [window.newly_covered_base_count for window in windows], dtype=np.float64
        )
        if int(weights.sum()) != len(sequence):
            raise RuntimeError(f"sequence {sequence_id} coverage weights do not sum to its length")
        pooled = np.average(np.vstack(vectors).astype(np.float64), axis=0, weights=weights)
        norm = float(np.linalg.norm(pooled))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"sequence {sequence_id} produced a zero or non-finite vector")
        pooled = np.asarray(pooled / norm, dtype=np.float32)
        if not np.isfinite(pooled).all():
            raise ValueError(f"sequence {sequence_id} produced a non-finite normalized vector")
        return EncodedSequence(
            vector=pooled,
            coverage=coverage,
            elapsed_seconds=time.perf_counter() - started,
        )

    def peak_device_memory_bytes(self) -> int | None:
        """Return peak allocated CUDA memory when CUDA is active."""
        if self.torch is None or not str(self.device).startswith("cuda"):
            return None
        if not self.torch.cuda.is_available():
            return None
        return int(self.torch.cuda.max_memory_allocated())

    def reset_peak_device_memory(self) -> None:
        """Reset the CUDA peak allocation counter before one precision pass."""
        if (
            self.torch is not None
            and str(self.device).startswith("cuda")
            and self.torch.cuda.is_available()
        ):
            self.torch.cuda.reset_peak_memory_stats()

    def _discover_maximum_content_bp(self) -> int:
        if self.tokenizer is None:
            raise RuntimeError("tokenizer is not loaded")
        low_units = 1
        high_units = self.recipe.model_max_tokens
        while low_units < high_units:
            middle = (low_units + high_units + 1) // 2
            content = "A" * (middle * self.recipe.tokenizer_unit_bp)
            input_ids, content_mask = self._tokenize(content)
            if len(input_ids) <= self.recipe.model_max_tokens and int(content_mask.sum()) == middle:
                low_units = middle
            else:
                high_units = middle - 1
        maximum_content_bp = low_units * self.recipe.tokenizer_unit_bp
        input_ids, content_mask = self._tokenize("A" * maximum_content_bp)
        if len(input_ids) > self.recipe.model_max_tokens or int(content_mask.sum()) != low_units:
            raise RuntimeError("failed to resolve a valid maximum content window")
        return maximum_content_bp

    def _tokenize(self, content: str) -> tuple[list[int], np.ndarray]:
        if self.tokenizer is None or self._out_of_vocabulary_token_id is None:
            raise RuntimeError("tokenizer is not loaded")
        prompt = f"{self.recipe.sequence_prefix}{content}{self.recipe.sequence_suffix}"
        encoded = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=False,
        )
        input_ids = [int(value) for value in encoded["input_ids"]]
        attention_mask = np.asarray(encoded["attention_mask"], dtype=bool)
        ids = np.asarray(input_ids, dtype=np.int64)
        if np.any(ids == self._out_of_vocabulary_token_id):
            raise ValueError("tokenizer emitted an out-of-vocabulary token")
        excluded = np.isin(ids, list(self._excluded_token_ids))
        content_mask = attention_mask & ~excluded
        expected_content_tokens = len(content) // self.recipe.tokenizer_unit_bp
        if len(content) % self.recipe.tokenizer_unit_bp:
            raise ValueError("content length is not aligned to the tokenizer unit")
        if int(content_mask.sum()) != expected_content_tokens:
            raise ValueError(
                "tokenizer content-token count changed: "
                f"expected {expected_content_tokens}, observed {int(content_mask.sum())}"
            )
        return input_ids, content_mask

    def _encode_window(self, content: str) -> tuple[np.ndarray, dict[str, int]]:
        if self.torch is None or self.base_model is None:
            raise RuntimeError("model is not loaded")
        input_ids, content_mask = self._tokenize(content)
        if len(input_ids) > self.recipe.model_max_tokens:
            raise ValueError(
                f"tokenized window has {len(input_ids)} tokens, "
                f"above the {self.recipe.model_max_tokens}-token model limit"
            )
        torch = self.torch
        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attention_tensor = torch.ones_like(ids_tensor)
        content_tensor = torch.tensor(content_mask, dtype=torch.bool, device=self.device)
        hidden = self._forward_hidden(ids_tensor, attention_tensor)
        content_hidden = hidden[0, content_tensor, :].float()
        if content_hidden.shape[0] != int(content_mask.sum()):
            raise RuntimeError("content mask and hidden-state rows differ")
        vector = content_hidden.mean(dim=0).detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("window pooling produced a non-finite vector")
        return vector, {
            "input_token_count": int(len(input_ids)),
            "content_token_count": int(content_mask.sum()),
            "special_token_count": int(len(input_ids) - content_mask.sum()),
            "out_of_vocabulary_token_count": 0,
        }

    def _forward_hidden(self, input_ids: Any, attention_mask: Any) -> Any:
        if self.torch is None or self.base_model is None:
            raise RuntimeError("model is not loaded")
        forward_parameters = inspect.signature(self.base_model.forward).parameters
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if "use_cache" in forward_parameters:
            kwargs["use_cache"] = False
        if "return_dict" in forward_parameters:
            kwargs["return_dict"] = True

        captured: list[Any] = []
        handles: list[Any] = []
        if self.recipe.pooling_layers == 4:
            layers = list(getattr(self.base_model, "layers", ()))
            if len(layers) < 4:
                raise TypeError("last-four pooling requires at least four base-model layers")

            def capture_layer(_module: Any, _inputs: Any, output: Any) -> None:
                captured.append(output[0] if isinstance(output, tuple) else output)

            handles = [layer.register_forward_hook(capture_layer) for layer in layers[-4:-1]]
        try:
            with self.torch.inference_mode():
                outputs = self.base_model(**kwargs)
        finally:
            for handle in handles:
                handle.remove()
        final_hidden = (
            outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        )
        if self.recipe.pooling_layers == 1:
            return final_hidden
        if len(captured) != 3:
            raise RuntimeError(f"last-four pooling captured {len(captured)} intermediate layers")
        return self.torch.stack([*captured, final_hidden], dim=0).mean(dim=0)
