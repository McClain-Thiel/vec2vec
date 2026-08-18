from __future__ import annotations

import pytest
from pydantic import ValidationError

from vec2vec.lib.dna_encoder import EncoderRecipe


def _recipe(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_id": "organization/model",
        "revision": "a" * 40,
        "model_class": "causal_lm",
        "trust_remote_code": True,
        "model_max_tokens": 128,
        "tokenizer_unit_bp": 6,
        "sequence_prefix": "<dna>",
        "sequence_suffix": "</dna>",
        "excluded_content_tokens": ["<dna>", "</dna>"],
        "out_of_vocabulary_token": "<oov>",
        "pooling_layers": 4,
        "attention_implementation": "sdpa",
    }
    values.update(overrides)
    return values


def test_encoder_recipe_requires_an_exact_revision() -> None:
    recipe = EncoderRecipe.model_validate(_recipe())
    assert recipe.revision == "a" * 40

    with pytest.raises(ValidationError, match="40-character lowercase Git SHA"):
        EncoderRecipe.model_validate(_recipe(revision="main"))


def test_encoder_recipe_rejects_unknown_configuration() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EncoderRecipe.model_validate(_recipe(unrecorded_pooling_rule="cls"))
