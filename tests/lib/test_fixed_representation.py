from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vec2vec.lib import fixed_representation
from vec2vec.lib.constraint_state import retrieval_population_sha256
from vec2vec.lib.sequences import sequence_sha256


def _sequence(length: int, offset: int) -> str:
    bases = "ACGT"
    return "".join(bases[(index + offset) % len(bases)] for index in range(length))


def _population(rows: int = 1_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    retrieval_rows = []
    split_rows = []
    for index in range(rows):
        length = 60 + index
        sequence = _sequence(length, index)
        sequence_id = f"sequence-{index:04d}"
        retrieval_rows.append(
            {
                "sequence_id": sequence_id,
                "sequence": sequence,
                "sequence_sha256": sequence_sha256(sequence),
                "description": f"plasmid description {index}",
                "length_bp": length,
                "leakage_component": index,
                "split_grouped": "train",
            }
        )
        split_rows.append(
            {
                "sequence_id": sequence_id,
                "similarity_component_primary": f"primary-{index:04d}",
                "leakage_component_v2": f"v2-{index:04d}",
                "split_grouped_v2": "train",
            }
        )
    return pd.DataFrame(retrieval_rows), pd.DataFrame(split_rows)


def test_panel_is_deterministic_component_unique_and_length_stratified() -> None:
    retrieval, split = _population()
    expected_hash = retrieval_population_sha256(retrieval)

    first, first_summary = fixed_representation.build_fixed_representation_panels(
        retrieval,
        split,
        expected_population_sha256=expected_hash,
        invariance_rows=512,
        numerical_smoke_rows=32,
        length_strata=10,
        selection_salt="panel-v1",
        eligible_sequence_alphabet="ACGT",
        expected_prior_panel_sha256=None,
        expected_panel_sha256=None,
    )
    second, second_summary = fixed_representation.build_fixed_representation_panels(
        retrieval.sample(frac=1.0, random_state=9),
        split.sample(frac=1.0, random_state=4),
        expected_population_sha256=expected_hash,
        invariance_rows=512,
        numerical_smoke_rows=32,
        length_strata=10,
        selection_salt="panel-v1",
        eligible_sequence_alphabet="ACGT",
        expected_prior_panel_sha256=None,
        expected_panel_sha256=None,
    )

    assert first_summary["panel_sha256"] == second_summary["panel_sha256"]
    assert first["similarity_component_primary"].nunique() == 512
    assert first["in_numerical_smoke_panel"].sum() == 32
    assert first.groupby("length_decile")["in_numerical_smoke_panel"].sum().tolist() == [
        4,
        4,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
    ]
    for stratum, group in first.groupby("length_decile"):
        eligible = retrieval.iloc[stratum * 100 : (stratum + 1) * 100]
        assert group["length_bp"].min() == eligible["length_bp"].min()
        assert group["length_bp"].max() == eligible["length_bp"].max()


def test_panel_rejects_a_sequence_hash_mismatch() -> None:
    retrieval, split = _population()
    expected_hash = retrieval_population_sha256(retrieval)
    retrieval.loc[0, "sequence"] = "A" * int(retrieval.loc[0, "length_bp"])

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fixed_representation.build_fixed_representation_panels(
            retrieval,
            split,
            expected_population_sha256=expected_hash,
            invariance_rows=512,
            numerical_smoke_rows=32,
            length_strata=10,
            selection_salt="panel-v1",
            eligible_sequence_alphabet="ACGT",
            expected_prior_panel_sha256=None,
            expected_panel_sha256=None,
        )


def test_panel_excludes_noneligible_iupac_rows_before_selection() -> None:
    retrieval, split = _population()
    ambiguous_sequence = "W" + str(retrieval.loc[0, "sequence"])[1:]
    retrieval.loc[0, "sequence"] = ambiguous_sequence
    retrieval.loc[0, "sequence_sha256"] = sequence_sha256(ambiguous_sequence)
    expected_hash = retrieval_population_sha256(retrieval)

    prior_panel, prior_summary = fixed_representation.build_fixed_representation_panels(
        retrieval,
        split,
        expected_population_sha256=expected_hash,
        invariance_rows=512,
        numerical_smoke_rows=32,
        length_strata=10,
        selection_salt="panel-v1",
        eligible_sequence_alphabet="ACGTRYSWKMBDHVN",
        expected_prior_panel_sha256=None,
        expected_panel_sha256=None,
    )

    panel, summary = fixed_representation.build_fixed_representation_panels(
        retrieval,
        split,
        expected_population_sha256=expected_hash,
        invariance_rows=512,
        numerical_smoke_rows=32,
        length_strata=10,
        selection_salt="panel-v1",
        eligible_sequence_alphabet="ACGT",
        expected_prior_panel_sha256=prior_summary["panel_sha256"],
        expected_panel_sha256=None,
    )

    prior_ids = set(prior_panel["sequence_id"])
    panel_ids = set(panel["sequence_id"])
    assert prior_ids - panel_ids == {"sequence-0000"}
    assert len(panel_ids - prior_ids) == 1
    assert panel["sequence"].map(lambda sequence: set(sequence) <= set("ACGT")).all()
    assert summary["sequence_eligibility"] == {
        "allowed_alphabet": "ACGT",
        "eligible_training_rows": 999,
        "excluded_training_rows": 1,
        "excluded_symbol_counts": {"W": 1},
        "excluded_rows_sha256": "9d029135b17ea8204d82b8316c27a8db166088998357dcb86c80ca7291b751ce",
    }
    assert summary["panel_amendment"]["replaced_rows"] == 1
    assert summary["panel_amendment"]["removed_rows"] == [
        {
            "sequence_id": "sequence-0000",
            "sequence_sha256": sequence_sha256(ambiguous_sequence),
            "length_decile": 0,
            "unsupported_symbol_counts": {"W": 1},
        }
    ]
    assert summary["panel_amendment"]["preserved_prior_rows"] == 511
    assert len(summary["panel_amendment"]["replacement_rows"]) == 1
    assert summary["panel_amendment"]["replacement_rows"][0]["length_decile"] == 0


def test_panel_rejects_a_changed_prior_selection_contract() -> None:
    retrieval, split = _population()

    with pytest.raises(ValueError, match="prior invariance panel changed"):
        fixed_representation.build_fixed_representation_panels(
            retrieval,
            split,
            expected_population_sha256=retrieval_population_sha256(retrieval),
            invariance_rows=512,
            numerical_smoke_rows=32,
            length_strata=10,
            selection_salt="panel-v1",
            eligible_sequence_alphabet="ACGT",
            expected_prior_panel_sha256="0" * 64,
            expected_panel_sha256=None,
        )


def test_panel_rejects_a_changed_amended_selection_contract() -> None:
    retrieval, split = _population()

    with pytest.raises(ValueError, match="amended invariance panel changed"):
        fixed_representation.build_fixed_representation_panels(
            retrieval,
            split,
            expected_population_sha256=retrieval_population_sha256(retrieval),
            invariance_rows=512,
            numerical_smoke_rows=32,
            length_strata=10,
            selection_salt="panel-v1",
            eligible_sequence_alphabet="ACGT",
            expected_prior_panel_sha256=None,
            expected_panel_sha256="0" * 64,
        )


def test_short_circular_window_wraps_to_the_tokenizer_unit() -> None:
    windows = fixed_representation.circular_window_plan(
        10,
        maximum_content_bp=48,
        tokenizer_unit_bp=6,
        overlap_fraction=0.25,
    )

    assert windows == [
        fixed_representation.CircularWindow(
            index=0,
            start_bp=0,
            input_base_count=12,
            newly_covered_base_count=10,
            wrapped_input_base_count=2,
        )
    ]
    assert fixed_representation.circular_subsequence("ACGTACGTAA", windows[0]) == "ACGTACGTAAAC"


def test_long_circular_windows_cover_each_base_without_changing_weight() -> None:
    windows = fixed_representation.circular_window_plan(
        100,
        maximum_content_bp=48,
        tokenizer_unit_bp=6,
        overlap_fraction=0.25,
    )

    assert [window.start_bp for window in windows] == [0, 36, 72]
    assert [window.newly_covered_base_count for window in windows] == [48, 36, 16]
    assert sum(window.newly_covered_base_count for window in windows) == 100
    assert windows[-1].wrapped_input_base_count == 20


def test_reverse_complement_rejects_an_ambiguous_base() -> None:
    assert fixed_representation.reverse_complement("AACCGT") == "ACGGTT"
    with pytest.raises(ValueError, match="unsupported bases"):
        fixed_representation.reverse_complement("AACNGT")


def test_circular_rotation_matches_the_frozen_quarter_offsets() -> None:
    sequence = "AACCGGTT"

    assert fixed_representation.circular_rotate(sequence, 0.25) == "CCGGTTAA"
    assert fixed_representation.circular_rotate(sequence, 0.50) == "GGTTAACC"
    assert fixed_representation.circular_rotate(sequence, 0.75) == "TTAACCGG"

    with pytest.raises(ValueError, match="must not be empty"):
        fixed_representation.circular_rotate("", 0.25)


def test_representation_geometry_reports_rank_and_pairwise_confounds() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    diagnostics = fixed_representation.representation_geometry(
        embeddings,
        lengths_bp=[100, 200, 400, 800],
        gc_fractions=[0.10, 0.30, 0.60, 0.90],
    )

    assert diagnostics["rows"] == 4
    assert diagnostics["embedding_dimension"] == 3
    assert 1.0 < diagnostics["effective_rank"] <= 3.0
    assert 0.0 < diagnostics["effective_rank_fraction"] <= 1.0
    assert diagnostics["mean_pairwise_cosine"] == pytest.approx(7 / 30)
    assert np.isfinite(diagnostics["pairwise_cosine_length_difference_pearson"])
    assert np.isfinite(diagnostics["pairwise_cosine_gc_difference_pearson"])


def test_representation_geometry_records_an_undefined_constant_confound() -> None:
    diagnostics = fixed_representation.representation_geometry(
        np.eye(3, dtype=np.float64),
        lengths_bp=[100, 100, 100],
        gc_fractions=[0.2, 0.4, 0.6],
    )

    assert diagnostics["pairwise_cosine_length_difference_pearson"] is None
    assert (
        diagnostics["pairwise_cosine_length_difference_pearson_status"]
        == "undefined_constant_input"
    )
