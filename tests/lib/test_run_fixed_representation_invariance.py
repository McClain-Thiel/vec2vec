from __future__ import annotations

import math

import pytest
from scripts import run_fixed_representation_invariance as runner
from scripts.run_fixed_representation_invariance import (
    _candidate_child_command,
    _remaining_candidate_seconds,
    _remaining_instance_hours,
    _required_transformers_version,
    _run_candidate_child,
)


def test_remaining_instance_hours_applies_one_batch_cap() -> None:
    assert _remaining_instance_hours(3.0, elapsed_seconds=3_600.0) == pytest.approx(2.0)

    with pytest.raises(TimeoutError, match="authorized instance-hour limit"):
        _remaining_instance_hours(3.0, elapsed_seconds=10_800.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_remaining_instance_hours_rejects_a_non_finite_cap(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _remaining_instance_hours(value, elapsed_seconds=0.0)

    with pytest.raises(ValueError, match="must be finite"):
        _remaining_instance_hours(1.0, elapsed_seconds=value)


def test_candidate_batch_requires_one_transformers_runtime() -> None:
    recipes = {
        "carbon": {"transformers_version": "5.12.1"},
        "generanno": {"transformers_version": "4.49.0"},
        "generator": {"transformers_version": "4.49.0"},
    }

    assert _required_transformers_version(("generanno", "generator"), recipes) == "4.49.0"
    with pytest.raises(ValueError, match="cannot mix"):
        _required_transformers_version(("carbon", "generanno"), recipes)

    with pytest.raises(ValueError, match="no configured encoder recipe"):
        _required_transformers_version(("missing",), recipes)


def test_candidate_deadline_includes_setup_and_reserves_shutdown_time() -> None:
    assert _remaining_candidate_seconds(
        1.0,
        elapsed_seconds=60.0,
        shutdown_reserve_seconds=30.0,
    ) == pytest.approx(3_510.0)

    with pytest.raises(TimeoutError, match="shutdown reserve"):
        _remaining_candidate_seconds(
            1.0,
            elapsed_seconds=3_580.0,
            shutdown_reserve_seconds=30.0,
        )


def test_candidate_child_wraps_the_complete_kedro_run_in_an_external_timeout(monkeypatch) -> None:
    observed = {}

    def fake_run(command, **kwargs) -> None:
        observed["command"] = command
        observed.update(kwargs)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    command = _candidate_child_command(
        candidate="candidate",
        approval_reference="approval",
        region="test-region-1",
        instance_type="test.instance",
        candidate_instance_hour_limit=0.5,
        batch_instance_hour_limit=1.0,
        observed_instance_price_usd_per_hour=2.0,
    )

    _run_candidate_child(command, timeout_seconds=1_800.0)

    assert "--internal-candidate-child" in observed["command"]
    assert observed["timeout"] == 1_800.0
    assert observed["cwd"] == runner.PROJECT_ROOT
    assert observed["check"] is True
    assert observed["start_new_session"] is True
