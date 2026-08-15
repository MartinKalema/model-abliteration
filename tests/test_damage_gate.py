"""Tests for the fail-closed candidate acceptance gate."""

from __future__ import annotations

import math

import pytest

from obliteratus.evaluation.damage_gate import (
    AcceptanceBudget,
    DamageBudget,
    DamageGateError,
    assess_candidate,
    paired_bootstrap_upper_bound,
    weighted_paired_bootstrap_upper_bound,
)


def _passing_metrics() -> dict[str, float | int]:
    return {
        "nll_increase_upper_ci": 0.01,
        "sampled_token_kl_upper_ci": 0.02,
        "sampled_token_kl_p95": 0.08,
        "top1_flip_rate": 0.01,
        "coherence_drop": 0.05,
        "new_degenerate_count": 0,
        "nonfinite_output_count": 0,
        "refusal_rate": 0.1,
        "refusal_eval_count": 30,
        "eval_prompt_count": 64,
        "eval_token_count": 1024,
        "sampled_token_count": 512,
    }


def test_candidate_passes_when_every_measurement_is_within_budget():
    assessment = assess_candidate(_passing_metrics(), AcceptanceBudget())

    assert assessment.accepted is True
    assert assessment.violations == ()
    assert assessment.inconclusive == ()


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("nll_increase_upper_ci", 0.051),
        ("sampled_token_kl_upper_ci", 0.051),
        ("sampled_token_kl_p95", 0.201),
        ("top1_flip_rate", 0.021),
        ("coherence_drop", 0.101),
        ("new_degenerate_count", 1),
        ("nonfinite_output_count", 1),
        ("refusal_rate", 0.21),
    ],
)
def test_each_enabled_limit_can_reject_a_candidate(metric, value):
    metrics = _passing_metrics()
    metrics[metric] = value

    assessment = assess_candidate(metrics, AcceptanceBudget())

    assert assessment.accepted is False
    assert assessment.violations


def test_missing_measurement_rejects_by_default():
    metrics = _passing_metrics()
    del metrics["sampled_token_kl_upper_ci"]

    assessment = assess_candidate(metrics, AcceptanceBudget())

    assert assessment.accepted is False
    assert "was not measured" in assessment.inconclusive[0]


def test_missing_measurement_requires_explicit_inconclusive_override():
    assessment = assess_candidate(
        {},
        AcceptanceBudget(
            damage=DamageBudget(unsafe_allow_inconclusive=True),
        ),
    )

    assert assessment.accepted is True
    assert assessment.inconclusive


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurement_always_rejects(value):
    metrics = _passing_metrics()
    metrics["sampled_token_kl_upper_ci"] = value

    assessment = assess_candidate(
        metrics,
        AcceptanceBudget(
            damage=DamageBudget(unsafe_allow_inconclusive=True),
        ),
    )

    assert assessment.accepted is False
    assert assessment.violations


def test_disabled_check_does_not_require_its_metric():
    metrics = _passing_metrics()
    del metrics["coherence_drop"]

    assessment = assess_candidate(
        metrics,
        AcceptanceBudget(
            damage=DamageBudget(max_coherence_drop=None),
        ),
    )

    assert assessment.accepted is True


def test_gate_error_keeps_structured_assessment():
    assessment = assess_candidate({}, AcceptanceBudget())
    error = DamageGateError(assessment)

    assert error.assessment is assessment
    assert "Candidate rejected" in str(error)


def test_paired_bootstrap_is_deterministic_and_tracks_constant_delta():
    deltas = [0.0125] * 40

    first = paired_bootstrap_upper_bound(deltas, seed=7)
    second = paired_bootstrap_upper_bound(deltas, seed=7)

    assert first == pytest.approx(0.0125)
    assert second == first


def test_paired_bootstrap_rejects_nonfinite_input_safely():
    assert math.isinf(paired_bootstrap_upper_bound([0.0, float("nan")]))


def test_weighted_bootstrap_uses_token_counts_not_prompt_averages():
    # Prompt 1 adds 1 nat across one token; prompt 2 adds 0 across 99 tokens.
    # The token-weighted increase is 0.01, not the prompt-average 0.5.
    upper = weighted_paired_bootstrap_upper_bound(
        [1.0, 0.0],
        [1, 99],
        n_resamples=2_000,
        seed=3,
    )

    # Some bootstrap draws contain only the short prompt, so the upper bound
    # is intentionally conservative; with a larger realistic sample the
    # weighting converges.  A constant repeated sample tests the exact ratio.
    exact = weighted_paired_bootstrap_upper_bound(
        [1.0] * 40,
        [100] * 40,
        seed=3,
    )
    assert upper >= 0.01
    assert exact == pytest.approx(0.01)


def test_budget_validation_rejects_invalid_policy():
    with pytest.raises(ValueError):
        DamageBudget(max_top1_flip_rate=1.1)
    with pytest.raises(ValueError):
        DamageBudget(max_nll_increase_upper_ci=-0.1)
