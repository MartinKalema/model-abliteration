"""Focused tests for paired benign-locality measurements."""

from __future__ import annotations

import json
import math

import pytest
import torch

from obliteratus.evaluation.locality import (
    capture_locality_baseline,
    combine_locality_measurements,
    compare_locality_candidate,
    measure_locality,
    select_evenly_spaced_positions,
)


def test_even_position_selection_uses_valid_ranks_and_includes_last_token():
    mask = torch.tensor([0, 1, 0, 1, 1, 0, 1, 1, 1, 0])

    assert select_evenly_spaced_positions(mask, 4) == (1, 4, 6, 8)
    assert select_evenly_spaced_positions(mask, 1) == (8,)
    assert select_evenly_spaced_positions(mask, 20) == (1, 3, 4, 6, 7, 8)


def test_identical_logits_have_zero_drift_and_padding_is_fully_excluded():
    baseline = torch.zeros((2, 5, 3), dtype=torch.float32)
    candidate = baseline.clone()
    # Non-finite padding would contaminate naive masked means because NaN * 0
    # is still NaN.  It must not enter either NLL, KL, or the hard-failure count.
    candidate[0, 3:, :] = float("nan")
    candidate[1, 0, :] = float("nan")
    input_ids = torch.tensor(
        [
            [0, 1, 2, 0, 0],
            [0, 2, 1, 0, 2],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
        ]
    )

    result = measure_locality(
        baseline,
        candidate,
        input_ids,
        attention_mask,
        max_kl_positions_per_prompt=2,
        bootstrap_resamples=100,
    )

    assert result.metrics["baseline_nll"] == pytest.approx(math.log(3))
    assert result.metrics["candidate_nll"] == pytest.approx(math.log(3))
    assert result.metrics["nll_increase"] == pytest.approx(0.0)
    assert result.metrics["nll_increase_upper_ci"] == pytest.approx(0.0)
    assert result.metrics["perplexity_ratio"] == pytest.approx(1.0)
    assert result.metrics["sampled_token_kl_mean"] == pytest.approx(0.0)
    assert result.metrics["sampled_token_kl_upper_ci"] == pytest.approx(0.0)
    assert result.metrics["sampled_token_kl_p95"] == pytest.approx(0.0)
    assert result.metrics["top1_flip_rate"] == pytest.approx(0.0)
    assert result.metrics["nonfinite_output_count"] == 0
    assert result.metrics["eval_prompt_count"] == 2
    assert result.metrics["eval_token_count"] == 5
    assert result.metrics["sampled_token_count"] == 4
    assert result.prompts[0].token_count == 2
    assert result.prompts[1].token_count == 3
    assert result.prompts[0].sampled_positions == (0, 2)
    assert result.prompts[1].sampled_positions == (1, 4)
    # The normal, finite result is ready for strict JSON metadata.
    json.dumps(result.to_dict(), allow_nan=False)


def test_two_phase_capture_retains_only_compact_cpu_baseline_artifacts():
    baseline_logits = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    candidate_logits = baseline_logits.clone()
    input_ids = torch.tensor([[0, 1, 2, 3, 0], [0, 0, 3, 2, 1]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [0, 0, 1, 1, 1]])
    expected_input_ids = input_ids.clone()
    expected_attention_mask = attention_mask.clone()

    baseline = capture_locality_baseline(
        baseline_logits,
        input_ids,
        attention_mask,
        max_kl_positions_per_prompt=2,
    )

    assert baseline.input_ids.device.type == "cpu"
    assert baseline.attention_mask.device.type == "cpu"
    assert torch.equal(baseline.input_ids, expected_input_ids)
    assert torch.equal(baseline.attention_mask, expected_attention_mask)
    assert not hasattr(baseline, "logits")
    assert [prompt.sampled_logits.shape for prompt in baseline.prompts] == [(2, 4), (2, 4)]
    assert all(prompt.sampled_logits.device.type == "cpu" for prompt in baseline.prompts)
    assert all(prompt.sampled_logits.dtype == torch.float32 for prompt in baseline.prompts)
    assert baseline.prompts[0].sampled_positions[-1] == 3
    assert baseline.prompts[1].sampled_positions[-1] == 4
    json.dumps(baseline.metadata_dict(), allow_nan=False)

    # Simulate the original model buffers and caller-owned encoding being
    # reused during an in-place edit.  The captured evidence remains immutable.
    baseline_logits.fill_(float("nan"))
    input_ids.zero_()
    attention_mask.zero_()
    result = compare_locality_candidate(
        baseline,
        candidate_logits,
        bootstrap_resamples=100,
    )

    assert result.metrics["nll_increase"] == pytest.approx(0.0)
    assert result.metrics["sampled_token_kl_mean"] == pytest.approx(0.0)
    assert result.metrics["nonfinite_output_count"] == 0


def test_compact_kl_comparison_quantizes_both_sides_symmetrically():
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn((2, 5, 11), generator=generator, dtype=torch.float32)
    input_ids = torch.randint(0, 11, (2, 5), generator=generator)
    attention_mask = torch.ones((2, 5), dtype=torch.long)

    baseline = capture_locality_baseline(logits, input_ids, attention_mask)
    result = compare_locality_candidate(
        baseline,
        logits.clone(),
        bootstrap_resamples=100,
    )

    assert result.metrics["sampled_token_kl_mean"] == pytest.approx(0.0)
    assert result.metrics["sampled_token_kl_upper_ci"] == pytest.approx(0.0)
    assert result.metrics["sampled_token_kl_p95"] == pytest.approx(0.0)
    assert result.metrics["top1_flip_rate"] == pytest.approx(0.0)


def test_nll_is_token_weighted_across_unequal_prompt_lengths():
    # The first prompt has one predicted token and the second has four.  Give
    # every valid target exactly +0.2 nat NLL, making the weighted mean and
    # every paired-bootstrap resample exactly +0.2.
    input_ids = torch.zeros((2, 5), dtype=torch.long)
    attention_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    baseline = torch.zeros((2, 5, 2), dtype=torch.float32)
    candidate = baseline.clone()
    correct_probability = 0.5 * math.exp(-0.2)
    shifted_logits = torch.tensor(
        [math.log(correct_probability), math.log(1.0 - correct_probability)]
    )
    candidate[0, :2] = shifted_logits
    candidate[1, :] = shifted_logits

    result = measure_locality(
        baseline,
        candidate,
        input_ids,
        attention_mask,
        bootstrap_resamples=100,
    )

    assert [prompt.token_count for prompt in result.prompts] == [1, 4]
    assert result.prompts[0].loss_delta_sum == pytest.approx(0.2)
    assert result.prompts[1].loss_delta_sum == pytest.approx(0.8)
    assert result.metrics["nll_increase"] == pytest.approx(0.2)
    assert result.metrics["nll_increase_upper_ci"] == pytest.approx(0.2)
    assert result.metrics["perplexity_ratio"] == pytest.approx(math.exp(0.2))

    batches = [
        measure_locality(
            baseline[index : index + 1],
            candidate[index : index + 1],
            input_ids[index : index + 1],
            attention_mask[index : index + 1],
            bootstrap_resamples=100,
        )
        for index in range(2)
    ]
    combined = combine_locality_measurements(
        batches,
        bootstrap_resamples=100,
    )
    assert combined.metrics == pytest.approx(result.metrics)
    assert [prompt.prompt_index for prompt in combined.prompts] == [0, 1]


def test_forward_kl_is_fp32_per_row_and_exposed_per_prompt():
    baseline_distribution = torch.tensor([0.75, 0.25], dtype=torch.float16)
    candidate_distribution = torch.tensor([0.5, 0.5], dtype=torch.float16)
    baseline = baseline_distribution.log().repeat(1, 2, 1)
    candidate = candidate_distribution.log().repeat(1, 2, 1)

    result = measure_locality(
        baseline,
        candidate,
        torch.tensor([[0, 1]]),
        torch.ones((1, 2), dtype=torch.long),
        max_kl_positions_per_prompt=2,
        bootstrap_resamples=100,
    )

    expected = 0.75 * math.log(0.75 / 0.5) + 0.25 * math.log(0.25 / 0.5)
    assert result.prompts[0].sampled_positions == (0, 1)
    assert result.prompts[0].sampled_token_kl == pytest.approx((expected, expected), abs=5e-4)
    assert result.prompts[0].sampled_token_kl_mean == pytest.approx(expected, abs=5e-4)
    assert result.metrics["sampled_token_kl_mean"] == pytest.approx(expected, abs=5e-4)
    assert result.metrics["sampled_token_kl_upper_ci"] == pytest.approx(expected, abs=5e-4)
    assert result.metrics["sampled_token_kl_p95"] == pytest.approx(expected, abs=5e-4)
    assert result.metrics["top1_flip_rate"] == 0.0


def test_last_prompt_position_is_measured_and_nonfinite_rows_fail_closed():
    baseline = torch.zeros((1, 3, 2), dtype=torch.float32)
    candidate = baseline.clone()
    candidate[0, -1, 1] = float("nan")

    result = measure_locality(
        baseline,
        candidate,
        torch.tensor([[0, 1, 0]]),
        torch.ones((1, 3), dtype=torch.long),
        max_kl_positions_per_prompt=2,
        bootstrap_resamples=100,
    )

    assert result.prompts[0].sampled_positions[-1] == 2
    assert math.isinf(result.prompts[0].sampled_token_kl[-1])
    assert math.isinf(result.metrics["sampled_token_kl_upper_ci"])
    assert math.isinf(result.metrics["sampled_token_kl_p95"])
    assert result.metrics["candidate_nonfinite_logit_count"] == 1
    assert result.metrics["nonfinite_output_count"] == 1
    assert result.metrics["top1_flip_rate"] >= 0.5


def test_bootstrap_metrics_are_deterministic_for_a_fixed_seed():
    generator = torch.Generator().manual_seed(7)
    baseline = torch.randn((6, 7, 5), generator=generator)
    candidate = baseline + 0.1 * torch.randn((6, 7, 5), generator=generator)
    input_ids = torch.randint(0, 5, (6, 7), generator=generator)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ]
    )

    first = measure_locality(
        baseline,
        candidate,
        input_ids,
        attention_mask,
        bootstrap_resamples=250,
        bootstrap_seed=123,
    )
    second = measure_locality(
        baseline,
        candidate,
        input_ids,
        attention_mask,
        bootstrap_resamples=250,
        bootstrap_seed=123,
    )

    assert first.metrics["nll_increase_upper_ci"] == second.metrics["nll_increase_upper_ci"]
    assert first.metrics["sampled_token_kl_upper_ci"] == second.metrics["sampled_token_kl_upper_ci"]


def test_invalid_or_empty_batches_are_rejected():
    with pytest.raises(ValueError, match="identical shapes"):
        measure_locality(
            torch.zeros((1, 2, 3)),
            torch.zeros((1, 2, 4)),
            torch.zeros((1, 2), dtype=torch.long),
            torch.ones((1, 2), dtype=torch.long),
            bootstrap_resamples=100,
        )

    with pytest.raises(ValueError, match="no real causal target tokens"):
        measure_locality(
            torch.zeros((1, 2, 3)),
            torch.zeros((1, 2, 3)),
            torch.zeros((1, 2), dtype=torch.long),
            torch.tensor([[0, 1]]),
            bootstrap_resamples=100,
        )
