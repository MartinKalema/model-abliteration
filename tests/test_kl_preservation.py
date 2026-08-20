"""Focused tests for measured baseline-vs-candidate KL preservation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from obliteratus.analysis.kl_preservation import (
    CausalLMBatch,
    KLPreservationError,
    KLPreservationThresholds,
    evaluate_kl_preservation,
    select_kl_preserving_candidate,
)


class _ToyCausalLM(nn.Module):
    """Small deterministic causal LM whose distribution is easy to perturb."""

    def __init__(
        self,
        drift: float = 0.0,
        *,
        drift_input_id: int | None = None,
        nonfinite: bool = False,
    ):
        super().__init__()
        self.logit_bias = nn.Parameter(torch.tensor([drift, -0.5 * drift, 0.0, 0.0]))
        self.drift_input_id = drift_input_id
        self.nonfinite = nonfinite
        self.grad_enabled_during_forward: list[bool] = []

    def forward(self, *, input_ids, attention_mask):
        del attention_mask
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        preferred = (input_ids + 1) % self.logit_bias.numel()
        logits = 2.0 * F.one_hot(
            preferred,
            num_classes=self.logit_bias.numel(),
        ).to(self.logit_bias.dtype)
        if self.drift_input_id is None:
            logits = logits + self.logit_bias
        else:
            selected = (input_ids == self.drift_input_id).unsqueeze(-1)
            logits = logits + selected * self.logit_bias
        if self.nonfinite:
            logits = logits.clone()
            logits[..., 0] = float("nan")
        return SimpleNamespace(logits=logits)


def _batch() -> CausalLMBatch:
    return CausalLMBatch(
        input_ids=torch.tensor(
            [
                [0, 1, 2, 3, 0],
                [2, 1, 0, 3, 1],
            ]
        ),
        attention_mask=torch.tensor(
            [
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
            ]
        ),
        response_mask=torch.tensor(
            [
                [0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1],
            ]
        ),
    )


def test_candidate_outputs_change_measured_response_token_kl():
    baseline = _ToyCausalLM()
    identical = _ToyCausalLM()
    changed = _ToyCausalLM(drift=1.75)

    no_op = evaluate_kl_preservation(baseline, identical, [_batch()])
    edited = evaluate_kl_preservation(baseline, changed, [_batch()])

    assert no_op.direction == "forward"
    assert no_op.mean_kl == pytest.approx(0.0, abs=1e-7)
    assert no_op.p95_kl == pytest.approx(0.0, abs=1e-7)
    assert edited.mean_kl > 0.0
    assert edited.p95_kl > 0.0
    assert edited.token_count == 5
    assert edited.batch_count == 1


def test_forward_reverse_and_symmetric_directions_are_explicit():
    baseline = _ToyCausalLM()
    changed = _ToyCausalLM(drift=1.25)

    forward = evaluate_kl_preservation(baseline, changed, [_batch()], direction="forward")
    reverse = evaluate_kl_preservation(baseline, changed, [_batch()], direction="reverse")
    symmetric = evaluate_kl_preservation(
        baseline,
        changed,
        [_batch()],
        direction="symmetric",
    )

    assert forward.mean_kl != pytest.approx(reverse.mean_kl)
    assert symmetric.mean_kl == pytest.approx(0.5 * (forward.mean_kl + reverse.mean_kl))
    with pytest.raises(ValueError, match="direction"):
        evaluate_kl_preservation(baseline, changed, [_batch()], direction="sideways")


def test_padding_is_excluded_by_causal_response_mask_alignment():
    padded = CausalLMBatch(
        input_ids=torch.tensor([[0, 1, 2, 3, 3, 3]]),
        attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0]]),
        response_mask=torch.tensor([[0, 1, 1, 0, 0, 0]]),
    )
    trimmed = CausalLMBatch(
        input_ids=torch.tensor([[0, 1, 2]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        response_mask=torch.tensor([[0, 1, 1]]),
    )
    baseline = _ToyCausalLM()
    changed = _ToyCausalLM(drift=0.8)

    padded_metrics = evaluate_kl_preservation(baseline, changed, [padded])
    trimmed_metrics = evaluate_kl_preservation(baseline, changed, [trimmed])

    assert padded_metrics.token_count == trimmed_metrics.token_count == 2
    assert padded_metrics.mean_kl == pytest.approx(trimmed_metrics.mean_kl)
    assert padded_metrics.p95_kl == pytest.approx(trimmed_metrics.p95_kl)


def test_only_explicit_response_token_distributions_contribute():
    baseline = _ToyCausalLM()
    changed = _ToyCausalLM(drift=2.0, drift_input_id=2)
    excludes_changed_distribution = CausalLMBatch(
        input_ids=torch.tensor([[0, 1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        response_mask=torch.tensor([[0, 1, 0, 0]]),
    )
    includes_changed_distribution = CausalLMBatch(
        input_ids=torch.tensor([[0, 1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1, 1]]),
        response_mask=torch.tensor([[0, 0, 0, 1]]),
    )

    excluded = evaluate_kl_preservation(
        baseline,
        changed,
        [excludes_changed_distribution],
    )
    included = evaluate_kl_preservation(
        baseline,
        changed,
        [includes_changed_distribution],
    )

    assert excluded.mean_kl == pytest.approx(0.0, abs=1e-7)
    assert included.mean_kl > 0.0


def test_hard_mean_and_p95_thresholds_reject_candidate():
    baseline = _ToyCausalLM()
    changed = _ToyCausalLM(drift=1.5)

    selection = select_kl_preserving_candidate(
        baseline,
        {"changed": changed},
        [_batch()],
        efficacy={"changed": 1.0},
        thresholds=KLPreservationThresholds(max_mean_kl=0.0, max_p95_kl=0.0),
    )

    assert selection.selected_name is None
    assert selection.eligible == ()
    assert len(selection.rejected) == 1
    assert selection.rejected[0].metrics is not None
    assert "mean KL" in selection.rejected[0].rejection_reason
    assert "p95 KL" in selection.rejected[0].rejection_reason


def test_selection_ranks_lower_kl_then_efficacy_and_name_deterministically():
    baseline = _ToyCausalLM()
    lower_kl = _ToyCausalLM(drift=0.1)
    higher_kl = _ToyCausalLM(drift=0.2)
    same_as_lower = _ToyCausalLM(drift=0.1)

    selection = select_kl_preserving_candidate(
        baseline,
        {
            "same-low-higher-efficacy": same_as_lower,
            "higher-kl": higher_kl,
            "lower-kl": lower_kl,
        },
        [_batch()],
        efficacy={
            "same-low-higher-efficacy": 0.9,
            "higher-kl": 1.0,
            "lower-kl": 0.2,
        },
        thresholds=KLPreservationThresholds(max_mean_kl=10.0, max_p95_kl=10.0),
    )

    assert selection.selected_name == "same-low-higher-efficacy"
    assert [result.name for result in selection.eligible] == [
        "same-low-higher-efficacy",
        "lower-kl",
        "higher-kl",
    ]


def test_nonfinite_selected_logits_fail_closed():
    baseline = _ToyCausalLM()
    broken = _ToyCausalLM(nonfinite=True)

    with pytest.raises(KLPreservationError, match="non-finite"):
        evaluate_kl_preservation(baseline, broken, [_batch()])

    selection = select_kl_preserving_candidate(
        baseline,
        {"broken": broken},
        [_batch()],
        efficacy={"broken": 10.0},
        thresholds=KLPreservationThresholds(max_mean_kl=10.0, max_p95_kl=10.0),
    )
    assert selection.selected_name is None
    assert selection.rejected[0].metrics is None
    assert "non-finite" in selection.rejected[0].rejection_reason


@pytest.mark.parametrize(
    "batches",
    [
        [],
        [
            CausalLMBatch(
                input_ids=torch.tensor([[0, 1]]),
                attention_mask=torch.tensor([[1, 1]]),
                response_mask=torch.tensor([[0, 0]]),
            )
        ],
        [
            CausalLMBatch(
                input_ids=torch.tensor([[0, 1]]),
                attention_mask=torch.tensor([[1, 0]]),
                response_mask=torch.tensor([[0, 1]]),
            )
        ],
    ],
)
def test_empty_or_malformed_response_inputs_fail_closed(batches):
    with pytest.raises(ValueError):
        evaluate_kl_preservation(_ToyCausalLM(), _ToyCausalLM(), batches)


def test_evaluation_uses_no_grad_restores_modes_and_never_changes_parameters():
    baseline = _ToyCausalLM()
    candidate = _ToyCausalLM(drift=0.4)
    baseline.train()
    candidate.eval()
    baseline_before = tuple(parameter.detach().clone() for parameter in baseline.parameters())
    candidate_before = tuple(parameter.detach().clone() for parameter in candidate.parameters())

    evaluate_kl_preservation(baseline, candidate, [_batch()])

    assert baseline.training is True
    assert candidate.training is False
    assert baseline.grad_enabled_during_forward == [False]
    assert candidate.grad_enabled_during_forward == [False]
    for before, after in zip(baseline_before, baseline.parameters(), strict=True):
        assert torch.equal(before, after.detach())
        assert after.grad is None
    for before, after in zip(candidate_before, candidate.parameters(), strict=True):
        assert torch.equal(before, after.detach())
        assert after.grad is None


@pytest.mark.parametrize(
    ("kwargs", "error_type"),
    [
        ({"max_mean_kl": -0.1, "max_p95_kl": 1.0}, ValueError),
        ({"max_mean_kl": 1.0, "max_p95_kl": float("nan")}, ValueError),
        ({"max_mean_kl": True, "max_p95_kl": 1.0}, TypeError),
    ],
)
def test_threshold_policy_must_be_finite_and_nonnegative(kwargs, error_type):
    with pytest.raises(error_type):
        KLPreservationThresholds(**kwargs)
