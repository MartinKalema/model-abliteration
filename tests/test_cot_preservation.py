"""Focused tests for explicit teacher-forced CoT preservation metrics."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from obliteratus.analysis.cot_preservation import (
    DEFAULT_COT_PRESERVATION_EXAMPLES,
    CoTPreservationError,
    CoTPreservationExample,
    compare_cot_score_snapshots,
    evaluate_cot_preservation,
    score_cot_references,
)


class _WordTokenizer:
    model_max_length = 64

    def __init__(self) -> None:
        tokens = ["p", "q", "r", "s", "a", "b", "x"]
        self.vocabulary = {token: index + 1 for index, token in enumerate(tokens)}
        self.calls: list[dict[str, object]] = []

    def encode(self, text: str, **kwargs) -> list[int]:
        self.calls.append(dict(kwargs))
        return [self.vocabulary[token] for token in text.split()]


class _TransitionModel(torch.nn.Module):
    def __init__(
        self,
        tokenizer: _WordTokenizer,
        *,
        degraded_sources: set[int] | None = None,
        nonfinite: bool = False,
    ) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.mode_marker = torch.nn.Dropout(p=0.25)
        self.config = SimpleNamespace(max_position_embeddings=64)
        self.degraded_sources = set(degraded_sources or ())
        self.nonfinite = nonfinite
        self.grad_enabled_during_forward: list[bool] = []
        self.training_states_during_forward: list[tuple[bool, bool]] = []
        vocab = tokenizer.vocabulary
        self.transitions = {
            vocab["p"]: vocab["q"],
            vocab["q"]: vocab["r"],
            vocab["r"]: vocab["s"],
            vocab["s"]: vocab["a"],
            vocab["a"]: vocab["b"],
        }
        self.vocab_size = max(vocab.values()) + 2

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.grad_enabled_during_forward.append(torch.is_grad_enabled())
        self.training_states_during_forward.append((self.training, self.mode_marker.training))
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        batch_size, sequence_length = input_ids.shape
        logits = torch.full(
            (batch_size, sequence_length, self.vocab_size),
            -4.0,
            device=input_ids.device,
        )
        logits = logits + self.anchor * 0.0
        for batch_index in range(batch_size):
            for position in range(sequence_length):
                source = int(input_ids[batch_index, position])
                target = self.transitions.get(source, 0)
                logits[batch_index, position, target] = 4.0
                if source in self.degraded_sources:
                    logits[batch_index, position, target] = -4.0
                    logits[batch_index, position, (target + 1) % self.vocab_size] = 4.0
        if self.nonfinite:
            logits[:, 1, :] = float("nan")
        return SimpleNamespace(logits=logits)


@pytest.fixture
def tokenizer() -> _WordTokenizer:
    return _WordTokenizer()


@pytest.fixture
def example() -> CoTPreservationExample:
    return CoTPreservationExample(
        prompt="p q",
        reference_reasoning=" r s",
        reference_answer=" a b",
        example_id="toy",
    )


def test_prompt_targets_are_masked_and_tokenizer_never_pads_or_truncates(
    tokenizer: _WordTokenizer,
    example: CoTPreservationExample,
):
    baseline = _TransitionModel(tokenizer)
    # p -> q is wholly inside the prompt.  Damaging it must not change either
    # response-span metric.
    candidate = _TransitionModel(tokenizer, degraded_sources={tokenizer.vocabulary["p"]})

    report = evaluate_cot_preservation(baseline, candidate, tokenizer, [example])

    assert report.reasoning_ce_delta == pytest.approx(0.0)
    assert report.answer_ce_delta == pytest.approx(0.0)
    assert tokenizer.calls
    assert all(call["padding"] is False for call in tokenizer.calls)
    assert all(call["truncation"] is False for call in tokenizer.calls)


def test_reasoning_and_answer_segments_independently_affect_their_metrics(
    tokenizer: _WordTokenizer,
    example: CoTPreservationExample,
):
    baseline = _TransitionModel(tokenizer)
    reasoning_damage = _TransitionModel(tokenizer, degraded_sources={tokenizer.vocabulary["q"]})
    answer_damage = _TransitionModel(tokenizer, degraded_sources={tokenizer.vocabulary["s"]})

    reasoning_report = evaluate_cot_preservation(baseline, reasoning_damage, tokenizer, [example])
    answer_report = evaluate_cot_preservation(baseline, answer_damage, tokenizer, [example])

    assert reasoning_report.reasoning_ce_delta > 1.0
    assert reasoning_report.answer_ce_delta == pytest.approx(0.0)
    assert answer_report.reasoning_ce_delta == pytest.approx(0.0)
    assert answer_report.answer_ce_delta > 1.0
    assert reasoning_report.baseline.reasoning_token_count == 2
    assert reasoning_report.baseline.answer_token_count == 2


def test_separate_snapshots_support_scoring_the_same_model_before_and_after_edit(
    tokenizer: _WordTokenizer,
    example: CoTPreservationExample,
):
    model = _TransitionModel(tokenizer)
    baseline = score_cot_references(model, tokenizer, [example])

    model.degraded_sources.add(tokenizer.vocabulary["q"])
    candidate = score_cot_references(model, tokenizer, [example])
    report = compare_cot_score_snapshots(baseline, candidate)

    assert report.reasoning_ce_delta > 1.0
    assert report.answer_ce_delta == pytest.approx(0.0)
    gate_metrics = report.as_gate_metrics()
    assert gate_metrics["cot_reasoning_ce_increase"] == pytest.approx(report.reasoning_ce_delta)
    assert gate_metrics["cot_answer_ce_increase"] == pytest.approx(report.answer_ce_delta)
    assert gate_metrics["cot_eval_example_count"] == 1
    with pytest.raises(FrozenInstanceError):
        baseline.aggregate = candidate.aggregate  # type: ignore[misc]


def test_models_are_scored_under_no_grad_without_parameter_or_mode_mutation(
    tokenizer: _WordTokenizer,
    example: CoTPreservationExample,
):
    baseline = _TransitionModel(tokenizer)
    candidate = _TransitionModel(tokenizer)
    baseline.train()
    candidate.train()
    baseline.mode_marker.eval()
    before = {
        "baseline": {name: value.detach().clone() for name, value in baseline.named_parameters()},
        "candidate": {name: value.detach().clone() for name, value in candidate.named_parameters()},
    }

    evaluate_cot_preservation(baseline, candidate, tokenizer, [example])

    assert baseline.training is True
    assert baseline.mode_marker.training is False
    assert candidate.training is True
    assert candidate.mode_marker.training is True
    assert baseline.grad_enabled_during_forward == [False]
    assert candidate.grad_enabled_during_forward == [False]
    assert baseline.training_states_during_forward == [(False, False)]
    assert candidate.training_states_during_forward == [(False, False)]
    for name, value in baseline.named_parameters():
        assert torch.equal(value, before["baseline"][name])
        assert value.grad is None
    for name, value in candidate.named_parameters():
        assert torch.equal(value, before["candidate"][name])
        assert value.grad is None


def test_injected_generator_scores_only_returned_final_answer_text(
    tokenizer: _WordTokenizer,
    example: CoTPreservationExample,
):
    model = _TransitionModel(tokenizer)
    generation_grad_states: list[bool] = []

    def generate_final_answer(received: CoTPreservationExample) -> str:
        assert received is example
        generation_grad_states.append(torch.is_grad_enabled())
        return " A   B "

    report = evaluate_cot_preservation(
        model,
        model,
        tokenizer,
        [example],
        candidate_answer_generator=generate_final_answer,
    )

    assert generation_grad_states == [False]
    assert report.exact_final_answer_match_rate == 0.0
    assert report.normalized_final_answer_match_rate == 1.0
    assert report.examples[0].generated_final_answer == " A   B "


@pytest.mark.parametrize("examples", [[], ()])
def test_empty_example_collection_fails_closed(tokenizer: _WordTokenizer, examples):
    model = _TransitionModel(tokenizer)
    with pytest.raises(CoTPreservationError, match="at least one"):
        score_cot_references(model, tokenizer, examples)


def test_slow_tokenizer_boundary_merge_fails_closed(
    tokenizer: _WordTokenizer,
):
    class BoundaryMergingTokenizer:
        model_max_length = 64

        @staticmethod
        def encode(text: str, **_kwargs) -> list[int]:
            encodings = {"p": [1], "p r": [8], "p r a": [9]}
            return encodings[text]

    example = CoTPreservationExample(
        prompt="p",
        reference_reasoning=" r",
        reference_answer=" a",
        example_id="boundary_merge",
    )
    model = _TransitionModel(tokenizer)

    with pytest.raises(CoTPreservationError, match="merges a token"):
        score_cot_references(model, BoundaryMergingTokenizer(), [example])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt", ""),
        ("reference_reasoning", "   "),
        ("reference_answer", ""),
    ],
)
def test_missing_explicit_segment_fails_closed(field: str, value: str):
    values = {
        "prompt": "p q",
        "reference_reasoning": "r s",
        "reference_answer": "a b",
    }
    values[field] = value
    with pytest.raises(CoTPreservationError, match=field):
        CoTPreservationExample(**values)


def test_nonfinite_response_logits_fail_closed(
    tokenizer: _WordTokenizer,
    example: CoTPreservationExample,
):
    with pytest.raises(CoTPreservationError, match="non-finite"):
        evaluate_cot_preservation(
            _TransitionModel(tokenizer),
            _TransitionModel(tokenizer, nonfinite=True),
            tokenizer,
            [example],
        )


def test_training_states_are_restored_when_scoring_fails(
    tokenizer: _WordTokenizer,
    example: CoTPreservationExample,
):
    model = _TransitionModel(tokenizer, nonfinite=True)
    model.train()
    model.mode_marker.eval()

    with pytest.raises(CoTPreservationError, match="non-finite"):
        score_cot_references(model, tokenizer, [example])

    assert model.training is True
    assert model.mode_marker.training is False


def test_context_overflow_is_not_silently_truncated(
    tokenizer: _WordTokenizer,
    example: CoTPreservationExample,
):
    model = _TransitionModel(tokenizer)
    with pytest.raises(CoTPreservationError, match="truncation"):
        score_cot_references(model, tokenizer, [example], max_length=5)


def test_default_references_are_immutable_explicit_and_harmless():
    assert isinstance(DEFAULT_COT_PRESERVATION_EXAMPLES, tuple)
    assert len(DEFAULT_COT_PRESERVATION_EXAMPLES) >= 8
    assert len({example.example_id for example in DEFAULT_COT_PRESERVATION_EXAMPLES}) == len(
        DEFAULT_COT_PRESERVATION_EXAMPLES
    )
    assert all(example.prompt.strip() for example in DEFAULT_COT_PRESERVATION_EXAMPLES)
    assert all(example.reference_reasoning.strip() for example in DEFAULT_COT_PRESERVATION_EXAMPLES)
    assert all(example.reference_answer.strip() for example in DEFAULT_COT_PRESERVATION_EXAMPLES)
