"""Integration checks for measured KL/CoT preservation pipeline controls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.analysis.cot_preservation import CoTPreservationExample
from obliteratus.evaluation.damage_gate import (
    AcceptanceBudget,
    DamageAssessment,
    DamageBudget,
    DamageGateError,
)


def _unique_prompt_pairs(count: int = 96) -> tuple[list[str], list[str]]:
    return (
        [f"harmful discovery prompt {index}" for index in range(count)],
        [f"benign preservation prompt {index}" for index in range(count)],
    )


def _search_pipeline(*, cot_aware: bool = False) -> AbliterationPipeline:
    harmful, harmless = _unique_prompt_pairs()
    return AbliterationPipeline(
        model_name="offline",
        method="basic",
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        use_kl_optimization=True,
        kl_budget=0.03,
        kl_search_steps=5,
        cot_aware=cot_aware,
        regularization=0.0,
        damage_eval_max_samples=64,
    )


def _assessment(
    pipeline: AbliterationPipeline,
    *,
    accepted: bool,
    mean_kl: float,
    p95_kl: float,
    refusal_rate: float,
) -> DamageAssessment:
    metrics = {
        "nll_increase_upper_ci": 0.0,
        "sampled_token_kl_mean": mean_kl,
        "sampled_token_kl_upper_ci": mean_kl,
        "sampled_token_kl_p95": p95_kl,
        "top1_flip_rate": 0.0,
        "coherence_drop": 0.0,
        "new_degenerate_count": 0,
        "nonfinite_output_count": 0,
        "refusal_rate": refusal_rate,
    }
    failures = () if accepted else ("candidate failed a hard gate",)
    return DamageAssessment(
        accepted=accepted,
        damage_accepted=accepted,
        efficacy_accepted=accepted,
        violations=failures,
        inconclusive=(),
        metrics=metrics,
        budget=pipeline.damage_budget,
    )


def test_constructor_enables_measured_preservation_and_declares_gate_limits() -> None:
    pipeline = _search_pipeline(cot_aware=True)

    assert pipeline.use_kl_optimization is True
    assert pipeline.cot_aware is True
    assert len(pipeline._cot_examples) == 8
    assert pipeline.damage_budget.damage.max_sampled_token_kl_upper_ci == 0.03
    assert pipeline.damage_budget.damage.max_cot_reasoning_ce_increase == 0.25
    assert pipeline.damage_budget.damage.max_cot_answer_ce_increase == 0.15
    assert pipeline.damage_budget.damage.min_cot_eval_examples == 4
    assert pipeline._kl_regularization_candidates() == pytest.approx(
        (0.0, 0.2375, 0.475, 0.7125, 0.95)
    )


def test_preservation_metadata_is_strict_json_and_contains_no_trace_text() -> None:
    pipeline = _search_pipeline(cot_aware=True)
    pipeline._kl_selected_regularization = 0.2375
    pipeline.regularization = 0.2375

    metadata = pipeline._build_metadata()
    encoded = json.dumps(metadata, allow_nan=False)

    assert metadata["kl_preservation"]["selected_regularization"] == pytest.approx(0.2375)
    assert metadata["damage_gate"]["assessment_scope"] == ("candidate_search_confirmation_only")
    assert len(metadata["cot_preservation"]["reference_text_fingerprints"]) == 8
    assert "Adding five to seven" not in encoded


def test_paper_som_metadata_separates_behavioral_search_from_kl_grid() -> None:
    pipeline = AbliterationPipeline(model_name="offline", method="som")

    pending = pipeline._build_metadata()
    assert pending["method_config"]["som_ordered_subset_size"] == 5
    assert pending["method_config"]["som_search_trials"] == 512
    assert pending["method_config"]["som_seed"] == 0
    assert pending["kl_preservation"]["algorithm"] == (
        "som_behavioral_tpe_winner_disjoint_damage_confirmation"
    )
    assert pending["kl_preservation"]["search_steps"] is None
    assert pending["kl_preservation"]["requested_regularization"] is None
    assert pending["som_paper_search"]["completed"] is False

    pipeline._som_source_layer = 7
    pipeline._som_confirmation_evidence = {"example_count": 32, "asr": 0.75}
    pipeline._som_paper_result = SimpleNamespace(
        to_metadata=lambda: {"method": "toy-paper-som", "search": {"trials": 2}}
    )
    completed = pipeline._build_metadata()

    assert completed["som_paper_search"]["completed"] is True
    assert completed["som_paper_search"]["source_layer"] == 7
    assert completed["som_paper_search"]["confirmation"]["example_count"] == 32
    json.dumps(completed, allow_nan=False)


def test_preservation_options_never_relax_stricter_predeclared_limits() -> None:
    harmful, harmless = _unique_prompt_pairs()
    budget = AcceptanceBudget(
        damage=DamageBudget(
            max_sampled_token_kl_upper_ci=0.01,
            max_cot_reasoning_ce_increase=0.08,
            max_cot_answer_ce_increase=0.04,
            min_cot_eval_examples=4,
        )
    )

    pipeline = AbliterationPipeline(
        model_name="offline",
        method="basic",
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        use_kl_optimization=True,
        kl_budget=0.05,
        cot_aware=True,
        cot_reasoning_ce_budget=0.25,
        cot_answer_ce_budget=0.15,
        damage_budget=budget,
        damage_eval_max_samples=64,
    )

    assert pipeline.kl_budget == pytest.approx(0.01)
    assert pipeline.cot_reasoning_ce_budget == pytest.approx(0.08)
    assert pipeline.cot_answer_ce_budget == pytest.approx(0.04)
    assert pipeline.damage_budget.damage.max_sampled_token_kl_upper_ci == pytest.approx(0.01)
    assert pipeline.damage_budget.damage.max_cot_reasoning_ce_increase == pytest.approx(0.08)
    assert pipeline.damage_budget.damage.max_cot_answer_ce_increase == pytest.approx(0.04)


def test_existing_cot_minimum_is_not_lowered_and_applies_to_both_search_halves() -> None:
    harmful, harmless = _unique_prompt_pairs()
    examples = [
        CoTPreservationExample(
            prompt=f"Question {index}?",
            reference_reasoning=f" Reasoning {index}.",
            reference_answer=f" Answer {index}",
            example_id=f"example-{index}",
        )
        for index in range(12)
    ]
    budget = AcceptanceBudget(damage=DamageBudget(min_cot_eval_examples=6))

    pipeline = AbliterationPipeline(
        model_name="offline",
        method="basic",
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        use_kl_optimization=True,
        cot_aware=True,
        cot_preservation_examples=examples,
        cot_min_eval_examples=4,
        damage_budget=budget,
        damage_eval_max_samples=64,
    )

    assert pipeline.cot_min_eval_examples == 6
    assert pipeline.damage_budget.damage.min_cot_eval_examples == 6


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_kl_optimization": "false"}, "must be a boolean"),
        ({"cot_aware": "false"}, "must be a boolean"),
        ({"kl_budget": True}, "finite non-negative"),
        ({"cot_reasoning_ce_budget": True}, "finite non-negative"),
    ],
)
def test_preservation_constructor_rejects_ambiguous_types(kwargs, message) -> None:
    harmful, harmless = _unique_prompt_pairs()

    with pytest.raises((TypeError, ValueError), match=message):
        AbliterationPipeline(
            model_name="offline",
            method="basic",
            harmful_prompts=harmful,
            harmless_prompts=harmless,
            damage_eval_max_samples=64,
            **kwargs,
        )


def test_cot_reference_duplicates_are_rejected_before_splitting() -> None:
    harmful, harmless = _unique_prompt_pairs()
    example = {
        "prompt": "Question?",
        "reference_reasoning": " Reasoning.",
        "reference_answer": " Answer",
        "example_id": "one",
    }
    duplicate = {**example, "example_id": "two", "prompt": "  question?  "}

    with pytest.raises(ValueError, match="duplicate normalized prompt"):
        AbliterationPipeline(
            model_name="offline",
            method="basic",
            harmful_prompts=harmful,
            harmless_prompts=harmless,
            cot_aware=True,
            cot_preservation_examples=[example, duplicate, example, duplicate],
        )


def test_explicit_cot_references_require_the_preservation_gate() -> None:
    harmful, harmless = _unique_prompt_pairs()

    with pytest.raises(ValueError, match="requires cot_aware=True"):
        AbliterationPipeline(
            model_name="offline",
            method="basic",
            harmful_prompts=harmful,
            harmless_prompts=harmless,
            cot_aware=False,
            cot_preservation_examples=[],
        )


def test_cot_gate_rejects_unsafe_inconclusive_policy() -> None:
    harmful, harmless = _unique_prompt_pairs()
    budget = AcceptanceBudget(damage=DamageBudget(unsafe_allow_inconclusive=True))

    with pytest.raises(ValueError, match="fail-closed"):
        AbliterationPipeline(
            model_name="offline",
            method="basic",
            harmful_prompts=harmful,
            harmless_prompts=harmless,
            cot_aware=True,
            damage_budget=budget,
        )


def test_kl_search_rejects_modes_that_override_or_remove_its_hard_gate() -> None:
    harmful, harmless = _unique_prompt_pairs()
    no_p95 = AcceptanceBudget(damage=DamageBudget(max_p95_sampled_token_kl=None))

    with pytest.raises(ValueError, match="p95"):
        AbliterationPipeline(
            model_name="offline",
            method="basic",
            harmful_prompts=harmful,
            harmless_prompts=harmless,
            use_kl_optimization=True,
            damage_budget=no_p95,
            damage_eval_max_samples=64,
        )
    with pytest.raises(ValueError, match="invert_refusal overrides"):
        AbliterationPipeline(
            model_name="offline",
            method="basic",
            harmful_prompts=harmful,
            harmless_prompts=harmless,
            use_kl_optimization=True,
            invert_refusal=True,
            damage_eval_max_samples=64,
        )


def test_kl_search_restores_every_trial_and_exactly_replays_lowest_kl_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _search_pipeline()
    pipeline.handle = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="right", pad_token_id=0)
    )
    pipeline._damage_baseline = ["all"]
    pipeline._baseline_generation_health = {"scope": "all"}
    pipeline._selection_generation_health_baseline = {"scope": "selection"}
    pipeline._confirmation_generation_health_baseline = {"scope": "confirmation"}
    pipeline._layer_excise_weights = {}
    pipeline._quality_metrics = {}
    pipeline._holdout_harmful = ["original harmful"]
    pipeline._holdout_harmless = ["original harmless"]
    pipeline._auto_selection_harmful = ["selection harmful"]
    pipeline._auto_selection_harmless = ["selection harmless"]
    pipeline._auto_confirmation_harmful = ["confirmation harmful"]
    pipeline._auto_confirmation_harmless = ["confirmation harmless"]

    restore_calls: list[float] = []
    excisions: list[float] = []

    monkeypatch.setattr(
        pipeline,
        "_assert_auto_projection_prerequisites",
        lambda purpose="": None,
    )
    monkeypatch.setattr(
        pipeline,
        "_split_auto_locality_baseline",
        lambda baseline: (["selection"], ["confirmation"]),
    )

    def restore(_weights, *, purpose=""):
        assert purpose == "use_kl_optimization"
        restore_calls.append(pipeline.regularization)

    def excise():
        excisions.append(pipeline.regularization)

    def verify():
        regularization = round(pipeline.regularization, 4)
        if pipeline._damage_baseline == ["confirmation"]:
            assert pipeline._generation_health_prompts == (
                pipeline._confirmation_generation_health_prompts
            )
            assert pipeline._baseline_generation_health == {"scope": "confirmation"}
            assert regularization == 0.2375
            return _assessment(
                pipeline,
                accepted=True,
                mean_kl=0.025,
                p95_kl=0.08,
                refusal_rate=0.12,
            )
        assert pipeline._generation_health_prompts == (
            pipeline._selection_generation_health_prompts
        )
        assert pipeline._baseline_generation_health == {"scope": "selection"}
        if regularization == 0.0:
            return _assessment(
                pipeline,
                accepted=True,
                mean_kl=0.028,
                p95_kl=0.10,
                refusal_rate=0.02,
            )
        if regularization == 0.2375:
            return _assessment(
                pipeline,
                accepted=True,
                mean_kl=0.015,
                p95_kl=0.07,
                refusal_rate=0.10,
            )
        return _assessment(
            pipeline,
            accepted=False,
            mean_kl=0.01,
            p95_kl=0.05,
            refusal_rate=0.40,
        )

    monkeypatch.setattr(pipeline, "_restore_auto_projection_baseline", restore)
    monkeypatch.setattr(pipeline, "_excise", excise)
    monkeypatch.setattr(pipeline, "_verify", verify)
    monkeypatch.setattr(pipeline, "_remove_activation_steering", lambda: 0)

    result = pipeline._run_kl_preservation_search()

    assert result.accepted is True
    assert pipeline._kl_selected_regularization == pytest.approx(0.2375)
    assert pipeline.regularization == pytest.approx(0.2375)
    assert excisions.count(0.2375) == 2
    assert len(restore_calls) == len(pipeline._kl_regularization_candidates()) + 1
    assert pipeline._damage_baseline == ["all"]
    assert pipeline._holdout_harmful == ["original harmful"]
    assert pipeline._holdout_harmless == ["original harmless"]
    assert pipeline._baseline_generation_health == {"scope": "all"}
    assert set(pipeline._selection_generation_health_prompts).isdisjoint(
        pipeline._confirmation_generation_health_prompts
    )


@pytest.mark.parametrize(
    "search_method",
    ["_run_auto_projection_search", "_run_kl_preservation_search"],
)
def test_search_wrapper_restores_split_state_when_cot_selection_baseline_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    search_method: str,
) -> None:
    pipeline = _search_pipeline(cot_aware=True)
    pipeline.handle = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="left", pad_token_id=17)
    )
    pipeline._damage_baseline = ["original baseline"]
    pipeline._baseline_generation_health = {"scope": "original"}
    pipeline._holdout_harmful = ["original harmful"]
    pipeline._holdout_harmless = ["original harmless"]
    pipeline.verify_sample_size = 19
    original_cot_examples = pipeline._cot_examples[-2:]
    original_cot_baseline = object()
    pipeline._cot_active_examples = original_cot_examples
    pipeline._cot_baseline = original_cot_baseline
    pipeline._cot_selection_baseline = None

    monkeypatch.setattr(
        pipeline,
        "_assert_auto_projection_prerequisites",
        lambda purpose="": None,
    )
    monkeypatch.setattr(
        pipeline,
        "_split_auto_locality_baseline",
        lambda baseline: (["selection"], ["confirmation"]),
    )

    with pytest.raises(RuntimeError, match="CoT selection baseline"):
        getattr(pipeline, search_method)()

    assert pipeline._damage_baseline == ["original baseline"]
    assert pipeline._baseline_generation_health == {"scope": "original"}
    assert pipeline._holdout_harmful == ["original harmful"]
    assert pipeline._holdout_harmless == ["original harmless"]
    assert pipeline.verify_sample_size == 19
    assert pipeline._cot_active_examples == original_cot_examples
    assert pipeline._cot_baseline is original_cot_baseline
    assert pipeline.handle.tokenizer.padding_side == "left"
    assert pipeline.handle.tokenizer.pad_token_id == 17
    assert pipeline._projection_auto_tokenizer_state is None


@pytest.mark.parametrize("failure_call", [1, 2])
def test_generation_baseline_capture_restores_prompt_scope_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    pipeline = _search_pipeline()

    class Tokenizer:
        def __call__(self, prompts, **_kwargs):
            input_ids = torch.ones((len(prompts), 2), dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, input_ids, attention_mask):
            logits = self.anchor + torch.zeros(
                (*input_ids.shape, 4),
                device=input_ids.device,
            )
            return SimpleNamespace(logits=logits)

    pipeline.handle = SimpleNamespace(model=Model(), tokenizer=Tokenizer())
    original_prompts = pipeline._generation_health_prompts
    calls = 0

    def generation_health():
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("generation baseline failed")
        return {"coherence": 1.0, "degenerate_count": 0}

    monkeypatch.setattr(
        pipeline,
        "_measure_benign_generation_health",
        generation_health,
    )

    with pytest.raises(RuntimeError, match="generation baseline failed"):
        pipeline._capture_damage_baseline()

    assert pipeline._generation_health_prompts == original_prompts
    assert pipeline._selection_generation_health_baseline is None
    assert pipeline._confirmation_generation_health_baseline is None


def test_kl_confirmation_rejection_resets_selected_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _search_pipeline()
    pipeline.handle = SimpleNamespace(
        tokenizer=SimpleNamespace(padding_side="right", pad_token_id=0)
    )
    pipeline._damage_baseline = ["selection"]
    pipeline._layer_excise_weights = {}
    pipeline._auto_confirmation_harmful = ["confirmation harmful"]
    pipeline._auto_confirmation_harmless = ["confirmation harmless"]
    pipeline._confirmation_generation_health_baseline = {"scope": "confirmation"}

    monkeypatch.setattr(
        pipeline,
        "_assert_auto_projection_prerequisites",
        lambda purpose="": None,
    )
    monkeypatch.setattr(
        pipeline,
        "_restore_auto_projection_baseline",
        lambda _weights, *, purpose="": None,
    )
    monkeypatch.setattr(pipeline, "_excise", lambda: None)
    monkeypatch.setattr(pipeline, "_remove_activation_steering", lambda: 0)

    def verify():
        if pipeline._damage_baseline == ["confirmation"]:
            return _assessment(
                pipeline,
                accepted=False,
                mean_kl=0.02,
                p95_kl=0.08,
                refusal_rate=0.1,
            )
        return _assessment(
            pipeline,
            accepted=round(pipeline.regularization, 4) == 0.2375,
            mean_kl=0.02,
            p95_kl=0.08,
            refusal_rate=0.1,
        )

    def reject(assessment):
        assert assessment.accepted is False
        assert pipeline._kl_selected_regularization is None
        assert pipeline.regularization == pipeline._requested_regularization
        raise DamageGateError(assessment)

    monkeypatch.setattr(pipeline, "_verify", verify)
    monkeypatch.setattr(pipeline, "_reject_and_restore", reject)

    with pytest.raises(DamageGateError):
        pipeline._run_kl_preservation_search_inner(["confirmation"])
