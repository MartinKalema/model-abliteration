"""Pipeline-level tests for fail-closed damage acceptance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.evaluation.damage_gate import (
    AcceptanceBudget,
    DamageBudget,
    DamageGateError,
    EfficacyBudget,
    assess_candidate,
)
from obliteratus.models.loader import ModelHandle


def _small_budget() -> AcceptanceBudget:
    return AcceptanceBudget(
        damage=DamageBudget(
            min_eval_prompts=1,
            min_eval_tokens=1,
            min_sampled_tokens=1,
        ),
        efficacy=EfficacyBudget(min_eval_prompts=1),
    )


def _pipeline(tmp_path: Path) -> AbliterationPipeline:
    harmful = [f"harmful {index}" for index in range(4)]
    harmless = [f"harmless {index}" for index in range(4)]
    return AbliterationPipeline(
        model_name="test-model",
        output_dir=str(tmp_path / "candidate"),
        harmful_prompts=harmful,
        harmless_prompts=harmless,
        damage_budget=_small_budget(),
    )


def _passing_metrics() -> dict[str, float | int]:
    return {
        "nll_increase_upper_ci": 0.0,
        "sampled_token_kl_upper_ci": 0.0,
        "sampled_token_kl_p95": 0.0,
        "top1_flip_rate": 0.0,
        "coherence_drop": 0.0,
        "new_degenerate_count": 0,
        "nonfinite_output_count": 0,
        "eval_prompt_count": 1,
        "eval_token_count": 8,
        "sampled_token_count": 1,
        "refusal_rate": 0.0,
        "refusal_eval_count": 1,
    }


def test_run_rejects_before_rebirth_when_gate_fails(tmp_path):
    pipeline = _pipeline(tmp_path)
    calls: list[str] = []

    for name in ("_summon", "_probe", "_distill", "_capture_damage_baseline", "_excise"):
        setattr(pipeline, name, lambda name=name: calls.append(name))
    pipeline._remove_activation_steering = lambda: 0
    pipeline._free_gpu_memory = lambda: None
    pipeline._verify = lambda: assess_candidate({}, pipeline.damage_budget)
    pipeline._rebirth = lambda: calls.append("_rebirth") or pipeline.output_dir

    with pytest.raises(DamageGateError):
        pipeline.run()

    assert calls == [
        "_summon",
        "_probe",
        "_distill",
        "_capture_damage_baseline",
        "_excise",
    ]
    assert not pipeline.output_dir.exists()


def test_rebirth_itself_cannot_bypass_missing_assessment(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline._gather_state_dict = MagicMock()

    with pytest.raises(DamageGateError):
        pipeline._rebirth()

    pipeline._gather_state_dict.assert_not_called()
    assert not pipeline.output_dir.exists()


def test_rejected_candidate_restores_exact_snapshot(tmp_path):
    pipeline = _pipeline(tmp_path)
    model = torch.nn.Linear(3, 2, bias=False)
    original = model.weight.detach().clone()
    handle = ModelHandle(
        model=model,
        tokenizer=MagicMock(),
        config=MagicMock(model_type="test"),
        model_name="test",
        task="causal_lm",
    )
    handle.snapshot()
    model.weight.data.add_(10.0)
    pipeline.handle = handle
    assessment = assess_candidate({}, pipeline.damage_budget)

    with pytest.raises(DamageGateError):
        pipeline._reject_and_restore(assessment)

    assert torch.equal(model.weight, original)


def test_runtime_hooks_are_removed_before_artifact_verification(tmp_path):
    pipeline = _pipeline(tmp_path)
    first = MagicMock()
    second = MagicMock()
    pipeline._steering_hooks = [first, second]
    pipeline._damage_assessment = assess_candidate(
        _passing_metrics(),
        pipeline.damage_budget,
    )

    assert pipeline._remove_activation_steering() == 2
    first.remove.assert_called_once_with()
    second.remove.assert_called_once_with()
    assert pipeline._steering_hooks == []
    assert pipeline._damage_assessment is None


def test_head_surgery_is_opt_in_and_tied_storage_is_detected(tmp_path):
    pipeline = _pipeline(tmp_path)
    embedding = torch.nn.Embedding(8, 4)
    head = torch.nn.Linear(4, 8, bias=False)
    head.weight = embedding.weight

    assert pipeline.project_lm_head is False
    assert pipeline.project_embeddings is False
    assert pipeline._tensors_share_storage(head.weight, embedding.weight)


def test_passing_assessment_allows_rebirth_guard(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline._damage_assessment = assess_candidate(
        _passing_metrics(),
        pipeline.damage_budget,
    )

    pipeline._require_damage_gate_passed()


def test_invalid_legacy_kl_correction_cannot_be_called(tmp_path):
    pipeline = _pipeline(tmp_path)

    with pytest.raises(RuntimeError, match="legacy KL correction is disabled"):
        pipeline._kl_optimize_corrections(torch.nn.ModuleList(), 0)


def test_iterative_verify_cannot_reuse_stale_passing_metrics(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.handle = ModelHandle(
        model=torch.nn.Linear(3, 2, bias=False),
        tokenizer=MagicMock(),
        config=MagicMock(model_type="test"),
        model_name="test",
        task="causal_lm",
    )
    pipeline._quality_metrics = _passing_metrics()
    pipeline._measure_candidate_locality = lambda: None
    pipeline._measure_benign_generation_health = lambda **_kwargs: None
    pipeline._free_gpu_memory = lambda: None
    pipeline._on_log = lambda _message: None
    pipeline._on_stage = lambda _result: None

    assessment = pipeline._verify()

    assert assessment.accepted is False
    assert pipeline._quality_metrics["perplexity"] is None
    assert "nll_increase_upper_ci" not in pipeline._quality_metrics
    assert pipeline._quality_metrics["acceptance"]["accepted"] is False


def test_new_degenerate_prompt_is_detected_even_when_total_is_unchanged():
    baseline = {
        "degenerate_count": 1,
        "degenerate_prompt_indices": [2],
    }
    candidate = {
        "degenerate_count": 1,
        "degenerate_prompt_indices": [7],
    }

    assert AbliterationPipeline._count_new_degenerate_outputs(
        baseline,
        candidate,
    ) == 1
