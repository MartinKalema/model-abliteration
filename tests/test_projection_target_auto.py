"""Focused correctness tests for projection orientation and auto target search."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

import obliteratus.abliterate as abliterate_module
from obliteratus.abliterate import METHODS, AbliterationPipeline
from obliteratus.evaluation.damage_gate import DamageGateError, assess_candidate
from obliteratus.models.loader import ModelHandle


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.o_proj = torch.nn.Linear(4, 4, bias=False)


class _FFN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _FFN()
        for parameter in self.parameters():
            values = torch.arange(1, parameter.numel() + 1, dtype=torch.float32)
            parameter.data.copy_(values.reshape_as(parameter))


class _TinyRegisteredModel(torch.nn.Module):
    """Minimal registered Llama layout plus a rollback sentinel parameter."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        )
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer()])


def _registered_handle() -> ModelHandle:
    config = SimpleNamespace(
        model_type="llama",
        quantization_config=None,
        text_config=None,
        num_hidden_layers=1,
        num_attention_heads=1,
        hidden_size=4,
        intermediate_size=8,
    )
    tokenizer = SimpleNamespace(padding_side="right", pad_token_id=None)
    return ModelHandle(
        model=_TinyRegisteredModel(),
        tokenizer=tokenizer,
        config=config,
        model_name="test",
        task="causal_lm",
    )


@pytest.mark.parametrize(
    ("target", "changed"),
    [
        ("output", {"o_proj", "down_proj"}),
        ("attention", {"q_proj", "o_proj", "down_proj"}),
        ("ffn", {"o_proj", "up_proj", "down_proj"}),
        ("all", {"q_proj", "o_proj", "up_proj", "down_proj"}),
    ],
)
def test_projection_targets_are_nested_writer_additions(target, changed):
    handle = _registered_handle()
    layer = handle.model.model.layers[0]
    originals = {
        "q_proj": layer.self_attn.q_proj.weight.detach().clone(),
        "o_proj": layer.self_attn.o_proj.weight.detach().clone(),
        "up_proj": layer.mlp.up_proj.weight.detach().clone(),
        "down_proj": layer.mlp.down_proj.weight.detach().clone(),
    }
    pipeline = AbliterationPipeline(
        model_name="test",
        method="basic",
        projection_target=target,
        refinement_passes=1,
        norm_preserve=False,
    )
    pipeline._strong_layers = [0]
    pipeline.refusal_subspaces = {0: torch.tensor([[1.0, 0.0, 0.0, 0.0]])}
    pipeline._free_gpu_memory = lambda: None
    pipeline.handle = handle
    pipeline._prepare_projection_manifests()
    pipeline._excise_inner(
        [layer],
        handle.architecture,
        handle.config,
        None,
        time.time(),
    )

    weights = {
        "q_proj": layer.self_attn.q_proj.weight,
        "o_proj": layer.self_attn.o_proj.weight,
        "up_proj": layer.mlp.up_proj.weight,
        "down_proj": layer.mlp.down_proj.weight,
    }
    for name, weight in weights.items():
        assert torch.equal(weight, originals[name]) is (name not in changed)


def test_preset_targets_match_intended_search_start_and_broad_modes():
    assert METHODS["basic"]["projection_target"] == "output"
    assert METHODS["advanced"]["projection_target"] == "output"
    for method in ("aggressive", "surgical", "inverted", "nuclear"):
        assert METHODS[method]["projection_target"] == "all"


def _auto_pipeline(candidates=("output", "all")):
    pipeline = AbliterationPipeline(
        model_name="test",
        method="basic",
        projection_target="auto",
        projection_auto_candidates=candidates,
    )
    pipeline.handle = _registered_handle()
    pipeline.handle.snapshot()
    pipeline._damage_baseline = [
        SimpleNamespace(prompts=tuple(range(8))) for _ in range(8)
    ]
    pipeline._free_gpu_memory = lambda: None
    pipeline._remove_activation_steering = lambda: 0
    pipeline._layer_excise_weights = {0: 0.75}
    return pipeline


def _metrics(refusal_rate, nll=0.01):
    return {
        "nll_increase_upper_ci": nll,
        "sampled_token_kl_upper_ci": 0.01,
        "sampled_token_kl_p95": 0.01,
        "top1_flip_rate": 0.01,
        "coherence_drop": 0.01,
        "new_degenerate_count": 0,
        "nonfinite_output_count": 0,
        "eval_prompt_count": 32,
        "eval_token_count": 300,
        "sampled_token_count": 128,
        "refusal_rate": refusal_rate,
        "refusal_eval_count": 32,
    }


def _install_fake_candidate_run(pipeline, selection, confirmation=None):
    original = pipeline.handle.model.weight.detach().clone()
    deltas = {"output": 1.0, "attention": 2.0, "ffn": 3.0, "all": 4.0}
    excise_starts = []
    verify_sets = []

    def excise():
        excise_starts.append(
            (
                pipeline.projection_target,
                pipeline.handle.model.weight.detach().clone(),
                pipeline.handle.tokenizer.padding_side,
            )
        )
        with torch.no_grad():
            pipeline.handle.model.weight.add_(deltas[pipeline.projection_target])

    def verify():
        is_confirmation = (
            pipeline._holdout_harmful == pipeline._auto_confirmation_harmful
        )
        verify_sets.append((is_confirmation, tuple(pipeline._holdout_harmful)))
        values = confirmation if is_confirmation and confirmation is not None else selection
        assessment = assess_candidate(values[pipeline.projection_target], pipeline.damage_budget)
        pipeline._damage_assessment = assessment
        # Simulate an exception-prone verifier mutation; rollback must undo it.
        pipeline.handle.tokenizer.padding_side = "left"
        return assessment

    pipeline._excise = excise
    pipeline._verify = verify
    return original, excise_starts, verify_sets, deltas


def test_auto_selects_lowest_refusal_from_fresh_baselines_then_confirms_once():
    pipeline = _auto_pipeline()
    original, starts, verify_sets, deltas = _install_fake_candidate_run(
        pipeline,
        {"output": _metrics(0.15), "all": _metrics(0.05)},
    )

    assessment = pipeline._run_auto_projection_search()

    assert assessment.accepted
    assert pipeline._projection_auto_selected == "all"
    assert [target for target, _, _ in starts] == ["output", "all", "all"]
    assert all(torch.equal(weight, original) for _, weight, _ in starts)
    assert all(padding_side == "right" for _, _, padding_side in starts)
    assert torch.equal(pipeline.handle.model.weight, original + deltas["all"])
    assert [is_confirmation for is_confirmation, _ in verify_sets] == [False, False, True]
    assert set(verify_sets[0][1]).isdisjoint(verify_sets[-1][1])
    assert pipeline.handle.tokenizer.padding_side == "right"


def test_auto_rejects_more_effective_candidate_that_breaks_damage_budget():
    pipeline = _auto_pipeline()
    original, starts, _, deltas = _install_fake_candidate_run(
        pipeline,
        {"output": _metrics(0.15), "all": _metrics(0.01, nll=0.5)},
    )

    pipeline._run_auto_projection_search()

    assert pipeline._projection_auto_selected == "output"
    assert [target for target, _, _ in starts] == ["output", "all", "output"]
    assert torch.equal(pipeline.handle.model.weight, original + deltas["output"])
    assert pipeline._projection_auto_results[1]["accepted"] is False


def test_auto_exact_refusal_tie_uses_lower_normalized_damage():
    pipeline = _auto_pipeline()
    _install_fake_candidate_run(
        pipeline,
        {"output": _metrics(0.05, nll=0.02), "all": _metrics(0.05, nll=0.01)},
    )
    pipeline._run_auto_projection_search()
    assert pipeline._projection_auto_selected == "all"


def test_auto_confirmation_failure_is_terminal_and_restores_untouched_model():
    pipeline = _auto_pipeline()
    original, starts, _, _ = _install_fake_candidate_run(
        pipeline,
        {"output": _metrics(0.10), "all": _metrics(0.05)},
        {"output": _metrics(0.05), "all": _metrics(0.80)},
    )

    with pytest.raises(DamageGateError):
        pipeline._run_auto_projection_search()

    assert [target for target, _, _ in starts] == ["output", "all", "all"]
    assert torch.equal(pipeline.handle.model.weight, original)
    assert pipeline._projection_auto_selected is None
    assert pipeline.projection_target == "auto"


def test_auto_requires_full_dense_snapshot_and_forces_it_during_load(monkeypatch):
    pipeline = _auto_pipeline()
    pipeline.handle._original_state = None
    with pytest.raises(RuntimeError, match="another model-size of RAM"):
        pipeline._assert_auto_projection_prerequisites()

    with pytest.raises(ValueError, match="dense FP16/BF16/FP32"):
        AbliterationPipeline(
            model_name="test",
            method="basic",
            projection_target="auto",
            quantization="4bit",
        )

    loaded = _auto_pipeline().handle
    calls = {}

    def fake_load_model(**kwargs):
        calls.update(kwargs)
        return loaded

    monkeypatch.setattr(abliterate_module, "load_model", fake_load_model)
    summon_pipeline = AbliterationPipeline(
        model_name="test", method="basic", projection_target="auto",
    )
    summon_pipeline._summon()
    assert calls["skip_snapshot"] is False
