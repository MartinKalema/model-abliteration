"""Fail-closed data-separation tests for Bayesian candidate tuning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from obliteratus.bayesian_optimizer import (
    _measure_kl_divergence,
    _measure_refusal_rate,
    run_bayesian_optimization,
)


class _Tokenizer:
    def __call__(self, _prompt, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def decode(self, _tokens, **_kwargs):
        return ""


class _Model:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    def generate(self, **_kwargs):
        if self.fail:
            raise RuntimeError("generation failed")
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def __call__(self, **_kwargs):
        if self.fail:
            raise RuntimeError("forward failed")
        return SimpleNamespace(logits=torch.zeros((1, 2, 4)))


class _Pipeline:
    def __init__(self, *, fail: bool = False):
        self._discovery_harmful = ["discovery-only"]
        self.harmful_prompts = ["must-not-be-read"]
        self.use_chat_template = False
        self.formatted: list[str] = []
        self.handle = SimpleNamespace(
            model=_Model(fail=fail),
            tokenizer=_Tokenizer(),
        )

    def _get_model_device(self, _model):
        return torch.device("cpu")

    def _maybe_apply_chat_template(self, prompts):
        self.formatted = list(prompts)
        return list(prompts)

    def _free_gpu_memory(self):
        return None


def test_bayesian_refusal_tuning_reads_discovery_only():
    pipeline = _Pipeline()

    # The tokenizer emits a blank completion, which is a target failure rather
    # than a falsely successful non-refusal.
    assert _measure_refusal_rate(pipeline, n_prompts=10) == 1.0
    assert pipeline.formatted == ["discovery-only"]


def test_bayesian_refusal_tuning_treats_generation_error_as_failure():
    pipeline = _Pipeline(fail=True)

    assert _measure_refusal_rate(pipeline) == 1.0


def test_bayesian_refusal_tuning_requires_an_explicit_discovery_set():
    pipeline = _Pipeline()
    del pipeline._discovery_harmful

    with pytest.raises(RuntimeError, match="direction-discovery"):
        _measure_refusal_rate(pipeline)


def test_bayesian_kl_measurement_fails_closed_when_forward_fails():
    pipeline = _Pipeline(fail=True)

    result = _measure_kl_divergence(
        pipeline,
        [torch.zeros(4)],
        ["benign"],
    )

    assert result == float("inf")


def test_bayesian_optimizer_fails_before_editing_weights():
    pipeline = _Pipeline()
    sentinel = torch.tensor([1.0, 2.0])
    pipeline.sentinel_weight = sentinel

    with pytest.raises(RuntimeError, match="exact winning-trial replay"):
        run_bayesian_optimization(pipeline, n_trials=1)

    assert torch.equal(sentinel, torch.tensor([1.0, 2.0]))


def test_bayesian_presets_advertise_fail_closed_status():
    from obliteratus.abliterate import METHODS, AbliterationPipeline

    for method in ("optimized", "heretic"):
        assert "Disabled" in METHODS[method]["label"]
        assert "fails closed" in METHODS[method]["description"]
        with pytest.raises(ValueError, match="disabled before model loading"):
            AbliterationPipeline(model_name="must-not-load", method=method)
