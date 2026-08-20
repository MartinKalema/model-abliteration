"""Toy-model tests for paper-style model-forward RDO."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from obliteratus.analysis.rdo import (
    RDOConfig,
    RDOError,
    RDOPromptSplit,
    generate_rdo_evidence,
    optimize_rdo_direction,
    run_rdo,
)


class _ToyTokenizer:
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    pad_token_id = 0

    def __init__(self, vocabulary_size: int = 29):
        self.vocabulary_size = vocabulary_size

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ):
        assert tokenize is False
        rendered = f"<user>{messages[0]['content']}</user>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text, *, return_tensors: str, add_special_tokens: bool):
        assert return_tensors == "pt"
        assert add_special_tokens is False
        ids = [2 + (ord(character) % (self.vocabulary_size - 2)) for character in text]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids, *, skip_special_tokens: bool = True):
        del skip_special_tokens
        return " ".join(f"t{token}" for token in token_ids)


class _ToyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return torch.tanh(self.projection(hidden_states)), None


class _ToyMLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        return torch.tanh(self.projection(hidden_states))


class _ToyLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = _ToyAttention(hidden_size)
        self.mlp = _ToyMLP(hidden_size)

    def forward(self, hidden_states):
        attention, cache = self.self_attn(hidden_states)
        hidden_states = hidden_states + 0.35 * attention
        hidden_states = hidden_states + 0.35 * self.mlp(hidden_states)
        return hidden_states, cache


class _NaNBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        del ctx
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        del ctx
        return torch.full_like(gradient, float("nan"))


class _ToyCausalLM(nn.Module):
    def __init__(self, *, seed: int = 11, vocabulary_size: int = 29, hidden_size: int = 6):
        super().__init__()
        generator_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.embedding = nn.Embedding(vocabulary_size, hidden_size)
            self.layers = nn.ModuleList([_ToyLayer(hidden_size) for _ in range(2)])
            self.lm_head = nn.Linear(hidden_size, vocabulary_size, bias=False)
        finally:
            torch.random.set_rng_state(generator_state)
        self.config = SimpleNamespace(is_encoder_decoder=False, quantization_config=None)
        self.frozen_flags_during_forward: list[bool] = []
        self.generate_calls = 0
        self.nan_backward = False

    def forward(self, *, input_ids, attention_mask, use_cache=False):
        del attention_mask, use_cache
        self.frozen_flags_during_forward.append(
            all(not parameter.requires_grad for parameter in self.parameters())
        )
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)[0]
        logits = self.lm_head(hidden_states)
        if self.nan_backward and logits.requires_grad:
            logits = _NaNBackward.apply(logits)
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        max_new_tokens,
        do_sample,
        pad_token_id=None,
    ):
        del pad_token_id
        assert do_sample is False
        self.generate_calls += 1
        tokens = input_ids.clone()
        mask = attention_mask.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(input_ids=tokens, attention_mask=mask).logits
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                tokens = torch.cat((tokens, next_token), dim=1)
                mask = torch.cat((mask, torch.ones_like(next_token)), dim=1)
        return tokens


def _splits():
    return (
        RDOPromptSplit(
            harmful=("harmful train alpha", "harmful train beta"),
            harmless=("harmless train alpha", "harmless train beta"),
        ),
        RDOPromptSplit(
            harmful=("harmful validation gamma",),
            harmless=("harmless validation gamma",),
        ),
    )


def _config(**overrides):
    values = {
        "addition_layer": 1,
        "addition_scale": 1.7,
        "steps": 3,
        "batch_size": 1,
        "learning_rate": 0.02,
        "target_new_tokens": 2,
        "retain_new_tokens": 1,
        "snapshot_window": 2,
        "seed": 19,
    }
    values.update(overrides)
    return RDOConfig(**values)


def _directions():
    target = torch.tensor([1.6, -0.5, 0.3, 0.1, -0.2, 0.4])
    initial = torch.tensor([0.2, 0.7, -0.1, 0.5, 0.3, -0.4])
    return target, initial


def _hooks_removed(model: _ToyCausalLM) -> bool:
    return all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in model.modules()
    )


def test_run_rdo_uses_real_gradients_without_mutating_model_parameters():
    model = _ToyCausalLM()
    tokenizer = _ToyTokenizer()
    train_split, validation_split = _splits()
    target, initial = _directions()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    requires_grad_before = [parameter.requires_grad for parameter in model.parameters()]

    result = run_rdo(
        model,
        tokenizer,
        model.layers,
        train_split=train_split,
        validation_split=validation_split,
        target_direction=target,
        config=_config(),
        initial_direction=initial,
    )

    assert result.selected_step in {2, 3}
    assert len(result.train_history) == 3
    assert len(result.snapshot_evidence) == 2
    assert all(
        evidence.tangent_grad_norm is not None and evidence.tangent_grad_norm > 0.0
        for evidence in result.train_history
    )
    assert result.direction.norm().item() == pytest.approx(1.0, abs=1e-6)
    assert not torch.allclose(result.direction, result.initial_direction)
    assert result.evidence_summary.train_examples == 2
    assert result.evidence_summary.validation_examples == 1
    assert result.evidence_summary.train_prompt_digest != (
        result.evidence_summary.validation_prompt_digest
    )
    assert all(model.frozen_flags_during_forward)
    assert [parameter.requires_grad for parameter in model.parameters()] == requires_grad_before
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())
    assert _hooks_removed(model)


def test_rdo_is_deterministic_for_the_same_model_data_and_seed():
    tokenizer = _ToyTokenizer()
    train_split, validation_split = _splits()
    target, _ = _directions()
    results = []
    for _ in range(2):
        model = _ToyCausalLM(seed=77)
        results.append(
            run_rdo(
                model,
                tokenizer,
                model.layers,
                train_split=train_split,
                validation_split=validation_split,
                target_direction=target,
                config=_config(steps=2, snapshot_window=2),
                initial_direction=None,
            )
        )

    assert torch.equal(results[0].direction, results[1].direction)
    assert results[0].train_history == results[1].train_history
    assert results[0].snapshot_evidence == results[1].snapshot_evidence


def test_generated_targets_are_exact_tokens_and_splits_are_disjoint():
    model = _ToyCausalLM()
    train_split, validation_split = _splits()
    target, _ = _directions()

    evidence = generate_rdo_evidence(
        model,
        _ToyTokenizer(),
        model.layers,
        train_split=train_split,
        validation_split=validation_split,
        target_direction=target,
        config=_config(),
    )

    assert len(evidence.train) == 2
    assert len(evidence.validation) == 1
    assert all(len(item.answer_after_ablation.response_token_ids) == 2 for item in evidence.train)
    assert all(len(item.refusal_after_addition.response_token_ids) == 2 for item in evidence.train)
    assert all(len(item.harmless_retain.response_token_ids) == 1 for item in evidence.train)
    assert model.generate_calls == 9
    assert _hooks_removed(model)


def test_overlapping_evidence_fails_before_generation():
    model = _ToyCausalLM()
    target, _ = _directions()
    train = RDOPromptSplit(("same prompt",), ("safe train",))
    validation = RDOPromptSplit(("  SAME   PROMPT ",), ("safe validation",))

    with pytest.raises(ValueError, match="must be disjoint"):
        generate_rdo_evidence(
            model,
            _ToyTokenizer(),
            model.layers,
            train_split=train,
            validation_split=validation,
            target_direction=target,
            config=_config(),
        )

    assert model.generate_calls == 0


def test_malformed_chat_template_fails_closed_before_generation():
    class _MalformedTokenizer(_ToyTokenizer):
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            del tokenize, add_generation_prompt
            return messages[0]["content"]

    model = _ToyCausalLM()
    train_split, validation_split = _splits()
    target, _ = _directions()
    with pytest.raises(RDOError, match="assistant generation marker"):
        generate_rdo_evidence(
            model,
            _MalformedTokenizer(),
            model.layers,
            train_split=train_split,
            validation_split=validation_split,
            target_direction=target,
            config=_config(),
        )
    assert model.generate_calls == 0


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("is_loaded_in_4bit", True, "quantized"),
        ("hf_device_map", {"first": "cpu", "second": "disk"}, "offloaded"),
    ],
)
def test_quantized_and_offloaded_models_fail_before_generation(attribute, value, match):
    model = _ToyCausalLM()
    setattr(model, attribute, value)
    train_split, validation_split = _splits()
    target, _ = _directions()
    with pytest.raises(RDOError, match=match):
        generate_rdo_evidence(
            model,
            _ToyTokenizer(),
            model.layers,
            train_split=train_split,
            validation_split=validation_split,
            target_direction=target,
            config=_config(),
        )
    assert model.generate_calls == 0


def test_nonfinite_direction_gradient_fails_and_cleans_hooks():
    model = _ToyCausalLM()
    train_split, validation_split = _splits()
    target, initial = _directions()
    config = _config(steps=1, snapshot_window=1)
    evidence = generate_rdo_evidence(
        model,
        _ToyTokenizer(),
        model.layers,
        train_split=train_split,
        validation_split=validation_split,
        target_direction=target,
        config=config,
    )
    model.nan_backward = True

    with pytest.raises(RDOError, match="gradient contains NaN"):
        optimize_rdo_direction(
            model,
            model.layers,
            evidence=evidence,
            config=config,
            initial_direction=initial,
        )

    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert _hooks_removed(model)
