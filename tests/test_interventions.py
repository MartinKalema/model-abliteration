"""Tests for differentiable, tuple-safe model-forward interventions."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from obliteratus.analysis.interventions import (
    DirectionalIntervention,
    InterventionError,
    ablate_direction,
    run_with_directional_ablation,
)


class _TupleAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sentinel = object()

    def forward(self, hidden_states: torch.Tensor):
        return self.projection(hidden_states), self.sentinel


class _TupleLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = _TupleAttention(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_attention_sentinel: object | None = None
        self.raise_after_attention = False

    def forward(self, hidden_states: torch.Tensor):
        attention, sentinel = self.self_attn(hidden_states)
        self.last_attention_sentinel = sentinel
        if self.raise_after_attention:
            raise RuntimeError("synthetic layer failure")
        mixed = hidden_states + 0.2 * attention
        return mixed + 0.2 * torch.tanh(self.mlp(mixed)), sentinel


class _TupleStack(nn.Module):
    def __init__(self, hidden_size: int = 4, depth: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([_TupleLayer(hidden_size) for _ in range(depth)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states, sentinel = layer(hidden_states)
            assert sentinel is layer.self_attn.sentinel
        return hidden_states


def _all_hooks_removed(model: _TupleStack) -> bool:
    modules = [model, *model.modules()]
    return all(
        not module._forward_hooks and not module._forward_pre_hooks
        for module in modules
    )


def test_all_layer_ablation_is_tuple_safe_and_differentiable():
    torch.manual_seed(3)
    model = _TupleStack()
    activation = torch.randn(2, 5, 4)
    direction = nn.Parameter(torch.tensor([0.8, -0.3, 0.2, 0.5]))

    output = run_with_directional_ablation(model, model.layers, direction, activation)
    loss = output.square().mean() + 0.3 * output[..., 0].mean()
    loss.backward()

    unit = direction.detach() / direction.detach().norm()
    assert torch.allclose(output.detach() @ unit, torch.zeros(2, 5), atol=1e-5)
    assert direction.grad is not None
    assert torch.isfinite(direction.grad).all()
    assert direction.grad.norm().item() > 0.0
    assert all(
        layer.last_attention_sentinel is layer.self_attn.sentinel for layer in model.layers
    )
    assert _all_hooks_removed(model)


def test_addition_hook_reaches_only_the_selected_layer_and_preserves_metadata():
    model = _TupleStack(depth=3)
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    activation = torch.zeros(1, 2, 4)

    with DirectionalIntervention(
        model.layers,
        direction,
        mode="add",
        addition_layer=1,
        addition_scale=2.5,
    ) as hooks:
        output = model(activation)
        hooks.assert_applied()
        counts = hooks.applied_counts

    assert torch.isfinite(output).all()
    assert set(counts) == {"layer[1].input"}
    assert counts["layer[1].input"] == 1
    assert all(
        layer.last_attention_sentinel is layer.self_attn.sentinel for layer in model.layers
    )
    assert _all_hooks_removed(model)


def test_hook_cleanup_is_fail_safe_when_model_forward_raises():
    model = _TupleStack(depth=2)
    model.layers[1].raise_after_attention = True
    direction = torch.tensor([0.5, 0.5, -0.5, -0.5])

    with pytest.raises(RuntimeError, match="synthetic layer failure"):
        run_with_directional_ablation(
            model,
            model.layers,
            direction,
            torch.randn(1, 3, 4),
        )

    assert _all_hooks_removed(model)


def test_direction_and_activation_validation_fail_closed():
    with pytest.raises(ValueError, match="non-zero"):
        ablate_direction(torch.ones(1, 2, 3), torch.zeros(3))
    with pytest.raises(InterventionError, match="hidden dimension"):
        ablate_direction(torch.ones(1, 2, 3), torch.ones(4))
    with pytest.raises(ValueError, match="addition_layer"):
        DirectionalIntervention(
            [nn.Identity()],
            torch.ones(3),
            mode="add",
            addition_layer=2,
        )
