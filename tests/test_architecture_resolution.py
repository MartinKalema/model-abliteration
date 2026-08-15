"""Offline tests for modern Hugging Face architecture module layouts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch import nn

from obliteratus.strategies.utils import (
    get_attention_module,
    get_ffn_module,
    get_layer_modules,
)


def _layers(count: int) -> nn.ModuleList:
    return nn.ModuleList(nn.Module() for _ in range(count))


def _handle(model: nn.Module, architecture: str, num_layers: int = 2):
    return SimpleNamespace(
        model=model,
        architecture=architecture,
        num_layers=num_layers,
    )


def test_nested_composite_prefers_language_layers_over_vision_layers():
    model = nn.Module()
    model.model = nn.Module()

    # Register vision first to prove traversal order cannot select it.
    model.model.vision_tower = nn.Module()
    model.model.vision_tower.layers = _layers(2)
    model.model.language_model = nn.Module()
    model.model.language_model.layers = _layers(2)

    resolved = get_layer_modules(_handle(model, "gemma4"))

    assert resolved is model.model.language_model.layers
    assert resolved is not model.model.vision_tower.layers


def test_unknown_architecture_rejects_ambiguous_structural_matches():
    model = nn.Module()
    model.first_stack = nn.Module()
    model.first_stack.layers = _layers(2)
    model.second_stack = nn.Module()
    model.second_stack.layers = _layers(2)

    with pytest.raises(RuntimeError, match="Ambiguous transformer layer stacks"):
        get_layer_modules(_handle(model, "custom_transformer"))


def test_llama4_resolves_feed_forward_module():
    model = nn.Module()
    model.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.feed_forward = nn.Module()
    model.model.layers = nn.ModuleList([layer, nn.Module()])

    layers = get_layer_modules(_handle(model, "llama4"))

    assert layers is model.model.layers
    assert get_attention_module(layer, "llama4") is layer.self_attn
    assert get_ffn_module(layer, "llama4") is layer.feed_forward


def test_granite_moe_resolves_block_sparse_moe():
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.block_sparse_moe = nn.Module()

    assert get_attention_module(layer, "granitemoe") is layer.self_attn
    assert get_ffn_module(layer, "granitemoe") is layer.block_sparse_moe


def test_dbrx_resolves_nested_attention_and_ffn():
    model = nn.Module()
    model.transformer = nn.Module()
    layer = nn.Module()
    layer.norm_attn_norm = nn.Module()
    layer.norm_attn_norm.attn = nn.Module()
    layer.ffn = nn.Module()
    model.transformer.blocks = nn.ModuleList([layer, nn.Module()])

    layers = get_layer_modules(_handle(model, "dbrx"))

    assert layers is model.transformer.blocks
    assert get_attention_module(layer, "dbrx") is layer.norm_attn_norm.attn
    assert get_ffn_module(layer, "dbrx") is layer.ffn


def test_qwen3_next_resolves_linear_attention_layer():
    layer = nn.Module()
    layer.linear_attn = nn.Module()
    layer.mlp = nn.Module()

    assert get_attention_module(layer, "qwen3_next") is layer.linear_attn
    assert get_ffn_module(layer, "qwen3_next") is layer.mlp


def test_deepseek_v4_fails_closed_before_layer_discovery():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = _layers(2)

    with pytest.raises(RuntimeError, match="mHC hyper-connections"):
        get_layer_modules(_handle(model, "deepseek_v4"))
