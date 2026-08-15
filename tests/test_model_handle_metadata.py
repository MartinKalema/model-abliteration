"""Focused metadata resolution tests for model handles."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from obliteratus.models.loader import ModelHandle


def _handle(config) -> ModelHandle:
    return ModelHandle(
        model=nn.Identity(),
        tokenizer=object(),
        config=config,
        model_name="offline-metadata-fixture",
        task="causal_lm",
    )


def test_model_handle_resolves_dbrx_native_metadata_names():
    handle = _handle(
        SimpleNamespace(
            model_type="dbrx",
            d_model=12,
            n_layers=3,
            n_heads=4,
            ffn_config=SimpleNamespace(ffn_hidden_size=20),
        )
    )

    assert handle.architecture == "dbrx"
    assert handle.hidden_size == 12
    assert handle.num_layers == 3
    assert handle.num_heads == 4
    assert handle.intermediate_size == 20


def test_nested_standard_text_metadata_precedes_wrapper_dbrx_names():
    handle = _handle(
        SimpleNamespace(
            model_type="composite_fixture",
            d_model=999,
            n_layers=99,
            n_heads=9,
            ffn_config=SimpleNamespace(ffn_hidden_size=9999),
            text_config=SimpleNamespace(
                hidden_size=16,
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=32,
            ),
        )
    )

    assert handle.hidden_size == 16
    assert handle.num_layers == 2
    assert handle.num_heads == 4
    assert handle.intermediate_size == 32


def test_model_handle_resolves_nested_dbrx_native_metadata_names():
    handle = _handle(
        SimpleNamespace(
            model_type="composite_dbrx_fixture",
            text_config=SimpleNamespace(
                d_model=24,
                n_layers=5,
                n_heads=6,
                ffn_config={"ffn_hidden_size": 48},
            ),
        )
    )

    assert handle.hidden_size == 24
    assert handle.num_layers == 5
    assert handle.num_heads == 6
    assert handle.intermediate_size == 48
