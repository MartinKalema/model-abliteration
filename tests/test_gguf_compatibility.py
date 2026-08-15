"""Focused tests for architecture-specific GGUF metadata compatibility."""

from __future__ import annotations

from types import SimpleNamespace

import gguf
import numpy as np
import torch
import transformers.modeling_gguf_pytorch_utils as gguf_utils
from transformers import configuration_utils

from obliteratus.gguf_backend import (
    _GGUF_FUSED_PARTS,
    _GGUF_LOAD_DTYPE,
    _build_project_tensor_processors,
    _GGUFCompatibilityMetadata,
    _repair_gemma4_config,
    _repair_gpt_oss_reader_fields,
    read_gguf_file_type,
    transformers_gguf_compatibility,
)


class _Field:
    def __init__(self, value):
        self.value = value
        self.parts = [object()]

    def contents(self):
        return self.value


class _Reader:
    def __init__(self, architecture: str, fields: dict[str, object]):
        self.fields = {
            "general.architecture": _Field(architecture),
            **{key: _Field(value) for key, value in fields.items()},
        }

    def get_field(self, key):
        return self.fields.get(key)


def _exact_gemma4_metadata() -> _GGUFCompatibilityMetadata:
    pattern = [True, True, True, True, True, False] * 5
    kv_heads = [8 if is_sliding else 2 for is_sliding in pattern]
    full_layers = [index for index, is_sliding in enumerate(pattern) if not is_sliding]
    return _GGUFCompatibilityMetadata(
        architecture="gemma4",
        fields={
            "gemma4.attention.head_count_kv": kv_heads,
            "gemma4.attention.sliding_window_pattern": pattern,
            "gemma4.attention.key_length": 512,
            "gemma4.attention.key_length_swa": 256,
            "gemma4.expert_count": 128,
            "gemma4.expert_used_count": 8,
            "gemma4.expert_feed_forward_length": 704,
            "gemma4.embedding_length_per_layer_input": 0,
            "gemma4.final_logit_softcapping": 30.0,
            "gemma4.attention.shared_kv_layers": 0,
        },
        tensor_names=frozenset(f"blk.{index}.attn_k.weight" for index in full_layers),
    )


def test_gpt_oss_rope_repair_uses_field_contents_not_key_length_part():
    reader = _Reader(
        "gpt-oss",
        {
            "gpt-oss.rope.scaling.type": "yarn",
            "gpt-oss.rope.scaling.factor": 32.0,
            "gpt-oss.rope.scaling.original_context_length": 4096,
            "gpt-oss.context_length": 131072,
        },
    )

    _repair_gpt_oss_reader_fields(reader)

    assert reader.fields["gpt-oss.rope.scaling.type"].parts[0] == "yarn"
    assert reader.fields["gpt-oss.rope.scaling.factor"].parts[0] == 32.0
    assert reader.fields["gpt-oss.rope.scaling.original_context_length"].parts[0] == 4096
    assert not isinstance(reader.fields["gpt-oss.context_length"].parts[0], int)


def test_gemma4_repair_builds_exact_text_and_moe_shape_config():
    config = {"model_type": "gemma4_text", "num_hidden_layers": 30}

    _repair_gemma4_config(config, _exact_gemma4_metadata())

    assert config["num_key_value_heads"] == 8
    assert config["num_global_key_value_heads"] == 2
    assert config["head_dim"] == 256
    assert config["global_head_dim"] == 512
    assert (
        config["layer_types"]
        == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        * 5
    )
    assert config["attention_k_eq_v"] is True
    assert config["enable_moe_block"] is True
    assert config["num_experts"] == 128
    assert config["top_k_experts"] == 8
    assert config["moe_intermediate_size"] == 704
    assert config["hidden_size_per_layer_input"] == 0
    assert config["final_logit_softcapping"] == 30.0
    assert config["num_kv_shared_layers"] == 0


def test_compatibility_context_is_scoped_and_repairs_loader_result(monkeypatch):
    original_config_loader = configuration_utils.load_gguf_checkpoint

    def fake_transformers_loader(_path, *args, **kwargs):
        return {"config": {"model_type": "gemma4_text", "num_hidden_layers": 30}}

    monkeypatch.setattr(gguf_utils, "load_gguf_checkpoint", fake_transformers_loader)
    monkeypatch.setattr(
        "obliteratus.gguf_backend._read_compatibility_metadata",
        lambda _path, _reader_class: _exact_gemma4_metadata(),
    )

    with transformers_gguf_compatibility():
        result = configuration_utils.load_gguf_checkpoint("metadata-only.gguf")
        assert result["config"]["num_key_value_heads"] == 8
        assert result["config"]["num_global_key_value_heads"] == 2

    assert configuration_utils.load_gguf_checkpoint is original_config_loader


def test_read_gguf_file_type_reports_quantization_enum(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.fields = {"general.file_type": _Field(15)}

        def get_field(self, key):
            return self.fields.get(key)

    monkeypatch.setattr(gguf, "GGUFReader", FakeReader)

    assert read_gguf_file_type(path) == "MOSTLY_Q4_K_M"


def test_gemma4_repair_rejects_ambiguous_kv_layout():
    metadata = _exact_gemma4_metadata()
    fields = dict(metadata.fields)
    fields["gemma4.attention.head_count_kv"] = [8, 4, 8, 8, 8, 2] * 5
    ambiguous = SimpleNamespace(
        architecture=metadata.architecture,
        fields=fields,
        tensor_names=metadata.tensor_names,
    )

    config = {"model_type": "gemma4_text", "num_hidden_layers": 30}
    try:
        _repair_gemma4_config(config, ambiguous)
    except RuntimeError as exc:
        assert "one KV-head count" in str(exc)
    else:
        raise AssertionError("ambiguous Gemma 4 KV metadata must fail closed")


def test_gpt_oss_processor_restores_transpose_and_gate_up_interleaving():
    processor = _build_project_tensor_processors(gguf_utils)["gpt-oss"]({})
    mapping: dict[str, str] = {}
    processor.perform_fallback_tensor_mapping(
        mapping,
        "",
        "",
        "model.layers.0.mlp.experts.gate_up_proj",
    )
    processor.perform_fallback_tensor_mapping(
        mapping,
        "",
        "",
        "model.layers.0.mlp.experts.gate_up_proj_bias",
    )
    assert mapping["blk.0.ffn_gate_exps.weight"] == ("model.layers.0.mlp.experts.gate_up_proj")
    assert mapping["blk.0.ffn_up_exps.bias"] == ("model.layers.0.mlp.experts.gate_up_proj_bias")

    parsed = {"tensors": {}}
    dtype_token = _GGUF_LOAD_DTYPE.set(torch.float16)
    fused_token = _GGUF_FUSED_PARTS.set({})
    try:
        gate = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        up = gate + 100
        processor.process(
            gate,
            "blk.0.ffn_gate_exps.weight",
            tensor_key_mapping=mapping,
            parsed_parameters=parsed,
        )
        processor.process(
            up,
            "blk.0.ffn_up_exps.weight",
            tensor_key_mapping=mapping,
            parsed_parameters=parsed,
        )
        fused = parsed["tensors"]["model.layers.0.mlp.experts.gate_up_proj"]
        assert fused.dtype is torch.float16
        torch.testing.assert_close(
            fused[..., 0::2].float(), torch.from_numpy(gate.swapaxes(-1, -2))
        )
        torch.testing.assert_close(fused[..., 1::2].float(), torch.from_numpy(up.swapaxes(-1, -2)))
        assert _GGUF_FUSED_PARTS.get() == {
            ("model.layers.0.mlp.experts.gate_up_proj", "weight"): {
                "gate",
                "up",
            }
        }
    finally:
        _GGUF_FUSED_PARTS.reset(fused_token)
        _GGUF_LOAD_DTYPE.reset(dtype_token)

    down = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    result = processor.process(down, "blk.0.ffn_down_exps.weight")
    assert result.name == "blk.0.ffn_down_exps"
    np.testing.assert_array_equal(result.weights, down.swapaxes(-1, -2))


def test_gemma4_processor_maps_router_scales_and_parameter_suffixes():
    processor = _build_project_tensor_processors(gguf_utils)["gemma4"]({})
    mapping: dict[str, str] = {}
    processor.perform_fallback_tensor_mapping(mapping, "", "", "model.layers.7.router.scale")
    processor.perform_fallback_tensor_mapping(
        mapping, "", "", "model.layers.7.router.per_expert_scale"
    )

    assert mapping == {
        "blk.7.ffn_gate_inp.scale": "model.layers.7.router.scale",
        "blk.7.ffn_down_exps.scale": "model.layers.7.router.per_expert_scale",
    }
    assert processor.normalize_source_name("blk.7.layer_output_scale.weight") == (
        "blk.7.layer_output_scale"
    )
    assert processor.normalize_source_name("blk.7.ffn_gate_up_exps.weight") == (
        "blk.7.ffn_gate_up_exps"
    )
