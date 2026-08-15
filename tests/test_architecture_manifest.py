"""Offline regression tests for genuine Transformers architecture manifests.

Every model in this module is constructed from a tiny local config.  No model
card, tokenizer, checkpoint, or Hub request is needed.
"""

from __future__ import annotations

import time
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.architecture_manifest import (
    ArchitectureCoverageError,
    build_projection_manifest,
)
from obliteratus.models.loader import ModelHandle
from obliteratus.strategies.utils import get_layer_modules


def _transformers_types(*names: str):
    transformers = pytest.importorskip("transformers")
    missing = [name for name in names if not hasattr(transformers, name)]
    if missing:
        pytest.skip(
            "installed Transformers lacks genuine fixture class(es): "
            + ", ".join(missing)
        )
    return tuple(getattr(transformers, name) for name in names)


def _handle(model: torch.nn.Module, config) -> ModelHandle:
    return ModelHandle(
        model=model,
        tokenizer=SimpleNamespace(padding_side="right", pad_token_id=None),
        config=config,
        model_name="offline-tiny",
        task="causal_lm",
    )


def _manifest_handle(
    model: torch.nn.Module,
    architecture: str,
    hidden_size: int,
    num_layers: int = 1,
):
    """Minimal manifest handle for configs with non-standard dimension names."""
    return SimpleNamespace(
        model=model,
        architecture=architecture,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )


def _qwen3_next():
    Config, Model = _transformers_types("Qwen3NextConfig", "Qwen3NextForCausalLM")
    config = Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        layer_types=["linear_attention"],
    )
    return Model(config), config


def _lfm2():
    Config, Model = _transformers_types("Lfm2Config", "Lfm2ForCausalLM")
    config = Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        block_multiple_of=1,
        full_attn_idxs=[1],
    )
    return Model(config), config


def _gemma4():
    Config, Model = _transformers_types("Gemma4TextConfig", "Gemma4ForCausalLM")
    config = Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        sliding_window=8,
        layer_types=["full_attention"],
        hidden_size_per_layer_input=0,
        enable_moe_block=True,
        num_experts=2,
        top_k_experts=1,
        moe_intermediate_size=4,
    )
    return Model(config), config


def _granite_moe_shared():
    Config, Model = _transformers_types(
        "GraniteMoeSharedConfig", "GraniteMoeSharedForCausalLM"
    )
    config = Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        shared_intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
    )
    return Model(config), config


def _jamba():
    Config, Model = _transformers_types("JambaConfig", "JambaForCausalLM")
    config = Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        num_experts=2,
        num_experts_per_tok=1,
        expert_layer_period=1,
        expert_layer_offset=0,
        attn_layer_period=2,
        attn_layer_offset=1,
        use_mamba_kernels=False,
        mamba_d_state=4,
        mamba_d_conv=2,
        mamba_expand=1,
        mamba_dt_rank=2,
    )
    return Model(config), config


def _nemotron_h():
    Config, Model = _transformers_types("NemotronHConfig", "NemotronHForCausalLM")
    config = Config(
        vocab_size=32,
        hidden_size=8,
        layers_block_type=["linear_attention", "moe", "full_attention", "mlp"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=32,
        intermediate_size=16,
        ssm_state_size=4,
        mamba_num_heads=2,
        mamba_head_dim=4,
        n_groups=1,
        conv_kernel=2,
        expand=1,
        use_mamba_kernels=False,
        n_routed_experts=2,
        n_shared_experts=1,
        moe_intermediate_size=4,
        moe_shared_expert_intermediate_size=4,
        num_experts_per_tok=1,
    )
    return Model(config), config


def _dbrx():
    Config, Model = _transformers_types("DbrxConfig", "DbrxForCausalLM")
    config = Config(
        vocab_size=37,
        d_model=12,
        n_heads=3,
        n_layers=1,
        max_seq_len=16,
        attn_config={
            "kv_n_heads": 1,
            "rope_theta": 10_000.0,
        },
        ffn_config={
            "ffn_hidden_size": 5,
            "moe_num_experts": 2,
            "moe_top_k": 1,
        },
        bos_token_id=None,
        eos_token_id=None,
    )
    return Model(config), config


def _gpt_oss():
    Config, Model = _transformers_types("GptOssConfig", "GptOssForCausalLM")
    config = Config(
        vocab_size=37,
        hidden_size=8,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
        sliding_window=8,
        layer_types=["full_attention"],
        bos_token_id=None,
        eos_token_id=None,
    )
    return Model(config), config


def _legacy_dbrx_manifest_handle():
    """Residual-aligned packed DBRX layout supported by the manifest."""
    model = torch.nn.Module()
    model.transformer = torch.nn.Module()
    model.transformer.wte = torch.nn.Embedding(37, 12)
    model.transformer.norm_f = torch.nn.LayerNorm(12)
    model.lm_head = torch.nn.Linear(12, 37, bias=False)

    block = torch.nn.Module()
    block.norm_attn_norm = torch.nn.Module()
    block.norm_attn_norm.norm_1 = torch.nn.LayerNorm(12)
    block.norm_attn_norm.norm_2 = torch.nn.LayerNorm(12)
    block.norm_attn_norm.attn = torch.nn.Module()
    block.norm_attn_norm.attn.Wqkv = torch.nn.Linear(12, 20, bias=False)
    block.norm_attn_norm.attn.out_proj = torch.nn.Linear(12, 12, bias=False)

    block.ffn = torch.nn.Module()
    block.ffn.router = torch.nn.Module()
    block.ffn.router.layer = torch.nn.Linear(12, 2, bias=False)
    block.ffn.experts = torch.nn.Module()
    block.ffn.experts.num_experts = 2
    block.ffn.experts.mlp = torch.nn.Module()
    block.ffn.experts.mlp.hidden_size = 12
    block.ffn.experts.mlp.ffn_hidden_size = 5
    block.ffn.experts.mlp.moe_num_experts = 2
    block.ffn.experts.mlp.w1 = torch.nn.Parameter(torch.empty(10, 12))
    block.ffn.experts.mlp.v1 = torch.nn.Parameter(torch.empty(10, 12))
    block.ffn.experts.mlp.w2 = torch.nn.Parameter(torch.empty(10, 12))
    model.transformer.blocks = torch.nn.ModuleList([block])
    return _manifest_handle(model, "dbrx", 12)


def _gpt2():
    Config, Model = _transformers_types("GPT2Config", "GPT2LMHeadModel")
    config = Config(
        vocab_size=37,
        n_embd=12,
        n_layer=1,
        n_head=3,
        n_positions=16,
        n_ctx=16,
        n_inner=20,
        bos_token_id=None,
        eos_token_id=None,
    )
    return Model(config), config


QWEN3_NEXT_NAMES = {
    "model.layers.0.linear_attn.in_proj_ba.weight",
    "model.layers.0.linear_attn.in_proj_qkvz.weight",
    "model.layers.0.linear_attn.out_proj.weight",
    "model.layers.0.mlp.experts.down_proj",
    "model.layers.0.mlp.experts.gate_up_proj",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.shared_expert.down_proj.weight",
    "model.layers.0.mlp.shared_expert.gate_proj.weight",
    "model.layers.0.mlp.shared_expert.up_proj.weight",
    "model.layers.0.mlp.shared_expert_gate.weight",
}

LFM2_NAMES = {
    "model.layers.0.conv.in_proj.weight",
    "model.layers.0.conv.out_proj.weight",
    "model.layers.0.feed_forward.w1.weight",
    "model.layers.0.feed_forward.w2.weight",
    "model.layers.0.feed_forward.w3.weight",
    "model.layers.1.feed_forward.w1.weight",
    "model.layers.1.feed_forward.w2.weight",
    "model.layers.1.feed_forward.w3.weight",
    "model.layers.1.self_attn.k_proj.weight",
    "model.layers.1.self_attn.out_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.v_proj.weight",
}

GEMMA4_NAMES = {
    "model.layers.0.experts.down_proj",
    "model.layers.0.experts.gate_up_proj",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.router.proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
}

GRANITE_MOE_SHARED_NAMES = {
    "model.layers.0.block_sparse_moe.experts.down_proj",
    "model.layers.0.block_sparse_moe.experts.gate_up_proj",
    "model.layers.0.block_sparse_moe.router.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.shared_mlp.input_linear.weight",
    "model.layers.0.shared_mlp.output_linear.weight",
}

JAMBA_NAMES = {
    "model.layers.0.feed_forward.experts.down_proj",
    "model.layers.0.feed_forward.experts.gate_up_proj",
    "model.layers.0.feed_forward.router.weight",
    "model.layers.0.mamba.in_proj.weight",
    "model.layers.0.mamba.out_proj.weight",
    "model.layers.1.feed_forward.experts.down_proj",
    "model.layers.1.feed_forward.experts.gate_up_proj",
    "model.layers.1.feed_forward.router.weight",
    "model.layers.1.self_attn.k_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
    "model.layers.1.self_attn.q_proj.weight",
    "model.layers.1.self_attn.v_proj.weight",
}

NEMOTRON_H_NAMES = {
    "model.layers.0.mixer.in_proj.weight",
    "model.layers.0.mixer.out_proj.weight",
    "model.layers.1.mixer.experts.down_proj",
    "model.layers.1.mixer.experts.up_proj",
    "model.layers.1.mixer.gate.weight",
    "model.layers.1.mixer.shared_experts.down_proj.weight",
    "model.layers.1.mixer.shared_experts.up_proj.weight",
    "model.layers.2.mixer.k_proj.weight",
    "model.layers.2.mixer.o_proj.weight",
    "model.layers.2.mixer.q_proj.weight",
    "model.layers.2.mixer.v_proj.weight",
    "model.layers.3.mixer.down_proj.weight",
    "model.layers.3.mixer.up_proj.weight",
}

DBRX_NAMES = {
    "transformer.blocks.0.ffn.experts.mlp.v1",
    "transformer.blocks.0.ffn.experts.mlp.w1",
    "transformer.blocks.0.ffn.experts.mlp.w2",
    "transformer.blocks.0.ffn.router.layer.weight",
    "transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight",
    "transformer.blocks.0.norm_attn_norm.attn.out_proj.weight",
}

GPT_OSS_NAMES = {
    "model.layers.0.mlp.experts.down_proj",
    "model.layers.0.mlp.experts.gate_up_proj",
    "model.layers.0.mlp.router.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
}

GPT2_NAMES = {
    "transformer.h.0.attn.c_attn.weight",
    "transformer.h.0.attn.c_proj.weight",
    "transformer.h.0.mlp.c_fc.weight",
    "transformer.h.0.mlp.c_proj.weight",
}


CASES = (
    (
        "qwen3_next",
        _qwen3_next,
        QWEN3_NEXT_NAMES,
        {
            "model.layers.0.linear_attn.A_log",
            "model.layers.0.linear_attn.dt_bias",
            "model.layers.0.linear_attn.conv1d.weight",
        },
    ),
    ("lfm2", _lfm2, LFM2_NAMES, {"model.layers.0.conv.conv.weight"}),
    ("gemma4", _gemma4, GEMMA4_NAMES, set()),
    ("granitemoeshared", _granite_moe_shared, GRANITE_MOE_SHARED_NAMES, set()),
    (
        "jamba",
        _jamba,
        JAMBA_NAMES,
        {
            "model.layers.0.mamba.A_log",
            "model.layers.0.mamba.D",
            "model.layers.0.mamba.conv1d.weight",
            "model.layers.0.mamba.conv1d.bias",
            "model.layers.0.mamba.dt_proj.weight",
            "model.layers.0.mamba.dt_proj.bias",
        },
    ),
    (
        "nemotron_h",
        _nemotron_h,
        NEMOTRON_H_NAMES,
        {
            "model.layers.0.mixer.A_log",
            "model.layers.0.mixer.D",
            "model.layers.0.mixer.dt_bias",
            "model.layers.0.mixer.conv1d.weight",
            "model.layers.0.mixer.conv1d.bias",
        },
    ),
    (
        "gpt_oss",
        _gpt_oss,
        GPT_OSS_NAMES,
        {
            "model.layers.0.mlp.experts.gate_up_proj_bias",
            "model.layers.0.mlp.experts.down_proj_bias",
            "model.layers.0.mlp.router.bias",
        },
    ),
    ("gpt2", _gpt2, GPT2_NAMES, set()),
)


@pytest.mark.parametrize(
    ("_label", "factory", "expected_names", "excluded_names"),
    CASES,
    ids=[case[0] for case in CASES],
)
def test_genuine_all_target_manifest_has_exact_names_and_exclusions(
    _label, factory, expected_names, excluded_names
):
    torch.manual_seed(0)
    model, config = factory()
    if config.model_type == "gpt2":
        handle = _manifest_handle(model, "gpt2", 12)
    else:
        handle = _handle(model, config)
    manifest = build_projection_manifest(handle, "all")
    manifest_names = {entry.qualified_name for entry in manifest.entries}

    assert manifest_names == expected_names
    model_parameter_names = {name for name, _ in model.named_parameters()}
    assert excluded_names <= model_parameter_names
    assert manifest_names.isdisjoint(excluded_names)


def test_gpt2_conv1d_manifest_uses_transposed_residual_axes():
    model, _ = _gpt2()
    manifest = build_projection_manifest(
        _manifest_handle(model, "gpt2", 12), "all"
    )
    entries = {entry.qualified_name: entry for entry in manifest.entries}

    expected = {
        "transformer.h.0.attn.c_attn.weight": ((12, 36), "reader", "input", 0),
        "transformer.h.0.attn.c_proj.weight": ((12, 12), "writer", "output", 1),
        "transformer.h.0.mlp.c_fc.weight": ((12, 20), "reader", "input", 0),
        "transformer.h.0.mlp.c_proj.weight": ((20, 12), "writer", "output", 1),
    }
    for name, (shape, role, orientation, residual_axis) in expected.items():
        entry = entries[name]
        assert entry.shape == shape
        assert entry.role == role
        assert entry.orientation == orientation
        assert entry.residual_axis == residual_axis
        assert entry.projection_kind == "module_weight"
        projection = entry.owner
        for part in entry.attribute_path.split("."):
            projection = getattr(projection, part)
        assert projection.__class__.__name__ == "Conv1D"

    assert not any(name.endswith(".bias") for name in entries)
    assert set(entries).isdisjoint(
        {
            "transformer.wte.weight",
            "transformer.wpe.weight",
            "transformer.h.0.ln_1.weight",
            "transformer.h.0.ln_2.weight",
            "transformer.ln_f.weight",
            "lm_head.weight",
        }
    )


def test_dbrx_manifest_records_nested_router_and_packed_expert_layout():
    handle = _legacy_dbrx_manifest_handle()

    manifest = build_projection_manifest(handle, "all")
    entries = {entry.qualified_name: entry for entry in manifest.entries}

    expected = {
        "transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight": (
            (20, 12), "input", 1, "module_weight"
        ),
        "transformer.blocks.0.norm_attn_norm.attn.out_proj.weight": (
            (12, 12), "output", 0, "module_weight"
        ),
        "transformer.blocks.0.ffn.router.layer.weight": (
            (2, 12), "input", 1, "module_weight"
        ),
        "transformer.blocks.0.ffn.experts.mlp.w1": (
            (10, 12), "input", 1, "parameter_axis"
        ),
        "transformer.blocks.0.ffn.experts.mlp.v1": (
            (10, 12), "input", 1, "parameter_axis"
        ),
        "transformer.blocks.0.ffn.experts.mlp.w2": (
            (10, 12), "output", 1, "parameter_axis"
        ),
    }
    for name, (shape, orientation, residual_axis, projection_kind) in expected.items():
        entry = entries[name]
        assert entry.shape == shape
        assert entry.orientation == orientation
        assert entry.residual_axis == residual_axis
        assert entry.projection_kind == projection_kind

    assert all(
        entry.expert_axis is None
        for name, entry in entries.items()
        if ".experts.mlp." in name
    )
    assert {
        (coverage["kind"], coverage["path"])
        for coverage in manifest.branch_coverage
    } == {("attention", "norm_attn_norm.attn"), ("ffn", "ffn")}
    assert set(entries).isdisjoint(
        {
            "transformer.wte.weight",
            "transformer.blocks.0.norm_attn_norm.norm_1.weight",
            "transformer.blocks.0.norm_attn_norm.norm_2.weight",
            "transformer.norm_f.weight",
            "lm_head.weight",
        }
    )


def test_genuine_dbrx_refactored_packed_layout_fails_closed_for_every_target():
    model, config = _dbrx()
    handle = _handle(model, config)

    assert handle.architecture == "dbrx"
    assert handle.hidden_size == 12
    assert handle.num_layers == 1
    assert handle.num_heads == 3
    assert handle.intermediate_size == 5

    for target in ("output", "attention", "ffn", "all"):
        with pytest.raises(
            ArchitectureCoverageError,
            match=r"DBRX router is not residual-aligned.*\(2, 5\).*12",
        ):
            build_projection_manifest(handle, target)


def test_gpt_oss_square_fused_experts_use_semantic_residual_axes():
    model, config = _gpt_oss()
    manifest = build_projection_manifest(_handle(model, config), "all")
    entries = {entry.qualified_name: entry for entry in manifest.entries}

    gate_up = entries["model.layers.0.mlp.experts.gate_up_proj"]
    assert gate_up.shape == (2, 8, 16)
    assert gate_up.residual_axis == 1
    assert gate_up.expert_axis == 0

    down = entries["model.layers.0.mlp.experts.down_proj"]
    assert down.shape == (2, 8, 8)
    assert down.residual_axis == 2
    assert down.expert_axis == 0


def test_fused_manifest_executes_once_per_direction_and_restores_norms():
    torch.manual_seed(17)
    model, config = _granite_moe_shared()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.normal_(mean=0.0, std=0.1)

    handle = _handle(model, config)
    pipeline = AbliterationPipeline(
        model_name="offline-tiny",
        method="basic",
        projection_target="all",
        refinement_passes=1,
        norm_preserve=True,
        regularization=0.9,
        project_biases=False,
        project_lm_head=False,
        project_embeddings=False,
        attention_head_surgery=False,
        safety_neuron_masking=False,
        use_sae_features=False,
    )
    pipeline.handle = handle
    pipeline._prepare_projection_manifests()
    pipeline._strong_layers = [0]
    pipeline.refusal_subspaces = {
        0: torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
    }
    pipeline._free_gpu_memory = lambda: None

    manifest = pipeline._projection_manifests["all"]
    before = {
        entry.storage_identity: entry.parameter.detach().clone()
        for entry in manifest.entries
    }
    before_norms = {
        storage_identity: tensor.float().norm().item()
        for storage_identity, tensor in before.items()
    }
    layer = get_layer_modules(handle)[0]
    captured = pipeline._capture_layer_weight_norms(layer)
    assert "block_sparse_moe.experts.gate_up_proj" in captured
    assert "block_sparse_moe.experts.down_proj" in captured

    calls: list[tuple[str, int]] = []
    project_entry = pipeline._project_manifest_entry

    def record_projection(entry, direction, **kwargs):
        calls.append((entry.storage_identity, kwargs["direction_index"]))
        return project_entry(entry, direction, **kwargs)

    pipeline._project_manifest_entry = record_projection
    pipeline._excise_inner(
        get_layer_modules(handle),
        handle.architecture,
        config,
        handle.num_heads,
        time.time(),
    )

    expected_calls = {
        (entry.storage_identity, direction_index)
        for entry in manifest.entries
        for direction_index in range(2)
    }
    assert Counter(calls) == Counter(expected_calls)
    for entry in manifest.entries:
        original = before[entry.storage_identity]
        assert not torch.equal(entry.parameter, original), entry.qualified_name
        assert entry.parameter.float().norm().item() == pytest.approx(
            before_norms[entry.storage_identity], rel=2e-5, abs=2e-6
        )
