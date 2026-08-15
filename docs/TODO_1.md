Validated offline with Transformers 5.15.0 / Torch 2.13.0. No files edited.

Use this shared handle helper because current `ModelHandle.__post_init__` does not infer DBRX’s `d_model` / `n_layers` fields:

```python
from types import SimpleNamespace

def manifest_handle(model, architecture, hidden_size, num_layers=1):
    return SimpleNamespace(
        model=model,
        architecture=architecture,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
```

DBRX genuine fixture:

```python
from transformers import DbrxConfig
from transformers.models.dbrx.modeling_dbrx import DbrxForCausalLM

config = DbrxConfig(
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
model = DbrxForCausalLM(config)
manifest = build_projection_manifest(
    manifest_handle(model, "dbrx", 12), "all"
)
entries = {entry.qualified_name: entry for entry in manifest.entries}
```

`rope_theta` must be inside `attn_config`; in Transformers 5.15.0, omitting it causes `DbrxAttentionConfig` construction to fail with `AttributeError`.

Exact DBRX assertions:

```python
assert set(entries) == {
    "transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight",
    "transformer.blocks.0.norm_attn_norm.attn.out_proj.weight",
    "transformer.blocks.0.ffn.router.layer.weight",
    "transformer.blocks.0.ffn.experts.mlp.w1",
    "transformer.blocks.0.ffn.experts.mlp.v1",
    "transformer.blocks.0.ffn.experts.mlp.w2",
}

expected = {
    "transformer.blocks.0.norm_attn_norm.attn.Wqkv.weight":
        ((20, 12), "input", 1, "module_weight"),
    "transformer.blocks.0.norm_attn_norm.attn.out_proj.weight":
        ((12, 12), "output", 0, "module_weight"),
    "transformer.blocks.0.ffn.router.layer.weight":
        ((2, 12), "input", 1, "module_weight"),
    "transformer.blocks.0.ffn.experts.mlp.w1":
        ((10, 12), "input", 1, "parameter_axis"),
    "transformer.blocks.0.ffn.experts.mlp.v1":
        ((10, 12), "input", 1, "parameter_axis"),
    "transformer.blocks.0.ffn.experts.mlp.w2":
        ((10, 12), "output", 1, "parameter_axis"),
}
for name, (shape, orientation, axis, kind) in expected.items():
    entry = entries[name]
    assert entry.shape == shape
    assert entry.orientation == orientation
    assert entry.residual_axis == axis
    assert entry.projection_kind == kind

assert {
    (coverage["kind"], coverage["path"])
    for coverage in manifest.branch_coverage
} == {
    ("attention", "norm_attn_norm.attn"),
    ("ffn", "ffn"),
}
assert all(
    entries[name].expert_axis is None
    for name in entries
    if ".experts.mlp." in name
)
```

Expected DBRX exclusions:

```python
assert set(entries).isdisjoint({
    "transformer.wte.weight",
    "transformer.blocks.0.norm_attn_norm.norm_1.weight",
    "transformer.blocks.0.norm_attn_norm.norm_2.weight",
    "transformer.norm_f.weight",
    "lm_head.weight",
})
```

The fused DBRX expert matrices are packed 2-D parameters, so `expert_axis=None` is currently correct manifest metadata. This also documents that they cannot receive individually selected per-expert directions.

GPT-2 genuine Conv1D fixture:

```python
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(
    vocab_size=37,
    n_embd=12,
    n_layer=1,
    n_head=3,
    n_inner=20,
    n_positions=16,
    n_ctx=16,
    bos_token_id=None,
    eos_token_id=None,
)
model = GPT2LMHeadModel(config)
manifest = build_projection_manifest(
    manifest_handle(model, "gpt2", 12), "all"
)
entries = {entry.qualified_name: entry for entry in manifest.entries}
```

Exact GPT-2 assertions:

```python
assert set(entries) == {
    "transformer.h.0.attn.c_attn.weight",
    "transformer.h.0.attn.c_proj.weight",
    "transformer.h.0.mlp.c_fc.weight",
    "transformer.h.0.mlp.c_proj.weight",
}

expected = {
    "transformer.h.0.attn.c_attn.weight":
        ((12, 36), "input", 0),
    "transformer.h.0.attn.c_proj.weight":
        ((12, 12), "output", 1),
    "transformer.h.0.mlp.c_fc.weight":
        ((12, 20), "input", 0),
    "transformer.h.0.mlp.c_proj.weight":
        ((20, 12), "output", 1),
}
for name, (shape, orientation, axis) in expected.items():
    entry = entries[name]
    assert entry.shape == shape
    assert entry.orientation == orientation
    assert entry.residual_axis == axis
    assert entry.projection_kind == "module_weight"

    obj = entry.owner
    for part in entry.attribute_path.split("."):
        obj = getattr(obj, part)
    assert obj.__class__.__name__ == "Conv1D"
```

Expected GPT-2 exclusions:

```python
assert not any(name.endswith(".bias") for name in entries)
assert set(entries).isdisjoint({
    "transformer.wte.weight",
    "transformer.wpe.weight",
    "transformer.h.0.ln_1.weight",
    "transformer.h.0.ln_2.weight",
    "transformer.ln_f.weight",
    "lm_head.weight",
})
```

The square attention `c_proj` is intentional: it ensures the test catches treating Conv1D like `nn.Linear`; its output residual axis must remain axis 1.