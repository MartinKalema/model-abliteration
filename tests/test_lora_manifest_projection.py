from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.architecture_manifest import (
    ArchitectureCoverageError,
    ProjectionManifest,
    ProjectionManifestEntry,
)
from obliteratus.lora_ablation import (
    apply_lora_adapters,
    compute_lora_adapters,
)


class _HybridLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Module()
        self.attention.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.recurrent = nn.Module()
        self.recurrent.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.packed = nn.Module()
        self.packed.down_proj = nn.Parameter(torch.randn(2, hidden_size * 2, hidden_size))


def _entry(
    layer: _HybridLayer,
    *,
    qualified_name: str,
    attribute_path: str,
    branch_kind: str,
    branch_path: str,
    residual_axis: int,
    projection_kind: str,
    expert_axis: int | None = None,
) -> ProjectionManifestEntry:
    target = layer
    for part in attribute_path.split("."):
        target = getattr(target, part)
    parameter = target.weight if projection_kind == "module_weight" else target
    return ProjectionManifestEntry(
        qualified_name=qualified_name,
        aliases=(qualified_name,),
        layer_indices=(0,),
        branch_kind=branch_kind,
        branch_paths=(branch_path,),
        component=f"{branch_kind}_output",
        role="writer",
        orientation="output",
        shape=tuple(parameter.shape),
        dtype=str(parameter.dtype),
        storage_identity=f"storage:{qualified_name}",
        residual_axis=residual_axis,
        expert_axis=expert_axis,
        projection_kind=projection_kind,
        owner=layer,
        attribute_path=attribute_path,
        parameter=parameter,
    )


def _fixture() -> tuple[
    AbliterationPipeline,
    ProjectionManifest,
    tuple[ProjectionManifestEntry, ...],
]:
    hidden_size = 4
    layer = _HybridLayer(hidden_size)
    entries = (
        _entry(
            layer,
            qualified_name="model.layers.0.attention.o_proj.weight",
            attribute_path="attention.o_proj",
            branch_kind="attention",
            branch_path="attention",
            residual_axis=0,
            projection_kind="module_weight",
        ),
        _entry(
            layer,
            qualified_name="model.layers.0.recurrent.out_proj.weight",
            attribute_path="recurrent.out_proj",
            branch_kind="attention",
            branch_path="recurrent",
            residual_axis=0,
            projection_kind="module_weight",
        ),
        _entry(
            layer,
            qualified_name="model.layers.0.mlp.down_proj.weight",
            attribute_path="mlp.down_proj",
            branch_kind="ffn",
            branch_path="mlp",
            residual_axis=0,
            projection_kind="module_weight",
        ),
        _entry(
            layer,
            qualified_name="model.layers.0.packed.down_proj",
            attribute_path="packed.down_proj",
            branch_kind="ffn",
            branch_path="packed",
            residual_axis=2,
            projection_kind="parameter_axis",
            expert_axis=0,
        ),
    )
    manifest = ProjectionManifest(
        architecture="hybrid_fixture",
        target="output",
        layer_path="model.layers",
        hidden_size=hidden_size,
        num_layers=1,
        entries=entries,
        branch_coverage=(),
    )

    pipeline = AbliterationPipeline(
        model_name="offline/hybrid-fixture",
        method="basic",
        use_lora_ablation=True,
        lora_rank=2,
        harmful_prompts=["harmful"],
        harmless_prompts=["harmless"],
    )
    pipeline.handle = SimpleNamespace(config=SimpleNamespace())
    pipeline._strong_layers = [0]
    directions = torch.linalg.qr(torch.randn(hidden_size, 2)).Q.T
    pipeline.refusal_subspaces = {0: directions}
    pipeline.projection_row_fraction = 0.5
    pipeline._projection_manifests = {"output": manifest}
    pipeline._on_log = lambda _message: None
    return pipeline, manifest, entries


def _expected_projection(
    parameter: torch.Tensor,
    residual_axis: int,
    subspace: torch.Tensor,
    expert_axis: int | None,
    regularization: float = 0.0,
) -> torch.Tensor:
    expected = parameter.detach().clone()
    for direction in subspace:
        if expert_axis is None:
            targets = ((expected, residual_axis),)
        else:
            targets = tuple(
                (
                    expected.select(expert_axis, expert_index),
                    residual_axis - int(expert_axis < residual_axis),
                )
                for expert_index in range(expected.shape[expert_axis])
            )
        for target, target_axis in targets:
            AbliterationPipeline._project_tensor_along_axis(
                target,
                direction,
                residual_axis=target_axis,
                norm_preserve=False,
                regularization=regularization,
                projection_row_fraction=0.5,
            )
    return expected


def test_manifest_lora_projects_every_hybrid_and_packed_target_once():
    pipeline, manifest, entries = _fixture()
    originals = {entry.qualified_name: entry.parameter.detach().clone() for entry in entries}
    expected = {
        entry.qualified_name: _expected_projection(
            entry.parameter,
            entry.residual_axis,
            pipeline.refusal_subspaces[0],
            entry.expert_axis,
        )
        for entry in entries
    }

    adapters = compute_lora_adapters(pipeline, rank=2, manifest=manifest)

    assert set(adapters) == {entry.qualified_name for entry in entries}
    assert len(adapters) == 4
    assert all(torch.equal(entry.parameter, originals[entry.qualified_name]) for entry in entries)
    packed_B, packed_A = adapters["model.layers.0.packed.down_proj"]
    assert packed_B.shape == (2 * 8, 2)
    assert packed_A.shape == (2, 4)

    applied = apply_lora_adapters(pipeline, adapters, manifest=manifest)

    assert applied == len(entries)
    assert set(pipeline._lora_adapters) == set(adapters)
    for entry in entries:
        assert torch.allclose(
            entry.parameter,
            expected[entry.qualified_name],
            atol=1e-5,
            rtol=1e-5,
        )


def test_manifest_lora_preserves_branch_specific_regularization():
    pipeline, manifest, entries = _fixture()
    pipeline.regularization = 0.2
    pipeline._bayesian_attn_scale = 0.5
    pipeline._bayesian_mlp_scale = 0.25
    originals = {entry.qualified_name: entry.parameter.detach().clone() for entry in entries}
    expected = {
        entry.qualified_name: _expected_projection(
            entry.parameter,
            entry.residual_axis,
            pipeline.refusal_subspaces[0],
            entry.expert_axis,
            regularization=(0.6 if entry.branch_kind == "attention" else 0.8),
        )
        for entry in entries
    }

    adapters = compute_lora_adapters(pipeline, rank=2, manifest=manifest)

    assert all(torch.equal(entry.parameter, originals[entry.qualified_name]) for entry in entries)
    assert apply_lora_adapters(pipeline, adapters, manifest=manifest) == len(entries)
    for entry in entries:
        assert torch.allclose(
            entry.parameter,
            expected[entry.qualified_name],
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize(
    "flag",
    [
        "norm_preserve",
        "project_biases",
        "attention_head_surgery",
        "safety_neuron_masking",
        "per_expert_directions",
        "invert_refusal",
        "project_lm_head",
        "project_embeddings",
        "use_sae_features",
        "expert_transplant",
        "activation_steering",
        "spectral_cascade",
        "true_iterative_refinement",
    ],
)
def test_incompatible_lora_semantics_fail_before_mutation(flag):
    pipeline, manifest, entries = _fixture()
    setattr(pipeline, flag, True)
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match=flag):
        compute_lora_adapters(pipeline, rank=2, manifest=manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


def test_insufficient_rank_fails_before_mutation():
    pipeline, manifest, entries = _fixture()
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match="rank 1"):
        compute_lora_adapters(pipeline, rank=1, manifest=manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


def test_quantized_configuration_fails_before_mutation():
    pipeline, manifest, entries = _fixture()
    pipeline.quantization = "4bit"
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match="quantization"):
        compute_lora_adapters(pipeline, rank=2, manifest=manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


@pytest.mark.parametrize("refinement_passes", [2, 1.5, True, "1"])
def test_lora_requires_exactly_one_integer_refinement_pass(refinement_passes):
    pipeline, manifest, entries = _fixture()
    pipeline.refinement_passes = refinement_passes
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match="refinement_passes"):
        compute_lora_adapters(pipeline, rank=2, manifest=manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


def test_lora_rejects_no_selected_strong_layers():
    pipeline, manifest, entries = _fixture()
    pipeline._strong_layers = []
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match="selected strong layer"):
        compute_lora_adapters(pipeline, rank=2, manifest=manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


def test_lora_rejects_a_foreign_manifest_before_mutation():
    pipeline, manifest, entries = _fixture()
    foreign_manifest = replace(manifest)
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match="stale or foreign"):
        compute_lora_adapters(pipeline, rank=2, manifest=foreign_manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


def test_lora_rejects_a_replaced_live_parameter_before_mutation():
    pipeline, manifest, entries = _fixture()
    stale_parameter = entries[0].parameter
    stale_original = stale_parameter.detach().clone()
    live_module = entries[0].owner.attention.o_proj
    live_module.weight = nn.Parameter(torch.randn_like(live_module.weight))
    live_original = live_module.weight.detach().clone()

    with pytest.raises(ArchitectureCoverageError, match="live model tensor"):
        compute_lora_adapters(pipeline, rank=2, manifest=manifest)

    assert torch.equal(live_module.weight, live_original)
    assert torch.equal(stale_parameter, stale_original)


def test_lora_requires_manifest_coverage_for_every_strong_layer():
    pipeline, manifest, entries = _fixture()
    pipeline._strong_layers = [0, 1]
    pipeline.refusal_subspaces[1] = pipeline.refusal_subspaces[0].clone()
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match=r"strong layers \[1\]"):
        compute_lora_adapters(pipeline, rank=2, manifest=manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


def test_lora_rejects_conflicting_expert_and_residual_axes():
    pipeline, manifest, entries = _fixture()
    bad_entry = replace(entries[-1], expert_axis=entries[-1].residual_axis)
    bad_manifest = replace(manifest, entries=(*entries[:-1], bad_entry))
    pipeline._projection_manifests["output"] = bad_manifest
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match="invalid expert axis"):
        compute_lora_adapters(pipeline, rank=2, manifest=bad_manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


def test_missing_apply_entry_fails_before_mutation():
    pipeline, manifest, entries = _fixture()
    adapters = compute_lora_adapters(pipeline, rank=2, manifest=manifest)
    adapters.pop(entries[-1].qualified_name)
    originals = [entry.parameter.detach().clone() for entry in entries]

    with pytest.raises(ArchitectureCoverageError, match="missing"):
        apply_lora_adapters(pipeline, adapters, manifest=manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )


def test_apply_rolls_back_every_manifest_tensor_on_a_later_failure(monkeypatch):
    pipeline, manifest, entries = _fixture()
    adapters = compute_lora_adapters(pipeline, rank=2, manifest=manifest)
    originals = [entry.parameter.detach().clone() for entry in entries]

    from obliteratus import lora_ablation

    real_merge = lora_ablation._merge_manifest_delta
    calls = 0

    def _fail_second_merge(entry, dense_delta):
        nonlocal calls
        if calls == 1:
            raise RuntimeError("synthetic second-target failure")
        calls += 1
        real_merge(entry, dense_delta)

    monkeypatch.setattr(lora_ablation, "_merge_manifest_delta", _fail_second_merge)

    with pytest.raises(ArchitectureCoverageError, match="restored exactly"):
        apply_lora_adapters(pipeline, adapters, manifest=manifest)

    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )
    assert pipeline._lora_adapters == {}


def test_excise_rejects_a_mismatched_actual_apply_count(monkeypatch):
    pipeline, manifest, _entries = _fixture()
    fake_adapters = {f"planned-{index}": (torch.ones(1, 1), torch.ones(1, 1)) for index in range(4)}

    monkeypatch.setattr(
        "obliteratus.lora_ablation.compute_lora_adapters",
        lambda *_args, **_kwargs: fake_adapters,
    )
    monkeypatch.setattr(
        "obliteratus.lora_ablation.apply_lora_adapters",
        lambda *_args, **_kwargs: 0,
    )

    with pytest.raises(ArchitectureCoverageError, match="applied 0.*requires 4"):
        pipeline._excise_inner([], "hybrid_fixture", pipeline.handle.config, None, time.time())

    assert pipeline._current_projection_manifest() is manifest


def test_excise_rejects_a_mismatched_computed_plan_count(monkeypatch):
    pipeline, _manifest, _entries = _fixture()
    fake_adapters = {f"planned-{index}": (torch.ones(1, 1), torch.ones(1, 1)) for index in range(3)}
    apply_called = False

    monkeypatch.setattr(
        "obliteratus.lora_ablation.compute_lora_adapters",
        lambda *_args, **_kwargs: fake_adapters,
    )

    def _unexpected_apply(*_args, **_kwargs):
        nonlocal apply_called
        apply_called = True
        return 3

    monkeypatch.setattr(
        "obliteratus.lora_ablation.apply_lora_adapters",
        _unexpected_apply,
    )

    with pytest.raises(ArchitectureCoverageError, match="computed 3.*requires 4"):
        pipeline._excise_inner([], "hybrid_fixture", pipeline.handle.config, None, time.time())

    assert apply_called is False


def test_excise_rejects_incompatible_mode_before_bayesian_trials(monkeypatch):
    pipeline, _manifest, entries = _fixture()
    pipeline.norm_preserve = True
    pipeline._bayesian_trials = 1
    originals = [entry.parameter.detach().clone() for entry in entries]
    bayesian_called = False

    def _unexpected_bayesian_call(*_args, **_kwargs):
        nonlocal bayesian_called
        bayesian_called = True
        return {}

    monkeypatch.setattr(
        "obliteratus.bayesian_optimizer.run_bayesian_optimization",
        _unexpected_bayesian_call,
    )

    with pytest.raises(ArchitectureCoverageError, match="norm_preserve"):
        pipeline._excise_inner([], "hybrid_fixture", pipeline.handle.config, None, time.time())

    assert bayesian_called is False
    assert all(
        torch.equal(entry.parameter, original)
        for entry, original in zip(entries, originals, strict=True)
    )
