"""Manifest-complete regression tests for informed sparse surgery."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from obliteratus.architecture_manifest import (
    ArchitectureCoverageError,
    ProjectionManifest,
    ProjectionManifestEntry,
)
from obliteratus.informed_pipeline import InformedAbliterationPipeline


class _SparseOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.primary = nn.Linear(5, 4, bias=False)
        self.attention.parallel = nn.Module()
        self.attention.parallel.out_proj = nn.Linear(6, 4, bias=False)
        self.moe = nn.Module()
        self.moe.packed_down = nn.Parameter(torch.empty(2, 3, 4))
        self.moe.flat_down = nn.Parameter(torch.empty(7, 4))
        self.reader = nn.Linear(4, 8, bias=False)

        torch.manual_seed(91)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.normal_(mean=0.2, std=0.4)


def _entry(
    owner: nn.Module,
    path: str,
    *,
    storage_identity: str,
    branch_kind: str,
    branch_path: str,
    role: str,
    orientation: str,
    residual_axis: int,
    projection_kind: str,
    expert_axis: int | None = None,
) -> ProjectionManifestEntry:
    projection = owner
    for part in path.split("."):
        projection = getattr(projection, part)
    parameter = (
        projection.weight
        if projection_kind == "module_weight"
        else projection
    )
    qualified_name = (
        f"layer.0.{path}.weight"
        if projection_kind == "module_weight"
        else f"layer.0.{path}"
    )
    return ProjectionManifestEntry(
        qualified_name=qualified_name,
        aliases=(qualified_name,),
        layer_indices=(0,),
        branch_kind=branch_kind,
        branch_paths=(branch_path,),
        component=f"{branch_kind}_{role}",
        role=role,
        orientation=orientation,
        shape=tuple(parameter.shape),
        dtype=str(parameter.dtype),
        storage_identity=storage_identity,
        residual_axis=residual_axis,
        expert_axis=expert_axis,
        projection_kind=projection_kind,
        owner=owner,
        attribute_path=path,
        parameter=parameter,
    )


def _manifest(entries, branch_paths) -> ProjectionManifest:
    return ProjectionManifest(
        architecture="synthetic_hybrid_moe",
        target="output",
        layer_path="layer",
        hidden_size=4,
        num_layers=1,
        entries=tuple(entries),
        branch_coverage=tuple(
            {
                "layer": 0,
                "kind": kind,
                "path": path,
                "writers": 1,
                "readers": 0,
            }
            for kind, path in branch_paths
        ),
    )


def _pipeline_with_manifest(tmp_path, manifest):
    pipeline = InformedAbliterationPipeline(
        model_name="offline-synthetic",
        output_dir=str(tmp_path / "out"),
        projection_target="output",
        on_log=lambda _message: None,
    )
    pipeline.norm_preserve = False
    pipeline._projection_manifests = {"output": manifest}
    pipeline.projection_target = "output"
    pipeline._strong_layers = [0]
    pipeline.refusal_subspaces = {
        0: torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
    }
    pipeline.refinement_passes = 1
    return pipeline


def test_sparse_surgery_executes_every_manifest_writer_and_residual_axis(
    tmp_path, monkeypatch
):
    owner = _SparseOwner()
    entries = [
        _entry(
            owner,
            "attention.primary",
            storage_identity="attn-primary",
            branch_kind="attention",
            branch_path="attention.primary",
            role="writer",
            orientation="output",
            residual_axis=0,
            projection_kind="module_weight",
        ),
        _entry(
            owner,
            "attention.parallel.out_proj",
            storage_identity="attn-parallel",
            branch_kind="attention",
            branch_path="attention.parallel",
            role="writer",
            orientation="output",
            residual_axis=0,
            projection_kind="module_weight",
        ),
        _entry(
            owner,
            "moe.packed_down",
            storage_identity="moe-packed",
            branch_kind="ffn",
            branch_path="moe",
            role="writer",
            orientation="output",
            residual_axis=2,
            expert_axis=0,
            projection_kind="parameter_axis",
        ),
        _entry(
            owner,
            "moe.flat_down",
            storage_identity="moe-flat",
            branch_kind="ffn",
            branch_path="moe",
            role="writer",
            orientation="output",
            residual_axis=1,
            projection_kind="parameter_axis",
        ),
        _entry(
            owner,
            "reader",
            storage_identity="reader",
            branch_kind="ffn",
            branch_path="moe",
            role="reader",
            orientation="input",
            residual_axis=1,
            projection_kind="module_weight",
        ),
    ]
    manifest = _manifest(
        entries,
        (
            ("attention", "attention.primary"),
            ("attention", "attention.parallel"),
            ("ffn", "moe"),
        ),
    )
    pipeline = _pipeline_with_manifest(tmp_path, manifest)
    originals = {
        "primary": owner.attention.primary.weight.detach().clone(),
        "parallel": owner.attention.parallel.out_proj.weight.detach().clone(),
        "packed": owner.moe.packed_down.detach().clone(),
        "flat": owner.moe.flat_down.detach().clone(),
        "reader": owner.reader.weight.detach().clone(),
    }
    events = []
    monkeypatch.setattr(
        pipeline,
        "_emit",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )

    pipeline._excise_sparse()

    assert not torch.equal(owner.attention.primary.weight, originals["primary"])
    assert not torch.equal(
        owner.attention.parallel.out_proj.weight, originals["parallel"]
    )
    assert not torch.equal(owner.moe.packed_down, originals["packed"])
    assert not torch.equal(owner.moe.flat_down, originals["flat"])
    assert torch.equal(owner.reader.weight, originals["reader"])
    for expert_index in range(owner.moe.packed_down.shape[0]):
        assert not torch.equal(
            owner.moe.packed_down[expert_index], originals["packed"][expert_index]
        )

    done = [event for event in events if event[0][:2] == ("excise", "done")]
    assert len(done) == 1
    # Four unique writer entries, each executed for both refusal directions.
    assert done[0][1]["modified_count"] == 8


def test_sparse_manifest_preflight_rejects_late_invalid_axis_before_mutation(
    tmp_path, monkeypatch
):
    owner = _SparseOwner()
    valid = _entry(
        owner,
        "attention.primary",
        storage_identity="valid",
        branch_kind="attention",
        branch_path="attention.primary",
        role="writer",
        orientation="output",
        residual_axis=0,
        projection_kind="module_weight",
    )
    invalid = _entry(
        owner,
        "attention.parallel.out_proj",
        storage_identity="invalid",
        branch_kind="attention",
        branch_path="attention.parallel",
        role="writer",
        orientation="output",
        # The actual shape is [4, 6], so this is not the hidden axis.
        residual_axis=1,
        projection_kind="module_weight",
    )
    pipeline = _pipeline_with_manifest(
        tmp_path,
        _manifest(
            (valid, invalid),
            (
                ("attention", "attention.primary"),
                ("attention", "attention.parallel"),
            ),
        ),
    )
    before = owner.attention.primary.weight.detach().clone()
    events = []
    monkeypatch.setattr(
        pipeline,
        "_emit",
        lambda *args, **kwargs: events.append((args, kwargs)),
    )

    with pytest.raises(ArchitectureCoverageError, match="residual axis"):
        pipeline._excise_sparse()

    assert torch.equal(owner.attention.primary.weight, before)
    assert not any(args[:2] == ("excise", "done") for args, _ in events)
