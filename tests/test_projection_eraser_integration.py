"""Regression tests for manifest-side oblique maps and row norms."""

from __future__ import annotations

import pytest
import torch

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.analysis.linear_eraser import ResidualEraser
from obliteratus.architecture_manifest import ProjectionManifestEntry
from obliteratus.informed_pipeline import InformedAbliterationPipeline


def _dual_factors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    left = torch.tensor(
        [[1.0, 0.0], [0.5, 1.0], [0.0, 0.5], [0.25, -0.5]],
        dtype=torch.float64,
    )
    # Construct a right inverse so R L = I and P=I-LR is idempotent.
    right = torch.linalg.pinv(left)
    projector = torch.eye(4, dtype=torch.float64) - left @ right
    return left, right, projector


def test_oblique_reader_projection_matches_right_multiplication() -> None:
    left, right, projector = _dual_factors()
    weight = torch.randn(7, 4, dtype=torch.float64)
    expected = weight @ projector

    AbliterationPipeline._project_tensor_along_axis(
        weight,
        left.T,
        removal_directions=right,
        residual_axis=1,
        role="reader",
        norm_preserve=False,
        regularization=0.0,
        projection_row_fraction=1.0,
    )

    assert torch.allclose(weight, expected, atol=1e-12)


def test_oblique_writer_projection_matches_left_multiplication() -> None:
    left, right, projector = _dual_factors()
    weight = torch.randn(4, 9, dtype=torch.float64)
    expected = projector @ weight

    AbliterationPipeline._project_tensor_along_axis(
        weight,
        right,
        removal_directions=left.T,
        residual_axis=0,
        role="writer",
        norm_preserve=False,
        regularization=0.0,
        projection_row_fraction=1.0,
    )

    assert torch.allclose(weight, expected, atol=1e-12)


def test_reader_norm_preservation_restores_every_output_row() -> None:
    generator = torch.Generator().manual_seed(11)
    weight = torch.randn(13, 6, generator=generator)
    direction = torch.nn.functional.normalize(torch.randn(6, generator=generator), dim=0)
    before = weight.norm(dim=1)

    AbliterationPipeline._project_tensor_along_axis(
        weight,
        direction,
        residual_axis=1,
        role="reader",
        norm_preserve=True,
        regularization=0.0,
        projection_row_fraction=1.0,
    )

    assert torch.allclose(weight.norm(dim=1), before, atol=1e-6)
    assert (weight @ direction).abs().max() < 2e-6


def test_writer_norm_preservation_restores_every_logical_output_row() -> None:
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(6, 10, generator=generator)
    direction = torch.nn.functional.normalize(torch.randn(6, generator=generator), dim=0)
    before = weight.norm(dim=1)

    AbliterationPipeline._project_tensor_along_axis(
        weight,
        direction,
        residual_axis=0,
        role="writer",
        norm_preserve=True,
        regularization=0.2,
        projection_row_fraction=1.0,
    )

    assert torch.allclose(weight.norm(dim=1), before, atol=1e-6)


def test_fully_annihilated_row_is_reported_and_left_zero() -> None:
    weight = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    direction = torch.tensor([1.0, 0.0])

    with pytest.warns(UserWarning, match="fully annihilated"):
        AbliterationPipeline._project_tensor_along_axis(
            weight,
            direction,
            residual_axis=1,
            role="reader",
            norm_preserve=True,
            regularization=0.0,
            projection_row_fraction=1.0,
        )

    assert torch.equal(weight[0], torch.zeros(2))
    assert weight[1].norm() == pytest.approx(2.0)


@pytest.mark.parametrize("role", ["reader", "writer"])
def test_manifest_entry_uses_both_oblique_factors(role: str) -> None:
    left, right, projector = _dual_factors()

    class Owner(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)

    owner = Owner()
    original = owner.proj.weight.detach().clone()
    residual_axis = 1 if role == "reader" else 0
    entry = ProjectionManifestEntry(
        qualified_name="layer.proj.weight",
        aliases=("layer.proj.weight",),
        layer_indices=(0,),
        branch_kind="attention",
        branch_paths=("attention",),
        component="attention_input" if role == "reader" else "attention_output",
        role=role,
        orientation="input" if role == "reader" else "output",
        shape=tuple(original.shape),
        dtype=str(original.dtype),
        storage_identity="test-storage",
        residual_axis=residual_axis,
        expert_axis=None,
        projection_kind="module_weight",
        owner=owner,
        attribute_path="proj",
        parameter=owner.proj.weight,
    )
    eraser = ResidualEraser(left, right)
    pipeline = AbliterationPipeline(
        model_name="offline",
        method="basic",
        project_biases=False,
    )

    pipeline._project_manifest_entry(
        entry,
        eraser.display_directions
        if eraser.display_directions is not None
        else left[:, 0],
        layer_idx=0,
        direction_index=0,
        regularization=0.0,
        norm_preserve=False,
        eraser=eraser,
    )

    expected = original @ projector if role == "reader" else projector @ original
    assert torch.allclose(owner.proj.weight, expected, atol=1e-12)


def test_head_selective_edit_restores_each_full_output_row() -> None:
    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = torch.nn.Linear(4, 4, bias=False)

    attention = Attention()
    before = attention.o_proj.weight.detach().norm(dim=1)
    direction = torch.nn.functional.normalize(torch.randn(4), dim=0)

    count = AbliterationPipeline._project_head_selective(
        attention,
        direction,
        head_scores=[(0, 2.0), (1, 1.0)],
        n_heads=2,
        head_fraction=0.5,
        norm_preserve=True,
    )

    assert count == 1
    assert torch.allclose(attention.o_proj.weight.norm(dim=1), before, atol=1e-6)


def test_fused_expert_edit_restores_rows_per_expert() -> None:
    class Experts(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = torch.nn.Parameter(torch.randn(2, 3, 4))

    experts = Experts()
    before = experts.up_proj.detach().norm(dim=-1)
    direction = torch.nn.functional.normalize(torch.randn(4, 1), dim=0)

    count = AbliterationPipeline._project_fused_3d(
        experts,
        direction,
        ["up_proj"],
        norm_preserve=True,
        scale=0.5,
    )

    assert count == 2
    assert torch.allclose(experts.up_proj.norm(dim=-1), before, atol=1e-6)


def test_informed_sparse_writer_restores_each_output_row() -> None:
    class Surgeon:
        @staticmethod
        def apply_sparse_projection(
            matrix: torch.Tensor,
            direction: torch.Tensor,
        ) -> torch.Tensor:
            direction = direction / direction.norm()
            return matrix - (matrix @ direction[:, None]) @ direction[None, :]

    pipeline = InformedAbliterationPipeline(model_name="offline")
    writer = torch.randn(4, 7)
    before = writer.norm(dim=1)
    direction = torch.nn.functional.normalize(torch.randn(4), dim=0)

    updated = pipeline._sparse_project_manifest_tensor(
        writer,
        direction.unsqueeze(0),
        residual_axis=0,
        expert_axis=None,
        surgeon=Surgeon(),
    )

    assert torch.allclose(updated.norm(dim=1), before, atol=1e-6)


@pytest.mark.parametrize(
    ("method", "direction_method"),
    [
        ("optimized", "svd"),
        ("heretic", "diff_means"),
        ("gabliteration", "svd"),
        ("rdo", "diff_means"),
        ("som", "som"),
    ],
)
def test_named_model_forward_method_builds_without_loading(
    method: str,
    direction_method: str,
) -> None:
    pipeline = AbliterationPipeline(model_name="offline", method=method)

    assert pipeline.method == method
    assert pipeline.direction_method == direction_method


@pytest.mark.parametrize(
    "method", ["gabliteration", "rdo", "som", "optimized", "heretic"]
)
def test_named_model_forward_methods_reject_quantization_before_loading(
    method: str,
) -> None:
    with pytest.raises(ValueError, match="requires dense writable weights"):
        AbliterationPipeline(
            model_name="offline",
            method=method,
            quantization="4bit",
        )


def test_direction_backend_contract_rejects_silent_relabeling() -> None:
    with pytest.raises(ValueError, match="requires n_directions=1"):
        AbliterationPipeline(
            model_name="offline",
            method="basic",
            direction_method="diff_means",
            n_directions=2,
        )
    with pytest.raises(ValueError, match="only through method='som_proxy'"):
        AbliterationPipeline(
            model_name="offline",
            method="basic",
            direction_method="som_proxy",
        )


@pytest.mark.parametrize(
    ("option", "message"),
    [({"rdo_refinement": True}, "available only through method='rdo'")],
)
def test_method_owned_weight_modifiers_fail_outside_their_method(
    option: dict[str, bool],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AbliterationPipeline(model_name="offline", method="basic", **option)


def test_rdo_method_owns_rdo_refinement() -> None:
    pipeline = AbliterationPipeline(
        model_name="offline",
        method="rdo",
        rdo_refinement=True,
    )

    assert pipeline.rdo_refinement is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_kl_optimization": False},
        {"direction_method": "svd"},
        {"n_directions": 1},
        {"n_directions": 8},
        {"projection_target": "attention"},
        {"norm_preserve": True},
        {"regularization": 0.2},
        {"project_biases": True},
        {"projection_row_fraction": 0.5},
        {"layer_selection": "knee"},
    ],
)
def test_paper_som_rejects_orchestration_bypasses(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="method='som' requires its exact scored orchestration"):
        AbliterationPipeline(model_name="offline", method="som", **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"projection_target": "all"},
        {"regularization": 0.3},
        {"norm_preserve": True},
        {"project_biases": True},
        {"use_kl_optimization": True},
    ],
)
def test_gabliteration_rejects_ignored_generic_overrides(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match="method='gabliteration' requires its exact scored orchestration",
    ):
        AbliterationPipeline(model_name="offline", method="gabliteration", **overrides)


def test_rdo_cannot_bypass_training_or_add_an_unscored_reprobe() -> None:
    with pytest.raises(ValueError, match="method='rdo' requires its model-forward training"):
        AbliterationPipeline(model_name="offline", method="rdo", rdo_refinement=False)
    with pytest.raises(ValueError, match="method='rdo' requires its model-forward training"):
        AbliterationPipeline(model_name="offline", method="rdo", refinement_passes=2)


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_directions": 2},
        {"direction_method": "svd"},
        {"regularization": 0.2},
        {"projection_row_fraction": 0.5},
        {"attention_head_surgery": True},
        {"project_embeddings": True},
    ],
)
def test_rdo_rejects_metadata_or_unscored_edit_mismatches(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match="method='rdo' requires its model-forward training",
    ):
        AbliterationPipeline(model_name="offline", method="rdo", **overrides)


@pytest.mark.parametrize("method", ["optimized", "heretic"])
def test_bayesian_named_method_cannot_disable_exact_search(method: str) -> None:
    with pytest.raises(ValueError, match="requires its exact scored orchestration"):
        AbliterationPipeline(
            model_name="offline",
            method=method,
            use_kl_optimization=False,
        )


@pytest.mark.parametrize("method", ["optimized", "heretic"])
@pytest.mark.parametrize(
    "overrides",
    [
        {"regularization": 0.2},
        {"refinement_passes": 2},
        {"attention_head_surgery": True},
        {"use_sae_features": True},
        {"project_embeddings": True},
        {"activation_steering": True},
    ],
)
def test_bayesian_named_methods_reject_unscored_generic_edits(
    method: str,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="requires its exact scored orchestration"):
        AbliterationPipeline(model_name="offline", method=method, **overrides)


def test_heretic_keeps_single_cross_layer_diff_means_contract() -> None:
    with pytest.raises(ValueError, match="requires its exact scored orchestration"):
        AbliterationPipeline(model_name="offline", method="heretic", n_directions=2)
