"""Unit tests for isolated paper-style Gabliteration tensor primitives."""

from __future__ import annotations

import pytest
import torch

from obliteratus.analysis.gabliteration import (
    average_projector_subspace,
    mean_separation_source_layer,
    paper_adaptive_layer_scales,
    ridge_subspace_update,
    shuffle_stabilized_svd_subspace,
)


def test_mean_separation_selects_largest_layer_and_reports_all_scores():
    harmful = {
        2: torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        5: torch.tensor([[0.0, 4.0], [0.0, 2.0], [0.0, 3.0]]),
        9: torch.tensor([[2.0, 0.0]]),
    }
    harmless = {
        2: torch.zeros(3, 2),
        5: torch.zeros(1, 2),
        9: torch.zeros(2, 2),
    }

    result = mean_separation_source_layer(harmful, harmless)

    assert result.source_layer == 5
    assert result.as_dict() == pytest.approx({2: 1.0, 5: 3.0, 9: 2.0})


def test_mean_separation_tie_chooses_smallest_layer_independent_of_mapping_order():
    harmful = {
        8: torch.tensor([[1.0, 0.0]]),
        3: torch.tensor([[0.0, -1.0]]),
    }
    harmless = {8: torch.zeros(1, 2), 3: torch.zeros(1, 2)}

    assert mean_separation_source_layer(harmful, harmless).source_layer == 3


def test_mean_separation_rejects_incomplete_layer_evidence():
    with pytest.raises(ValueError, match="identical layers"):
        mean_separation_source_layer(
            {0: torch.ones(2, 3)},
            {1: torch.zeros(2, 3)},
        )


def _projector(directions: torch.Tensor) -> torch.Tensor:
    return directions.T @ directions


def test_average_projector_is_sign_order_and_basis_rotation_invariant():
    base = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    angle = torch.tensor(0.37, dtype=torch.float64)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle)],
            [torch.sin(angle), torch.cos(angle)],
        ]
    )
    equivalent = rotation @ base
    signed_and_reordered = base[[1, 0]] * torch.tensor([[-1.0], [1.0]])

    result = average_projector_subspace(
        [base, equivalent, signed_and_reordered],
        n_directions=2,
    )

    assert torch.allclose(_projector(result.directions), _projector(base), atol=1e-12)
    assert torch.allclose(result.eigenvalues, torch.ones(2, dtype=torch.float64), atol=1e-12)


def test_average_projector_rejects_non_tensor_basis_cleanly():
    with pytest.raises(TypeError, match="only torch.Tensor"):
        average_projector_subspace([[[1.0, 0.0]]])  # type: ignore[list-item]


def test_shuffle_stabilized_svd_is_deterministic_and_does_not_touch_global_rng():
    generator = torch.Generator().manual_seed(123)
    harmless = torch.randn(14, 6, generator=generator)
    harmful = harmless + torch.tensor([3.0, -2.0, 0.5, 0.0, 0.0, 0.0])

    torch.manual_seed(777)
    rng_before = torch.random.get_rng_state().clone()
    first = shuffle_stabilized_svd_subspace(
        harmful,
        harmless,
        n_directions=2,
        n_shuffles=5,
        seed=19,
    )
    rng_after = torch.random.get_rng_state()
    second = shuffle_stabilized_svd_subspace(
        harmful,
        harmless,
        n_directions=2,
        n_shuffles=5,
        seed=19,
    )

    assert torch.equal(rng_before, rng_after)
    assert torch.equal(first.directions, second.directions)
    assert torch.equal(first.projector_eigenvalues, second.projector_eigenvalues)
    assert first.shuffle_singular_values.shape == (5, 2)
    assert first.n_paired == 14
    assert first.n_shuffles == 5
    assert first.seed == 19
    assert torch.allclose(
        first.directions @ first.directions.T,
        torch.eye(2),
        atol=1e-6,
    )


def test_shuffle_stabilized_svd_pairs_to_shorter_dataset():
    harmful = torch.tensor(
        [[3.0, 0.0], [2.0, 1.0], [1.0, 2.0], [0.0, 3.0]],
    )
    harmless = torch.tensor([[0.0, 0.0], [0.2, -0.1], [-0.1, 0.2]])

    result = shuffle_stabilized_svd_subspace(
        harmful,
        harmless,
        n_directions=2,
        n_shuffles=3,
        seed=4,
    )

    assert result.n_paired == 3
    assert result.directions.shape == (2, 2)


def test_ridge_update_matches_explicit_projector_on_arbitrary_axis():
    tensor = torch.arange(1, 25, dtype=torch.float64).reshape(2, 4, 3)
    directions = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.5]],
        dtype=torch.float64,
    )
    original = tensor.clone()
    alpha = 0.3
    ridge_lambda = 0.1

    updated = ridge_subspace_update(
        tensor,
        directions,
        residual_axis=1,
        alpha=alpha,
        ridge_lambda=ridge_lambda,
    )

    r = directions.T
    gram = r.T @ r + ridge_lambda * torch.eye(2, dtype=torch.float64)
    projector = r @ torch.linalg.solve(gram, r.T)
    moved = tensor.movedim(1, -1)
    expected = (moved - alpha * (moved @ projector)).movedim(-1, 1)
    assert torch.allclose(updated, expected, atol=1e-12)
    assert torch.equal(tensor, original), "the pure helper must not mutate its input"


def test_ridge_and_alpha_remain_separate_for_orthonormal_direction():
    tensor = torch.tensor([[2.0, 5.0]], dtype=torch.float64)
    direction = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    updated = ridge_subspace_update(
        tensor,
        direction,
        residual_axis=-1,
        alpha=0.3,
        ridge_lambda=0.1,
    )

    # Paper defaults remove 0.3 / 1.1 of the aligned component, preserving
    # 0.7272727...; they do not preserve 0.231 as the old preset claimed.
    assert updated[0, 0] / tensor[0, 0] == pytest.approx(1.0 - 0.3 / 1.1)
    assert updated[0, 1] == tensor[0, 1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"residual_axis": 2, "alpha": 0.3, "ridge_lambda": 0.1}, "out of range"),
        ({"residual_axis": 1, "alpha": 1.1, "ridge_lambda": 0.1}, "alpha"),
        ({"residual_axis": 1, "alpha": 0.3, "ridge_lambda": -0.1}, "ridge_lambda"),
    ],
)
def test_ridge_update_validates_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ridge_subspace_update(
            torch.ones(2, 3),
            torch.tensor([[1.0, 0.0, 0.0]]),
            **kwargs,
        )


def test_paper_adaptive_scales_use_rank_within_noncontiguous_layers():
    result = paper_adaptive_layer_scales(
        [30, 2, 18, 11, 7],
        alpha_base=0.3,
        beta=0.5,
    )

    assert result.layers == (2, 7, 11, 18, 30)
    assert result.normalized_positions == pytest.approx((-1.0, -0.5, 0.0, 0.5, 1.0))
    assert result.alphas == pytest.approx((0.3, 0.375, 0.45, 0.375, 0.3))
    assert result.as_dict()[11] == pytest.approx(0.45)


def test_paper_adaptive_singleton_receives_maximum_scale():
    result = paper_adaptive_layer_scales([17], alpha_base=0.3, beta=0.5)

    assert result.normalized_positions == (0.0,)
    assert result.alphas == pytest.approx((0.45,))


def test_paper_adaptive_scales_reject_duplicate_or_overstrong_configuration():
    with pytest.raises(ValueError, match="duplicates"):
        paper_adaptive_layer_scales([2, 2])
    with pytest.raises(ValueError, match="must not exceed 1"):
        paper_adaptive_layer_scales([2, 3, 4], alpha_base=0.8, beta=0.5)
