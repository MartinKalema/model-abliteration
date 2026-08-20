"""Focused regression tests for exact oblique activation erasers."""

from __future__ import annotations

import torch

from obliteratus.analysis.leace import LEACEExtractor
from obliteratus.analysis.linear_eraser import ResidualEraser
from obliteratus.analysis.whitened_svd import WhitenedSVDExtractor


def test_residual_eraser_matches_dense_affine_map() -> None:
    left = torch.tensor([[2.0], [1.0]], dtype=torch.float64)
    right = torch.tensor([[0.4, 0.2]], dtype=torch.float64)
    center = torch.tensor([3.0, -2.0], dtype=torch.float64)
    eraser = ResidualEraser(left, right, center=center, method="test")
    activations = torch.tensor(
        [[[1.0, 4.0], [2.0, -3.0]], [[0.0, 2.0], [5.0, 1.0]]],
        dtype=torch.float64,
    )

    expected = center + (activations - center) @ eraser.projector.T

    assert torch.allclose(eraser.apply(activations), expected)
    assert torch.allclose(eraser(center), center)
    assert torch.allclose(eraser.affine_bias, center - eraser.projector @ center)


def _anisotropic_binary_data() -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    scale = torch.tensor([8.0, 0.3, 2.0])
    harmless_tensor = torch.randn(160, 3, generator=generator) * scale
    delta = torch.tensor([1.5, 1.0, -0.5])
    harmful_tensor = harmless_tensor + delta
    return list(harmful_tensor), list(harmless_tensor), delta


def test_leace_is_an_exact_oblique_affine_eraser() -> None:
    harmful, harmless, delta = _anisotropic_binary_data()
    result = LEACEExtractor(regularization_eps=1e-5).extract(
        harmful,
        harmless,
        layer_idx=7,
    )
    identity = torch.eye(delta.numel())
    harmful_tensor = torch.stack(harmful)
    harmless_tensor = torch.stack(harmless)

    assert result.layer_idx == 7
    assert result.proj_left.shape == (3, 1)
    assert result.proj_right.shape == (1, 3)
    assert torch.allclose(result.proj_right @ result.proj_left, torch.ones(1, 1), atol=2e-5)
    assert torch.allclose(result.projector @ result.projector, result.projector, atol=2e-5)
    assert torch.allclose(result.projector @ delta, torch.zeros_like(delta), atol=2e-5)
    assert not torch.allclose(result.projector, result.projector.T, atol=1e-3)

    transformed_harmful = result.apply(harmful_tensor)
    transformed_harmless = result.apply(harmless_tensor)
    assert torch.allclose(
        transformed_harmful.mean(dim=0),
        transformed_harmless.mean(dim=0),
        atol=2e-5,
    )
    assert torch.allclose(result.apply(result.center), result.center, atol=1e-6)

    # A symmetric projection using the legacy scoring direction is not LEACE
    # and leaves a substantial component of the empirical mean difference.
    symmetric = identity - torch.outer(result.direction, result.direction)
    assert (symmetric @ delta).norm() > 0.25 * delta.norm()


def test_leace_has_no_more_empirical_distortion_than_orthogonal_erasure() -> None:
    harmful, harmless, delta = _anisotropic_binary_data()
    result = LEACEExtractor(regularization_eps=0.0).extract(harmful, harmless)
    activations = torch.cat((torch.stack(harmful), torch.stack(harmless)))
    orthogonal = torch.eye(3) - torch.outer(delta, delta) / delta.square().sum()
    orthogonal_output = result.center + (activations - result.center) @ orthogonal.T
    leace_loss = (activations - result.apply(activations)).square().sum(dim=-1).mean()
    orthogonal_loss = (activations - orthogonal_output).square().sum(dim=-1).mean()

    assert leace_loss <= orthogonal_loss + 1e-5
    assert abs(result.erasure_loss - leace_loss.item()) < 1e-5


def test_leace_handles_singular_covariance_and_zero_concept() -> None:
    generator = torch.Generator().manual_seed(9)
    harmless_tensor = torch.randn(3, 12, generator=generator)
    delta = torch.randn(12, generator=generator)
    harmful_tensor = harmless_tensor + delta
    singular = LEACEExtractor(regularization_eps=0.0).extract(
        list(harmful_tensor),
        list(harmless_tensor),
    )

    assert torch.isfinite(singular.proj_left).all()
    assert torch.isfinite(singular.proj_right).all()
    assert torch.allclose(singular.projector @ delta, torch.zeros_like(delta), atol=2e-4)

    identical = LEACEExtractor().extract(list(harmless_tensor), list(harmless_tensor))
    assert identical.eraser.rank == 0
    assert torch.equal(identical.apply(harmless_tensor), harmless_tensor)


def _whitened_data() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(77)
    scale = torch.tensor([0.15, 1.5, 7.0, 0.5])
    harmless = torch.randn(192, 4, generator=generator) * scale
    coefficient_a = torch.randn(192, 1, generator=generator)
    coefficient_b = torch.randn(192, 1, generator=generator)
    difference = (
        coefficient_a * torch.tensor([[1.0, 0.2, 0.1, -0.3]])
        + coefficient_b * torch.tensor([[0.2, -1.0, 0.4, 0.5]])
        + torch.tensor([[0.7, -0.1, 0.3, 0.2]])
    )
    return list(harmless + difference), list(harmless)


def test_whitened_svd_factors_match_whiten_project_unwhiten() -> None:
    harmful, harmless = _whitened_data()
    result = WhitenedSVDExtractor(
        regularization_eps=1e-5,
        min_variance_ratio=0.0,
    ).extract(harmful, harmless, n_directions=2)
    query = torch.randn(3, 5, 4, generator=torch.Generator().manual_seed(3))
    centered = query - result.center
    q = result.whitened_directions
    explicit = query - (
        (centered @ result.whitening_projection @ q.T)
        @ q
        @ result.unwhitening_projection.T
    )

    assert torch.allclose(
        result.proj_left,
        result.unwhitening_projection @ q.T,
        atol=1e-6,
    )
    assert torch.allclose(
        result.proj_right,
        q @ result.whitening_projection.T,
        atol=1e-6,
    )
    assert torch.allclose(result.apply(query), explicit, atol=2e-5)


def test_whitened_svd_eraser_is_idempotent_and_removes_coefficients() -> None:
    harmful, harmless = _whitened_data()
    result = WhitenedSVDExtractor(min_variance_ratio=0.0).extract(
        harmful,
        harmless,
        n_directions=2,
    )
    identity_rank = torch.eye(result.eraser.rank)
    activations = torch.stack(harmful)
    transformed = result.apply(activations)
    remaining_coefficients = (
        (transformed - result.center)
        @ result.whitening_projection
        @ result.whitened_directions.T
    )

    assert torch.allclose(result.proj_right @ result.proj_left, identity_rank, atol=2e-5)
    assert torch.allclose(result.projector @ result.projector, result.projector, atol=2e-5)
    assert remaining_coefficients.abs().max() < 2e-4
    assert torch.allclose(result.apply(transformed), transformed, atol=2e-4)

    permutation = torch.tensor([1, 0])
    reordered = ResidualEraser(
        result.proj_left[:, permutation],
        result.proj_right[permutation],
        center=result.center,
    )
    assert torch.allclose(reordered.apply(activations), transformed, atol=2e-5)


def test_whitened_svd_is_not_euclidean_display_direction_projection() -> None:
    harmful, harmless = _whitened_data()
    result = WhitenedSVDExtractor(min_variance_ratio=0.0).extract(
        harmful,
        harmless,
        n_directions=1,
    )
    activations = torch.stack(harmful[:20])
    centered = activations - result.center
    display = result.directions
    euclidean = activations - (centered @ display.T) @ display

    assert not torch.allclose(result.apply(activations), euclidean, atol=1e-3)


def test_whitened_svd_zero_covariance_returns_identity_eraser() -> None:
    zero = [torch.zeros(6) for _ in range(4)]
    result = WhitenedSVDExtractor().extract(zero, zero, n_directions=3)
    query = torch.randn(5, 6, generator=torch.Generator().manual_seed(5))

    assert result.eraser.rank == 0
    assert result.directions.shape == (0, 6)
    assert result.whitened_directions.shape == (0, 0)
    assert torch.equal(result.apply(query), query)
