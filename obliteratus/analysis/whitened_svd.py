"""Covariance-whitened SVD with an exact original-space eraser.

Let ``A = V Lambda^-1/2`` whiten row activations into the supported harmless
covariance eigenspace and let ``B = V Lambda^1/2`` invert that map.  If ``Q``
contains orthonormal SVD directions in whitened coordinates, the corresponding
original-space eraser is

    P = I - (B Q.T) (Q A.T).

The two factors are generally not transposes.  Normalizing the mapped vectors
and applying an ordinary Euclidean projection therefore does not implement the
whitened eraser.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from obliteratus.analysis.linear_eraser import ResidualEraser


def _stack_rows(activations: list[torch.Tensor], name: str) -> torch.Tensor:
    if not activations:
        raise ValueError(f"{name} must contain at least one activation")
    rows = torch.stack(activations).float()
    if rows.ndim == 3 and rows.shape[1] == 1:
        rows = rows.squeeze(1)
    if rows.ndim != 2:
        raise ValueError(
            f"{name} activations must stack to (samples, hidden_dim), "
            f"got {tuple(rows.shape)}"
        )
    if not torch.isfinite(rows).all():
        raise ValueError(f"{name} activations contain NaN or infinity")
    return rows


@dataclass
class WhitenedSVDResult:
    """Result of whitened SVD extraction for a single layer.

    ``directions`` remains a unit-normalized original-space display basis for
    compatibility.  Exact application requires ``eraser`` (or the proxied
    ``proj_left`` and ``proj_right`` fields).
    """

    layer_idx: int
    directions: torch.Tensor
    whitened_directions: torch.Tensor
    singular_values: torch.Tensor
    variance_explained: float
    condition_number: float
    effective_rank: float
    eraser: ResidualEraser
    whitening_projection: torch.Tensor
    unwhitening_projection: torch.Tensor

    @property
    def proj_left(self) -> torch.Tensor:
        return self.eraser.proj_left

    @property
    def proj_right(self) -> torch.Tensor:
        return self.eraser.proj_right

    @property
    def center(self) -> torch.Tensor | None:
        return self.eraser.center

    @property
    def projector(self) -> torch.Tensor:
        return self.eraser.projector

    def apply(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply the affine whiten/project/unwhiten map to row activations."""
        return self.eraser.apply(activations)


class WhitenedSVDExtractor:
    """Extract an SVD subspace relative to harmless activation covariance."""

    def __init__(
        self,
        regularization_eps: float = 1e-4,
        min_variance_ratio: float = 0.01,
    ):
        if regularization_eps < 0:
            raise ValueError("regularization_eps must be non-negative")
        if not 0.0 <= min_variance_ratio <= 1.0:
            raise ValueError("min_variance_ratio must be between 0 and 1")
        self.regularization_eps = regularization_eps
        self.min_variance_ratio = min_variance_ratio

    def extract(
        self,
        harmful_activations: list[torch.Tensor],
        harmless_activations: list[torch.Tensor],
        n_directions: int = 4,
        layer_idx: int = 0,
    ) -> WhitenedSVDResult:
        """Extract whitened directions and their exact affine eraser."""
        if n_directions < 0:
            raise ValueError("n_directions must be non-negative")
        harmful = _stack_rows(harmful_activations, "harmful")
        harmless = _stack_rows(harmless_activations, "harmless")
        if harmful.shape != harmless.shape:
            raise ValueError(
                "whitened paired differences require harmful and harmless "
                "activations with identical shapes"
            )

        n_samples, hidden_dim = harmless.shape
        center = harmless.mean(dim=0)
        harmless_centered = harmless - center
        covariance = (
            harmless_centered.T @ harmless_centered
        ) / max(n_samples - 1, 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.clamp(min=0)
        max_eigenvalue = eigenvalues.max().item() if eigenvalues.numel() else 0.0
        support_tolerance = (
            max_eigenvalue * max(hidden_dim, 1) * torch.finfo(covariance.dtype).eps
        )
        supported = eigenvalues[eigenvalues > support_tolerance]
        if supported.numel():
            condition_number = (supported.max() / supported.min()).item()
        else:
            condition_number = float("inf")

        eigenvalue_sum = eigenvalues.sum()
        if eigenvalue_sum > 0:
            probabilities = eigenvalues / eigenvalue_sum
            probabilities = probabilities[probabilities > 0]
            effective_rank = torch.exp(
                -(probabilities * probabilities.log()).sum()
            ).item()
        else:
            effective_rank = 0.0

        threshold = max(max_eigenvalue * self.min_variance_ratio, support_tolerance)
        valid_mask = eigenvalues > threshold
        valid_values = eigenvalues[valid_mask]
        valid_vectors = eigenvectors[:, valid_mask]
        supported_dim = valid_values.numel()

        if supported_dim:
            regularized_values = valid_values + self.regularization_eps
            whitening_projection = valid_vectors * torch.rsqrt(
                regularized_values
            ).unsqueeze(0)
            unwhitening_projection = valid_vectors * torch.sqrt(
                regularized_values
            ).unsqueeze(0)

            harmful_whitened = (harmful - center) @ whitening_projection
            harmless_whitened = harmless_centered @ whitening_projection
            whitened_difference = harmful_whitened - harmless_whitened
            _, all_singular_values, right_vectors = torch.linalg.svd(
                whitened_difference,
                full_matrices=False,
            )
            rank = min(
                n_directions,
                whitened_difference.shape[0],
                whitened_difference.shape[1],
            )
            whitened_directions = right_vectors[:rank]
            singular_values = all_singular_values[:rank]
        else:
            whitening_projection = covariance.new_empty((hidden_dim, 0))
            unwhitening_projection = covariance.new_empty((hidden_dim, 0))
            all_singular_values = covariance.new_empty((0,))
            whitened_directions = covariance.new_empty((0, 0))
            singular_values = covariance.new_empty((0,))
            rank = 0

        # B Q.T and Q A.T are the distinct factors of the original-space
        # oblique projector.  Their product must be applied as one rank-k map.
        proj_left = unwhitening_projection @ whitened_directions.T
        proj_right = whitened_directions @ whitening_projection.T

        original_directions = proj_left.T
        if rank:
            original_directions = original_directions / original_directions.norm(
                dim=-1,
                keepdim=True,
            ).clamp(min=1e-12)

        total_variance = all_singular_values.square().sum().item()
        selected_variance = singular_values.square().sum().item()
        variance_explained = selected_variance / max(total_variance, 1e-12)

        eraser = ResidualEraser(
            proj_left=proj_left,
            proj_right=proj_right,
            center=center,
            display_directions=original_directions,
            method="whitened_svd",
            diagnostics={
                "variance_explained": variance_explained,
                "condition_number": condition_number,
                "effective_rank": effective_rank,
                "supported_covariance_rank": supported_dim,
            },
        )
        return WhitenedSVDResult(
            layer_idx=layer_idx,
            directions=original_directions,
            whitened_directions=whitened_directions,
            singular_values=singular_values,
            variance_explained=variance_explained,
            condition_number=condition_number,
            effective_rank=effective_rank,
            eraser=eraser,
            whitening_projection=whitening_projection,
            unwhitening_projection=unwhitening_projection,
        )

    def extract_all_layers(
        self,
        harmful_acts: dict[int, list[torch.Tensor]],
        harmless_acts: dict[int, list[torch.Tensor]],
        n_directions: int = 4,
    ) -> dict[int, WhitenedSVDResult]:
        """Extract whitened erasers for every layer in both mappings."""
        results = {}
        for idx in sorted(harmful_acts):
            if idx in harmless_acts:
                results[idx] = self.extract(
                    harmful_acts[idx],
                    harmless_acts[idx],
                    n_directions=n_directions,
                    layer_idx=idx,
                )
        return results

    @staticmethod
    def compare_with_standard(
        whitened_result: WhitenedSVDResult,
        standard_direction: torch.Tensor,
    ) -> dict[str, float]:
        """Compare display directions with a standard Euclidean SVD basis."""
        if standard_direction.ndim == 1:
            standard_direction = standard_direction.unsqueeze(0)
        normalized_standard = standard_direction / standard_direction.norm(
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-8)
        whitened = whitened_result.directions
        if whitened.shape[0] == 0 or normalized_standard.shape[0] == 0:
            primary_cosine = 0.0
            average_max_cosine = 0.0
            principal_cosine = 0.0
        else:
            cosine_matrix = (whitened @ normalized_standard.T).abs()
            primary_cosine = cosine_matrix[0, 0].item()
            average_max_cosine = cosine_matrix.max(dim=-1).values.mean().item()
            if whitened.shape[0] > 1 and normalized_standard.shape[0] > 1:
                overlap_singular_values = torch.linalg.svdvals(
                    whitened @ normalized_standard.T
                )
                principal_cosine = overlap_singular_values[0].clamp(max=1.0).item()
            else:
                principal_cosine = primary_cosine

        return {
            "primary_direction_cosine": primary_cosine,
            "avg_max_direction_cosine": average_max_cosine,
            "subspace_principal_cosine": principal_cosine,
            "whitened_condition_number": whitened_result.condition_number,
            "whitened_effective_rank": whitened_result.effective_rank,
        }
