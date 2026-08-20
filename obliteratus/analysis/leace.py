"""Closed-form linear concept erasure (LEACE) for binary concepts.

For a binary concept, the concept cross-covariance has the same span as the
class-mean difference ``delta``.  With total activation covariance ``Sigma``,
the minimum-distortion affine eraser can therefore be written

    v = Sigma^+ delta
    P = I - delta v.T / (delta.T v)
    x' = mu + P (x - mu).

Unlike an ordinary direction projection, ``P`` is generally oblique.  The
implementation returns its distinct low-rank left and right factors so callers
do not lose the LEACE geometry by replacing it with ``I - d d.T``.

Reference:
    Belrose et al. (2023), "LEACE: Perfect linear concept erasure in closed
    form", NeurIPS 2023.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from obliteratus.analysis.linear_eraser import ResidualEraser


def _supported_eigensystem(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the supported PSD eigensystem and its condition number."""
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = eigenvalues.clamp(min=0)
    if eigenvalues.numel() == 0:
        return eigenvalues, eigenvectors[:, :0], float("inf")
    max_eigenvalue = eigenvalues.max()
    if max_eigenvalue <= 0:
        return eigenvalues[:0], eigenvectors[:, :0], float("inf")
    tolerance = max_eigenvalue * max(matrix.shape) * torch.finfo(matrix.dtype).eps
    support_mask = eigenvalues > tolerance
    supported_values = eigenvalues[support_mask]
    supported_vectors = eigenvectors[:, support_mask]
    if supported_values.numel() == 0:
        return supported_values, supported_vectors, float("inf")
    condition = (supported_values.max() / supported_values.min()).item()
    return supported_values, supported_vectors, condition


def _condition_number(matrix: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(matrix).clamp(min=0)
    if eigenvalues.numel() == 0:
        return float("inf")
    max_eigenvalue = eigenvalues.max()
    if max_eigenvalue <= 0:
        return float("inf")
    tolerance = max_eigenvalue * max(matrix.shape) * torch.finfo(matrix.dtype).eps
    supported = eigenvalues[eigenvalues > tolerance]
    if supported.numel() == 0:
        return float("inf")
    return (supported.max() / supported.min()).item()


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
class LEACEResult:
    """Result of exact binary LEACE extraction for one layer.

    ``direction`` remains the normalized covariance-aware scoring direction
    for compatibility with existing diagnostics.  It is not sufficient to
    apply an oblique eraser; use ``eraser`` or ``proj_left``/``proj_right``.
    """

    layer_idx: int
    direction: torch.Tensor
    generalized_eigenvalue: float
    within_class_condition: float
    mean_diff_norm: float
    erasure_loss: float
    eraser: ResidualEraser
    total_covariance_condition: float

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
        """Apply the affine LEACE map to row activations."""
        return self.eraser.apply(activations)


class LEACEExtractor:
    """Fit the minimum-distortion affine LEACE eraser for a binary concept.

    Shrinkage and Tikhonov regularization are applied to the *total*
    activation covariance.  Consequently the returned operator is exact for
    that regularized covariance metric and still erases the empirical binary
    mean difference exactly.
    """

    def __init__(
        self,
        regularization_eps: float = 1e-4,
        shrinkage: float = 0.0,
    ):
        if regularization_eps < 0:
            raise ValueError("regularization_eps must be non-negative")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must be between 0 and 1")
        self.regularization_eps = regularization_eps
        self.shrinkage = shrinkage

    def extract(
        self,
        harmful_activations: list[torch.Tensor],
        harmless_activations: list[torch.Tensor],
        layer_idx: int = 0,
    ) -> LEACEResult:
        """Extract an exact rank-one binary LEACE eraser for one layer."""
        harmful = _stack_rows(harmful_activations, "harmful")
        harmless = _stack_rows(harmless_activations, "harmless")
        if harmful.shape[1] != harmless.shape[1]:
            raise ValueError("harmful and harmless activations must have the same hidden dimension")

        n_h, hidden_dim = harmful.shape
        n_b = harmless.shape[0]
        mean_h = harmful.mean(dim=0)
        mean_b = harmless.mean(dim=0)
        mean_difference = mean_h - mean_b
        mean_diff_norm = mean_difference.norm().item()

        harmful_centered = harmful - mean_h
        harmless_centered = harmless - mean_b
        cov_h = (harmful_centered.T @ harmful_centered) / max(n_h - 1, 1)
        cov_b = (harmless_centered.T @ harmless_centered) / max(n_b - 1, 1)
        within_covariance = (cov_h + cov_b) / 2.0

        all_activations = torch.cat((harmful, harmless), dim=0)
        center = all_activations.mean(dim=0)
        all_centered = all_activations - center
        total_covariance = (
            all_centered.T @ all_centered
        ) / max(all_activations.shape[0] - 1, 1)

        identity = torch.eye(hidden_dim, device=total_covariance.device)
        if self.shrinkage > 0:
            total_scale = total_covariance.trace() / hidden_dim
            total_covariance = (
                (1.0 - self.shrinkage) * total_covariance
                + self.shrinkage * total_scale * identity
            )
            within_scale = within_covariance.trace() / hidden_dim
            within_covariance = (
                (1.0 - self.shrinkage) * within_covariance
                + self.shrinkage * within_scale * identity
            )

        covariance = total_covariance + self.regularization_eps * identity
        within_regularized = within_covariance + self.regularization_eps * identity
        total_condition = _condition_number(covariance)
        within_condition = _condition_number(within_regularized)

        # Honor every regularized covariance dimension when the matrix is
        # positive definite.  Fall back to an explicit Hermitian
        # Moore-Penrose inverse in singular d > n settings.
        cholesky, cholesky_info = torch.linalg.cholesky_ex(covariance)
        if torch.count_nonzero(cholesky_info) == 0:
            score = torch.cholesky_solve(
                mean_difference.unsqueeze(1),
                cholesky,
            ).squeeze(1)
        else:
            supported_values, supported_vectors, _ = _supported_eigensystem(covariance)
            if supported_values.numel():
                score_coordinates = supported_vectors.T @ mean_difference
                score = supported_vectors @ (score_coordinates / supported_values)
            else:
                score = covariance.new_zeros(hidden_dim)
        discriminability = mean_difference @ score
        tolerance = (
            torch.finfo(covariance.dtype).eps
            * hidden_dim
            * max(mean_diff_norm * mean_diff_norm, 1.0)
        )

        if mean_diff_norm == 0.0 or discriminability <= tolerance:
            proj_left = covariance.new_empty((hidden_dim, 0))
            proj_right = covariance.new_empty((0, hidden_dim))
            direction = covariance.new_zeros(hidden_dim)
            display_directions = covariance.new_empty((0, hidden_dim))
            discriminability_value = 0.0
        else:
            normalizer = discriminability.sqrt()
            proj_left = mean_difference.unsqueeze(1) / normalizer
            proj_right = score.unsqueeze(0) / normalizer
            score_norm = score.norm()
            direction = score / score_norm if score_norm > 0 else score
            display_directions = (
                mean_difference / mean_difference.norm()
            ).unsqueeze(0)
            discriminability_value = discriminability.item()

        eraser = ResidualEraser(
            proj_left=proj_left,
            proj_right=proj_right,
            center=center,
            display_directions=display_directions,
            method="leace",
            diagnostics={
                "generalized_eigenvalue": discriminability_value,
                "total_covariance_condition": total_condition,
                "within_class_condition": within_condition,
                "mean_diff_norm": mean_diff_norm,
            },
        )
        removed = all_activations - eraser.apply(all_activations)
        erasure_loss = removed.square().sum(dim=-1).mean().item()

        return LEACEResult(
            layer_idx=layer_idx,
            direction=direction,
            generalized_eigenvalue=discriminability_value,
            within_class_condition=within_condition,
            mean_diff_norm=mean_diff_norm,
            erasure_loss=erasure_loss,
            eraser=eraser,
            total_covariance_condition=total_condition,
        )

    def extract_all_layers(
        self,
        harmful_acts: dict[int, list[torch.Tensor]],
        harmless_acts: dict[int, list[torch.Tensor]],
    ) -> dict[int, LEACEResult]:
        """Extract LEACE erasers for every layer present in both mappings."""
        results = {}
        for idx in sorted(harmful_acts):
            if idx in harmless_acts:
                results[idx] = self.extract(
                    harmful_acts[idx],
                    harmless_acts[idx],
                    layer_idx=idx,
                )
        return results

    @staticmethod
    def compare_with_diff_of_means(
        leace_result: LEACEResult,
        harmful_mean: torch.Tensor,
        harmless_mean: torch.Tensor,
    ) -> dict[str, float]:
        """Compare the covariance-aware scoring axis with mean difference."""
        difference = harmful_mean.squeeze() - harmless_mean.squeeze()
        difference_norm = difference.norm()
        normalized = difference / difference_norm if difference_norm > 1e-8 else difference
        cosine = (leace_result.direction @ normalized).abs().item()
        return {
            "cosine_similarity": cosine,
            "leace_eigenvalue": leace_result.generalized_eigenvalue,
            "leace_erasure_loss": leace_result.erasure_loss,
            "within_class_condition": leace_result.within_class_condition,
            "total_covariance_condition": leace_result.total_covariance_condition,
            "mean_diff_norm": leace_result.mean_diff_norm,
        }
