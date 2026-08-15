"""Self-organizing-map extraction of multi-direction refusal manifolds.

This module implements the direction-construction step from Piras et al.,
"SOM Directions Are Better than One" (AAAI 2026): train a SOM on harmful
activations, then subtract the harmless centroid from every SOM prototype.

The paper selects a subset of prototype directions with a downstream model
evaluation.  OBLITERATUS needs a cheap, per-layer selection rule during its
pipeline, so this implementation ranks candidates by local refusal signal,
harmless-distribution distortion, prototype support, and directional diversity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

_EPS = 1e-8


@dataclass
class SOMDirectionResult:
    """SOM refusal directions and diagnostics for one transformer layer."""

    layer_idx: int
    directions: torch.Tensor
    direction_scores: torch.Tensor
    coverage_score: float
    quantization_error: float
    prototypes: torch.Tensor
    selected_indices: torch.Tensor
    support_counts: torch.Tensor
    signal_to_noise: torch.Tensor
    grid_shape: tuple[int, int]


class SOMDirectionExtractor:
    """Extract related refusal directions from a harmful-activation manifold.

    The SOM is deliberately trained on harmful activations only.  Each learned
    prototype is translated by the harmless centroid and normalized, producing
    related (not forcibly orthogonal) candidate directions.
    """

    def __init__(
        self,
        n_iterations: int = 200,
        learning_rate: float = 0.4,
        sigma: float | None = None,
        candidate_count: int | None = None,
        harmless_pc_count: int = 0,
        distortion_aware: bool = True,
        diversity_penalty: float = 1.0,
        min_signal_to_noise: float = 0.0,
        seed: int = 0,
    ) -> None:
        if n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if sigma is not None and sigma <= 0.0:
            raise ValueError("sigma must be positive when provided")
        if candidate_count is not None and candidate_count <= 0:
            raise ValueError("candidate_count must be positive when provided")
        if harmless_pc_count < 0:
            raise ValueError("harmless_pc_count must be non-negative")
        if diversity_penalty < 0.0:
            raise ValueError("diversity_penalty must be non-negative")
        if min_signal_to_noise < 0.0:
            raise ValueError("min_signal_to_noise must be non-negative")

        self.n_iterations = int(n_iterations)
        self.learning_rate = float(learning_rate)
        self.sigma = float(sigma) if sigma is not None else None
        self.candidate_count = int(candidate_count) if candidate_count is not None else None
        self.harmless_pc_count = int(harmless_pc_count)
        self.distortion_aware = bool(distortion_aware)
        self.diversity_penalty = float(diversity_penalty)
        self.min_signal_to_noise = float(min_signal_to_noise)
        self.seed = int(seed)

    @staticmethod
    def _stack_activations(
        activations: list[torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        if not activations:
            raise ValueError(f"{name} must contain at least one activation")

        rows = []
        for activation in activations:
            if not isinstance(activation, torch.Tensor):
                raise TypeError(f"{name} must contain only torch.Tensor values")
            row = activation.detach()
            if row.ndim == 2 and row.shape[0] == 1:
                row = row.squeeze(0)
            if row.ndim != 1:
                raise ValueError(
                    f"{name} activations must have shape (hidden_dim,) or "
                    f"(1, hidden_dim), got {tuple(activation.shape)}"
                )
            rows.append(row.to(device="cpu", dtype=torch.float32))

        matrix = torch.stack(rows)
        if not torch.isfinite(matrix).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return matrix

    @staticmethod
    def _grid_coordinates(count: int) -> tuple[torch.Tensor, tuple[int, int]]:
        """Return a compact hexagonal-lattice embedding for ``count`` neurons."""
        rows = max(1, int(math.sqrt(count)))
        cols = math.ceil(count / rows)
        indices = torch.arange(count)
        row = torch.div(indices, cols, rounding_mode="floor")
        col = indices.remainder(cols)
        x = col.float() + 0.5 * row.remainder(2).float()
        y = row.float() * (math.sqrt(3.0) / 2.0)
        return torch.stack([x, y], dim=1), (rows, cols)

    def _train_som(
        self,
        harmful: torch.Tensor,
        candidate_count: int,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
        coordinates, grid_shape = self._grid_coordinates(candidate_count)

        if candidate_count == 1:
            prototype = harmful.mean(dim=0, keepdim=True)
            assignments = torch.zeros(harmful.shape[0], dtype=torch.long)
            support = torch.tensor([harmful.shape[0]], dtype=torch.long)
            return prototype, assignments, support, grid_shape

        generator = torch.Generator(device="cpu")
        layer_seed = (self.seed + 1_000_003 * int(layer_idx)) % (2**63 - 1)
        generator.manual_seed(layer_seed)
        initial = torch.randperm(harmful.shape[0], generator=generator)[:candidate_count]
        prototypes = harmful[initial].clone()

        grid_delta = coordinates[:, None, :] - coordinates[None, :, :]
        grid_distance_sq = grid_delta.square().sum(dim=-1)
        sigma_start = self.sigma
        if sigma_start is None:
            sigma_start = max(grid_shape) / 2.0

        for step in range(self.n_iterations):
            sample_idx = torch.randint(
                harmful.shape[0],
                (1,),
                generator=generator,
            ).item()
            sample = harmful[sample_idx]
            bmu = (prototypes - sample).square().sum(dim=1).argmin().item()

            progress = step / max(self.n_iterations, 1)
            decay = 1.0 + 2.0 * progress
            learning_rate = self.learning_rate / decay
            sigma = max(sigma_start / decay, 1e-4)
            neighborhood = torch.exp(
                -grid_distance_sq[bmu] / (2.0 * sigma * sigma)
            )
            prototypes.add_(
                learning_rate * neighborhood.unsqueeze(1) * (sample - prototypes)
            )

        distances = torch.cdist(harmful, prototypes)
        assignments = distances.argmin(dim=1)
        support = torch.bincount(assignments, minlength=candidate_count)
        return prototypes, assignments, support, grid_shape

    def _leading_harmless_pcs(self, centered: torch.Tensor) -> torch.Tensor:
        # Mean-centering limits rank to n_samples - 1.  Respect that analytic
        # cap before the numerical-rank check below; otherwise zero-singular
        # vectors from tiny custom datasets can erase arbitrary feature axes.
        rank_cap = min(max(centered.shape[0] - 1, 0), centered.shape[1])
        count = min(self.harmless_pc_count, rank_cap)
        if count == 0:
            return centered.new_empty((0, centered.shape[1]))

        # Exact SVD is deterministic and cheap for the usual 33-prompt probe.
        # Use randomized low-rank PCA for unusually large prompt collections.
        if min(centered.shape) <= 128:
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            tolerance = (
                max(centered.shape)
                * torch.finfo(singular_values.dtype).eps
                * singular_values.max()
            )
            numerical_rank = int((singular_values > tolerance).sum().item())
            return vh[:min(count, numerical_rank)]

        q = min(min(centered.shape), count + 2)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            _, singular_values, eigenvectors = torch.pca_lowrank(
                centered,
                q=q,
                center=False,
                niter=3,
            )
        tolerance = (
            max(centered.shape)
            * torch.finfo(singular_values.dtype).eps
            * singular_values.max()
        )
        numerical_rank = int((singular_values > tolerance).sum().item())
        return eigenvectors[:, :min(count, numerical_rank)].T

    def _score_candidates(
        self,
        harmful: torch.Tensor,
        harmless: torch.Tensor,
        harmless_centroid: torch.Tensor,
        prototypes: torch.Tensor,
        assignments: torch.Tensor,
        support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = prototypes - harmless_centroid
        harmless_centered = harmless - harmless_centroid

        harmless_pcs = self._leading_harmless_pcs(harmless_centered)
        if harmless_pcs.numel() > 0:
            candidates = candidates - (candidates @ harmless_pcs.T) @ harmless_pcs

        norms = candidates.norm(dim=1)
        directions = candidates / norms.unsqueeze(1).clamp(min=_EPS)
        scores = torch.zeros(prototypes.shape[0], dtype=torch.float32)
        signal_to_noise = torch.zeros_like(scores)
        valid = (norms > _EPS) & (support > 0)

        for candidate_idx in valid.nonzero(as_tuple=False).flatten().tolist():
            local_harmful = harmful[assignments == candidate_idx]
            local_centroid = local_harmful.mean(dim=0)
            direction = directions[candidate_idx]

            signal = torch.dot(local_centroid - harmless_centroid, direction)
            if signal < 0:
                direction = -direction
                directions[candidate_idx] = direction
                signal = -signal

            local_projection = (local_harmful - local_centroid) @ direction
            harmless_projection = harmless_centered @ direction
            pooled_noise = torch.sqrt(
                0.5
                * (
                    local_projection.square().mean()
                    + harmless_projection.square().mean()
                )
                + _EPS
            )
            snr = signal.clamp(min=0.0) / pooled_noise
            support_fraction = support[candidate_idx].float() / harmful.shape[0]

            if self.distortion_aware:
                score = support_fraction * snr.square()
            else:
                score = support_fraction * signal.clamp(min=0.0).square()

            signal_to_noise[candidate_idx] = snr
            scores[candidate_idx] = score

        valid &= torch.isfinite(scores)
        valid &= torch.isfinite(signal_to_noise)
        valid &= scores > 0
        valid &= signal_to_noise >= self.min_signal_to_noise
        return directions, scores, signal_to_noise, valid

    def _select_candidates(
        self,
        directions: torch.Tensor,
        scores: torch.Tensor,
        valid: torch.Tensor,
        count: int,
    ) -> list[int]:
        remaining = valid.nonzero(as_tuple=False).flatten().tolist()
        selected: list[int] = []

        while remaining and len(selected) < count:
            ranked = []
            for candidate_idx in remaining:
                adjusted_score = scores[candidate_idx]
                if selected:
                    similarity = (
                        directions[candidate_idx] @ directions[selected].T
                    ).abs().max()
                    adjusted_score = adjusted_score / (
                        1.0 + self.diversity_penalty * similarity
                    )
                ranked.append((float(adjusted_score), -candidate_idx, candidate_idx))

            best = max(ranked)[2]
            selected.append(best)
            remaining.remove(best)

        return selected

    @torch.no_grad()
    def extract(
        self,
        harmful_activations: list[torch.Tensor],
        harmless_activations: list[torch.Tensor],
        n_directions: int = 3,
        layer_idx: int = 0,
    ) -> SOMDirectionResult:
        """Train a harmful-activation SOM and return its best directions."""
        if n_directions <= 0:
            raise ValueError("n_directions must be positive")

        harmful = self._stack_activations(harmful_activations, "harmful_activations")
        harmless = self._stack_activations(harmless_activations, "harmless_activations")
        if harmful.shape[1] != harmless.shape[1]:
            raise ValueError(
                "harmful and harmless activations must have the same hidden dimension"
            )

        requested_candidates = self.candidate_count
        if requested_candidates is None:
            requested_candidates = max(16, 2 * n_directions)
        if requested_candidates < n_directions:
            raise ValueError("candidate_count must be at least n_directions")
        candidate_count = min(requested_candidates, harmful.shape[0])

        prototypes, assignments, support, grid_shape = self._train_som(
            harmful,
            candidate_count,
            layer_idx,
        )
        harmless_centroid = harmless.mean(dim=0)
        directions, scores, signal_to_noise, valid = self._score_candidates(
            harmful,
            harmless,
            harmless_centroid,
            prototypes,
            assignments,
            support,
        )
        selected = self._select_candidates(
            directions,
            scores,
            valid,
            min(n_directions, candidate_count),
        )
        if not selected:
            raise ValueError(
                "SOM produced no finite directions above the signal-to-noise threshold"
            )

        selected_indices = torch.tensor(selected, dtype=torch.long)
        selected_support = support[selected_indices].sum().item()
        coverage = selected_support / harmful.shape[0]
        quantization_error = (
            harmful - prototypes[assignments]
        ).norm(dim=1).mean().item()

        return SOMDirectionResult(
            layer_idx=layer_idx,
            directions=directions[selected_indices].contiguous(),
            direction_scores=scores[selected_indices].contiguous(),
            coverage_score=float(coverage),
            quantization_error=float(quantization_error),
            prototypes=prototypes,
            selected_indices=selected_indices,
            support_counts=support,
            signal_to_noise=signal_to_noise[selected_indices].contiguous(),
            grid_shape=grid_shape,
        )

    def extract_all_layers(
        self,
        harmful_acts: dict[int, list[torch.Tensor]],
        harmless_acts: dict[int, list[torch.Tensor]],
        n_directions: int = 3,
    ) -> dict[int, SOMDirectionResult]:
        """Extract SOM directions for every layer present in both mappings."""
        results = {}
        for layer_idx in sorted(harmful_acts):
            if layer_idx not in harmless_acts:
                continue
            results[layer_idx] = self.extract(
                harmful_acts[layer_idx],
                harmless_acts[layer_idx],
                n_directions=n_directions,
                layer_idx=layer_idx,
            )
        return results
