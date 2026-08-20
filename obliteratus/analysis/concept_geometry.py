"""Descriptive category-direction dispersion analysis.

This module compares harmful-minus-harmless activation directions after prompts
have been grouped by an explicit category label.  It reports signed directional
similarity, angular spread, and the effective rank of the category-direction
matrix.  These are descriptive activation statistics only.

In particular, this module does *not* optimize non-negative combinations of
directions, intervene on model behavior, or validate that the measured region
is a causal refusal concept cone.  Results therefore carry
``causally_validated=False`` and must not be presented as a reproduction of the
RCO/concept-cone procedure in Wollschlager et al. (2025).

The legacy ``ConceptConeAnalyzer`` and cone-named result attributes remain only
as deprecated compatibility shims.  New code should use
``CategoryDirectionDispersionAnalyzer`` and its explicitly descriptive fields.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

# Historical labels for the first 30 built-in prompts.  This mapping is kept as
# an exported convenience, but it is never selected implicitly: callers must
# pass the labels that correspond to the activations being analyzed.
DEFAULT_HARM_CATEGORIES = {
    0: "weapons", 1: "weapons", 2: "weapons",
    3: "cyber", 4: "cyber", 5: "cyber", 6: "cyber",
    7: "cyber", 8: "cyber", 9: "cyber", 10: "cyber", 11: "cyber",
    12: "fraud", 13: "fraud", 14: "fraud", 15: "fraud",
    16: "intrusion", 17: "intrusion", 18: "intrusion", 19: "intrusion",
    20: "substances", 21: "substances",
    22: "extremism", 23: "stalking",
    24: "privacy", 25: "privacy",
    26: "manipulation", 27: "manipulation",
    28: "self_harm", 29: "self_harm",
}


def _warn_legacy(name: str, replacement: str) -> None:
    warnings.warn(
        f"{name} is a deprecated compatibility alias; use {replacement}. "
        "The returned statistics are descriptive and are not causal concept-cone evidence.",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass
class CategoryDirection:
    """A descriptive harmful-minus-harmless direction for one labeled category."""

    category: str
    direction: torch.Tensor
    strength: float
    n_prompts: int
    directional_distinctiveness: float

    @property
    def specificity(self) -> float:
        """Deprecated alias for ``directional_distinctiveness``.

        The value is ``(1 - mean_signed_cosine) / 2`` and is only a geometric
        distinctiveness statistic.  It does not establish category-specific
        causal control.
        """

        _warn_legacy("CategoryDirection.specificity", "directional_distinctiveness")
        return self.directional_distinctiveness

    @specificity.setter
    def specificity(self, value: float) -> None:
        _warn_legacy("CategoryDirection.specificity", "directional_distinctiveness")
        self.directional_distinctiveness = value


@dataclass
class CategoryDirectionDispersionResult:
    """Descriptive category-direction statistics for one layer.

    Pairwise cosines retain their sign.  Angles are ordinary geodesic angles
    between unit vectors and are valid in any hidden dimension; no solid angle
    or cone volume is estimated.
    """

    layer_idx: int
    category_directions: list[CategoryDirection]
    pairwise_cosines: dict[tuple[str, str], float]
    pairwise_angles_degrees: dict[tuple[str, str], float]
    effective_rank: float
    mean_pairwise_cosine: float | None
    mean_pairwise_angle_degrees: float | None
    max_pairwise_angle_degrees: float | None
    angular_dispersion: float
    is_directionally_coherent: bool
    is_directionally_disperse: bool
    general_direction: torch.Tensor
    category_count: int
    causally_validated: bool = field(default=False, init=False)
    analysis_kind: str = field(
        default="descriptive_category_direction_dispersion",
        init=False,
    )

    @property
    def cone_dimensionality(self) -> float:
        """Deprecated alias for the descriptive squared-SVD effective rank."""

        _warn_legacy("cone_dimensionality", "effective_rank")
        return self.effective_rank

    @property
    def cone_solid_angle(self) -> float:
        """Deprecated numeric shim; this is *not* a solid angle.

        For compatibility, this returns the mean pairwise angle in radians.
        It must not be interpreted as steradians or as a cone-volume estimate.
        """

        _warn_legacy(
            "cone_solid_angle",
            "mean_pairwise_angle_degrees (this value is not a solid angle)",
        )
        if self.mean_pairwise_angle_degrees is None:
            return 0.0
        return math.radians(self.mean_pairwise_angle_degrees)

    @property
    def is_linear(self) -> bool:
        """Deprecated alias for signed directional coherence."""

        _warn_legacy("is_linear", "is_directionally_coherent")
        return self.is_directionally_coherent

    @property
    def is_polyhedral(self) -> bool:
        """Deprecated alias for descriptive directional dispersion."""

        _warn_legacy("is_polyhedral", "is_directionally_disperse")
        return self.is_directionally_disperse


@dataclass
class MultiLayerDirectionDispersionResult:
    """Descriptive category-direction dispersion across multiple layers."""

    per_layer: dict[int, CategoryDirectionDispersionResult]
    highest_dispersion_layer: int | None
    lowest_dispersion_layer: int | None
    effective_rank_by_layer: dict[int, float]
    mean_effective_rank: float
    causally_validated: bool = field(default=False, init=False)
    analysis_kind: str = field(
        default="descriptive_category_direction_dispersion",
        init=False,
    )

    @property
    def most_polyhedral_layer(self) -> int:
        _warn_legacy("most_polyhedral_layer", "highest_dispersion_layer")
        return 0 if self.highest_dispersion_layer is None else self.highest_dispersion_layer

    @property
    def most_linear_layer(self) -> int:
        _warn_legacy("most_linear_layer", "lowest_dispersion_layer")
        return 0 if self.lowest_dispersion_layer is None else self.lowest_dispersion_layer

    @property
    def cone_complexity_by_layer(self) -> dict[int, float]:
        _warn_legacy("cone_complexity_by_layer", "effective_rank_by_layer")
        return self.effective_rank_by_layer

    @property
    def mean_cone_dimensionality(self) -> float:
        _warn_legacy("mean_cone_dimensionality", "mean_effective_rank")
        return self.mean_effective_rank


class CategoryDirectionDispersionAnalyzer:
    """Measure descriptive dispersion among explicitly labeled directions."""

    def __init__(
        self,
        category_map: Mapping[int, str],
        min_category_size: int = 2,
    ):
        if not isinstance(min_category_size, int) or isinstance(min_category_size, bool):
            raise TypeError("min_category_size must be an integer")
        if min_category_size < 1:
            raise ValueError("min_category_size must be at least 1")

        labels = dict(category_map)
        invalid_indices = [idx for idx in labels if not isinstance(idx, int) or isinstance(idx, bool)]
        if invalid_indices:
            raise TypeError("category_map keys must be integer prompt indices")
        invalid_labels = [
            idx for idx, label in labels.items()
            if not isinstance(label, str) or not label.strip()
        ]
        if invalid_labels:
            raise ValueError(
                "category_map values must be non-empty strings; invalid indices: "
                f"{invalid_labels[:10]}"
            )

        self.category_map = labels
        self.min_category_size = min_category_size

    def _require_explicit_labels(self, n_prompts: int) -> None:
        missing = [idx for idx in range(n_prompts) if idx not in self.category_map]
        if missing:
            preview = ", ".join(str(idx) for idx in missing[:10])
            suffix = "..." if len(missing) > 10 else ""
            raise ValueError(
                "Explicit category labels are required for every activation pair; "
                f"missing prompt indices: {preview}{suffix}"
            )

    @staticmethod
    def _as_vector(activation: torch.Tensor, *, prompt_index: int, kind: str) -> torch.Tensor:
        vector = activation.detach().float().squeeze()
        if vector.ndim != 1:
            raise ValueError(
                f"{kind} activation {prompt_index} must reduce to one hidden vector; "
                f"got shape {tuple(vector.shape)}"
            )
        if not torch.isfinite(vector).all():
            raise ValueError(f"{kind} activation {prompt_index} contains non-finite values")
        return vector

    def analyze_layer(
        self,
        harmful_activations: list[torch.Tensor],
        harmless_activations: list[torch.Tensor],
        layer_idx: int = 0,
    ) -> CategoryDirectionDispersionResult:
        """Analyze signed category-direction dispersion at a single layer."""

        if len(harmful_activations) != len(harmless_activations):
            raise ValueError("harmful and harmless activation lists must have equal length")
        n_prompts = len(harmful_activations)
        self._require_explicit_labels(n_prompts)

        categories: dict[str, list[int]] = {}
        for idx in range(n_prompts):
            categories.setdefault(self.category_map[idx], []).append(idx)

        category_directions: list[CategoryDirection] = []
        direction_vectors: dict[str, torch.Tensor] = {}

        for category, indices in sorted(categories.items()):
            if len(indices) < self.min_category_size:
                continue

            harmful_vectors = [
                self._as_vector(harmful_activations[idx], prompt_index=idx, kind="harmful")
                for idx in indices
            ]
            harmless_vectors = [
                self._as_vector(harmless_activations[idx], prompt_index=idx, kind="harmless")
                for idx in indices
            ]
            shapes = {tuple(vector.shape) for vector in harmful_vectors + harmless_vectors}
            if len(shapes) != 1:
                raise ValueError(
                    f"category {category!r} contains incompatible activation shapes: "
                    f"{sorted(shapes)}"
                )

            difference = torch.stack(harmful_vectors).mean(dim=0) - torch.stack(
                harmless_vectors
            ).mean(dim=0)
            strength = float(difference.norm().item())
            if strength <= 1e-8:
                continue

            direction = difference / difference.norm()
            direction_vectors[category] = direction
            category_directions.append(
                CategoryDirection(
                    category=category,
                    direction=direction,
                    strength=strength,
                    n_prompts=len(indices),
                    directional_distinctiveness=0.0,
                )
            )

        pairwise_cosines, pairwise_angles = self._pairwise_diagnostics(direction_vectors)
        cosine_values = list(pairwise_cosines.values())
        angle_values = list(pairwise_angles.values())
        mean_cosine = (
            sum(cosine_values) / len(cosine_values) if cosine_values else None
        )
        mean_angle = sum(angle_values) / len(angle_values) if angle_values else None
        max_angle = max(angle_values) if angle_values else None
        angular_dispersion = 0.0 if mean_angle is None else mean_angle / 180.0

        for category_direction in category_directions:
            signed_cosines = [
                float(category_direction.direction @ other.direction)
                for other in category_directions
                if other.category != category_direction.category
            ]
            if signed_cosines:
                mean_other_cosine = sum(signed_cosines) / len(signed_cosines)
                category_direction.directional_distinctiveness = max(
                    0.0,
                    min(1.0, (1.0 - mean_other_cosine) / 2.0),
                )

        general_direction = self._general_direction(direction_vectors)
        effective_rank = self._squared_singular_value_effective_rank(direction_vectors)
        enough_categories = len(category_directions) >= 2
        is_coherent = bool(
            enough_categories
            and mean_cosine is not None
            and mean_cosine > 0.9
            and effective_rank < 1.5
        )
        is_disperse = bool(
            enough_categories
            and mean_cosine is not None
            and (mean_cosine < 0.8 or effective_rank > 2.0)
        )

        return CategoryDirectionDispersionResult(
            layer_idx=layer_idx,
            category_directions=category_directions,
            pairwise_cosines=pairwise_cosines,
            pairwise_angles_degrees=pairwise_angles,
            effective_rank=effective_rank,
            mean_pairwise_cosine=mean_cosine,
            mean_pairwise_angle_degrees=mean_angle,
            max_pairwise_angle_degrees=max_angle,
            angular_dispersion=angular_dispersion,
            is_directionally_coherent=is_coherent,
            is_directionally_disperse=is_disperse,
            general_direction=general_direction,
            category_count=len(category_directions),
        )

    def analyze_all_layers(
        self,
        harmful_acts: dict[int, list[torch.Tensor]],
        harmless_acts: dict[int, list[torch.Tensor]],
        strong_layers: list[int] | None = None,
    ) -> MultiLayerDirectionDispersionResult:
        """Analyze descriptive direction dispersion across selected layers."""

        layers = strong_layers if strong_layers is not None else sorted(harmful_acts)
        per_layer: dict[int, CategoryDirectionDispersionResult] = {}
        for layer_idx in layers:
            if layer_idx not in harmful_acts or layer_idx not in harmless_acts:
                continue
            per_layer[layer_idx] = self.analyze_layer(
                harmful_acts[layer_idx],
                harmless_acts[layer_idx],
                layer_idx=layer_idx,
            )

        if not per_layer:
            return MultiLayerDirectionDispersionResult(
                per_layer={},
                highest_dispersion_layer=None,
                lowest_dispersion_layer=None,
                effective_rank_by_layer={},
                mean_effective_rank=0.0,
            )

        effective_ranks = {
            layer_idx: result.effective_rank for layer_idx, result in per_layer.items()
        }
        highest = max(
            per_layer,
            key=lambda idx: (per_layer[idx].angular_dispersion, per_layer[idx].effective_rank),
        )
        lowest = min(
            per_layer,
            key=lambda idx: (per_layer[idx].angular_dispersion, per_layer[idx].effective_rank),
        )
        return MultiLayerDirectionDispersionResult(
            per_layer=per_layer,
            highest_dispersion_layer=highest,
            lowest_dispersion_layer=lowest,
            effective_rank_by_layer=effective_ranks,
            mean_effective_rank=sum(effective_ranks.values()) / len(effective_ranks),
        )

    @staticmethod
    def _pairwise_diagnostics(
        direction_vectors: Mapping[str, torch.Tensor],
    ) -> tuple[
        dict[tuple[str, str], float],
        dict[tuple[str, str], float],
    ]:
        pairwise_cosines: dict[tuple[str, str], float] = {}
        pairwise_angles: dict[tuple[str, str], float] = {}
        categories = sorted(direction_vectors)
        for left_index, left_category in enumerate(categories):
            left = direction_vectors[left_category]
            for right_category in categories[left_index + 1:]:
                cosine = float(left @ direction_vectors[right_category])
                cosine = max(-1.0, min(1.0, cosine))
                key = (left_category, right_category)
                pairwise_cosines[key] = cosine
                pairwise_angles[key] = math.degrees(math.acos(cosine))
        return pairwise_cosines, pairwise_angles

    @staticmethod
    def _general_direction(direction_vectors: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if not direction_vectors:
            return torch.zeros(1, dtype=torch.float32)
        mean_direction = torch.stack(list(direction_vectors.values())).mean(dim=0)
        norm = mean_direction.norm()
        if not torch.isfinite(norm) or norm <= 1e-8:
            return torch.zeros_like(mean_direction)
        return mean_direction / norm

    @staticmethod
    def _squared_singular_value_effective_rank(
        direction_vectors: Mapping[str, torch.Tensor],
    ) -> float:
        """Return entropy effective rank using normalized ``singular_value**2``."""

        if not direction_vectors:
            return 0.0
        direction_matrix = torch.stack(list(direction_vectors.values())).float()
        singular_values = torch.linalg.svdvals(direction_matrix)
        squared = singular_values.square()
        squared = squared[squared > 1e-12]
        if squared.numel() == 0:
            return 0.0
        probabilities = squared / squared.sum()
        entropy = -(probabilities * probabilities.log()).sum()
        return float(torch.exp(entropy).item())

    @staticmethod
    def format_report(result: CategoryDirectionDispersionResult) -> str:
        """Format a descriptive report without causal cone or solid-angle claims."""

        if result.is_directionally_coherent:
            geometry = "DIRECTIONALLY COHERENT"
        elif result.is_directionally_disperse:
            geometry = "DIRECTIONALLY DISPERSE"
        else:
            geometry = "INTERMEDIATE / INSUFFICIENT"

        lines = [
            f"Category Direction Dispersion — Layer {result.layer_idx}",
            "=" * 48,
            "DESCRIPTIVE ONLY — NOT CAUSALLY VALIDATED",
            "Legacy Concept Cone terminology is deprecated for these statistics.",
            "",
            f"Geometry: {geometry}",
            f"Effective dimensionality (s²-weighted rank): {result.effective_rank:.2f}",
            f"Angular dispersion (mean angle / 180°): {result.angular_dispersion:.3f}",
            f"Categories analyzed: {result.category_count}",
        ]
        if result.mean_pairwise_cosine is not None:
            lines.append(
                f"Mean pairwise signed cosine: {result.mean_pairwise_cosine:.3f}"
            )
        if result.mean_pairwise_angle_degrees is not None:
            lines.append(
                f"Mean / max pairwise angle: "
                f"{result.mean_pairwise_angle_degrees:.1f}° / "
                f"{result.max_pairwise_angle_degrees:.1f}°"
            )
        lines.extend(("", "Per-Category Descriptive Directions:"))
        for category_direction in sorted(
            result.category_directions,
            key=lambda item: -item.strength,
        ):
            lines.append(
                f"  {category_direction.category:15s}  "
                f"strength={category_direction.strength:.3f}  "
                f"distinctiveness={category_direction.directional_distinctiveness:.3f}  "
                f"(n={category_direction.n_prompts})"
            )

        if result.pairwise_cosines:
            lines.extend(("", "Pairwise Signed Cosines / Angles:"))
            for key in sorted(result.pairwise_cosines):
                left, right = key
                cosine = result.pairwise_cosines[key]
                angle = result.pairwise_angles_degrees[key]
                lines.append(
                    f"  {left:12s} ↔ {right:12s}: cos={cosine:+.3f}, "
                    f"angle={angle:.1f}°"
                )
        return "\n".join(lines)


class ConceptConeAnalyzer(CategoryDirectionDispersionAnalyzer):
    """Deprecated compatibility name for category-direction dispersion analysis."""

    def __init__(
        self,
        category_map: Mapping[int, str] | None = None,
        min_category_size: int = 2,
    ):
        _warn_legacy("ConceptConeAnalyzer", "CategoryDirectionDispersionAnalyzer")
        # An empty map preserves construction/empty-input compatibility, while
        # any non-empty analysis still fails until the caller supplies labels.
        super().__init__({} if category_map is None else category_map, min_category_size)


# Deprecated type aliases.  Result instances expose warning-emitting legacy
# properties, while new code should import the descriptive names above.
ConeConeResult = CategoryDirectionDispersionResult
MultiLayerConeResult = MultiLayerDirectionDispersionResult
