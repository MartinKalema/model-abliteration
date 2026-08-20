"""Paper-style Gabliteration primitives and fail-closed behavioral replay.

The live OBLITERATUS pipeline historically used the name *Gabliteration* for a
different per-layer SVD heuristic.  This module keeps the paper implementation
isolated behind an explicit, validated integration API so it cannot silently
fall back to that heuristic.

Conventions
-----------
Activation matrices have shape ``(samples, hidden_dim)``.  Refusal subspaces
are stored row-wise with shape ``(n_directions, hidden_dim)``; this is the
transpose of the paper's column-wise matrix :math:`R`.

The shuffle stabilizer averages subspace projectors rather than signed vectors.
If ``Q_s`` is one shuffle's row-orthonormal basis, its projector is
``Q_s.T @ Q_s``.  The leading eigenspace of the average projector is computed
without materializing a ``hidden_dim x hidden_dim`` matrix: stack
``Q_s / sqrt(n_shuffles)`` and take its right singular vectors.

The tensor helpers perform no model forward passes and mutate none of their
inputs.  :func:`run_gabliteration_search` adds the paper's unmodified-model
source-layer forward passes, isolated one-layer generation trials, strict
effectiveness selection, and exact final replay.  Trials reuse the repository's
``ModelHandle.snapshot`` / ``ModelHandle.restore`` API and verify a byte-level
hash of the *complete* state dict after every rollback.

Primary-source provenance and explicit assumptions
--------------------------------------------------
The implementation follows arXiv:2512.18901v3, sections 2.1, 2.3, 2.5--2.9
and Algorithms 1--4, plus the author's upstream package at commit
``1498fc747772550cb0fca5b0b8e593b8326532af``.  The paper says to average
right-singular directions over 3--5 random pairings, but singular vectors have
arbitrary signs and a repeated singular subspace has arbitrary rotations.  We
therefore average the sign/rotation-invariant projectors and take their leading
eigenspace.  This is the only stable interpretation of that instruction that
does not make the result depend on an SVD implementation's basis convention.

The complete assumptions are exported as
:data:`GABLITERATION_IMPLEMENTATION_ASSUMPTIONS` and embedded verbatim in replay
metadata.  In particular, ``alpha`` is the fraction removed and remains
separate from ridge ``lambda``; one-layer trials use ``alpha_base``; final edits
use ordinal position within the selected effective-layer set; every selected
layer is edited exactly once; and unsupported/ambiguous manifests, quantized
weights, shared cross-layer storage, missing forward evidence, or incomplete
rollback all abort before a checkpoint can be treated as valid.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import torch
from torch import nn

from obliteratus.architecture_manifest import (
    ArchitectureCoverageError,
    ProjectionManifest,
    ProjectionManifestEntry,
)


@dataclass(frozen=True)
class MeanSeparationResult:
    """Mean-separation scores and the deterministically selected source layer."""

    source_layer: int
    scores: tuple[tuple[int, float], ...]

    def as_dict(self) -> dict[int, float]:
        """Return the scores as a new mutable mapping."""

        return dict(self.scores)


@dataclass(frozen=True)
class ProjectorAverageResult:
    """Leading eigenspace of an average of sign-invariant projectors.

    ``directions`` contains row-orthonormal vectors.  ``eigenvalues`` are the
    corresponding eigenvalues of the average projector and therefore lie in
    ``[0, 1]`` up to floating-point noise.
    """

    directions: torch.Tensor
    eigenvalues: torch.Tensor


@dataclass(frozen=True)
class ShuffleStabilizedSubspace:
    """A refusal subspace estimated from deterministic random pairings."""

    directions: torch.Tensor
    projector_eigenvalues: torch.Tensor
    shuffle_singular_values: torch.Tensor
    n_paired: int
    n_shuffles: int
    seed: int


@dataclass(frozen=True)
class AdaptiveLayerScales:
    """Paper position coordinates and partial-removal scales by layer rank."""

    layers: tuple[int, ...]
    normalized_positions: tuple[float, ...]
    alphas: tuple[float, ...]

    def as_dict(self) -> dict[int, float]:
        """Return ``layer -> alpha`` as a new mutable mapping."""

        return dict(zip(self.layers, self.alphas, strict=True))


def _validate_activation_matrix(matrix: torch.Tensor, name: str) -> None:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if matrix.ndim != 2:
        raise ValueError(f"{name} must have shape (samples, hidden_dim)")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not matrix.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not torch.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def mean_separation_source_layer(
    harmful_by_layer: Mapping[int, torch.Tensor],
    harmless_by_layer: Mapping[int, torch.Tensor],
) -> MeanSeparationResult:
    """Select the source layer by harmful/harmless mean separation.

    For every layer ``ell`` this computes

    ``S_ell = ||mean(harmful_ell) - mean(harmless_ell)||_2``.

    The layer with the largest score is selected.  Exact ties choose the
    numerically smallest layer, making the result independent of mapping order.
    The two mappings must cover exactly the same layers; silently ignoring
    missing evidence would make a claimed source-layer search incomplete.
    """

    harmful_layers = set(harmful_by_layer)
    harmless_layers = set(harmless_by_layer)
    if not harmful_layers:
        raise ValueError("at least one layer of activations is required")
    if harmful_layers != harmless_layers:
        missing_harmful = sorted(harmless_layers - harmful_layers)
        missing_harmless = sorted(harmful_layers - harmless_layers)
        raise ValueError(
            "harmful and harmless mappings must contain identical layers "
            f"(missing harmful={missing_harmful}, missing harmless={missing_harmless})"
        )

    scores: list[tuple[int, float]] = []
    for layer_idx in sorted(harmful_layers):
        harmful = harmful_by_layer[layer_idx]
        harmless = harmless_by_layer[layer_idx]
        _validate_activation_matrix(harmful, f"harmful_by_layer[{layer_idx}]")
        _validate_activation_matrix(harmless, f"harmless_by_layer[{layer_idx}]")
        if harmful.shape[1] != harmless.shape[1]:
            raise ValueError(f"layer {layer_idx} harmful and harmless hidden dimensions differ")

        # Compute the statistic in at least float32 so half-precision activation
        # caches do not underflow during averaging or norm evaluation.
        work_dtype = (
            torch.float64
            if harmful.dtype == torch.float64 or harmless.dtype == torch.float64
            else torch.float32
        )
        harmful_mean = harmful.to(dtype=work_dtype).mean(dim=0)
        harmless_mean = harmless.to(device=harmful.device, dtype=work_dtype).mean(dim=0)
        score = torch.linalg.vector_norm(harmful_mean - harmless_mean).item()
        scores.append((int(layer_idx), float(score)))

    source_layer = min(scores, key=lambda item: (-item[1], item[0]))[0]
    return MeanSeparationResult(source_layer=source_layer, scores=tuple(scores))


def _canonicalize_row_signs(directions: torch.Tensor) -> torch.Tensor:
    """Choose a stable representative for each direction's arbitrary SVD sign."""

    result = directions.clone()
    for row_idx in range(result.shape[0]):
        pivot = result[row_idx].abs().argmax()
        if result[row_idx, pivot] < 0:
            result[row_idx].neg_()
    return result


def average_projector_subspace(
    subspaces: Sequence[torch.Tensor],
    *,
    n_directions: int | None = None,
) -> ProjectorAverageResult:
    """Return the leading eigenspace of an average of subspace projectors.

    Each input has row-wise directions with a common hidden dimension.  Rows
    need not already be orthonormal: a reduced QR factorization first converts
    each input to an orthonormal basis.  Averaging projectors makes the result
    invariant to sign flips, row permutations, and rotations within an input
    subspace.

    The computation is matrix-free with respect to hidden size; it never forms
    an explicit ``hidden_dim x hidden_dim`` projector.
    """

    if not subspaces:
        raise ValueError("subspaces must contain at least one basis")
    if not all(isinstance(basis, torch.Tensor) for basis in subspaces):
        raise TypeError("subspaces must contain only torch.Tensor values")

    hidden_dim: int | None = None
    orthonormal_bases: list[torch.Tensor] = []
    common_device = subspaces[0].device
    work_dtype = (
        torch.float64 if any(s.dtype == torch.float64 for s in subspaces) else torch.float32
    )

    for index, basis in enumerate(subspaces):
        if basis.ndim != 2 or basis.shape[0] == 0 or basis.shape[1] == 0:
            raise ValueError(f"subspaces[{index}] must have shape (rank, hidden_dim)")
        if not basis.is_floating_point():
            raise TypeError(f"subspaces[{index}] must have a floating-point dtype")
        if not torch.isfinite(basis).all():
            raise ValueError(f"subspaces[{index}] contains NaN or infinite values")
        if basis.device != common_device:
            raise ValueError("all subspaces must be on the same device")
        if hidden_dim is None:
            hidden_dim = basis.shape[1]
        elif basis.shape[1] != hidden_dim:
            raise ValueError("all subspaces must have the same hidden dimension")

        # QR on the transposed row basis gives orthonormal columns; transpose
        # back to the module's row-wise direction convention.
        q, r = torch.linalg.qr(basis.to(dtype=work_dtype).T, mode="reduced")
        diag = torch.diagonal(r).abs()
        tolerance = max(basis.shape) * torch.finfo(work_dtype).eps * diag.max().clamp(min=1.0)
        numerical_rank = int((diag > tolerance).sum().item())
        if numerical_rank != basis.shape[0]:
            raise ValueError(f"subspaces[{index}] is rank deficient")
        orthonormal_bases.append(q.T.contiguous())

    assert hidden_dim is not None
    max_rank = min(hidden_dim, sum(basis.shape[0] for basis in orthonormal_bases))
    if n_directions is None:
        n_directions = min(basis.shape[0] for basis in orthonormal_bases)
    if not isinstance(n_directions, int) or isinstance(n_directions, bool):
        raise TypeError("n_directions must be an integer")
    if not 1 <= n_directions <= max_rank:
        raise ValueError(f"n_directions must be in [1, {max_rank}]")

    stacked = torch.cat(orthonormal_bases, dim=0) / math.sqrt(len(orthonormal_bases))
    _, singular_values, vh = torch.linalg.svd(stacked, full_matrices=False)
    directions = _canonicalize_row_signs(vh[:n_directions]).contiguous()
    eigenvalues = singular_values[:n_directions].square().clamp(min=0.0, max=1.0)
    return ProjectorAverageResult(directions=directions, eigenvalues=eigenvalues)


def shuffle_stabilized_svd_subspace(
    harmful: torch.Tensor,
    harmless: torch.Tensor,
    *,
    n_directions: int = 2,
    n_shuffles: int = 5,
    seed: int = 0,
) -> ShuffleStabilizedSubspace:
    """Extract a stable paired-difference SVD subspace.

    Each shuffle independently permutes both activation sets using a local CPU
    generator, pairs ``min(n_harmful, n_harmless)`` rows, and extracts the top
    right singular vectors of the difference matrix.  The final basis is the
    leading eigenspace of the average shuffle projector, so arbitrary SVD signs
    and within-subspace basis order cannot affect it.

    The function does not touch PyTorch's global random state.
    """

    _validate_activation_matrix(harmful, "harmful")
    _validate_activation_matrix(harmless, "harmless")
    if harmful.shape[1] != harmless.shape[1]:
        raise ValueError("harmful and harmless hidden dimensions must match")
    if harmful.device != harmless.device:
        raise ValueError("harmful and harmless activations must be on the same device")
    if not isinstance(n_directions, int) or isinstance(n_directions, bool):
        raise TypeError("n_directions must be an integer")
    if not isinstance(n_shuffles, int) or isinstance(n_shuffles, bool):
        raise TypeError("n_shuffles must be an integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if n_shuffles <= 0:
        raise ValueError("n_shuffles must be positive")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be in [0, 2**63)")

    n_paired = min(harmful.shape[0], harmless.shape[0])
    max_directions = min(n_paired, harmful.shape[1])
    if not 1 <= n_directions <= max_directions:
        raise ValueError(f"n_directions must be in [1, {max_directions}]")

    work_dtype = (
        torch.float64
        if harmful.dtype == torch.float64 or harmless.dtype == torch.float64
        else torch.float32
    )
    harmful_work = harmful.to(dtype=work_dtype)
    harmless_work = harmless.to(dtype=work_dtype)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    bases: list[torch.Tensor] = []
    singular_values_by_shuffle: list[torch.Tensor] = []
    for shuffle_idx in range(n_shuffles):
        harmful_indices = torch.randperm(harmful.shape[0], generator=generator)[:n_paired]
        harmless_indices = torch.randperm(harmless.shape[0], generator=generator)[:n_paired]
        harmful_indices = harmful_indices.to(device=harmful.device)
        harmless_indices = harmless_indices.to(device=harmless.device)
        differences = harmful_work.index_select(0, harmful_indices) - harmless_work.index_select(
            0, harmless_indices
        )

        _, singular_values, vh = torch.linalg.svd(differences, full_matrices=False)
        tolerance = (
            max(differences.shape)
            * torch.finfo(work_dtype).eps
            * singular_values.max().clamp(min=1.0)
        )
        numerical_rank = int((singular_values > tolerance).sum().item())
        if numerical_rank < n_directions:
            raise ValueError(
                f"shuffle {shuffle_idx} paired-difference matrix has rank "
                f"{numerical_rank}, below requested {n_directions}"
            )
        bases.append(vh[:n_directions].contiguous())
        singular_values_by_shuffle.append(singular_values[:n_directions].contiguous())

    averaged = average_projector_subspace(bases, n_directions=n_directions)
    return ShuffleStabilizedSubspace(
        directions=averaged.directions,
        projector_eigenvalues=averaged.eigenvalues,
        shuffle_singular_values=torch.stack(singular_values_by_shuffle),
        n_paired=n_paired,
        n_shuffles=n_shuffles,
        seed=int(seed),
    )


def ridge_subspace_update(
    tensor: torch.Tensor,
    directions: torch.Tensor,
    *,
    residual_axis: int,
    alpha: float,
    ridge_lambda: float,
) -> torch.Tensor:
    """Return the exact ridge-subspace update without forming a large projector.

    For row-wise directions ``Q`` (the paper's ``R.T``), this computes along an
    arbitrary residual axis:

    ``X' = X - alpha * X Q.T (Q Q.T + ridge_lambda I)^-1 Q``.

    Only the ``n_directions x n_directions`` Gram matrix is materialized.  The
    input tensor and directions are never modified.  ``alpha`` is the paper's
    partial-removal strength; ``ridge_lambda`` is the separate ridge parameter.
    Keeping these values separate prevents conflating numerical regularization
    with the fraction of the refusal component to remove.
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(directions, torch.Tensor):
        raise TypeError("tensor and directions must be torch.Tensor values")
    if tensor.ndim == 0:
        raise ValueError("tensor must have at least one dimension")
    if directions.ndim != 2 or directions.shape[0] == 0 or directions.shape[1] == 0:
        raise ValueError("directions must have shape (n_directions, hidden_dim)")
    if not tensor.is_floating_point() or not directions.is_floating_point():
        raise TypeError("tensor and directions must have floating-point dtypes")
    if tensor.device != directions.device:
        raise ValueError("tensor and directions must be on the same device")
    if not torch.isfinite(tensor).all() or not torch.isfinite(directions).all():
        raise ValueError("tensor and directions must contain only finite values")
    if not isinstance(residual_axis, int) or isinstance(residual_axis, bool):
        raise TypeError("residual_axis must be an integer")
    if not -tensor.ndim <= residual_axis < tensor.ndim:
        raise ValueError("residual_axis is out of range")
    if not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    if not math.isfinite(float(ridge_lambda)) or float(ridge_lambda) < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")

    axis = residual_axis % tensor.ndim
    hidden_dim = tensor.shape[axis]
    if directions.shape[1] != hidden_dim:
        raise ValueError(
            f"directions hidden dimension {directions.shape[1]} does not match "
            f"tensor residual axis {hidden_dim}"
        )
    if directions.shape[0] > hidden_dim:
        raise ValueError("n_directions cannot exceed hidden_dim")

    work_dtype = (
        torch.float64
        if tensor.dtype == torch.float64 or directions.dtype == torch.float64
        else torch.float32
    )
    moved = tensor.to(dtype=work_dtype).movedim(axis, -1)
    flat = moved.reshape(-1, hidden_dim)
    q = directions.to(dtype=work_dtype)
    gram = q @ q.T
    gram = gram + float(ridge_lambda) * torch.eye(q.shape[0], device=q.device, dtype=q.dtype)
    coefficients = flat @ q.T
    try:
        solved = torch.linalg.solve(gram, coefficients.T).T
    except torch.linalg.LinAlgError as exc:
        raise ValueError(
            "directions are rank deficient; use positive ridge_lambda or an "
            "independent subspace basis"
        ) from exc
    projected = solved @ q
    updated = flat - float(alpha) * projected
    return updated.reshape(moved.shape).movedim(-1, axis).to(dtype=tensor.dtype)


def paper_adaptive_layer_scales(
    effective_layers: Sequence[int],
    *,
    alpha_base: float = 0.3,
    beta: float = 0.5,
) -> AdaptiveLayerScales:
    """Compute Gabliteration's position-adaptive partial-removal strengths.

    Layer identifiers are sorted, then treated by ordinal rank within the
    effective-layer set.  This is necessary for non-contiguous transformer
    layer indices: substituting raw indices into the paper's one-based position
    formula would not stay in ``[-1, 1]``.

    Boundary layers receive ``alpha_base``; middle layers receive up to
    ``alpha_base * (1 + beta)``.  A singleton receives the maximum scale, as in
    the paper's explicit one-layer case.
    """

    if not math.isfinite(float(alpha_base)) or not 0.0 <= float(alpha_base) <= 1.0:
        raise ValueError("alpha_base must be finite and in [0, 1]")
    if not math.isfinite(float(beta)) or float(beta) < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if float(alpha_base) * (1.0 + float(beta)) > 1.0 + 1e-12:
        raise ValueError("alpha_base * (1 + beta) must not exceed 1")
    if not effective_layers:
        raise ValueError("effective_layers must not be empty")
    if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in effective_layers):
        raise TypeError("effective_layers must contain integer layer indices")

    layers = tuple(sorted(effective_layers))
    if len(set(layers)) != len(layers):
        raise ValueError("effective_layers must not contain duplicates")

    if len(layers) == 1:
        positions = (0.0,)
        alphas = (float(alpha_base) * (1.0 + float(beta)),)
    else:
        positions = tuple(2.0 * rank / (len(layers) - 1) - 1.0 for rank in range(len(layers)))
        alphas = tuple(
            float(alpha_base) * (1.0 + float(beta) * (1.0 - abs(position)))
            for position in positions
        )

    return AdaptiveLayerScales(
        layers=layers,
        normalized_positions=positions,
        alphas=alphas,
    )


# ---------------------------------------------------------------------------
# Behavioral layer search and exact replay
# ---------------------------------------------------------------------------

GABLITERATION_PAPER = "https://arxiv.org/abs/2512.18901v3"
GABLITERATION_UPSTREAM = "https://github.com/Goekdeniz-Guelmez/gabliteration"
GABLITERATION_UPSTREAM_COMMIT = "1498fc747772550cb0fca5b0b8e593b8326532af"
GABLITERATION_REPLAY_SCHEMA = "obliteratus.gabliteration.replay.v1"

GABLITERATION_IMPLEMENTATION_ASSUMPTIONS: tuple[str, ...] = (
    (
        "Source-layer selection uses last-token post-block hidden states from the untouched "
        "model and chooses the maximum harmful/harmless mean-separation norm, with the "
        "smallest layer index breaking exact ties."
    ),
    (
        "The paper's average over shuffled SVD directions is represented by the leading "
        "eigenspace of the average sign- and rotation-invariant shuffle projectors."
    ),
    (
        "A layer-effectiveness trial edits every validated attention-output and MLP-output "
        "writer in exactly one candidate layer using alpha_base, then evaluates generated "
        "responses with the strict paper criterion refusal_rate < threshold."
    ),
    (
        "Every trial starts from and returns to the same complete ModelHandle state_dict "
        "snapshot; byte-level hashes are checked before edit, after evaluation, and after "
        "rollback. Python, NumPy (when loaded), and Torch RNG states are also reset."
    ),
    (
        "The final effective-layer edit is a single pass from the untouched snapshot. "
        "Adaptive alpha uses ordinal position in the sorted effective-layer set, not raw "
        "transformer indices."
    ),
    (
        "Alpha is the fraction of the ridge projection removed. It is not converted into, "
        "combined with, or substituted for the independent ridge lambda."
    ),
    (
        "Only dense, materialized floating-point writer parameters with an exact output-only "
        "ProjectionManifest are supported. Quantized, meta, incomplete, ambiguous, or "
        "cross-layer shared entries fail before mutation."
    ),
    (
        "Replay is permitted only on a byte-equivalent baseline state and an identical "
        "semantic manifest; direction, plan, baseline, per-parameter, and final-state hashes "
        "are all verified."
    ),
)

DEFAULT_REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot",
    "i can't",
    "i apologize",
    "i'm sorry",
    "i am sorry",
    "i'm unable",
    "i am unable",
    "cannot assist",
    "can't assist",
    "cannot help",
    "can't help",
)


class GabliterationError(RuntimeError):
    """Base class for fail-closed Gabliteration orchestration failures."""


class GabliterationValidationError(GabliterationError):
    """Raised before or during search when exact semantics cannot be established."""


class GabliterationRollbackError(GabliterationError):
    """Raised when a complete trial rollback cannot be proven byte-equivalent."""


class SnapshotModelHandle(Protocol):
    """The subset of :class:`ModelHandle` required for exact trial rollback."""

    model: nn.Module
    _original_state: Mapping[str, torch.Tensor] | None

    def snapshot(self) -> None:
        """Capture the complete model state."""

    def restore(self) -> None:
        """Restore the complete captured model state."""


@dataclass(frozen=True)
class HiddenStateBatch:
    """One already-tokenized batch for source-layer forward extraction.

    ``model_kwargs`` are passed directly to ``model(**kwargs)`` with
    ``output_hidden_states=True``.  If ``last_token_indices`` is absent, the
    final true position is inferred from ``attention_mask``; without a mask,
    the final sequence position is used.
    """

    model_kwargs: Mapping[str, Any]
    last_token_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class GabliterationSearchConfig:
    """Fully explicit parameters for one paper-style layer search."""

    candidate_layers: tuple[int, ...]
    n_directions: int = 2
    n_shuffles: int = 5
    seed: int = 0
    alpha_base: float = 0.3
    beta: float = 0.5
    ridge_lambda: float = 0.1
    effectiveness_threshold: float = 0.8
    refusal_markers: tuple[str, ...] = DEFAULT_REFUSAL_MARKERS

    def validated(self) -> GabliterationSearchConfig:
        """Validate and return ``self`` for fluent call sites."""

        if not self.candidate_layers:
            raise GabliterationValidationError("candidate_layers must not be empty")
        if any(
            not isinstance(layer, int) or isinstance(layer, bool)
            for layer in self.candidate_layers
        ):
            raise GabliterationValidationError(
                "candidate_layers must contain integer layer indices"
            )
        if tuple(sorted(self.candidate_layers)) != self.candidate_layers:
            raise GabliterationValidationError(
                "candidate_layers must be strictly increasing for deterministic replay"
            )
        if len(set(self.candidate_layers)) != len(self.candidate_layers):
            raise GabliterationValidationError("candidate_layers must not contain duplicates")
        if not isinstance(self.n_directions, int) or isinstance(self.n_directions, bool):
            raise GabliterationValidationError("n_directions must be an integer")
        if self.n_directions <= 0:
            raise GabliterationValidationError("n_directions must be positive")
        if not isinstance(self.n_shuffles, int) or isinstance(self.n_shuffles, bool):
            raise GabliterationValidationError("n_shuffles must be an integer")
        if self.n_shuffles <= 0:
            raise GabliterationValidationError("n_shuffles must be positive")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise GabliterationValidationError("seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise GabliterationValidationError("seed must be in [0, 2**63)")
        for name, value in (
            ("alpha_base", self.alpha_base),
            ("beta", self.beta),
            ("ridge_lambda", self.ridge_lambda),
            ("effectiveness_threshold", self.effectiveness_threshold),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise GabliterationValidationError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise GabliterationValidationError(f"{name} must be finite")
        if not 0.0 <= float(self.alpha_base) <= 1.0:
            raise GabliterationValidationError("alpha_base must be in [0, 1]")
        if float(self.beta) < 0.0:
            raise GabliterationValidationError("beta must be non-negative")
        if float(self.alpha_base) * (1.0 + float(self.beta)) > 1.0 + 1e-12:
            raise GabliterationValidationError(
                "alpha_base * (1 + beta) must not exceed 1"
            )
        if float(self.ridge_lambda) <= 0.0:
            raise GabliterationValidationError(
                "ridge_lambda must be positive for fail-closed paper replay"
            )
        if not 0.0 <= float(self.effectiveness_threshold) <= 1.0:
            raise GabliterationValidationError("effectiveness_threshold must be in [0, 1]")
        if not self.refusal_markers or any(
            not isinstance(marker, str) or not marker.strip()
            for marker in self.refusal_markers
        ):
            raise GabliterationValidationError(
                "refusal_markers must contain non-empty strings"
            )
        return self

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable configuration record."""

        return {
            "candidate_layers": list(self.candidate_layers),
            "n_directions": self.n_directions,
            "n_shuffles": self.n_shuffles,
            "seed": self.seed,
            "alpha_base": float(self.alpha_base),
            "beta": float(self.beta),
            "ridge_lambda": float(self.ridge_lambda),
            "effectiveness_threshold": float(self.effectiveness_threshold),
            "refusal_markers": list(self.refusal_markers),
        }


@dataclass(frozen=True)
class LayerEffectivenessTrial:
    """Auditable evidence for one isolated paper Phase-4 layer trial."""

    layer: int
    refusal_count: int
    response_count: int
    refusal_rate: float
    effective: bool
    before_state_sha256: str
    edited_state_sha256: str
    restored_state_sha256: str
    response_sha256: str
    edited_parameter_sha256: tuple[tuple[str, str], ...]

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable trial record without raw responses."""

        return {
            "layer": self.layer,
            "refusal_count": self.refusal_count,
            "response_count": self.response_count,
            "refusal_rate": self.refusal_rate,
            "effective": self.effective,
            "before_state_sha256": self.before_state_sha256,
            "edited_state_sha256": self.edited_state_sha256,
            "restored_state_sha256": self.restored_state_sha256,
            "response_sha256": self.response_sha256,
            "edited_parameter_sha256": dict(self.edited_parameter_sha256),
        }


@dataclass(frozen=True)
class GabliterationReplayPlan:
    """Self-verifying, exact data needed to replay the selected final edit."""

    schema: str
    source_layer: int
    candidate_layers: tuple[int, ...]
    effective_layers: tuple[int, ...]
    layer_alphas: tuple[tuple[int, float], ...]
    ridge_lambda: float
    n_directions: int
    n_shuffles: int
    seed: int
    directions: torch.Tensor
    direction_sha256: str
    manifest_sha256: str
    baseline_state_sha256: str
    expected_state_sha256: str
    baseline_parameter_sha256: tuple[tuple[str, str], ...]
    expected_parameter_sha256: tuple[tuple[str, str], ...]
    plan_sha256: str

    def _integrity_metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "paper": GABLITERATION_PAPER,
            "upstream": GABLITERATION_UPSTREAM,
            "upstream_commit": GABLITERATION_UPSTREAM_COMMIT,
            "assumptions": list(GABLITERATION_IMPLEMENTATION_ASSUMPTIONS),
            "source_layer": self.source_layer,
            "candidate_layers": list(self.candidate_layers),
            "effective_layers": list(self.effective_layers),
            "layer_alphas": [
                {"layer": layer, "alpha": alpha} for layer, alpha in self.layer_alphas
            ],
            "ridge_lambda": self.ridge_lambda,
            "n_directions": self.n_directions,
            "n_shuffles": self.n_shuffles,
            "seed": self.seed,
            "directions": {
                "shape": list(self.directions.shape),
                "dtype": str(self.directions.dtype),
                "sha256": self.direction_sha256,
            },
            "manifest_sha256": self.manifest_sha256,
            "baseline_state_sha256": self.baseline_state_sha256,
            "expected_state_sha256": self.expected_state_sha256,
            "baseline_parameter_sha256": dict(self.baseline_parameter_sha256),
            "expected_parameter_sha256": dict(self.expected_parameter_sha256),
        }

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable replay metadata (directions are hash-addressed)."""

        metadata = self._integrity_metadata()
        metadata["plan_sha256"] = self.plan_sha256
        return metadata

    def to_payload(self) -> dict[str, Any]:
        """Return a ``torch.save``-ready payload containing exact replay data."""

        return {
            "metadata": self.to_metadata(),
            "directions": self.directions.detach().to(device="cpu", copy=True),
        }

    def validate_integrity(self) -> None:
        """Reject mutated directions or metadata before touching model weights."""

        if self.schema != GABLITERATION_REPLAY_SCHEMA:
            raise GabliterationValidationError(
                f"unsupported replay schema {self.schema!r}"
            )
        if (
            not self.candidate_layers
            or tuple(sorted(self.candidate_layers)) != self.candidate_layers
            or len(set(self.candidate_layers)) != len(self.candidate_layers)
        ):
            raise GabliterationValidationError(
                "replay candidate_layers must be non-empty, unique, and sorted"
            )
        if self.source_layer not in self.candidate_layers:
            raise GabliterationValidationError(
                "replay source_layer is not in candidate_layers"
            )
        if (
            not self.effective_layers
            or tuple(sorted(self.effective_layers)) != self.effective_layers
            or not set(self.effective_layers).issubset(self.candidate_layers)
        ):
            raise GabliterationValidationError(
                "replay effective_layers must be a non-empty sorted candidate subset"
            )
        alpha_layers = tuple(layer for layer, _alpha in self.layer_alphas)
        if alpha_layers != self.effective_layers or any(
            not math.isfinite(float(alpha)) or not 0.0 <= float(alpha) <= 1.0
            for _layer, alpha in self.layer_alphas
        ):
            raise GabliterationValidationError(
                "replay layer_alphas must exactly cover effective_layers in [0, 1]"
            )
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda <= 0.0:
            raise GabliterationValidationError("replay ridge_lambda must be positive")
        if self.n_directions <= 0 or self.n_shuffles <= 0:
            raise GabliterationValidationError(
                "replay direction and shuffle counts must be positive"
            )
        if not 0 <= self.seed < 2**63:
            raise GabliterationValidationError("replay seed is out of range")
        baseline_names = tuple(name for name, _digest in self.baseline_parameter_sha256)
        expected_names = tuple(name for name, _digest in self.expected_parameter_sha256)
        if (
            not baseline_names
            or len(set(baseline_names)) != len(baseline_names)
            or baseline_names != expected_names
        ):
            raise GabliterationValidationError(
                "replay parameter hashes must uniquely cover the same non-empty entries"
            )
        if self.directions.device.type != "cpu" or not self.directions.is_contiguous():
            raise GabliterationValidationError(
                "replay directions must be contiguous CPU tensors"
            )
        if (
            self.directions.ndim != 2
            or self.directions.shape[0] != self.n_directions
            or not self.directions.is_floating_point()
            or not torch.isfinite(self.directions).all()
        ):
            raise GabliterationValidationError(
                "replay directions have an invalid shape, dtype, or value"
            )
        actual_direction_hash = tensor_sha256(self.directions)
        if actual_direction_hash != self.direction_sha256:
            raise GabliterationValidationError(
                "replay directions do not match direction_sha256"
            )
        actual_plan_hash = _json_sha256(self._integrity_metadata())
        if actual_plan_hash != self.plan_sha256:
            raise GabliterationValidationError("replay metadata does not match plan_sha256")


@dataclass(frozen=True)
class GabliterationSearchResult:
    """Source selection, behavioral trials, and exact final replay artifact."""

    replay_plan: GabliterationReplayPlan
    source_scores: tuple[tuple[int, float], ...]
    projector_eigenvalues: tuple[float, ...]
    shuffle_singular_values: tuple[tuple[float, ...], ...]
    trials: tuple[LayerEffectivenessTrial, ...]
    applied: bool
    final_state_sha256: str

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable audit record."""

        return {
            "replay_plan": self.replay_plan.to_metadata(),
            "source_scores": dict(self.source_scores),
            "projector_eigenvalues": list(self.projector_eigenvalues),
            "shuffle_singular_values": [list(row) for row in self.shuffle_singular_values],
            "trials": [trial.to_metadata() for trial in self.trials],
            "applied": self.applied,
            "final_state_sha256": self.final_state_sha256,
        }


@dataclass(frozen=True)
class _RNGSnapshot:
    python_state: object
    numpy_state: object | None
    torch_cpu_state: torch.Tensor
    torch_cuda_states: tuple[torch.Tensor, ...]
    torch_mps_state: torch.Tensor | None


def _capture_rng_state() -> _RNGSnapshot:
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except (ImportError, AttributeError):
        pass
    cuda_states: tuple[torch.Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    mps_state = None
    if torch.backends.mps.is_available():
        mps_state = torch.mps.get_rng_state().clone()
    return _RNGSnapshot(
        python_state=random.getstate(),
        numpy_state=numpy_state,
        torch_cpu_state=torch.random.get_rng_state().clone(),
        torch_cuda_states=cuda_states,
        torch_mps_state=mps_state,
    )


def _restore_rng_state(snapshot: _RNGSnapshot) -> None:
    random.setstate(snapshot.python_state)
    if snapshot.numpy_state is not None:
        try:
            import numpy as np

            np.random.set_state(snapshot.numpy_state)
        except (ImportError, AttributeError) as exc:
            raise GabliterationRollbackError("NumPy RNG rollback failed") from exc
    torch.random.set_rng_state(snapshot.torch_cpu_state)
    if snapshot.torch_cuda_states:
        if not torch.cuda.is_available():
            raise GabliterationRollbackError("CUDA RNG state cannot be restored")
        torch.cuda.set_rng_state_all(list(snapshot.torch_cuda_states))
    if snapshot.torch_mps_state is not None:
        if not torch.backends.mps.is_available():
            raise GabliterationRollbackError("MPS RNG state cannot be restored")
        torch.mps.set_rng_state(snapshot.torch_mps_state)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor shape, dtype, and exact contiguous CPU bytes."""

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    if tensor.device.type == "meta":
        raise GabliterationValidationError("meta tensors cannot be hashed for exact replay")
    cpu = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(json.dumps(list(cpu.shape), separators=(",", ":")).encode("ascii"))
    digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_dict_sha256(
    state_dict: Mapping[str, torch.Tensor],
    *,
    overrides: Mapping[str, torch.Tensor] | None = None,
) -> str:
    """Hash a full named state mapping, optionally substituting planned values."""

    if not state_dict:
        raise GabliterationValidationError("state_dict must not be empty")
    overrides = overrides or {}
    unknown = set(overrides) - set(state_dict)
    if unknown:
        raise GabliterationValidationError(
            f"state hash overrides contain unknown keys: {sorted(unknown)}"
        )
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        value = overrides.get(name, state_dict[name])
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise GabliterationValidationError(
                "state_dict must map string names to tensors"
            )
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_sha256(value).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def gabliteration_manifest_sha256(manifest: ProjectionManifest) -> str:
    """Hash semantic manifest fields while excluding process-local storage pointers."""

    entries = []
    for entry in manifest.entries:
        metadata = entry.to_metadata()
        metadata.pop("storage_identity", None)
        entries.append(metadata)
    return _json_sha256(
        {
            "architecture": manifest.architecture,
            "target": manifest.target,
            "layer_path": manifest.layer_path,
            "hidden_size": manifest.hidden_size,
            "num_layers": manifest.num_layers,
            "entries": entries,
            "branch_coverage": list(manifest.branch_coverage),
        }
    )


def _model_hidden_states(output: Any) -> Sequence[torch.Tensor]:
    hidden_states = (
        output.get("hidden_states") if isinstance(output, Mapping) else getattr(output, "hidden_states", None)
    )
    if not isinstance(hidden_states, (tuple, list)) or not hidden_states:
        raise GabliterationValidationError(
            "model forward did not return a non-empty hidden_states sequence"
        )
    if not all(isinstance(value, torch.Tensor) for value in hidden_states):
        raise GabliterationValidationError("hidden_states must contain only tensors")
    return hidden_states


def _last_token_indices(
    batch: HiddenStateBatch,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    indices = batch.last_token_indices
    if indices is None:
        attention_mask = batch.model_kwargs.get("attention_mask")
        if attention_mask is None:
            return torch.full(
                (batch_size,), sequence_length - 1, dtype=torch.long, device=device
            )
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
            raise GabliterationValidationError(
                "attention_mask must be a rank-2 tensor when used for token selection"
            )
        if tuple(attention_mask.shape) != (batch_size, sequence_length):
            raise GabliterationValidationError(
                "attention_mask shape does not match hidden-state batch/sequence axes"
            )
        positions = torch.arange(sequence_length, device=attention_mask.device)
        masked_positions = torch.where(
            attention_mask.to(dtype=torch.bool), positions.unsqueeze(0), -1
        )
        indices = masked_positions.max(dim=1).values
        if (indices < 0).any():
            raise GabliterationValidationError("attention_mask contains an empty sequence")
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise GabliterationValidationError("last_token_indices must be a rank-1 tensor")
    if indices.shape[0] != batch_size:
        raise GabliterationValidationError(
            "last_token_indices length does not match forward batch size"
        )
    indices = indices.to(device=device, dtype=torch.long)
    if ((indices < 0) | (indices >= sequence_length)).any():
        raise GabliterationValidationError("last_token_indices contains an invalid position")
    return indices


def extract_last_token_hidden_states(
    model: nn.Module,
    batches: Sequence[HiddenStateBatch],
    candidate_layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Run actual model forwards and collect last-token post-layer states."""

    if not batches:
        raise GabliterationValidationError("at least one hidden-state batch is required")
    layers = tuple(candidate_layers)
    if not layers:
        raise GabliterationValidationError("candidate_layers must not be empty")
    collected: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            if not isinstance(batch, HiddenStateBatch):
                raise GabliterationValidationError(
                    f"batches[{batch_index}] must be a HiddenStateBatch"
                )
            kwargs = dict(batch.model_kwargs)
            if kwargs.get("output_hidden_states") is False:
                raise GabliterationValidationError(
                    "output_hidden_states=False conflicts with source-layer extraction"
                )
            kwargs["output_hidden_states"] = True
            output = model(**kwargs)
            hidden_states = _model_hidden_states(output)
            max_layer = max(layers)
            if len(hidden_states) <= max_layer + 1:
                raise GabliterationValidationError(
                    f"forward returned {len(hidden_states)} hidden states, insufficient for "
                    f"post-layer index {max_layer + 1}"
                )
            for layer in layers:
                value = hidden_states[layer + 1]
                if value.ndim != 3 or value.shape[0] == 0 or value.shape[1] == 0:
                    raise GabliterationValidationError(
                        f"hidden state for layer {layer} must have shape "
                        "(batch, sequence, hidden)"
                    )
                indices = _last_token_indices(
                    batch,
                    batch_size=value.shape[0],
                    sequence_length=value.shape[1],
                    device=value.device,
                )
                rows = value[
                    torch.arange(value.shape[0], device=value.device), indices
                ]
                if not rows.is_floating_point() or not torch.isfinite(rows).all():
                    raise GabliterationValidationError(
                        f"hidden state for layer {layer} is non-floating or non-finite"
                    )
                collected[layer].append(rows.detach().to(device="cpu"))
    return {layer: torch.cat(parts, dim=0) for layer, parts in collected.items()}


def _validate_manifest(
    handle: SnapshotModelHandle,
    manifest: ProjectionManifest,
    candidate_layers: Sequence[int],
) -> None:
    if manifest.target != "output":
        raise ArchitectureCoverageError(
            "paper Gabliteration requires an output-only writer manifest"
        )
    if manifest.hidden_size <= 0 or manifest.num_layers <= 0:
        raise ArchitectureCoverageError("manifest dimensions must be positive")
    if any(layer < 0 or layer >= manifest.num_layers for layer in candidate_layers):
        raise ArchitectureCoverageError("candidate layer is outside the manifest range")
    current_state = handle.model.state_dict()
    named_parameters = dict(handle.model.named_parameters(remove_duplicate=False))
    supported_weight_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    for module in handle.model.modules():
        if hasattr(module, "qweight"):
            raise ArchitectureCoverageError(
                "paper Gabliteration exact replay does not support quantized modules"
            )
    for entry in manifest.entries:
        if entry.role != "writer" or entry.orientation != "output":
            raise ArchitectureCoverageError(
                f"manifest entry {entry.qualified_name!r} is not an output writer"
            )
        parameter = entry.parameter
        if parameter.device.type == "meta" or parameter.dtype not in supported_weight_dtypes:
            raise ArchitectureCoverageError(
                f"manifest entry {entry.qualified_name!r} must be dense FP16/BF16/FP32"
            )
        if tuple(parameter.shape) != entry.shape:
            raise ArchitectureCoverageError(
                f"manifest entry {entry.qualified_name!r} changed shape"
            )
        if entry.dtype != str(parameter.dtype):
            raise ArchitectureCoverageError(
                f"manifest entry {entry.qualified_name!r} changed dtype"
            )
        axis = entry.residual_axis % parameter.ndim
        if parameter.shape[axis] != manifest.hidden_size:
            raise ArchitectureCoverageError(
                f"manifest entry {entry.qualified_name!r} has an invalid residual axis"
            )
        if not torch.isfinite(parameter).all():
            raise ArchitectureCoverageError(
                f"manifest entry {entry.qualified_name!r} contains NaN/Inf"
            )
        for alias in entry.aliases:
            if alias not in current_state or tuple(current_state[alias].shape) != entry.shape:
                raise ArchitectureCoverageError(
                    f"manifest alias {alias!r} is absent or shape-incompatible"
                )
            actual = named_parameters.get(alias)
            if actual is None:
                raise ArchitectureCoverageError(
                    f"manifest alias {alias!r} is not a named model parameter"
                )
            try:
                same_storage = (
                    actual.untyped_storage().data_ptr()
                    == parameter.untyped_storage().data_ptr()
                    and actual.storage_offset() == parameter.storage_offset()
                    and tuple(actual.stride()) == tuple(parameter.stride())
                )
            except (AttributeError, RuntimeError):
                same_storage = actual is parameter
            if not same_storage:
                raise ArchitectureCoverageError(
                    f"manifest alias {alias!r} does not resolve to the live model storage"
                )

    for layer in candidate_layers:
        entries = manifest.entries_for_layer(layer)
        if not entries:
            raise ArchitectureCoverageError(
                f"candidate layer {layer} has no validated output writers"
            )
        branch_kinds = {entry.branch_kind for entry in entries}
        if not {"attention", "ffn"}.issubset(branch_kinds):
            raise ArchitectureCoverageError(
                f"candidate layer {layer} lacks complete attention + FFN writers"
            )
        for entry in entries:
            if entry.layer_indices != (layer,):
                raise ArchitectureCoverageError(
                    f"entry {entry.qualified_name!r} shares storage across layers; "
                    "one-layer behavioral isolation is impossible"
                )


def _validate_snapshot(handle: SnapshotModelHandle) -> tuple[Mapping[str, torch.Tensor], str]:
    snapshot = getattr(handle, "_original_state", None)
    if snapshot is None:
        try:
            handle.snapshot()
        except Exception as exc:
            raise GabliterationValidationError(
                "could not create the complete ModelHandle snapshot required for trials"
            ) from exc
        snapshot = getattr(handle, "_original_state", None)
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise GabliterationValidationError(
            "ModelHandle snapshot is missing or incomplete"
        )
    current = handle.model.state_dict()
    if set(snapshot) != set(current):
        raise GabliterationValidationError(
            "ModelHandle snapshot keys do not exactly match the current model"
        )
    for name, value in snapshot.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.shape != current[name].shape
            or value.dtype != current[name].dtype
        ):
            raise GabliterationValidationError(
                f"ModelHandle snapshot tensor {name!r} is not an exact CPU peer"
            )
    snapshot_hash = state_dict_sha256(snapshot)
    if state_dict_sha256(current) != snapshot_hash:
        raise GabliterationValidationError(
            "current model is not byte-equivalent to the untouched ModelHandle snapshot"
        )
    return snapshot, snapshot_hash


def _restore_exact(handle: SnapshotModelHandle, expected_hash: str) -> None:
    try:
        handle.restore()
    except Exception as exc:
        raise GabliterationRollbackError("ModelHandle.restore() failed") from exc
    restored_hash = state_dict_sha256(handle.model.state_dict())
    if restored_hash != expected_hash:
        raise GabliterationRollbackError(
            "ModelHandle.restore() did not recreate the exact untouched state"
        )


def _entry_hashes(entries: Sequence[ProjectionManifestEntry]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (entry.qualified_name, tensor_sha256(entry.parameter))
        for entry in sorted(entries, key=lambda item: item.qualified_name)
    )


def _materialize_updates(
    manifest: ProjectionManifest,
    directions: torch.Tensor,
    layer_alphas: Mapping[int, float],
    ridge_lambda: float,
) -> dict[str, torch.Tensor]:
    updates: dict[str, torch.Tensor] = {}
    for layer in sorted(layer_alphas):
        entries = manifest.entries_for_layer(layer)
        if not entries:
            raise ArchitectureCoverageError(
                f"replay layer {layer} has no manifest entries"
            )
        for entry in entries:
            if entry.qualified_name in updates:
                raise ArchitectureCoverageError(
                    f"replay entry {entry.qualified_name!r} is scheduled more than once"
                )
            updates[entry.qualified_name] = ridge_subspace_update(
                entry.parameter.detach(),
                directions.to(device=entry.parameter.device),
                residual_axis=entry.residual_axis,
                alpha=float(layer_alphas[layer]),
                ridge_lambda=float(ridge_lambda),
            )
    return updates


def _apply_updates(
    manifest: ProjectionManifest,
    updates: Mapping[str, torch.Tensor],
) -> int:
    entries = {entry.qualified_name: entry for entry in manifest.entries}
    if not updates:
        raise GabliterationValidationError("an empty Gabliteration edit is not valid")
    unknown = set(updates) - set(entries)
    if unknown:
        raise ArchitectureCoverageError(f"updates contain unknown entries: {sorted(unknown)}")
    with torch.no_grad():
        for name in sorted(updates):
            entry = entries[name]
            value = updates[name]
            if value.shape != entry.parameter.shape or value.dtype != entry.parameter.dtype:
                raise ArchitectureCoverageError(
                    f"prepared update for {name!r} changed shape or dtype"
                )
            if not torch.isfinite(value).all():
                raise ArchitectureCoverageError(
                    f"prepared update for {name!r} contains NaN/Inf"
                )
        for name in sorted(updates):
            entry = entries[name]
            entry.parameter.data.copy_(updates[name])
    return len(updates)


def _response_statistics(
    responses: Sequence[str],
    markers: Sequence[str],
) -> tuple[int, float, str]:
    if not responses:
        raise GabliterationValidationError("behavioral evaluator returned no responses")
    if any(not isinstance(response, str) for response in responses):
        raise GabliterationValidationError(
            "behavioral evaluator responses must all be strings"
        )
    normalized_markers = tuple(marker.casefold() for marker in markers)
    normalized = tuple(response.casefold() for response in responses)
    refusal_count = sum(
        any(marker in response for marker in normalized_markers) for response in normalized
    )
    response_hash = _json_sha256(list(responses))
    return refusal_count, refusal_count / len(responses), response_hash


def _record_model_forward(
    evidence: list[int], _module: nn.Module, _args: tuple[Any, ...], _output: Any
) -> None:
    evidence[0] += 1


def _build_replay_plan(
    *,
    manifest: ProjectionManifest,
    snapshot: Mapping[str, torch.Tensor],
    source_layer: int,
    config: GabliterationSearchConfig,
    effective_layers: tuple[int, ...],
    layer_alphas: tuple[tuple[int, float], ...],
    directions: torch.Tensor,
    baseline_state_sha256: str,
) -> GabliterationReplayPlan:
    directions_cpu = directions.detach().to(device="cpu").contiguous()
    updates = _materialize_updates(
        manifest,
        directions_cpu,
        dict(layer_alphas),
        config.ridge_lambda,
    )
    entry_by_name = {entry.qualified_name: entry for entry in manifest.entries}
    baseline_parameter_hashes = tuple(
        (name, tensor_sha256(entry_by_name[name].parameter)) for name in sorted(updates)
    )
    expected_parameter_hashes = tuple(
        (name, tensor_sha256(updates[name])) for name in sorted(updates)
    )
    state_overrides: dict[str, torch.Tensor] = {}
    for name, value in updates.items():
        for alias in entry_by_name[name].aliases:
            state_overrides[alias] = value
    expected_state_hash = state_dict_sha256(snapshot, overrides=state_overrides)
    provisional = GabliterationReplayPlan(
        schema=GABLITERATION_REPLAY_SCHEMA,
        source_layer=source_layer,
        candidate_layers=config.candidate_layers,
        effective_layers=effective_layers,
        layer_alphas=layer_alphas,
        ridge_lambda=float(config.ridge_lambda),
        n_directions=config.n_directions,
        n_shuffles=config.n_shuffles,
        seed=config.seed,
        directions=directions_cpu,
        direction_sha256=tensor_sha256(directions_cpu),
        manifest_sha256=gabliteration_manifest_sha256(manifest),
        baseline_state_sha256=baseline_state_sha256,
        expected_state_sha256=expected_state_hash,
        baseline_parameter_sha256=baseline_parameter_hashes,
        expected_parameter_sha256=expected_parameter_hashes,
        plan_sha256="",
    )
    return GabliterationReplayPlan(
        **{
            **provisional.__dict__,
            "plan_sha256": _json_sha256(provisional._integrity_metadata()),
        }
    )


def apply_gabliteration_replay(
    *,
    handle: SnapshotModelHandle,
    manifest: ProjectionManifest,
    plan: GabliterationReplayPlan,
) -> int:
    """Apply a self-verifying replay plan once, rolling back any failed replay."""

    plan.validate_integrity()
    _validate_manifest(handle, manifest, plan.candidate_layers)
    if gabliteration_manifest_sha256(manifest) != plan.manifest_sha256:
        raise GabliterationValidationError(
            "live semantic manifest does not match the replay plan"
        )
    current_hash = state_dict_sha256(handle.model.state_dict())
    if current_hash != plan.baseline_state_sha256:
        raise GabliterationValidationError(
            "live model is not the byte-equivalent baseline required by replay"
        )
    entries = {entry.qualified_name: entry for entry in manifest.entries}
    for name, expected_hash in plan.baseline_parameter_sha256:
        entry = entries.get(name)
        if entry is None or tensor_sha256(entry.parameter) != expected_hash:
            raise GabliterationValidationError(
                f"baseline parameter {name!r} does not match the replay plan"
            )
    updates = _materialize_updates(
        manifest,
        plan.directions,
        dict(plan.layer_alphas),
        plan.ridge_lambda,
    )
    expected_update_hashes = dict(plan.expected_parameter_sha256)
    if set(updates) != set(expected_update_hashes):
        raise GabliterationValidationError(
            "materialized replay entries do not match the plan"
        )
    for name, value in updates.items():
        if tensor_sha256(value) != expected_update_hashes[name]:
            raise GabliterationValidationError(
                f"materialized replay value for {name!r} does not match the plan"
            )

    try:
        applied = _apply_updates(manifest, updates)
        final_hash = state_dict_sha256(handle.model.state_dict())
        if final_hash != plan.expected_state_sha256:
            raise GabliterationValidationError(
                "applied replay did not produce the plan's exact final state"
            )
        for name, expected_hash in plan.expected_parameter_sha256:
            if tensor_sha256(entries[name].parameter) != expected_hash:
                raise GabliterationValidationError(
                    f"applied parameter {name!r} does not match its expected hash"
                )
        return applied
    except Exception as exc:
        try:
            _restore_exact(handle, plan.baseline_state_sha256)
        except GabliterationRollbackError as rollback_exc:
            raise GabliterationRollbackError(
                "Gabliteration replay failed and exact rollback also failed"
            ) from rollback_exc
        if isinstance(exc, GabliterationError):
            raise
        raise GabliterationValidationError(
            "Gabliteration replay failed; untouched baseline was restored"
        ) from exc


def run_gabliteration_search(
    *,
    handle: SnapshotModelHandle,
    manifest: ProjectionManifest,
    harmful_batches: Sequence[HiddenStateBatch],
    harmless_batches: Sequence[HiddenStateBatch],
    evaluation_prompts: Sequence[Any],
    response_generator: Callable[[nn.Module, Sequence[Any]], Sequence[str]],
    config: GabliterationSearchConfig,
    apply_final: bool = True,
) -> GabliterationSearchResult:
    """Run the complete paper source/effective-layer search and exact replay.

    ``response_generator`` must generate one response for every evaluation
    prompt using the *model object supplied as its first argument*.  This keeps
    tokenization/chat-template policy in the owning pipeline while ensuring the
    evaluator sees the actual temporarily edited model.  Any state-dict change
    made by the callback itself is detected and rejected.

    On success, ``apply_final=True`` leaves the exact plan applied once to the
    live model.  ``False`` leaves the untouched baseline and returns the same
    replay artifact for deferred application.
    """

    if not isinstance(config, GabliterationSearchConfig):
        raise TypeError("config must be a GabliterationSearchConfig")
    config.validated()
    if not isinstance(handle.model, nn.Module):
        raise GabliterationValidationError("handle.model must be a torch module")
    if not callable(response_generator):
        raise TypeError("response_generator must be callable")
    prompts = tuple(evaluation_prompts)
    if not prompts:
        raise GabliterationValidationError("evaluation_prompts must not be empty")
    _validate_manifest(handle, manifest, config.candidate_layers)
    snapshot, baseline_hash = _validate_snapshot(handle)

    module_modes = tuple((module, module.training) for module in handle.model.modules())
    original_rng = _capture_rng_state()
    trial_rng = _capture_rng_state()
    trials: list[LayerEffectivenessTrial] = []
    final_applied = False
    try:
        handle.model.eval()
        harmful_by_layer = extract_last_token_hidden_states(
            handle.model, harmful_batches, config.candidate_layers
        )
        harmless_by_layer = extract_last_token_hidden_states(
            handle.model, harmless_batches, config.candidate_layers
        )
        if state_dict_sha256(handle.model.state_dict()) != baseline_hash:
            raise GabliterationValidationError(
                "source-layer forwards mutated the model state_dict"
            )
        source = mean_separation_source_layer(harmful_by_layer, harmless_by_layer)
        subspace = shuffle_stabilized_svd_subspace(
            harmful_by_layer[source.source_layer],
            harmless_by_layer[source.source_layer],
            n_directions=config.n_directions,
            n_shuffles=config.n_shuffles,
            seed=config.seed,
        )
        if subspace.directions.shape[1] != manifest.hidden_size:
            raise GabliterationValidationError(
                "extracted refusal directions do not match manifest hidden size"
            )

        for layer in config.candidate_layers:
            _restore_exact(handle, baseline_hash)
            _restore_rng_state(trial_rng)
            before_hash = state_dict_sha256(handle.model.state_dict())
            entries = manifest.entries_for_layer(layer)
            updates = _materialize_updates(
                manifest,
                subspace.directions,
                {layer: float(config.alpha_base)},
                config.ridge_lambda,
            )
            try:
                _apply_updates(manifest, updates)
                edited_hash = state_dict_sha256(handle.model.state_dict())
                edited_parameter_hashes = _entry_hashes(entries)
                forward_evidence = [0]
                hook = handle.model.register_forward_hook(
                    partial(_record_model_forward, forward_evidence)
                )
                try:
                    responses = tuple(response_generator(handle.model, prompts))
                finally:
                    hook.remove()
                if forward_evidence[0] == 0:
                    raise GabliterationValidationError(
                        "response_generator returned without forwarding the live trial model"
                    )
                if len(responses) != len(prompts):
                    raise GabliterationValidationError(
                        "response_generator must return exactly one response per prompt"
                    )
                if state_dict_sha256(handle.model.state_dict()) != edited_hash:
                    raise GabliterationValidationError(
                        "response_generator mutated the trial model state_dict"
                    )
                refusal_count, refusal_rate, response_hash = _response_statistics(
                    responses, config.refusal_markers
                )
            finally:
                _restore_exact(handle, baseline_hash)
            restored_hash = state_dict_sha256(handle.model.state_dict())
            effective = refusal_rate < float(config.effectiveness_threshold)
            trials.append(
                LayerEffectivenessTrial(
                    layer=layer,
                    refusal_count=refusal_count,
                    response_count=len(responses),
                    refusal_rate=refusal_rate,
                    effective=effective,
                    before_state_sha256=before_hash,
                    edited_state_sha256=edited_hash,
                    restored_state_sha256=restored_hash,
                    response_sha256=response_hash,
                    edited_parameter_sha256=edited_parameter_hashes,
                )
            )

        effective_layers = tuple(trial.layer for trial in trials if trial.effective)
        if not effective_layers:
            raise GabliterationValidationError(
                "no candidate satisfied the strict paper effectiveness threshold; "
                "refusing an empty named Gabliteration edit"
            )
        scales = paper_adaptive_layer_scales(
            effective_layers,
            alpha_base=config.alpha_base,
            beta=config.beta,
        )
        layer_alphas = tuple(zip(scales.layers, scales.alphas, strict=True))
        plan = _build_replay_plan(
            manifest=manifest,
            snapshot=snapshot,
            source_layer=source.source_layer,
            config=config,
            effective_layers=effective_layers,
            layer_alphas=layer_alphas,
            directions=subspace.directions,
            baseline_state_sha256=baseline_hash,
        )
        plan.validate_integrity()
        if apply_final:
            _restore_exact(handle, baseline_hash)
            apply_gabliteration_replay(handle=handle, manifest=manifest, plan=plan)
            final_applied = True
            final_hash = state_dict_sha256(handle.model.state_dict())
        else:
            _restore_exact(handle, baseline_hash)
            final_hash = baseline_hash
        return GabliterationSearchResult(
            replay_plan=plan,
            source_scores=source.scores,
            projector_eigenvalues=tuple(
                float(value) for value in subspace.projector_eigenvalues.tolist()
            ),
            shuffle_singular_values=tuple(
                tuple(float(value) for value in row)
                for row in subspace.shuffle_singular_values.tolist()
            ),
            trials=tuple(trials),
            applied=final_applied,
            final_state_sha256=final_hash,
        )
    except Exception:
        # Never leave a partially searched/failed named method in memory.
        current_hash = state_dict_sha256(handle.model.state_dict())
        if current_hash != baseline_hash:
            _restore_exact(handle, baseline_hash)
        raise
    finally:
        _restore_rng_state(original_rng)
        for module, training in module_modes:
            module.train(training)


__all__ = [
    "DEFAULT_REFUSAL_MARKERS",
    "GABLITERATION_IMPLEMENTATION_ASSUMPTIONS",
    "GABLITERATION_PAPER",
    "GABLITERATION_REPLAY_SCHEMA",
    "GABLITERATION_UPSTREAM",
    "GABLITERATION_UPSTREAM_COMMIT",
    "AdaptiveLayerScales",
    "GabliterationError",
    "GabliterationReplayPlan",
    "GabliterationRollbackError",
    "GabliterationSearchConfig",
    "GabliterationSearchResult",
    "GabliterationValidationError",
    "HiddenStateBatch",
    "LayerEffectivenessTrial",
    "MeanSeparationResult",
    "ProjectorAverageResult",
    "ShuffleStabilizedSubspace",
    "apply_gabliteration_replay",
    "average_projector_subspace",
    "extract_last_token_hidden_states",
    "gabliteration_manifest_sha256",
    "mean_separation_source_layer",
    "paper_adaptive_layer_scales",
    "ridge_subspace_update",
    "run_gabliteration_search",
    "shuffle_stabilized_svd_subspace",
    "state_dict_sha256",
    "tensor_sha256",
]
