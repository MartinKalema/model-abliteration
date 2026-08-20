"""Low-rank linear concept erasers.

The erasers in this module use the column-vector convention

    P = I - L R,

where ``L`` has shape ``(d, k)`` and ``R`` has shape ``(k, d)``.  Model
activations are normally stored as row vectors, so :meth:`ResidualEraser.apply`
evaluates the equivalent affine map without materialising the dense ``d x d``
projector.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch


@dataclass(frozen=True)
class ResidualEraser:
    """A low-rank affine eraser acting on the last axis of a tensor.

    ``proj_left`` and ``proj_right`` define the linear part ``P = I - LR``.
    If ``center`` is present, row activations are transformed as

    ``x -> center + (x - center) @ P.T``.

    ``display_directions`` is deliberately diagnostic only.  For an oblique
    eraser it is not sufficient to reconstruct or apply the eraser.
    """

    proj_left: torch.Tensor
    proj_right: torch.Tensor
    center: torch.Tensor | None = None
    display_directions: torch.Tensor | None = None
    method: str = "linear"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.proj_left.ndim != 2:
            raise ValueError("proj_left must have shape (hidden_dim, rank)")
        if self.proj_right.ndim != 2:
            raise ValueError("proj_right must have shape (rank, hidden_dim)")
        if self.proj_left.shape[1] != self.proj_right.shape[0]:
            raise ValueError("proj_left and proj_right have incompatible ranks")
        if self.proj_left.shape[0] != self.proj_right.shape[1]:
            raise ValueError("proj_left and proj_right have incompatible hidden dimensions")
        if self.center is not None and (
            self.center.ndim != 1 or self.center.shape[0] != self.hidden_dim
        ):
            raise ValueError("center must have shape (hidden_dim,)")
        if self.display_directions is not None:
            if self.display_directions.ndim != 2:
                raise ValueError("display_directions must have shape (n_directions, hidden_dim)")
            if self.display_directions.shape[1] != self.hidden_dim:
                raise ValueError("display_directions has the wrong hidden dimension")

    @property
    def hidden_dim(self) -> int:
        return self.proj_left.shape[0]

    @property
    def rank(self) -> int:
        return self.proj_left.shape[1]

    @property
    def projector(self) -> torch.Tensor:
        """Return the dense column-vector linear map ``I - LR``."""
        identity = torch.eye(
            self.hidden_dim,
            dtype=self.proj_left.dtype,
            device=self.proj_left.device,
        )
        right = self.proj_right.to(
            dtype=self.proj_left.dtype,
            device=self.proj_left.device,
        )
        return identity - self.proj_left @ right

    @property
    def affine_bias(self) -> torch.Tensor:
        """Return ``b`` for the equivalent column map ``x -> Px + b``."""
        if self.center is None:
            return self.proj_left.new_zeros(self.hidden_dim)
        center = self.center.to(
            dtype=self.proj_left.dtype,
            device=self.proj_left.device,
        )
        right = self.proj_right.to(
            dtype=self.proj_left.dtype,
            device=self.proj_left.device,
        )
        return self.proj_left @ (right @ center)

    def apply(self, activations: torch.Tensor) -> torch.Tensor:
        """Apply the affine eraser to row activations along their last axis.

        The returned tensor keeps the activation tensor's dtype and device.
        Factors are converted as needed, which makes extraction in float32
        compatible with float16/bfloat16 model activations.
        """
        if activations.ndim == 0 or activations.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"activations must end in hidden dimension {self.hidden_dim}, "
                f"got shape {tuple(activations.shape)}"
            )
        if not (activations.is_floating_point() or activations.is_complex()):
            raise TypeError("activations must have a floating-point or complex dtype")

        left = self.proj_left.to(device=activations.device, dtype=activations.dtype)
        right = self.proj_right.to(device=activations.device, dtype=activations.dtype)
        if self.center is None:
            centered = activations
        else:
            center = self.center.to(device=activations.device, dtype=activations.dtype)
            centered = activations - center
        removed = (centered @ right.transpose(-2, -1)) @ left.transpose(-2, -1)
        return activations - removed

    __call__ = apply

    def to(self, *args: Any, **kwargs: Any) -> ResidualEraser:
        """Return a copy with all tensor fields moved like ``Tensor.to``."""
        return replace(
            self,
            proj_left=self.proj_left.to(*args, **kwargs),
            proj_right=self.proj_right.to(*args, **kwargs),
            center=None if self.center is None else self.center.to(*args, **kwargs),
            display_directions=(
                None
                if self.display_directions is None
                else self.display_directions.to(*args, **kwargs)
            ),
        )
