"""Differentiable residual-stream interventions for model-forward analysis.

The helpers in this module deliberately use ordinary PyTorch hooks rather than
cached activations.  Consequently, losses computed from an intervened forward
pass retain a gradient path to the intervention direction.  Model weights are
never changed.

Directional ablation follows the RDO/Arditi residual-stream operation

``x <- x - (x . r_hat) r_hat``

at every decoder layer and token position.  Hooks cover the layer input,
attention/MLP outputs when those submodules are discoverable, and the layer
output.  The final layer-output hook also gives a safe architecture-independent
boundary when a decoder layer does not expose conventional submodule names.

Activation addition follows paper RDO and adds ``alpha * r_hat`` to every token
at the input of one selected decoder layer.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import copy
from typing import Any, Literal

import torch
from torch import nn

InterventionMode = Literal["ablate", "add"]


class InterventionError(RuntimeError):
    """Raised when an intervention cannot be applied completely and safely."""


def _validate_direction(direction: torch.Tensor) -> None:
    if not isinstance(direction, torch.Tensor):
        raise TypeError("direction must be a torch.Tensor")
    if direction.ndim != 1 or direction.numel() == 0:
        raise ValueError("direction must have shape [hidden_size]")
    if not direction.is_floating_point():
        raise TypeError("direction must use a floating-point dtype")
    detached = direction.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError("direction contains NaN or infinite values")
    norm = torch.linalg.vector_norm(detached.float())
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise ValueError("direction must have a finite non-zero norm")


def unit_direction_for(direction: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """Return a differentiable unit direction matching an activation tensor."""

    _validate_direction(direction)
    if not isinstance(activation, torch.Tensor):
        raise TypeError("activation must be a torch.Tensor")
    if not activation.is_floating_point():
        raise InterventionError("intervention activation must be floating-point")
    if activation.ndim == 0 or activation.shape[-1] != direction.numel():
        raise InterventionError(
            "activation hidden dimension does not match intervention direction "
            f"({activation.shape[-1] if activation.ndim else 'scalar'} != {direction.numel()})"
        )
    if direction.device != activation.device:
        raise InterventionError(
            "direction and activation must be on one device; sharded/offloaded "
            "intervention paths are unsupported"
        )

    work = direction.to(dtype=activation.dtype)
    return work / torch.linalg.vector_norm(work).clamp_min(torch.finfo(work.dtype).eps)


def ablate_direction(activation: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Differentiably project ``activation`` away from ``direction``."""

    unit = unit_direction_for(direction, activation)
    coefficients = torch.tensordot(activation, unit, dims=([-1], [0]))
    return activation - coefficients.unsqueeze(-1) * unit


def add_direction(
    activation: torch.Tensor,
    direction: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Differentiably add a scaled unit direction at every token position."""

    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise TypeError("scale must be a finite number")
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be a finite number")
    unit = unit_direction_for(direction, activation)
    return activation + float(scale) * unit


def _replace_primary_tensor(value: object, transform: Any, *, location: str) -> object:
    """Apply ``transform`` to a module output without discarding tuple metadata."""

    if isinstance(value, torch.Tensor):
        return transform(value)

    if isinstance(value, tuple):
        if not value or not isinstance(value[0], torch.Tensor):
            raise InterventionError(f"{location} tuple output has no leading activation tensor")
        transformed = transform(value[0])
        if hasattr(value, "_fields") and hasattr(value, "_replace"):
            return value._replace(**{value._fields[0]: transformed})
        return (transformed, *value[1:])

    if isinstance(value, list):
        if not value or not isinstance(value[0], torch.Tensor):
            raise InterventionError(f"{location} list output has no leading activation tensor")
        return [transform(value[0]), *value[1:]]

    if isinstance(value, Mapping):
        for key in ("hidden_states", "last_hidden_state"):
            if key in value and isinstance(value[key], torch.Tensor):
                transformed = transform(value[key])
                try:
                    updated = copy(value)
                    updated[key] = transformed
                    if hasattr(updated, key):
                        setattr(updated, key, transformed)
                    return updated
                except (AttributeError, TypeError, ValueError):
                    return {**value, key: transformed}
        raise InterventionError(f"{location} mapping output has no recognized activation tensor")

    raise InterventionError(
        f"{location} output type {type(value).__name__!r} does not expose an activation tensor"
    )


def _validate_layers(decoder_layers: Sequence[nn.Module]) -> tuple[nn.Module, ...]:
    if isinstance(decoder_layers, (str, bytes)):
        raise TypeError("decoder_layers must be a sequence of torch modules")
    layers = tuple(decoder_layers)
    if not layers:
        raise ValueError("at least one decoder layer is required")
    if not all(isinstance(layer, nn.Module) for layer in layers):
        raise TypeError("decoder_layers must contain only torch modules")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("decoder_layers must not contain duplicate modules")
    return layers


class DirectionalIntervention:
    """Context-managed, fail-safe differentiable intervention hook set.

    Call :meth:`assert_applied` after a forward/generation call.  It verifies
    that every requested decoder layer was actually reached; this prevents a
    silently partial intervention when an incorrect layer list is supplied.
    Hooks are removed even when the model forward or this assertion raises.
    """

    _ATTENTION_NAMES = ("self_attn", "attention", "attn")
    _MLP_NAMES = ("mlp", "feed_forward", "ffn")

    def __init__(
        self,
        decoder_layers: Sequence[nn.Module],
        direction: torch.Tensor,
        *,
        mode: InterventionMode,
        addition_layer: int | None = None,
        addition_scale: float = 1.0,
    ) -> None:
        self.layers = _validate_layers(decoder_layers)
        _validate_direction(direction)
        self.direction = direction
        if mode not in {"ablate", "add"}:
            raise ValueError("mode must be 'ablate' or 'add'")
        self.mode = mode

        if mode == "add":
            if not isinstance(addition_layer, int) or isinstance(addition_layer, bool):
                raise TypeError("addition_layer must be an integer for addition")
            if not 0 <= addition_layer < len(self.layers):
                raise ValueError(
                    f"addition_layer must be in [0, {len(self.layers) - 1}]"
                )
        elif addition_layer is not None:
            raise ValueError("addition_layer is only valid for addition mode")
        if isinstance(addition_scale, bool) or not isinstance(addition_scale, (int, float)):
            raise TypeError("addition_scale must be a finite number")
        if not math.isfinite(float(addition_scale)):
            raise ValueError("addition_scale must be a finite number")

        self.addition_layer = addition_layer
        self.addition_scale = float(addition_scale)
        self._handles: list[Any] = []
        self._expected: set[str] = set()
        self._counts: dict[str, int] = {}
        self._active = False

    @property
    def applied_counts(self) -> dict[str, int]:
        """Return a copy of per-hook application counts."""

        return dict(self._counts)

    def _mark(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1

    def _pre_hook(self, name: str, transform: Any):
        def hook(
            _module: nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            self._mark(name)
            if args and isinstance(args[0], torch.Tensor):
                return (transform(args[0]), *args[1:]), kwargs
            if isinstance(kwargs.get("hidden_states"), torch.Tensor):
                updated = dict(kwargs)
                updated["hidden_states"] = transform(kwargs["hidden_states"])
                return args, updated
            raise InterventionError(
                f"{name} input has no positional or 'hidden_states' activation tensor"
            )

        return hook

    def _output_hook(self, name: str, transform: Any):
        def hook(_module: nn.Module, _args: tuple[object, ...], output: object) -> object:
            self._mark(name)
            return _replace_primary_tensor(output, transform, location=name)

        return hook

    @staticmethod
    def _first_named_module(layer: nn.Module, names: Sequence[str]) -> nn.Module | None:
        for name in names:
            module = getattr(layer, name, None)
            if isinstance(module, nn.Module):
                return module
        return None

    def _register_pre(self, layer: nn.Module, name: str, transform: Any) -> None:
        self._handles.append(
            layer.register_forward_pre_hook(self._pre_hook(name, transform), with_kwargs=True)
        )
        self._expected.add(name)

    def _register_output(
        self,
        module: nn.Module,
        name: str,
        transform: Any,
        *,
        required: bool,
    ) -> None:
        self._handles.append(module.register_forward_hook(self._output_hook(name, transform)))
        if required:
            self._expected.add(name)

    def __enter__(self) -> DirectionalIntervention:  # noqa: PYI034 - Python 3.10 support
        if self._active:
            raise RuntimeError("directional intervention is already active")
        self._active = True
        try:
            if self.mode == "ablate":
                transform = lambda value: ablate_direction(value, self.direction)
                for index, layer in enumerate(self.layers):
                    self._register_pre(layer, f"layer[{index}].input", transform)

                    attention = self._first_named_module(layer, self._ATTENTION_NAMES)
                    if attention is not None:
                        self._register_output(
                            attention,
                            f"layer[{index}].attention.output",
                            transform,
                            required=False,
                        )
                    mlp = self._first_named_module(layer, self._MLP_NAMES)
                    if mlp is not None:
                        self._register_output(
                            mlp,
                            f"layer[{index}].mlp.output",
                            transform,
                            required=False,
                        )

                    # Required final boundary: catches unconventional layers and
                    # verifies that every requested layer participated.
                    self._register_output(
                        layer,
                        f"layer[{index}].output",
                        transform,
                        required=True,
                    )
            else:
                assert self.addition_layer is not None
                transform = lambda value: add_direction(
                    value,
                    self.direction,
                    scale=self.addition_scale,
                )
                layer = self.layers[self.addition_layer]
                self._register_pre(
                    layer,
                    f"layer[{self.addition_layer}].input",
                    transform,
                )
        except Exception:
            self.remove()
            raise
        return self

    def assert_applied(self) -> None:
        """Fail if any requested decoder-layer hook was never reached."""

        missing = sorted(name for name in self._expected if self._counts.get(name, 0) == 0)
        if missing:
            raise InterventionError(
                "model forward did not reach all requested intervention points: "
                + ", ".join(missing)
            )

    def remove(self) -> None:
        """Remove all installed hooks; safe to call repeatedly."""

        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._active = False

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.remove()


def run_with_directional_ablation(
    model: nn.Module,
    decoder_layers: Sequence[nn.Module],
    direction: torch.Tensor,
    *model_args: object,
    **model_kwargs: object,
) -> object:
    """Run one real model forward with all-layer directional ablation."""

    with DirectionalIntervention(decoder_layers, direction, mode="ablate") as hooks:
        output = model(*model_args, **model_kwargs)
        hooks.assert_applied()
        return output


def run_with_directional_addition(
    model: nn.Module,
    decoder_layers: Sequence[nn.Module],
    direction: torch.Tensor,
    *,
    addition_layer: int,
    addition_scale: float,
    model_args: Sequence[object] = (),
    model_kwargs: Mapping[str, object] | None = None,
) -> object:
    """Run one real model forward with paper-style single-layer addition."""

    kwargs = dict(model_kwargs or {})
    with DirectionalIntervention(
        decoder_layers,
        direction,
        mode="add",
        addition_layer=addition_layer,
        addition_scale=addition_scale,
    ) as hooks:
        output = model(*tuple(model_args), **kwargs)
        hooks.assert_applied()
        return output
