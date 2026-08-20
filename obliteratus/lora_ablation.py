"""LoRA-based reversible ablation mode.

Instead of permanent in-place weight surgery, applies ablation via rank-1
LoRA adapters.  This provides:

1. **Reversibility**: LoRA adapters can be removed to restore original model
2. **Composability**: Multiple ablation adapters can be stacked/blended
3. **Auditability**: Exact factors are saved alongside the edited checkpoint

Inspired by Heretic (p-e-w, 2025) which pioneered LoRA-mediated ablation.
OBLITERATUS extends this with:
- Multi-direction rank-k adapters (not just rank-1)
- Manifest-complete hybrid and MoE tensor discovery
- Explicit residual-axis factorization for nested and fused parameters
- Fail-closed rejection of semantics that are not exactly factorable as LoRA

The manifest declares the residual axis explicitly. Moving that axis last and
flattening all leading axes gives a matrix ``M`` for every supported layout:

    In-place:  M' = M - scale * (M @ d) @ d^T
    LoRA:      M' = M + B @ A
               B = -scale * (M @ d), A = d^T

This covers ordinary Linear/Conv1D weights as well as nested and fused expert
parameters without inferring orientation from coincidental dimensions.

References:
    - Hu et al. (2022): LoRA: Low-Rank Adaptation of Large Language Models
    - Heretic (p-e-w, 2025): LoRA-mediated directional ablation
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from obliteratus.architecture_manifest import (
    ArchitectureCoverageError,
    ProjectionManifest,
    ProjectionManifestEntry,
)

if TYPE_CHECKING:
    from obliteratus.abliterate import AbliterationPipeline

logger = logging.getLogger(__name__)


LoRAAdapters = dict[str, tuple[torch.Tensor, torch.Tensor]]


def _manifest_entries_owned_by_strong_layers(
    pipeline: AbliterationPipeline,
    manifest: ProjectionManifest,
) -> tuple[tuple[ProjectionManifestEntry, int], ...]:
    """Return each manifest tensor once, assigned to its first strong owner."""
    try:
        active_manifest = pipeline._current_projection_manifest()
    except (AttributeError, ArchitectureCoverageError) as exc:
        raise ArchitectureCoverageError(
            "LoRA projection requires the pipeline's active validated manifest"
        ) from exc
    if manifest is not active_manifest:
        raise ArchitectureCoverageError(
            "LoRA projection refused a stale or foreign manifest; use the active "
            "validated manifest. No model weights were modified."
        )
    if manifest.target != pipeline.projection_target:
        raise ArchitectureCoverageError(
            f"LoRA manifest target {manifest.target!r} does not match active "
            f"projection target {pipeline.projection_target!r}"
        )

    strong_layers = set(pipeline._strong_layers)
    planned: list[tuple[ProjectionManifestEntry, int]] = []
    seen_storage: set[str] = set()
    seen_names: set[str] = set()
    covered_layers: set[int] = set()
    for entry in manifest.entries:
        owners = strong_layers.intersection(entry.layer_indices)
        if not owners:
            continue
        covered_layers.update(owners)
        if entry.storage_identity in seen_storage:
            raise ArchitectureCoverageError(
                f"LoRA manifest schedules storage {entry.storage_identity} more than once"
            )
        if entry.qualified_name in seen_names:
            raise ArchitectureCoverageError(
                f"LoRA manifest contains duplicate target {entry.qualified_name!r}"
            )
        seen_storage.add(entry.storage_identity)
        seen_names.add(entry.qualified_name)
        planned.append((entry, min(owners)))

    missing_layers = strong_layers - covered_layers
    if missing_layers:
        raise ArchitectureCoverageError(
            "LoRA projection manifest has no target entries for selected strong "
            f"layers {sorted(missing_layers)}. No model weights were modified."
        )
    if not planned:
        raise ArchitectureCoverageError(
            "LoRA projection manifest has no entries owned by the selected strong layers"
        )
    return tuple(planned)


def _resolve_entry_object(entry: ProjectionManifestEntry):
    obj = entry.owner
    try:
        for part in entry.attribute_path.split("."):
            obj = getattr(obj, part)
    except AttributeError as exc:
        raise ArchitectureCoverageError(
            f"LoRA manifest target {entry.qualified_name!r} no longer resolves"
        ) from exc
    return obj


def _same_tensor_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left is right:
        return True
    try:
        metadata_matches = (
            left.device == right.device
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and left.stride() == right.stride()
            and left.storage_offset() == right.storage_offset()
        )
    except RuntimeError:
        return False
    if not metadata_matches:
        return False
    try:
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return left.data_ptr() == right.data_ptr()


def _validate_exact_lora_plan(
    pipeline: AbliterationPipeline,
    manifest: ProjectionManifest,
    rank: int,
) -> tuple[tuple[ProjectionManifestEntry, int], ...]:
    """Reject any requested behavior not exactly representable by this plan.

    Validation is deliberately complete before factor computation or mutation;
    LoRA mode must never quietly omit a configured secondary edit.
    """
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")

    incompatible_flags = {
        "norm_preserve": "per-logical-row norm restoration is generally full-rank",
        "project_biases": "bias deltas are not weight LoRA adapters",
        "attention_head_surgery": "selective head surgery is a secondary mutation",
        "safety_neuron_masking": "neuron masking is a secondary mutation",
        "use_sae_features": "SAE directions are secondary manifest mutations",
        "per_expert_directions": "expert-specific directions need distinct expert adapters",
        "invert_refusal": "expert-aware inversion uses non-uniform reflection semantics",
        "project_lm_head": "the output head is outside the primary layer manifest",
        "project_embeddings": "input embeddings are outside the primary layer manifest",
        "expert_transplant": "expert transplantation is not a low-rank projection",
        "activation_steering": "runtime steering hooks are not persisted LoRA edits",
        "spectral_cascade": "spectral cascade changes layer strengths after this branch",
        "true_iterative_refinement": "iterative re-probing requires interleaved mutations",
    }
    enabled = [
        f"{name} ({reason})"
        for name, reason in incompatible_flags.items()
        if bool(getattr(pipeline, name, False))
    ]
    refinement_passes = getattr(pipeline, "refinement_passes", 1)
    if (
        not isinstance(refinement_passes, int)
        or isinstance(refinement_passes, bool)
        or refinement_passes != 1
    ):
        enabled.append(
            f"refinement_passes={refinement_passes!r} (LoRA mode requires the integer 1)"
        )

    handle = getattr(pipeline, "handle", None)
    config = getattr(handle, "config", None)
    if (
        getattr(pipeline, "quantization", None) is not None
        or getattr(config, "quantization_config", None) is not None
    ):
        enabled.append("quantization (exact merged-adapter requantization is unsupported)")
    if enabled:
        raise ArchitectureCoverageError(
            "Exact LoRA manifest projection is incompatible with: "
            + "; ".join(enabled)
            + ". No model weights were modified."
        )

    planned = _manifest_entries_owned_by_strong_layers(pipeline, manifest)
    supported_kinds = {"module_weight", "parameter_axis"}
    quantized_module_names = {
        "QuantLinear",
        "WQLinear",
        "WQLinear_GEMM",
        "WQLinear_GEMV",
    }
    for entry, owner in planned:
        if entry.projection_kind not in supported_kinds:
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} uses unsupported projection "
                f"kind {entry.projection_kind!r}; no weights were modified"
            )
        parameter = entry.parameter
        if not isinstance(parameter, torch.Tensor):
            raise ArchitectureCoverageError(f"LoRA target {entry.qualified_name!r} is not a tensor")
        if parameter.ndim < 2:
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} must have at least two dimensions"
            )
        if parameter.device.type == "meta":
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} is still on the meta device"
            )
        if parameter.layout != torch.strided or not parameter.is_floating_point():
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} must be a dense floating tensor; "
                f"got layout={parameter.layout}, dtype={parameter.dtype}"
            )
        target_obj = _resolve_entry_object(entry)
        live_parameter = (
            getattr(target_obj, "weight", None)
            if entry.projection_kind == "module_weight"
            else target_obj
        )
        if not isinstance(live_parameter, torch.Tensor) or not _same_tensor_storage(
            live_parameter,
            parameter,
        ):
            raise ArchitectureCoverageError(
                f"LoRA manifest target {entry.qualified_name!r} no longer references "
                "the live model tensor. No model weights were modified."
            )
        if tuple(parameter.shape) != entry.shape or str(parameter.dtype) != entry.dtype:
            raise ArchitectureCoverageError(
                f"LoRA manifest metadata for {entry.qualified_name!r} is stale: "
                f"manifest shape/dtype={entry.shape}/{entry.dtype}, live="
                f"{tuple(parameter.shape)}/{parameter.dtype}. No model weights were modified."
            )
        if (
            pipeline._is_quantized_param(parameter)
            or target_obj.__class__.__name__ in quantized_module_names
        ):
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} is quantized; exact merged "
                "adapter requantization is unsupported. No weights were modified."
            )
        if (
            not isinstance(entry.residual_axis, int)
            or isinstance(entry.residual_axis, bool)
            or not 0 <= entry.residual_axis < parameter.ndim
        ):
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} has invalid residual axis "
                f"{entry.residual_axis!r} for shape {tuple(parameter.shape)}"
            )
        if entry.expert_axis is not None and (
            not isinstance(entry.expert_axis, int)
            or isinstance(entry.expert_axis, bool)
            or not 0 <= entry.expert_axis < parameter.ndim
            or entry.expert_axis == entry.residual_axis
        ):
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} has invalid expert axis "
                f"{entry.expert_axis!r} for residual axis {entry.residual_axis}"
            )
        axis = entry.residual_axis
        if parameter.shape[axis] != manifest.hidden_size:
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} shape {tuple(parameter.shape)} "
                f"does not match declared residual axis {entry.residual_axis} and "
                f"hidden size {manifest.hidden_size}"
            )
        if not torch.isfinite(parameter).all():
            raise ArchitectureCoverageError(
                f"LoRA target {entry.qualified_name!r} contains NaN/Inf values"
            )
        subspace = pipeline.refusal_subspaces.get(owner)
        if not isinstance(subspace, torch.Tensor) or subspace.ndim != 2:
            raise ArchitectureCoverageError(
                f"Strong layer {owner} has no two-dimensional refusal subspace"
            )
        if subspace.shape[0] < 1 or subspace.shape[1] != manifest.hidden_size:
            raise ArchitectureCoverageError(
                f"Strong layer {owner} refusal subspace has shape "
                f"{tuple(subspace.shape)}, expected [directions, {manifest.hidden_size}]"
            )
        if rank < subspace.shape[0]:
            raise ArchitectureCoverageError(
                f"LoRA rank {rank} cannot exactly represent all "
                f"{subspace.shape[0]} refusal directions at strong layer {owner}; "
                "increase lora_rank. No weights were modified."
            )
        if not torch.isfinite(subspace).all():
            raise ArchitectureCoverageError(
                f"Strong layer {owner} refusal subspace contains NaN/Inf values"
            )
    return planned


def validate_lora_manifest_plan(
    pipeline: AbliterationPipeline,
    manifest: ProjectionManifest,
    rank: int,
) -> int:
    """Validate an exact manifest-backed LoRA plan without mutating the model."""
    return len(_validate_exact_lora_plan(pipeline, manifest, rank))


def _layer_regularization(
    pipeline: AbliterationPipeline,
    layer_idx: int,
    bayesian_regularizations: dict[int, float],
) -> float:
    if layer_idx in bayesian_regularizations:
        regularization = bayesian_regularizations[layer_idx]
    else:
        regularization = pipeline.regularization
        if pipeline.layer_adaptive_strength and layer_idx in pipeline._layer_excise_weights:
            weight = pipeline._layer_excise_weights[layer_idx]
            regularization += (1.0 - weight) * (1.0 - regularization) * 0.15
    if pipeline.float_layer_interpolation and layer_idx in pipeline._float_layer_weights:
        float_weight = pipeline._float_layer_weights[layer_idx]
        regularization += (1.0 - float_weight) * (1.0 - regularization) * 0.3
    if not math.isfinite(regularization):
        raise ArchitectureCoverageError(f"LoRA regularization for layer {layer_idx} is not finite")
    return regularization


def _entry_regularization(
    pipeline: AbliterationPipeline,
    entry: ProjectionManifestEntry,
    layer_regularization: float,
) -> float:
    if entry.branch_kind == "attention":
        component_scale = getattr(pipeline, "_bayesian_attn_scale", None)
    elif entry.branch_kind == "ffn":
        component_scale = getattr(pipeline, "_bayesian_mlp_scale", None)
    else:
        raise ArchitectureCoverageError(
            f"LoRA target {entry.qualified_name!r} has unknown branch kind {entry.branch_kind!r}"
        )
    if component_scale is not None and component_scale < 1.0:
        return 1.0 - (1.0 - layer_regularization) * component_scale
    return layer_regularization


def _factor_manifest_update(
    pipeline: AbliterationPipeline,
    entry: ProjectionManifestEntry,
    subspace: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor the exact sequential residual-axis projection as ``B @ A``."""
    parameter = entry.parameter
    axis = entry.residual_axis % parameter.ndim
    moved = parameter.detach().movedim(axis, -1)
    moved_shape = moved.shape
    working = moved.reshape(-1, moved_shape[-1]).clone()
    scale = 1.0 - regularization
    b_parts: list[torch.Tensor] = []
    a_parts: list[torch.Tensor] = []

    moved_expert_axis = None
    if entry.expert_axis is not None:
        expert_axis = entry.expert_axis % parameter.ndim
        moved_expert_axis = expert_axis if expert_axis < axis else expert_axis - 1

    for direction in subspace:
        d = direction.to(device=working.device, dtype=working.dtype).reshape(-1)
        coefficients = torch.tensordot(
            working.reshape(moved_shape),
            d,
            dims=([-1], [0]),
        )
        if moved_expert_axis is None:
            coefficients = pipeline._select_projection_coefficients(
                coefficients,
                pipeline.projection_row_fraction,
            )
        else:
            # The ordinary manifest editor projects each packed expert
            # independently, so selective-row top-k masks must also be chosen
            # independently rather than across the flattened expert tensor.
            selected = torch.zeros_like(coefficients)
            for expert_index in range(coefficients.shape[moved_expert_axis]):
                expert_coefficients = coefficients.select(
                    moved_expert_axis,
                    expert_index,
                )
                selected.select(moved_expert_axis, expert_index).copy_(
                    pipeline._select_projection_coefficients(
                        expert_coefficients,
                        pipeline.projection_row_fraction,
                    )
                )
            coefficients = selected
        b_part = -scale * coefficients.reshape(-1, 1)
        a_part = d.reshape(1, -1)
        if not torch.isfinite(b_part).all() or not torch.isfinite(a_part).all():
            raise ArchitectureCoverageError(
                f"LoRA factors for {entry.qualified_name!r} contain NaN/Inf values"
            )
        b_parts.append(b_part)
        a_parts.append(a_part)
        working.add_(b_part @ a_part)

    return torch.cat(b_parts, dim=1), torch.cat(a_parts, dim=0)


def compute_lora_adapters(
    pipeline: AbliterationPipeline,
    rank: int = 1,
    *,
    manifest: ProjectionManifest | None = None,
    bayesian_regularizations: dict[int, float] | None = None,
) -> LoRAAdapters:
    """Compute LoRA adapter pairs (B, A) for refusal direction ablation.

    For each target weight matrix W with refusal direction d:
        A = d^T @ W    (rank-1: shape (1, in_features) or (rank, in_features))
        B = -scale * d  (rank-1: shape (out_features, 1) or (out_features, rank))

    So that  W + B @ A  ≈  W - scale * (d @ d^T) @ W

    Args:
        pipeline: Initialized pipeline (post-DISTILL, pre-EXCISE).
        rank: LoRA rank (1 = rank-1 ablation, >1 = multi-direction).

    Returns:
        Dict mapping each manifest parameter name to its exact ``(B, A)`` pair.
    """
    if not pipeline.handle:
        raise RuntimeError("Cannot compute LoRA adapters without a loaded model")
    if not pipeline._strong_layers:
        raise ArchitectureCoverageError(
            "LoRA projection requires at least one selected strong layer; "
            "no model weights were modified"
        )
    manifest = manifest or pipeline._current_projection_manifest()
    planned = _validate_exact_lora_plan(pipeline, manifest, rank)
    bayesian_regularizations = bayesian_regularizations or {}
    adapters: LoRAAdapters = {}
    for entry, owner in planned:
        layer_reg = _layer_regularization(
            pipeline,
            owner,
            bayesian_regularizations,
        )
        entry_reg = _entry_regularization(pipeline, entry, layer_reg)
        adapters[entry.qualified_name] = _factor_manifest_update(
            pipeline,
            entry,
            pipeline.refusal_subspaces[owner],
            entry_reg,
        )

    expected_names = {entry.qualified_name for entry, _ in planned}
    if set(adapters) != expected_names:
        missing = expected_names - set(adapters)
        extra = set(adapters) - expected_names
        raise ArchitectureCoverageError(
            "LoRA adapter computation did not exactly cover the validated manifest "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    pipeline.log(
        f"Computed {len(adapters)} manifest-complete LoRA adapter pairs (maximum rank={rank})"
    )
    return adapters


def _merge_manifest_delta(
    entry: ProjectionManifestEntry,
    dense_delta: torch.Tensor,
) -> None:
    """Merge one validated flattened residual-axis delta into its live tensor."""
    target = entry.parameter.data.movedim(entry.residual_axis, -1)
    target.add_(dense_delta.reshape(target.shape))


def apply_lora_adapters(
    pipeline: AbliterationPipeline,
    adapters: LoRAAdapters,
    *,
    manifest: ProjectionManifest | None = None,
) -> int:
    """Apply pre-computed LoRA adapters by modifying weights in-place.

    This is equivalent to merging the LoRA into the base model.
    The adapters dict is stored in pipeline._lora_adapters for potential
    later unmerging.
    """
    if not pipeline.handle:
        raise RuntimeError("Cannot apply LoRA adapters without a loaded model")

    manifest = manifest or pipeline._current_projection_manifest()
    planned = _validate_exact_lora_plan(pipeline, manifest, pipeline.lora_rank)
    expected = {entry.qualified_name: (entry, owner) for entry, owner in planned}
    if set(adapters) != set(expected):
        missing = set(expected) - set(adapters)
        extra = set(adapters) - set(expected)
        raise ArchitectureCoverageError(
            "LoRA apply plan does not exactly match the validated manifest "
            f"(missing={sorted(missing)}, extra={sorted(extra)}). "
            "No model weights were modified."
        )

    # Validate every factor and materialize each device/dtype-correct low-rank
    # product once before the first mutation. This catches factor failures
    # before taking the transactional snapshots used for exact rollback.
    for name, (entry, owner) in expected.items():
        lora_B, lora_A = adapters[name]
        if not isinstance(lora_B, torch.Tensor) or not isinstance(lora_A, torch.Tensor):
            raise ArchitectureCoverageError(
                f"LoRA factors for {name!r} must be tensors; no weights were modified"
            )
        parameter = entry.parameter
        axis = entry.residual_axis
        rows = parameter.numel() // parameter.shape[axis]
        expected_b_shape = (rows, lora_A.shape[0] if lora_A.ndim == 2 else -1)
        if (
            lora_B.ndim != 2
            or lora_A.ndim != 2
            or lora_B.shape[1] != lora_A.shape[0]
            or lora_B.shape[0] != rows
            or lora_A.shape[1] != parameter.shape[axis]
            or lora_A.shape[0] != pipeline.refusal_subspaces[owner].shape[0]
        ):
            raise ArchitectureCoverageError(
                f"LoRA factors for {name!r} have incompatible shapes "
                f"B={tuple(lora_B.shape)}, A={tuple(lora_A.shape)}; expected "
                f"B rows={expected_b_shape[0]} and A columns={parameter.shape[axis]}. "
                "No model weights were modified."
            )
        if not torch.isfinite(lora_B).all() or not torch.isfinite(lora_A).all():
            raise ArchitectureCoverageError(
                f"LoRA factors for {name!r} contain NaN/Inf values; no model weights were modified"
            )
        try:
            dense_delta = (lora_B @ lora_A).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        except (RuntimeError, TypeError) as exc:
            raise ArchitectureCoverageError(
                f"LoRA factors for {name!r} cannot be multiplied before mutation: {exc}"
            ) from exc
        if dense_delta.shape != (rows, parameter.shape[axis]):
            raise ArchitectureCoverageError(
                f"LoRA delta for {name!r} has shape {tuple(dense_delta.shape)}; "
                "no model weights were modified"
            )
        if not torch.isfinite(dense_delta).all():
            raise ArchitectureCoverageError(
                f"LoRA delta for {name!r} contains NaN/Inf values; no model weights were modified"
            )
        del dense_delta

    try:
        originals = {
            name: entry.parameter.detach().to(device="cpu", copy=True)
            for name, (entry, _owner) in expected.items()
        }
    except (RuntimeError, TypeError) as exc:
        raise ArchitectureCoverageError(
            f"Could not snapshot the complete LoRA manifest before mutation: {exc}"
        ) from exc

    applied = 0
    try:
        with torch.no_grad():
            for name, (entry, _owner) in expected.items():
                lora_B, lora_A = adapters[name]
                delta = (lora_B @ lora_A).to(
                    device=entry.parameter.device,
                    dtype=entry.parameter.dtype,
                )
                _merge_manifest_delta(entry, delta)
                applied += 1
                del delta
        if applied != len(expected):
            raise RuntimeError(
                f"LoRA applied {applied} updates but the manifest requires {len(expected)}"
            )
    except Exception as exc:
        rollback_errors: list[str] = []
        with torch.no_grad():
            for name, (entry, _owner) in expected.items():
                try:
                    entry.parameter.data.copy_(originals[name])
                except (RuntimeError, TypeError) as rollback_exc:
                    rollback_errors.append(f"{name}: {rollback_exc}")
        pipeline._lora_adapters = {}
        if rollback_errors:
            raise RuntimeError(
                "LoRA merge failed and exact rollback was incomplete: " + "; ".join(rollback_errors)
            ) from exc
        raise ArchitectureCoverageError(
            f"LoRA merge failed after {applied} updates; all manifest tensors "
            f"were restored exactly: {exc}"
        ) from exc
    finally:
        del originals

    pipeline._lora_adapters = dict(adapters)
    pipeline.log(f"Applied {applied} LoRA adapters (merged into weights)")
    return applied


def save_lora_adapters(
    adapters: dict[str, tuple[torch.Tensor, torch.Tensor]],
    output_dir: str | Path,
):
    """Save LoRA adapters to disk for later use.

    Saves as a simple dict of {key: (B, A)} tensors using torch.save.
    Can be loaded and applied to the original model for reversible ablation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_dict = {}
    for key, (B, A) in adapters.items():
        save_dict[f"{key}.lora_B"] = B
        save_dict[f"{key}.lora_A"] = A

    adapter_path = output_path / "abliteration_lora_adapters.pt"
    torch.save(save_dict, adapter_path)

    # Save metadata for OBLITERATUS reapplication/audit. Fused residual-axis
    # factors are intentionally not advertised as generic PEFT module adapters.
    import json

    config = {
        "adapter_type": "obliteratus_abliteration_lora",
        "factor_layout": "manifest_residual_axis_flattened",
        "n_adapters": len(adapters),
        "target_modules": sorted(
            {
                parts[-2] if parts[-1] == "weight" else parts[-1]
                for key in adapters
                if (parts := key.split("."))
            }
        ),
        "description": (
            "Reversible abliteration LoRA adapters generated by OBLITERATUS. "
            "These adapters remove refusal directions from the model when merged."
        ),
    }
    (output_path / "abliteration_lora_config.json").write_text(json.dumps(config, indent=2))

    return adapter_path
