"""Exact model-forward TPE search for dense checkpoint projections.

The runnable ``optimized`` and ``heretic`` presets search a compact pair of
piecewise-linear attention/FFN layer kernels plus a continuous direction
coordinate.  Optimized interpolates adjacent SVD components per layer;
Heretic uses one continuously interpolated cross-layer difference direction.
Every trial begins from the immutable full-model snapshot, materializes a
complete candidate plan, runs the ordinary held-out gate, and records exact
direction/manifest/tensor hashes.  The winning plan is restored and replayed
byte-identically before a separate confirmation gate.

This is an independent dense-projection baseline inspired by Heretic
(p-e-w, 2025), not parity with Heretic's optional LoRA execution path.  The
disabled legacy prototype remains only as a short compatibility failure point;
it is never called by a public preset.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from obliteratus.abliterate import AbliterationPipeline

logger = logging.getLogger(__name__)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's exact shape, dtype, and bytes on CPU."""

    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class LayerKernel:
    """Heretic-style piecewise-linear removal kernel."""

    max_weight: float
    peak_position: float
    min_weight: float
    min_weight_distance: float

    def __post_init__(self) -> None:
        values = {
            "max_weight": self.max_weight,
            "peak_position": self.peak_position,
            "min_weight": self.min_weight,
            "min_weight_distance": self.min_weight_distance,
        }
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Layer-kernel parameters must be finite")
        if not 0.0 <= self.min_weight <= self.max_weight <= 1.0:
            raise ValueError("Layer-kernel weights must satisfy 0 <= min <= max <= 1")
        if not 0.0 <= self.peak_position <= 1.0:
            raise ValueError("peak_position must be between zero and one")
        if not 0.0 < self.min_weight_distance <= 1.0:
            raise ValueError("min_weight_distance must be in (0, 1]")

    def removal_weight(self, layer_idx: int, n_layers: int) -> float:
        return _parametric_layer_weight(
            layer_idx,
            n_layers,
            self.max_weight,
            self.peak_position,
            self.min_weight,
            self.min_weight_distance,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "max_weight": self.max_weight,
            "peak_position": self.peak_position,
            "min_weight": self.min_weight,
            "min_weight_distance": self.min_weight_distance,
        }


@dataclass(frozen=True)
class ExactProjectionCandidate:
    """Complete, replayable checkpoint edit evaluated by the optimizer.

    Directions are copied to CPU when the candidate is built.  The plan also
    records separate attention/FFN regularizations and the exact manifest
    storage identities; no averaged scalar is reconstructed after scoring.
    """

    method: str
    trial_index: int
    direction_index: float
    attention_kernel: LayerKernel
    ffn_kernel: LayerKernel
    directions: tuple[tuple[int, torch.Tensor], ...] = field(repr=False, compare=False)
    attention_regularizations: tuple[tuple[int, float], ...]
    ffn_regularizations: tuple[tuple[int, float], ...]
    direction_hashes: tuple[tuple[int, str], ...]
    manifest_target: str
    manifest_fingerprint: str
    target_storage_ids: tuple[str, ...]
    norm_preserve: bool
    project_biases: bool
    projection_row_fraction: float
    parameters: tuple[tuple[str, float], ...]

    def direction_map(self) -> dict[int, torch.Tensor]:
        return {layer: direction for layer, direction in self.directions}

    def attention_map(self) -> dict[int, float]:
        return dict(self.attention_regularizations)

    def ffn_map(self) -> dict[int, float]:
        return dict(self.ffn_regularizations)

    def to_metadata(self) -> dict[str, object]:
        return {
            "method": self.method,
            "trial_index": self.trial_index,
            "direction_index": self.direction_index,
            "attention_kernel": self.attention_kernel.to_dict(),
            "ffn_kernel": self.ffn_kernel.to_dict(),
            "attention_regularizations": {
                str(layer): value for layer, value in self.attention_regularizations
            },
            "ffn_regularizations": {
                str(layer): value for layer, value in self.ffn_regularizations
            },
            "direction_hashes": {
                str(layer): value for layer, value in self.direction_hashes
            },
            "manifest_target": self.manifest_target,
            "manifest_fingerprint": self.manifest_fingerprint,
            "target_storage_ids": list(self.target_storage_ids),
            "norm_preserve": self.norm_preserve,
            "project_biases": self.project_biases,
            "projection_row_fraction": self.projection_row_fraction,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class CandidateApplication:
    """Evidence produced by one exact candidate application."""

    modified_count: int
    target_state_hash: str


@dataclass(frozen=True)
class BayesianTrialRecord:
    """Selection evidence for one model-forward trial."""

    candidate: ExactProjectionCandidate
    objective: float
    accepted: bool
    refusal_rate: float | None
    mean_kl: float | None
    p95_kl: float | None
    target_state_hash: str | None
    assessment: dict[str, object] | None
    error: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.to_metadata(),
            "objective": self.objective if math.isfinite(self.objective) else None,
            "accepted": self.accepted,
            "refusal_rate": self.refusal_rate,
            "mean_kl": self.mean_kl,
            "p95_kl": self.p95_kl,
            "target_state_hash": self.target_state_hash,
            "assessment": self.assessment,
            "error": self.error,
        }


@dataclass(frozen=True)
class BayesianOptimizationResult:
    """Winner plus all selection evidence and exact confirmation hash."""

    winner: ExactProjectionCandidate
    trials: tuple[BayesianTrialRecord, ...]
    baseline_state_hash: str
    selection_state_hash: str
    confirmation_state_hash: str | None = None

    def to_metadata(self) -> dict[str, object]:
        return {
            "algorithm": "exact_model_forward_tpe_projection_search",
            "winner": self.winner.to_metadata(),
            "baseline_state_hash": self.baseline_state_hash,
            "selection_state_hash": self.selection_state_hash,
            "confirmation_state_hash": self.confirmation_state_hash,
            "trials": [trial.to_metadata() for trial in self.trials],
        }


def _measure_refusal_rate(
    pipeline: AbliterationPipeline,
    n_prompts: int = 10,
    max_new_tokens: int = 128,
) -> float:
    """Measure refusal rate on a subset of harmful prompts.

    Applies the model's chat template (when available) so the model is in
    chat mode — matching how users actually interact with it.  Uses token-ID
    slicing for robust completion extraction regardless of special tokens.
    """
    from obliteratus.evaluation.advanced_metrics import _is_refusal_detailed

    model = pipeline.handle.model
    tokenizer = pipeline.handle.tokenizer
    device = pipeline._get_model_device(model)

    # Hyperparameter search is part of training/selection.  It must never read
    # the locked acceptance prompts, otherwise the final gate is no longer a
    # holdout.  Every current pipeline constructs ``_discovery_harmful`` before
    # optimization; the explicit failure keeps custom callers from silently
    # falling back to the full (possibly mixed) corpus.
    discovery_prompts = getattr(pipeline, "_discovery_harmful", None)
    if discovery_prompts is None:
        raise RuntimeError(
            "Bayesian optimization requires a direction-discovery prompt set"
        )
    test_prompts = list(discovery_prompts[:n_prompts])

    # Always apply chat template for refusal measurement — refusal circuitry
    # is activated by chat formatting, so raw prompts give false 0% rates.
    saved_flag = pipeline.use_chat_template
    pipeline.use_chat_template = True
    try:
        formatted_prompts = pipeline._maybe_apply_chat_template(test_prompts)
    finally:
        pipeline.use_chat_template = saved_flag
    if not formatted_prompts:
        return 1.0

    target_failures = 0

    for prompt in formatted_prompts:
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256,
            )
            input_len = inputs["input_ids"].shape[1]
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            # Token-ID slicing: robust regardless of special tokens
            completion_ids = output[0][input_len:]
            completion = tokenizer.decode(
                completion_ids, skip_special_tokens=True,
            ).strip()
            refused, reason = _is_refusal_detailed(completion, mode="combined")
            # Empty/repetitive output is not successful refusal removal.  Count
            # it with refusals so a damaged trial cannot win the optimizer.
            if refused or reason == "DEGENERATE":
                target_failures += 1
            del inputs, output
        except Exception as exc:  # noqa: BLE001 - model backends raise heterogeneous errors
            # A failed measurement is worst-case evidence, not compliance.
            logger.debug("Bayesian refusal measurement failed: %s", exc)
            target_failures += 1

    pipeline._free_gpu_memory()
    return target_failures / len(formatted_prompts)


def _measure_kl_divergence(
    pipeline: AbliterationPipeline,
    reference_logits: list[torch.Tensor],
    prompts: list[str],
) -> float:
    """Measure KL divergence from reference (pre-ablation) logits."""
    model = pipeline.handle.model
    tokenizer = pipeline.handle.tokenizer
    device = pipeline._get_model_device(model)

    total_kl = 0.0
    n_valid = 0

    for i, prompt in enumerate(prompts):
        if i >= len(reference_logits):
            break
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=64,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                new_logits = outputs.logits[:, -1, :].detach().cpu().float()

            ref = reference_logits[i]
            log_p = F.log_softmax(ref, dim=-1)
            log_q = F.log_softmax(new_logits.squeeze(0), dim=-1)
            p = F.softmax(ref, dim=-1)
            kl = (p * (log_p - log_q)).sum().item()
            total_kl += max(kl, 0.0)  # Clamp negative KL (numerical noise)
            n_valid += 1
            del inputs, outputs, new_logits
        except Exception as exc:  # noqa: BLE001 - model backends raise heterogeneous errors
            # Partial/missing KL evidence must not make a broken trial appear
            # closer to the base model than a fully measured trial.
            logger.debug("Bayesian KL measurement failed: %s", exc)
            pipeline._free_gpu_memory()
            return float("inf")

    pipeline._free_gpu_memory()
    return total_kl / n_valid if n_valid else float("inf")


def _parametric_layer_weight(
    layer_idx: int,
    n_layers: int,
    max_weight: float,
    peak_position: float,
    min_weight: float,
    spread: float,
) -> float:
    """Compute ablation weight for a layer using a piecewise-linear tent kernel.

    Faithful reproduction of Heretic's parametric kernel (p-e-w/heretic):
    - max_weight: peak ablation strength at peak_position
    - peak_position: normalized position of peak (0..1)
    - min_weight: weight at the edges of the tent
    - spread: normalized distance from peak to tent edge (min_weight_distance)

    Layers beyond ``spread`` from the peak get weight 0 (skipped entirely).
    Within the tent, weight drops linearly from max_weight to min_weight.
    This matches Heretic's actual formula::

        distance = abs(layer_index - max_weight_position)
        if distance > min_weight_distance: skip
        weight = max_weight + (distance / min_weight_distance) * (min_weight - max_weight)
    """
    if n_layers <= 1:
        return max_weight

    normalized_pos = layer_idx / (n_layers - 1)
    dist = abs(normalized_pos - peak_position)
    min_weight_distance = max(spread, 0.01)

    # Hard cutoff: layers outside the tent get 0 weight (Heretic skips them)
    if dist > min_weight_distance:
        return 0.0

    # Linear interpolation: max_weight at peak → min_weight at edges
    return max_weight + (dist / min_weight_distance) * (min_weight - max_weight)


def _interpolate_direction(
    pipeline: AbliterationPipeline,
    layer_idx: int,
    float_dir_idx: float,
) -> torch.Tensor:
    """Get an interpolated refusal direction from a float-valued layer index.

    Faithful reproduction of Heretic's direction interpolation: the index
    selects which *layer's* diff-of-means direction to use, with float
    values interpolating between adjacent layers' directions.  This is
    fundamentally different from interpolating between SVD components
    within a single layer — it searches across the layer axis.

    From Heretic source (model.py)::

        weight, index = math.modf(direction_index + 1)
        refusal_direction = F.normalize(
            refusal_directions[int(index)].lerp(
                refusal_directions[int(index) + 1], weight), p=2, dim=0)

    Args:
        pipeline: Pipeline with extracted refusal directions per layer.
        layer_idx: The layer being projected (used as fallback).
        float_dir_idx: Continuous direction index — selects which layer's
            direction to use (e.g., 5.3 interpolates 70% layer-5 + 30% layer-6).

    Returns:
        Normalized direction tensor.
    """
    # Build sorted list of layer indices that have refusal directions
    sorted_layers = sorted(pipeline.refusal_directions.keys())
    if not sorted_layers:
        return pipeline.refusal_directions.get(layer_idx, torch.zeros(1))

    n_layers_with_dirs = len(sorted_layers)

    # Heretic uses direction_index + 1 offset; we map float_dir_idx into
    # the sorted layer list, clamped to valid range.
    float_dir_idx = max(0.0, min(float_dir_idx, n_layers_with_dirs - 1))

    lo = int(float_dir_idx)
    hi = min(lo + 1, n_layers_with_dirs - 1)

    lo_layer = sorted_layers[lo]
    hi_layer = sorted_layers[hi]

    d_lo = pipeline.refusal_directions[lo_layer]
    d_hi = pipeline.refusal_directions[hi_layer]

    if lo == hi:
        d = d_lo
    else:
        # Linear interpolation between adjacent layers' directions
        alpha = float_dir_idx - lo
        d = (1.0 - alpha) * d_lo + alpha * d_hi

    norm = d.norm()
    if norm > 1e-8:
        d = d / norm
    return d


def _unsafe_legacy_run_bayesian_optimization(
    pipeline: AbliterationPipeline,
    n_trials: int = 50,
    n_refusal_prompts: int = 30,
    n_kl_prompts: int = 5,
) -> dict[int, float]:
    """Run Bayesian optimization to find optimal ablation parameters.

    Uses Optuna TPE with a Heretic-style parametric layer kernel to search
    a compact parameter space:
    - 4 kernel params (max_weight, peak_position, min_weight, spread)
    - 1 float direction index (interpolated between SVD directions)
    - 2 component weights (attention vs MLP scaling)
    Total: 7 parameters regardless of model size (vs O(n_layers) before).

    Also optimizes per-layer independently when the kernel doesn't fit well
    (uses kernel as warm-start for per-layer refinement in phase 2).

    Args:
        pipeline: Initialized AbliterationPipeline (post-DISTILL stage).
        n_trials: Number of optimization trials.
        n_refusal_prompts: Number of harmful prompts for refusal measurement.
        n_kl_prompts: Number of harmless prompts for KL measurement.

    Returns:
        Dict mapping layer_idx -> optimal regularization value.
    """
    raise RuntimeError(
        "Unsafe legacy Bayesian replay is permanently disabled; it must not edit weights"
    )

    # Retained temporarily as implementation history while the exact structured
    # plan is developed. This block is unreachable by design.
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.warning(
            "Optuna not installed — skipping Bayesian optimization. "
            "Install with: pip install optuna"
        )
        return {}

    if not pipeline.handle or not pipeline._strong_layers:
        return {}

    model = pipeline.handle.model
    tokenizer = pipeline.handle.tokenizer
    device = pipeline._get_model_device(model)

    pipeline.log("Bayesian optimization: collecting reference logits...")

    # Collect reference (pre-ablation) logits for KL measurement
    kl_prompts = [
        "The weather today is",
        "In mathematics, the concept of",
        "The history of ancient Rome",
        "How to cook a simple pasta",
        "The process of photosynthesis involves",
    ][:n_kl_prompts]

    reference_logits: list[torch.Tensor] = []
    measured_kl_prompts: list[str] = []
    for prompt in kl_prompts:
        try:
            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=64,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                reference_logits.append(
                    outputs.logits[:, -1, :].detach().cpu().float().squeeze(0)
                )
                measured_kl_prompts.append(prompt)
            del inputs, outputs
        except Exception as exc:  # noqa: BLE001 - model backends raise heterogeneous errors
            logger.debug("Bayesian reference-logit capture failed: %s", exc)
    pipeline._free_gpu_memory()

    if not reference_logits:
        pipeline.log("  Failed to collect reference logits — skipping optimization")
        return {}

    from obliteratus.abliterate import _ATTN_OUT_NAMES, _FFN_OUT_NAMES
    from obliteratus.strategies.utils import (
        get_attention_module,
        get_ffn_module,
        get_layer_modules,
    )

    layer_modules = get_layer_modules(pipeline.handle)
    arch = pipeline.handle.architecture
    n_total_layers = len(layer_modules)

    # Save weight tensors for rollback — clone to CPU to free GPU memory
    original_params: list[tuple[torch.Tensor, torch.Tensor]] = []
    seen_data_ptrs: set[int] = set()

    for idx in pipeline._strong_layers:
        try:
            attn = get_attention_module(layer_modules[idx], arch)
            for attr_name in _ATTN_OUT_NAMES:
                proj = getattr(attn, attr_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    ptr = proj.weight.data.data_ptr()
                    if ptr not in seen_data_ptrs:
                        original_params.append((proj.weight.data, proj.weight.data.clone().cpu()))
                        seen_data_ptrs.add(ptr)
                    if hasattr(proj, "bias") and proj.bias is not None:
                        bptr = proj.bias.data.data_ptr()
                        if bptr not in seen_data_ptrs:
                            original_params.append((proj.bias.data, proj.bias.data.clone().cpu()))
                            seen_data_ptrs.add(bptr)
        except (AttributeError, RuntimeError):
            pass
        try:
            ffn = get_ffn_module(layer_modules[idx], arch)
            for attr_name in _FFN_OUT_NAMES:
                proj = getattr(ffn, attr_name, None)
                if proj is not None and hasattr(proj, "weight"):
                    ptr = proj.weight.data.data_ptr()
                    if ptr not in seen_data_ptrs:
                        original_params.append((proj.weight.data, proj.weight.data.clone().cpu()))
                        seen_data_ptrs.add(ptr)
                    if hasattr(proj, "bias") and proj.bias is not None:
                        bptr = proj.bias.data.data_ptr()
                        if bptr not in seen_data_ptrs:
                            original_params.append((proj.bias.data, proj.bias.data.clone().cpu()))
                            seen_data_ptrs.add(bptr)
        except (AttributeError, RuntimeError):
            pass

    del seen_data_ptrs
    total_saved_mb = sum(clone.nelement() * clone.element_size() for _, clone in original_params) / 1e6
    pipeline.log(f"  Saved {len(original_params)} weight tensors for rollback ({total_saved_mb:.0f} MB, on CPU)")

    def _restore_all():
        for live_data, saved_clone in original_params:  # noqa: F821
            live_data.copy_(saved_clone.to(live_data.device))

    # Warm-start values for the parametric kernel.
    # If the informed pipeline provided analysis-derived warm-start params,
    # use those (they're much better than the default heuristic).
    informed_warm = getattr(pipeline, "_informed_warm_start", None)
    if informed_warm:
        warm_peak = informed_warm.get("peak_position", 0.5)
        pipeline.log(f"  Using analysis-informed warm-start (peak={warm_peak:.2f})")
    elif pipeline._strong_layers:
        peak_layer = pipeline._strong_layers[0]
        warm_peak = peak_layer / max(n_total_layers - 1, 1)
    else:
        warm_peak = 0.5

    best_result: dict[int, float] = {}
    best_score = float("inf")

    # Suppress Optuna's verbose logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Max layers with directions (for float direction interpolation)
    n_layers_with_dirs = len([
        idx for idx in pipeline._strong_layers
        if idx in pipeline.refusal_directions
    ])

    # ── Phase 1: Parametric kernel optimization (compact search space) ──
    # Heretic uses SEPARATE kernel parameters for attention and MLP,
    # allowing them to peak at different layers (8 params + 1 dir_idx = 9).

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        """Multi-objective: minimize (refusal_rate, kl_divergence)."""
        _restore_all()

        # Attention kernel: 4 params
        attn_max = trial.suggest_float("attn_max_weight", 0.5, 1.0)
        attn_peak = trial.suggest_float("attn_peak_position", 0.1, 0.9)
        attn_min = trial.suggest_float("attn_min_weight", 0.0, 0.3)
        attn_spread = trial.suggest_float("attn_spread", 0.1, 0.6)

        # MLP kernel: 4 params (separate — can peak at a different layer)
        mlp_max = trial.suggest_float("mlp_max_weight", 0.3, 1.0)
        mlp_peak = trial.suggest_float("mlp_peak_position", 0.1, 0.9)
        mlp_min = trial.suggest_float("mlp_min_weight", 0.0, 0.3)
        mlp_spread = trial.suggest_float("mlp_spread", 0.1, 0.6)

        # Float direction index (cross-layer interpolation, Heretic-style)
        dir_idx = trial.suggest_float("dir_idx", 0.0, max(n_layers_with_dirs - 1, 0.0))

        # Compute per-layer, per-component regularization from kernels
        attn_regs: dict[int, float] = {}
        mlp_regs: dict[int, float] = {}
        for idx in pipeline._strong_layers:
            attn_w = _parametric_layer_weight(idx, n_total_layers, attn_max, attn_peak, attn_min, attn_spread)
            mlp_w = _parametric_layer_weight(idx, n_total_layers, mlp_max, mlp_peak, mlp_min, mlp_spread)
            attn_regs[idx] = 1.0 - attn_w
            mlp_regs[idx] = 1.0 - mlp_w

        # Apply projection with trial's parameters
        for idx in pipeline._strong_layers:
            if idx not in pipeline.refusal_directions:
                continue

            # Use cross-layer interpolated direction
            direction = _interpolate_direction(pipeline, idx, dir_idx)
            d_col = direction.to(device=next(layer_modules[idx].parameters()).device)
            d_col = d_col.unsqueeze(-1) if d_col.dim() == 1 else d_col

            # Attention projection (with per-component kernel)
            attn_reg = attn_regs[idx]
            try:
                attn = get_attention_module(layer_modules[idx], arch)
                pipeline._project_out_advanced(
                    attn, d_col, _ATTN_OUT_NAMES,
                    norm_preserve=pipeline.norm_preserve,
                    regularization=attn_reg,
                )
            except (AttributeError, RuntimeError):
                pass

            # MLP/FFN projection (with per-component kernel)
            mlp_reg = mlp_regs[idx]
            try:
                ffn = get_ffn_module(layer_modules[idx], arch)
                count = pipeline._project_out_advanced(
                    ffn, d_col, _FFN_OUT_NAMES,
                    norm_preserve=pipeline.norm_preserve,
                    regularization=mlp_reg,
                )
                if count == 0:
                    pipeline._project_moe_experts(
                        ffn, d_col,
                        norm_preserve=pipeline.norm_preserve,
                        regularization=mlp_reg,
                        project_biases=False,
                    )
            except (AttributeError, RuntimeError):
                pass

        # Measure objectives
        refusal = _measure_refusal_rate(pipeline, n_prompts=n_refusal_prompts)
        kl = _measure_kl_divergence(
            pipeline,
            reference_logits,
            measured_kl_prompts,
        )

        # Track best combined score (use average of attn/mlp regs for layer_regs)
        nonlocal best_score, best_result
        combined = refusal + 0.5 * kl
        if combined < best_score:
            best_score = combined
            best_result = {
                idx: (attn_regs[idx] + mlp_regs[idx]) / 2.0
                for idx in pipeline._strong_layers
            }

        pipeline.log(
            f"  Trial {trial.number + 1}/{n_trials}: "
            f"refusal={refusal:.0%}, KL={kl:.4f} "
            f"(attn_peak={attn_peak:.2f}, mlp_peak={mlp_peak:.2f}, dir={dir_idx:.2f})"
        )

        return refusal, kl

    sampler = TPESampler(seed=42, n_startup_trials=min(5, n_trials // 3))
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
        study_name="obliteratus_parametric_optimization",
    )

    # Enqueue warm-start trial with analysis-derived estimates.
    # Translate informed pipeline params to the new per-component format.
    if informed_warm:
        iw = informed_warm
        warm_params = {
            "attn_max_weight": iw.get("max_weight", 0.9),
            "attn_peak_position": iw.get("peak_position", warm_peak),
            "attn_min_weight": iw.get("min_weight", 0.05),
            "attn_spread": iw.get("spread", 0.3),
            "mlp_max_weight": iw.get("max_weight", 0.9) * iw.get("mlp_scale", 0.6),
            "mlp_peak_position": iw.get("peak_position", warm_peak),
            "mlp_min_weight": iw.get("min_weight", 0.05),
            "mlp_spread": iw.get("spread", 0.3),
            "dir_idx": iw.get("dir_idx", 0.0),
        }
    else:
        warm_params = {
            "attn_max_weight": 0.9,
            "attn_peak_position": warm_peak,
            "attn_min_weight": 0.05,
            "attn_spread": 0.3,
            "mlp_max_weight": 0.6,
            "mlp_peak_position": warm_peak,
            "mlp_min_weight": 0.05,
            "mlp_spread": 0.3,
            "dir_idx": 0.0,
        }
    study.enqueue_trial(warm_params)

    pipeline.log(f"Bayesian optimization: running {n_trials} trials (parametric kernel)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Restore model and apply best result
    _restore_all()

    # Get best trial from Pareto front (prefer low refusal)
    pareto = [
        trial
        for trial in study.best_trials
        if trial.values is not None
        and all(math.isfinite(float(value)) for value in trial.values)
    ]
    if pareto:
        pareto.sort(key=lambda t: (t.values[0], t.values[1]))
        best_trial = pareto[0]

        # Reconstruct per-layer regs from best kernel params
        p = best_trial.params
        best_result = {}
        for idx in pipeline._strong_layers:
            attn_w = _parametric_layer_weight(
                idx, n_total_layers,
                p["attn_max_weight"], p["attn_peak_position"],
                p["attn_min_weight"], p["attn_spread"],
            )
            mlp_w = _parametric_layer_weight(
                idx, n_total_layers,
                p["mlp_max_weight"], p["mlp_peak_position"],
                p["mlp_min_weight"], p["mlp_spread"],
            )
            best_result[idx] = (attn_w + mlp_w) / 2.0  # average for layer-level reg
            best_result[idx] = 1.0 - best_result[idx]

        pipeline.log(
            f"  Best trial: refusal={best_trial.values[0]:.0%}, "
            f"KL={best_trial.values[1]:.4f}"
        )
        pipeline.log(
            f"  Attn kernel: peak={p['attn_peak_position']:.2f}, "
            f"spread={p['attn_spread']:.2f}, max={p['attn_max_weight']:.2f}"
        )
        pipeline.log(
            f"  MLP kernel:  peak={p['mlp_peak_position']:.2f}, "
            f"spread={p['mlp_spread']:.2f}, max={p['mlp_max_weight']:.2f}"
        )
        pipeline.log(f"  dir_idx={p['dir_idx']:.2f}")

        # Store the best direction index for use during EXCISE
        best_dir_idx = p.get("dir_idx", 0.0)
        if best_dir_idx > 0.1:
            pipeline.log(f"  Applying interpolated direction (idx={best_dir_idx:.2f})...")
            for idx in pipeline._strong_layers:
                new_dir = _interpolate_direction(pipeline, idx, best_dir_idx)
                pipeline.refusal_directions[idx] = new_dir

        # Store component scales for use in EXCISE (backward compat)
        pipeline._bayesian_attn_scale = p.get("attn_max_weight", 1.0)
        pipeline._bayesian_mlp_scale = p.get("mlp_max_weight", 1.0)

    elif best_result:
        pipeline.log(f"  Using best combined score: {best_score:.4f}")
    else:
        pipeline.log(
            "  Bayesian optimization produced no fully measured finite trial; "
            "ignoring its candidate settings"
        )

    # Clean up
    del original_params
    pipeline._free_gpu_memory()

    return best_result


def _manifest_fingerprint(manifest: Any) -> str:
    """Fingerprint the exact manifest layout and runtime storage identities."""

    payload = {
        "architecture": manifest.architecture,
        "target": manifest.target,
        "hidden_size": manifest.hidden_size,
        "num_layers": manifest.num_layers,
        "entries": [
            {
                "name": entry.qualified_name,
                "layers": list(entry.layer_indices),
                "component": entry.component,
                "role": entry.role,
                "orientation": entry.orientation,
                "shape": list(entry.shape),
                "storage": entry.storage_identity,
                "axis": entry.residual_axis,
            }
            for entry in manifest.entries
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _target_state_hash(
    pipeline: AbliterationPipeline,
    manifest: Any,
    *,
    project_biases: bool,
) -> str:
    """Hash every manifest tensor and every writer bias touched by a plan."""

    digest = hashlib.sha256()
    seen_biases: set[int] = set()
    for entry in sorted(manifest.entries, key=lambda item: item.qualified_name):
        tensor = entry.parameter.detach().cpu().contiguous()
        digest.update(entry.qualified_name.encode("utf-8"))
        digest.update(_tensor_sha256(tensor).encode("ascii"))
        if not project_biases or entry.role != "writer":
            continue
        obj = pipeline._resolve_dotted_projection(entry.owner, entry.attribute_path)
        bias = getattr(obj, "bias", None)
        if not isinstance(bias, torch.Tensor) or id(bias) in seen_biases:
            continue
        seen_biases.add(id(bias))
        digest.update(f"{entry.qualified_name}:bias".encode())
        digest.update(_tensor_sha256(bias).encode("ascii"))
    return digest.hexdigest()


def _normalized_direction(value: torch.Tensor, *, context: str) -> torch.Tensor:
    direction = value.detach().cpu().float().reshape(-1).contiguous()
    if not torch.isfinite(direction).all():
        raise RuntimeError(f"{context} contains non-finite values")
    norm = float(direction.norm().item())
    if not math.isfinite(norm) or norm <= 1e-8:
        raise RuntimeError(f"{context} is degenerate")
    return (direction / norm).contiguous()


def _optimized_layer_direction(
    pipeline: AbliterationPipeline,
    layer_idx: int,
    direction_index: float,
) -> torch.Tensor:
    """Interpolate adjacent SVD components without mutating distilled state."""

    subspace = pipeline.refusal_subspaces.get(layer_idx)
    if subspace is None or subspace.ndim != 2 or subspace.shape[0] == 0:
        try:
            return _normalized_direction(
                pipeline.refusal_directions[layer_idx],
                context=f"layer {layer_idx} refusal direction",
            )
        except KeyError as exc:
            raise RuntimeError(f"Layer {layer_idx} has no distilled direction") from exc

    upper = subspace.shape[0] - 1
    position = min(max(float(direction_index), 0.0), float(upper))
    lower_index = math.floor(position)
    upper_index = min(lower_index + 1, upper)
    alpha = position - lower_index
    value = (1.0 - alpha) * subspace[lower_index] + alpha * subspace[upper_index]
    return _normalized_direction(value, context=f"layer {layer_idx} SVD interpolation")


def _heretic_shared_direction(
    pipeline: AbliterationPipeline,
    direction_index: float,
) -> torch.Tensor:
    value = _interpolate_direction(pipeline, 0, direction_index)
    return _normalized_direction(value, context="Heretic cross-layer interpolation")


def _candidate_direction_upper_bound(
    pipeline: AbliterationPipeline,
    method: str,
) -> float:
    if method == "heretic":
        return float(max(0, len(pipeline.refusal_directions) - 1))
    ranks = [
        int(subspace.shape[0])
        for layer, subspace in pipeline.refusal_subspaces.items()
        if layer in pipeline._strong_layers and subspace.ndim == 2 and subspace.shape[0]
    ]
    return float(max(ranks, default=1) - 1)


def build_exact_projection_candidate(
    pipeline: AbliterationPipeline,
    *,
    trial_index: int,
    parameters: dict[str, float],
) -> ExactProjectionCandidate:
    """Materialize all tensors and strengths needed to replay one trial."""

    method = pipeline.method
    if method not in {"optimized", "heretic"}:
        raise ValueError("Exact Bayesian projection supports optimized or heretic")
    if not pipeline._strong_layers:
        raise RuntimeError("Bayesian projection needs at least one selected layer")
    manifest = pipeline._current_projection_manifest()
    layers = tuple(sorted(dict.fromkeys(pipeline._strong_layers)))
    if any(layer < 0 or layer >= manifest.num_layers for layer in layers):
        raise RuntimeError("Selected layer lies outside the validated manifest")

    def kernel(prefix: str) -> LayerKernel:
        maximum = float(parameters[f"{prefix}_max_weight"])
        return LayerKernel(
            max_weight=maximum,
            peak_position=float(parameters[f"{prefix}_peak_position"]),
            min_weight=min(float(parameters[f"{prefix}_min_weight"]), maximum),
            min_weight_distance=float(parameters[f"{prefix}_min_weight_distance"]),
        )

    attention_kernel = kernel("attention")
    ffn_kernel = kernel("ffn")
    direction_index = float(parameters["direction_index"])
    shared_direction = (
        _heretic_shared_direction(pipeline, direction_index)
        if method == "heretic"
        else None
    )
    directions: list[tuple[int, torch.Tensor]] = []
    attention_regularizations: list[tuple[int, float]] = []
    ffn_regularizations: list[tuple[int, float]] = []
    for layer_idx in layers:
        direction = (
            shared_direction.clone()
            if shared_direction is not None
            else _optimized_layer_direction(pipeline, layer_idx, direction_index)
        )
        directions.append((layer_idx, direction))
        attention_regularizations.append(
            (
                layer_idx,
                1.0 - attention_kernel.removal_weight(layer_idx, manifest.num_layers),
            )
        )
        ffn_regularizations.append(
            (
                layer_idx,
                1.0 - ffn_kernel.removal_weight(layer_idx, manifest.num_layers),
            )
        )

    direction_hashes = tuple(
        (layer_idx, _tensor_sha256(direction)) for layer_idx, direction in directions
    )
    return ExactProjectionCandidate(
        method=method,
        trial_index=trial_index,
        direction_index=direction_index,
        attention_kernel=attention_kernel,
        ffn_kernel=ffn_kernel,
        directions=tuple(directions),
        attention_regularizations=tuple(attention_regularizations),
        ffn_regularizations=tuple(ffn_regularizations),
        direction_hashes=direction_hashes,
        manifest_target=manifest.target,
        manifest_fingerprint=_manifest_fingerprint(manifest),
        target_storage_ids=tuple(entry.storage_identity for entry in manifest.entries),
        norm_preserve=bool(pipeline.norm_preserve),
        project_biases=bool(pipeline.project_biases),
        projection_row_fraction=float(pipeline.projection_row_fraction),
        parameters=tuple(sorted((name, float(value)) for name, value in parameters.items())),
    )


def apply_exact_projection_candidate(
    pipeline: AbliterationPipeline,
    candidate: ExactProjectionCandidate,
    *,
    expected_state_hash: str | None = None,
) -> CandidateApplication:
    """Apply only the stored candidate and verify exact replay when requested."""

    if pipeline.method != candidate.method:
        raise RuntimeError("Candidate method does not match the active pipeline")
    manifest = pipeline._current_projection_manifest()
    if manifest.target != candidate.manifest_target:
        raise RuntimeError("Candidate projection target changed before replay")
    if _manifest_fingerprint(manifest) != candidate.manifest_fingerprint:
        raise RuntimeError("Candidate manifest changed before replay")
    if tuple(entry.storage_identity for entry in manifest.entries) != candidate.target_storage_ids:
        raise RuntimeError("Candidate target storage identities changed before replay")
    if bool(pipeline.norm_preserve) != candidate.norm_preserve:
        raise RuntimeError("Candidate norm-preservation mode changed before replay")
    if bool(pipeline.project_biases) != candidate.project_biases:
        raise RuntimeError("Candidate bias-projection mode changed before replay")
    if not math.isclose(
        float(pipeline.projection_row_fraction),
        candidate.projection_row_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Candidate row-selection fraction changed before replay")

    directions = candidate.direction_map()
    actual_hashes = tuple(
        (layer, _tensor_sha256(direction)) for layer, direction in sorted(directions.items())
    )
    if actual_hashes != candidate.direction_hashes:
        raise RuntimeError("Candidate direction tensor changed before replay")
    attention = candidate.attention_map()
    ffn = candidate.ffn_map()
    strong_layers = set(directions)
    edited: set[tuple[str, int]] = set()
    modified = 0
    with torch.no_grad():
        for layer_idx in sorted(strong_layers):
            modified += pipeline._project_manifest_layer_direction(
                manifest,
                layer_idx=layer_idx,
                direction_index=0,
                direction=directions[layer_idx],
                attention_regularization=attention[layer_idx],
                ffn_regularization=ffn[layer_idx],
                norm_preserve=candidate.norm_preserve,
                edited=edited,
                strong_layers=strong_layers,
            )

    expected_entries: set[tuple[str, int]] = set()
    for entry in manifest.entries:
        owners = strong_layers.intersection(entry.layer_indices)
        if owners:
            expected_entries.add((entry.storage_identity, 0))
    if edited != expected_entries:
        missing = sorted(expected_entries - edited)
        unexpected = sorted(edited - expected_entries)
        raise RuntimeError(
            "Exact candidate did not match its manifest plan: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    state_hash = _target_state_hash(
        pipeline,
        manifest,
        project_biases=candidate.project_biases,
    )
    if expected_state_hash is not None and state_hash != expected_state_hash:
        raise RuntimeError(
            "Exact winner replay hash mismatch; restored model will not be saved"
        )
    return CandidateApplication(modified_count=modified, target_state_hash=state_hash)


class ExactTPESampler:
    """Small deterministic Parzen-density sampler with no optional dependency.

    Startup trials uniformly explore the bounded space.  Later suggestions are
    drawn from kernels around the best quartile and selected by the standard
    TPE density ratio l(x)/g(x).
    """

    def __init__(self, *, direction_upper: float, seed: int = 42) -> None:
        if not math.isfinite(direction_upper) or direction_upper < 0.0:
            raise ValueError("direction_upper must be finite and non-negative")
        self.direction_upper = float(direction_upper)
        self._rng = random.Random(seed)
        self._observations: list[tuple[dict[str, float], float]] = []
        self._bounds = {
            "attention_max_weight": (0.35, 1.0),
            "attention_peak_position": (0.0, 1.0),
            "attention_min_weight": (0.0, 0.35),
            "attention_min_weight_distance": (0.1, 1.0),
            "ffn_max_weight": (0.25, 1.0),
            "ffn_peak_position": (0.0, 1.0),
            "ffn_min_weight": (0.0, 0.35),
            "ffn_min_weight_distance": (0.1, 1.0),
            "direction_index": (0.0, self.direction_upper),
        }

    @property
    def observations(self) -> tuple[tuple[dict[str, float], float], ...]:
        return tuple((dict(params), objective) for params, objective in self._observations)

    def _warm_start(self) -> dict[str, float]:
        return {
            "attention_max_weight": 0.85,
            "attention_peak_position": 0.5,
            "attention_min_weight": 0.05,
            "attention_min_weight_distance": 0.35,
            "ffn_max_weight": 0.60,
            "ffn_peak_position": 0.5,
            "ffn_min_weight": 0.05,
            "ffn_min_weight_distance": 0.35,
            "direction_index": 0.0,
        }

    def _uniform(self) -> dict[str, float]:
        return {
            name: lower if lower == upper else self._rng.uniform(lower, upper)
            for name, (lower, upper) in self._bounds.items()
        }

    @staticmethod
    def _density(value: float, samples: list[float], bandwidth: float) -> float:
        if not samples:
            return 1e-12
        scale = max(bandwidth, 1e-12)
        return sum(
            math.exp(-0.5 * ((value - sample) / scale) ** 2)
            for sample in samples
        ) / (len(samples) * scale)

    def _tpe_value(self, name: str, lower: float, upper: float) -> float:
        if lower == upper:
            return lower
        ordered = sorted(self._observations, key=lambda item: item[1])
        good_count = max(2, math.ceil(len(ordered) * 0.25))
        good = [params[name] for params, _ in ordered[:good_count]]
        bad = [params[name] for params, _ in ordered[good_count:]]
        span = upper - lower
        mean = sum(good) / len(good)
        variance = sum((value - mean) ** 2 for value in good) / max(1, len(good) - 1)
        good_bandwidth = max(span * 0.04, math.sqrt(variance) * 0.8)
        bad_bandwidth = max(span * 0.08, good_bandwidth)
        choices: list[tuple[float, float]] = []
        for _ in range(32):
            anchor = self._rng.choice(good)
            value = min(upper, max(lower, self._rng.gauss(anchor, good_bandwidth)))
            good_density = self._density(value, good, good_bandwidth)
            bad_density = self._density(value, bad, bad_bandwidth)
            choices.append((good_density / max(bad_density, 1e-12), value))
        return max(choices, key=lambda item: item[0])[1]

    def suggest(self) -> dict[str, float]:
        if not self._observations:
            return self._warm_start()
        if len(self._observations) < 8:
            return self._uniform()
        return {
            name: self._tpe_value(name, lower, upper)
            for name, (lower, upper) in self._bounds.items()
        }

    def observe(self, parameters: dict[str, float], objective: float) -> None:
        if not math.isfinite(float(objective)):
            objective = 1e9
        if set(parameters) != set(self._bounds):
            raise ValueError("TPE observation parameter set does not match its search space")
        self._observations.append((dict(parameters), float(objective)))


def run_bayesian_optimization(
    pipeline: AbliterationPipeline,
    n_trials: int = 50,
    n_refusal_prompts: int = 30,
    n_kl_prompts: int = 5,
) -> BayesianOptimizationResult:
    """Run exact model-forward TPE selection and leave the winner replayed.

    This function assumes the pipeline has already switched to its selection
    evidence partition.  Every suggestion starts from the immutable full-model
    snapshot, is applied directly from a complete candidate plan, and is scored
    by the ordinary held-out gate.  The selected plan is restored and replayed
    exactly; callers must still run it once on a disjoint confirmation split.
    """

    del n_refusal_prompts, n_kl_prompts  # the ordinary gate owns evidence sizes
    if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if pipeline.method not in {"optimized", "heretic"}:
        raise ValueError("Bayesian checkpoint search is only defined for optimized/heretic")
    purpose = f"method={pipeline.method!r} exact Bayesian search"
    pipeline._assert_auto_projection_prerequisites(purpose)
    baseline_layer_weights = dict(pipeline._layer_excise_weights)
    manifest = pipeline._current_projection_manifest()
    sampler = ExactTPESampler(
        direction_upper=_candidate_direction_upper_bound(pipeline, pipeline.method),
        seed=int(pipeline.damage_eval_seed),
    )
    records: list[BayesianTrialRecord] = []
    accepted: list[BayesianTrialRecord] = []
    baseline_hash: str | None = None

    try:
        for trial_index in range(n_trials):
            pipeline._restore_auto_projection_baseline(
                baseline_layer_weights,
                purpose=purpose,
            )
            restored_hash = _target_state_hash(
                pipeline,
                manifest,
                project_biases=bool(pipeline.project_biases),
            )
            if baseline_hash is None:
                baseline_hash = restored_hash
            elif restored_hash != baseline_hash:
                raise RuntimeError(
                    "Full-snapshot rollback did not recreate the same Bayesian baseline"
                )

            parameters = sampler.suggest()
            candidate = build_exact_projection_candidate(
                pipeline,
                trial_index=trial_index,
                parameters=parameters,
            )
            pipeline.log(
                f"  Bayesian trial {trial_index + 1}/{n_trials}: "
                f"direction={candidate.direction_index:.3f}"
            )
            try:
                application = apply_exact_projection_candidate(pipeline, candidate)
                assessment = pipeline._verify()
                metrics = assessment.metrics
                refusal_value = metrics.get("refusal_rate")
                mean_kl_value = metrics.get("sampled_token_kl_mean")
                p95_kl_value = metrics.get("sampled_token_kl_p95")
                refusal = (
                    float(refusal_value)
                    if refusal_value is not None and math.isfinite(float(refusal_value))
                    else None
                )
                mean_kl = (
                    float(mean_kl_value)
                    if mean_kl_value is not None and math.isfinite(float(mean_kl_value))
                    else None
                )
                p95_kl = (
                    float(p95_kl_value)
                    if p95_kl_value is not None and math.isfinite(float(p95_kl_value))
                    else None
                )
                normalized_damage = pipeline._normalized_projection_damage(assessment)
                objective = (
                    (refusal if refusal is not None else 1.0)
                    + (normalized_damage if math.isfinite(normalized_damage) else 10.0)
                    + (0.0 if assessment.accepted else 10.0)
                )
                record = BayesianTrialRecord(
                    candidate=candidate,
                    objective=objective,
                    accepted=bool(assessment.accepted),
                    refusal_rate=refusal,
                    mean_kl=mean_kl,
                    p95_kl=p95_kl,
                    target_state_hash=application.target_state_hash,
                    assessment=assessment.to_dict(),
                )
            except Exception as exc:  # noqa: BLE001 - heterogeneous model backend failures
                objective = 1e9
                record = BayesianTrialRecord(
                    candidate=candidate,
                    objective=objective,
                    accepted=False,
                    refusal_rate=None,
                    mean_kl=None,
                    p95_kl=None,
                    target_state_hash=None,
                    assessment=None,
                    error=f"{type(exc).__name__}: {exc}",
                )
                pipeline.log(f"    trial failed closed: {record.error}")
            sampler.observe(parameters, objective)
            records.append(record)
            if record.accepted and record.target_state_hash is not None:
                accepted.append(record)

        if not accepted:
            raise RuntimeError(
                "Every exact Bayesian candidate failed the selection gate; "
                "the untouched snapshot has been restored"
            )
        winner_record = min(
            accepted,
            key=lambda item: (
                item.objective,
                item.refusal_rate if item.refusal_rate is not None else float("inf"),
                item.mean_kl if item.mean_kl is not None else float("inf"),
                item.candidate.trial_index,
            ),
        )
        pipeline._restore_auto_projection_baseline(
            baseline_layer_weights,
            purpose=purpose,
        )
        restored_hash = _target_state_hash(
            pipeline,
            manifest,
            project_biases=bool(pipeline.project_biases),
        )
        if restored_hash != baseline_hash:
            raise RuntimeError("Winner replay did not begin from the immutable baseline")
        replay = apply_exact_projection_candidate(
            pipeline,
            winner_record.candidate,
            expected_state_hash=winner_record.target_state_hash,
        )
        return BayesianOptimizationResult(
            winner=winner_record.candidate,
            trials=tuple(records),
            baseline_state_hash=baseline_hash,
            selection_state_hash=replay.target_state_hash,
        )
    except BaseException:
        pipeline._restore_auto_projection_baseline(
            baseline_layer_weights,
            purpose=purpose,
        )
        raise
