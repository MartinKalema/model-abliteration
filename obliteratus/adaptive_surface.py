"""Validation boundary for architecture/telemetry-selected UI overrides."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any


def _bool(value: Any) -> bool:
    return isinstance(value, bool)


def _integer(minimum: int, maximum: int) -> Callable[[Any], bool]:
    return lambda value: (
        isinstance(value, int)
        and not isinstance(value, bool)
        and minimum <= value <= maximum
    )


def _number(
    minimum: float,
    maximum: float | None = None,
    *,
    minimum_open: bool = False,
) -> Callable[[Any], bool]:
    def validate(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        numeric = float(value)
        if not math.isfinite(numeric):
            return False
        if numeric < minimum or (minimum_open and numeric == minimum):
            return False
        return maximum is None or numeric <= maximum

    return validate


def _choice(*choices: str) -> Callable[[Any], bool]:
    allowed = frozenset(choices)
    return lambda value: isinstance(value, str) and value in allowed


# This is deliberately narrower than every AbliterationPipeline constructor
# argument. Adaptive data may tune edit behavior, but it cannot replace model,
# output, prompt, callback, checkpoint, or acceptance-policy arguments.
_ADAPTIVE_OVERRIDE_VALIDATORS: dict[str, Callable[[Any], bool]] = {
    "n_directions": _integer(1, 64),
    "direction_method": _choice("diff_means", "svd", "leace", "som"),
    "norm_preserve": _bool,
    "regularization": _number(0.0, 1.0),
    "refinement_passes": _integer(1, 16),
    "project_biases": _bool,
    "use_chat_template": _bool,
    "use_whitened_svd": _bool,
    "true_iterative_refinement": _bool,
    "use_jailbreak_contrast": _bool,
    "layer_adaptive_strength": _bool,
    "safety_neuron_masking": _bool,
    "per_expert_directions": _bool,
    "attention_head_surgery": _bool,
    "use_sae_features": _bool,
    "invert_refusal": _bool,
    "reflection_strength": _number(0.0, 3.0),
    "project_lm_head": _bool,
    "project_embeddings": _bool,
    "embed_regularization": _number(0.0, 1.0),
    "activation_steering": _bool,
    "steering_strength": _number(0.0, 1.0),
    "expert_transplant": _bool,
    "transplant_blend": _number(0.0, 1.0),
    "n_sae_features": _integer(1, 4096),
    "winsorize_activations": _bool,
    "winsorize_percentile": _number(0.0, 1.0),
    "use_lora_ablation": _bool,
    "lora_rank": _integer(1, 4096),
    "use_kl_optimization": _bool,
    "kl_budget": _number(0.0),
    "float_layer_interpolation": _bool,
    "cot_aware": _bool,
    "layer_selection": _choice(
        "knee_cosmic", "all", "all_except_first", "middle60", "top_k", "knee"
    ),
    "min_layer_fraction": _number(0.0, 1.0),
    "max_layer_fraction": _number(0.0, 1.0),
    "harmless_pc_count": _integer(0, 4096),
    "shield_concept_count": _integer(0, 4096),
    "shield_ridge": _number(0.0),
    "shield_residualize": _bool,
    "shield_layer_penalty": _number(0.0),
    "projection_target": _choice("auto", "all", "attention", "ffn", "output"),
    "projection_row_fraction": _number(0.0, 1.0, minimum_open=True),
    "rdo_refinement": _bool,
    "use_wasserstein_optimal": _bool,
    "spectral_cascade": _bool,
    "spectral_bands": _integer(1, 64),
    "spectral_threshold": _number(0.0),
}


def validate_adaptive_overrides(
    overrides: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return pipeline-safe edit overrides and reasons for discarded entries."""
    accepted: dict[str, Any] = {}
    rejected: dict[str, str] = {}
    for key, value in (overrides or {}).items():
        validator = _ADAPTIVE_OVERRIDE_VALIDATORS.get(key)
        if validator is None:
            rejected[key] = "not an adaptive edit parameter"
        elif not validator(value):
            rejected[key] = "invalid value"
        else:
            accepted[key] = value
    return accepted, rejected
