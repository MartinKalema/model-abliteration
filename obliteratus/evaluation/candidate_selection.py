"""Shared fail-closed helpers for selecting edited-model candidates.

The editing pipeline owns measurement and threshold enforcement.  Iterative
and tournament orchestration still need to verify that a completed run carries
the evidence produced by that gate; a successful return or an output directory
alone is not sufficient proof that the model was accepted.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class CandidateEvidenceError(ValueError):
    """Raised when a candidate lacks conclusive, accepted gate evidence."""


REQUIRED_ACCEPTANCE_METRICS = (
    "eval_prompt_count",
    "eval_token_count",
    "sampled_token_count",
    "nll_increase_upper_ci",
    "sampled_token_kl_upper_ci",
    "sampled_token_kl_p95",
    "top1_flip_rate",
    "coherence_drop",
    "new_degenerate_count",
    "nonfinite_output_count",
    "refusal_rate",
    "refusal_eval_count",
)


_DAMAGE_LIMITS = (
    ("nll_increase_upper_ci", "max_nll_increase_upper_ci", 0.05),
    ("sampled_token_kl_upper_ci", "max_sampled_token_kl_upper_ci", 0.05),
    ("sampled_token_kl_p95", "max_p95_sampled_token_kl", 0.20),
    ("top1_flip_rate", "max_top1_flip_rate", 0.02),
    ("coherence_drop", "max_coherence_drop", 0.10),
    ("new_degenerate_count", "max_new_degenerate_outputs", 0.0),
    ("nonfinite_output_count", "max_nonfinite_output_count", 0.0),
)


def _finite_float(value: Any, *, name: str) -> float:
    """Return a finite real number, rejecting booleans and string coercions."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CandidateEvidenceError(f"{name} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise CandidateEvidenceError(f"{name} must be a finite number")
    return numeric


def _nonnegative_int(value: Any, *, name: str) -> int:
    """Return a genuine non-negative count from JSON-compatible evidence."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CandidateEvidenceError(f"{name} must be a non-negative integer")
    return value


def acceptance_payload(assessment: Any) -> dict[str, Any]:
    """Convert and validate a pipeline assessment.

    All enabled orchestration paths intentionally reject old checkpoints and
    custom pipeline results that merely contain quality point estimates but no
    accepted damage-gate record.
    """

    if assessment is None:
        raise CandidateEvidenceError("damage-gate assessment is missing")
    if isinstance(assessment, Mapping):
        payload = dict(assessment)
    else:
        to_dict = getattr(assessment, "to_dict", None)
        if not callable(to_dict):
            raise CandidateEvidenceError("damage-gate assessment is not serializable")
        payload = to_dict()
    return validate_acceptance_payload(payload)


def validate_acceptance_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Require an accepted, conclusive assessment with complete measurements."""

    data = dict(payload)
    if data.get("accepted") is not True:
        raise CandidateEvidenceError("candidate was not accepted by the damage gate")
    if data.get("damage_accepted") is not True:
        raise CandidateEvidenceError("candidate failed the collateral-damage budget")
    if data.get("efficacy_accepted") is not True:
        raise CandidateEvidenceError("candidate failed the refusal-removal budget")
    if data.get("violations"):
        raise CandidateEvidenceError("candidate assessment contains budget violations")
    if data.get("inconclusive"):
        raise CandidateEvidenceError("candidate assessment is inconclusive")

    metrics = data.get("metrics")
    if not isinstance(metrics, Mapping):
        raise CandidateEvidenceError("candidate assessment metrics are missing")
    missing = [name for name in REQUIRED_ACCEPTANCE_METRICS if metrics.get(name) is None]
    if missing:
        raise CandidateEvidenceError(
            "candidate assessment is missing required metrics: " + ", ".join(missing)
        )
    numeric_metrics = {
        name: _finite_float(metrics[name], name=name)
        for name in REQUIRED_ACCEPTANCE_METRICS
    }
    for count_name in (
        "eval_prompt_count",
        "eval_token_count",
        "sampled_token_count",
        "new_degenerate_count",
        "nonfinite_output_count",
        "refusal_eval_count",
    ):
        _nonnegative_int(metrics[count_name], name=count_name)

    for probability_name in ("top1_flip_rate", "refusal_rate"):
        if not 0.0 <= numeric_metrics[probability_name] <= 1.0:
            raise CandidateEvidenceError(f"{probability_name} must be between 0 and 1")
    for nonnegative_name in (
        "sampled_token_kl_upper_ci",
        "sampled_token_kl_p95",
    ):
        if numeric_metrics[nonnegative_name] < 0.0:
            raise CandidateEvidenceError(f"{nonnegative_name} cannot be negative")
    if not -1.0 <= numeric_metrics["coherence_drop"] <= 1.0:
        raise CandidateEvidenceError("coherence_drop must be between -1 and 1")

    budget = data.get("budget")
    if not isinstance(budget, Mapping):
        raise CandidateEvidenceError("candidate assessment budget is missing")
    damage_budget = budget.get("damage")
    efficacy_budget = budget.get("efficacy")
    if not isinstance(damage_budget, Mapping) or not isinstance(
        efficacy_budget,
        Mapping,
    ):
        raise CandidateEvidenceError("candidate assessment budget is incomplete")

    for metric_name, minimum_name in (
        ("eval_prompt_count", "min_eval_prompts"),
        ("eval_token_count", "min_eval_tokens"),
        ("sampled_token_count", "min_sampled_tokens"),
    ):
        minimum = damage_budget.get(minimum_name)
        minimum_value = _nonnegative_int(minimum, name=minimum_name)
        if minimum_value < 1:
            raise CandidateEvidenceError(f"declared minimum {minimum_name} is invalid")
        if metrics[metric_name] < minimum_value:
            raise CandidateEvidenceError(
                f"candidate does not satisfy declared evidence minimum {minimum_name}"
            )

    for metric_name, limit_name, _ in _DAMAGE_LIMITS:
        limit = damage_budget.get(limit_name)
        if limit is None:
            raise CandidateEvidenceError(
                f"candidate orchestration requires enabled limit {limit_name}"
            )
        limit_value = _finite_float(limit, name=limit_name)
        if limit_value < 0.0:
            raise CandidateEvidenceError(f"declared limit {limit_name} is invalid")
        if numeric_metrics[metric_name] > limit_value:
            raise CandidateEvidenceError(
                f"candidate exceeds declared limit {limit_name}"
            )

    refusal_limit = efficacy_budget.get("max_refusal_rate")
    refusal_minimum = efficacy_budget.get("min_eval_prompts")
    if refusal_limit is None or refusal_minimum is None:
        raise CandidateEvidenceError("candidate efficacy budget is incomplete")
    refusal_limit_value = _finite_float(
        refusal_limit,
        name="max_refusal_rate",
    )
    if not 0.0 <= refusal_limit_value <= 1.0:
        raise CandidateEvidenceError("declared max_refusal_rate is invalid")
    refusal_minimum_value = _nonnegative_int(
        refusal_minimum,
        name="efficacy min_eval_prompts",
    )
    if refusal_minimum_value < 1:
        raise CandidateEvidenceError("declared efficacy min_eval_prompts is invalid")
    if numeric_metrics["refusal_rate"] > refusal_limit_value:
        raise CandidateEvidenceError("candidate exceeds its refusal-rate limit")
    if metrics["refusal_eval_count"] < refusal_minimum_value:
        raise CandidateEvidenceError("candidate refusal evaluation is undersized")
    return data


def damage_severity(payload: Mapping[str, Any]) -> float:
    """Return mean fraction of the declared collateral-damage budget consumed.

    Zero is best.  One means that, on average, the accepted candidate sits at
    the limits of its declared budget.  Improvements over baseline are clipped
    to zero rather than rewarded, so efficacy cannot hide collateral damage.
    """

    data = validate_acceptance_payload(payload)
    metrics = data["metrics"]
    damage_budget = data["budget"]["damage"]

    fractions: list[float] = []
    for metric_name, limit_name, fallback_limit in _DAMAGE_LIMITS:
        value = max(0.0, float(metrics[metric_name]))
        limit = damage_budget.get(limit_name, fallback_limit)
        if limit is None:
            limit = fallback_limit
        limit_value = float(limit)
        if limit_value == 0.0:
            fractions.append(0.0 if value == 0.0 else float("inf"))
        else:
            fractions.append(value / limit_value)
    return sum(fractions) / len(fractions)


def add_acceptance_evidence(
    metrics: Mapping[str, Any],
    assessment: Any,
) -> dict[str, Any]:
    """Copy quality metrics and attach validated, serializable gate evidence."""

    payload = acceptance_payload(assessment)
    enriched = dict(metrics)
    for metric_name in REQUIRED_ACCEPTANCE_METRICS:
        enriched[metric_name] = payload["metrics"][metric_name]
    enriched["acceptance"] = payload
    enriched["acceptance_passed"] = True
    enriched["damage_accepted"] = True
    enriched["efficacy_accepted"] = True
    enriched["damage_severity"] = damage_severity(payload)
    return enriched


__all__ = [
    "REQUIRED_ACCEPTANCE_METRICS",
    "CandidateEvidenceError",
    "acceptance_payload",
    "add_acceptance_evidence",
    "damage_severity",
    "validate_acceptance_payload",
]
