"""Measured KL-preservation evaluation for causal language-model candidates.

This module deliberately separates *measurement and selection* from model
editing.  It compares an untouched baseline with already-created candidates on
explicitly supplied benign token batches.  A response-token mask identifies
which next-token distributions belong to the benign response and therefore
count toward the preservation decision.

The default divergence is forward KL, ``KL(baseline || candidate)``.  Reverse
KL and the half-Jeffreys symmetric form
``0.5 * (KL(baseline || candidate) + KL(candidate || baseline))`` are available
for experiments, but callers should declare the direction before evaluating
candidates and use the same held-out batches and thresholds for every candidate.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

KLDirection = Literal["forward", "reverse", "symmetric"]
_VALID_DIRECTIONS = frozenset({"forward", "reverse", "symmetric"})
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


class KLPreservationError(RuntimeError):
    """Raised when a model pair cannot produce a trustworthy KL measurement."""


@dataclass(frozen=True)
class CausalLMBatch:
    """A tokenized benign batch with an explicit response-token mask.

    ``response_mask[b, t] == 1`` means token ``input_ids[b, t]`` is part of
    the response to preserve.  Because causal logits at position ``t - 1``
    predict token ``t``, the evaluator aligns ``response_mask[:, 1:]`` with
    ``logits[:, :-1]``.  Attention-masked padding and transitions across
    padding are always excluded.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor


@dataclass(frozen=True)
class KLPreservationThresholds:
    """Hard candidate-acceptance limits in nats per selected response token."""

    max_mean_kl: float
    max_p95_kl: float

    def __post_init__(self) -> None:
        for name, value in (
            ("max_mean_kl", self.max_mean_kl),
            ("max_p95_kl", self.max_p95_kl),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite non-negative number")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class KLPreservationMetrics:
    """Aggregate response-token KL statistics for one candidate.

    ``p95_kl`` uses the conservative nearest-rank definition over individual
    selected response-token divergences.
    """

    direction: KLDirection
    mean_kl: float
    p95_kl: float
    token_count: int
    batch_count: int


@dataclass(frozen=True)
class CandidateKLResult:
    """One candidate's measured preservation result and hard-gate decision."""

    name: str
    efficacy: float
    accepted: bool
    metrics: KLPreservationMetrics | None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class KLCandidateSelection:
    """Deterministic result of KL-gating and ranking candidate models.

    ``eligible`` is ordered by lower mean KL, then lower p95 KL, then higher
    caller-supplied efficacy, and finally candidate name.  ``selected_name`` is
    ``None`` when every candidate fails closed.
    """

    selected_name: str | None
    eligible: tuple[CandidateKLResult, ...]
    rejected: tuple[CandidateKLResult, ...]
    thresholds: KLPreservationThresholds
    direction: KLDirection


@dataclass(frozen=True)
class _ValidatedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    prediction_mask: torch.Tensor


def _validate_direction(direction: str) -> KLDirection:
    if direction not in _VALID_DIRECTIONS:
        choices = ", ".join(sorted(_VALID_DIRECTIONS))
        raise ValueError(f"direction must be one of: {choices}")
    return direction  # type: ignore[return-value]


def _binary_mask(name: str, value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if value.dtype != torch.bool and value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use a boolean or integer dtype")

    cpu_value = value.detach().to("cpu").clone()
    if not bool(((cpu_value == 0) | (cpu_value == 1)).all().item()):
        raise ValueError(f"{name} must be binary")
    return cpu_value.bool()


def _validate_batch(batch: CausalLMBatch | Mapping[str, torch.Tensor]) -> _ValidatedBatch:
    if isinstance(batch, CausalLMBatch):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        response_mask = batch.response_mask
    elif isinstance(batch, Mapping):
        missing = {
            name for name in ("input_ids", "attention_mask", "response_mask") if name not in batch
        }
        if missing:
            raise ValueError(
                "benign token batch is missing required fields: " + ", ".join(sorted(missing))
            )
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
    else:
        raise TypeError("each benign batch must be CausalLMBatch or a tensor mapping")

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.shape[0] <= 0 or input_ids.shape[1] < 2:
        raise ValueError("each benign batch needs at least one row and two tokens")
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("input_ids must use an integer dtype")
    if bool((input_ids < 0).any().item()):
        raise ValueError("input_ids cannot contain negative token ids")

    cpu_input_ids = input_ids.detach().to(device="cpu", dtype=torch.long).clone()
    cpu_attention = _binary_mask("attention_mask", attention_mask, input_ids.shape)
    cpu_response = _binary_mask("response_mask", response_mask, input_ids.shape)
    if bool((cpu_response & ~cpu_attention).any().item()):
        raise ValueError("response_mask cannot select attention-masked padding")

    # logits[:, t] predict input_ids[:, t + 1].  Both sides of the transition
    # must be real tokens, and the predicted token must be an explicit response
    # token.  This works for either left or right padding.
    prediction_mask = cpu_response[:, 1:] & cpu_attention[:, 1:] & cpu_attention[:, :-1]
    per_row_counts = prediction_mask.sum(dim=1)
    if bool((per_row_counts == 0).any().item()):
        raise ValueError("every benign batch row must contain a predicted response token")

    return _ValidatedBatch(
        input_ids=cpu_input_ids,
        attention_mask=cpu_attention,
        response_mask=cpu_response,
        prediction_mask=prediction_mask,
    )


def _materialize_batches(
    batches: Iterable[CausalLMBatch | Mapping[str, torch.Tensor]],
) -> tuple[_ValidatedBatch, ...]:
    if isinstance(batches, (torch.Tensor, Mapping, CausalLMBatch)):
        raise TypeError("benign_batches must be an iterable of token batches")
    try:
        raw_batches = tuple(batches)
    except TypeError as exc:
        raise TypeError("benign_batches must be an iterable of token batches") from exc
    if not raw_batches:
        raise ValueError("at least one benign token batch is required")
    return tuple(_validate_batch(batch) for batch in raw_batches)


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    declared_device = getattr(model, "device", None)
    if declared_device is not None:
        device = torch.device(declared_device)
        if device.type != "meta":
            return device
    for tensor in chain(model.parameters(), model.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    return fallback


def _extract_logits(outputs: object) -> torch.Tensor:
    logits: object
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    elif isinstance(outputs, Mapping):
        if "logits" not in outputs:
            raise KLPreservationError("model output mapping has no 'logits' field")
        logits = outputs["logits"]
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    else:
        raise KLPreservationError("model output does not expose causal-LM logits")

    if not isinstance(logits, torch.Tensor):
        raise KLPreservationError("model logits must be a torch.Tensor")
    return logits


def _selected_logit_rows(model: nn.Module, batch: _ValidatedBatch) -> torch.Tensor:
    device = _model_device(model, batch.input_ids.device)
    input_ids = batch.input_ids.to(device=device).clone()
    attention_mask = batch.attention_mask.to(device=device, dtype=torch.long).clone()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = _extract_logits(outputs)

    expected_prefix = tuple(batch.input_ids.shape)
    if logits.ndim != 3 or tuple(logits.shape[:2]) != expected_prefix:
        raise KLPreservationError(
            "model logits must have shape [batch, sequence, vocabulary] matching input_ids"
        )
    if logits.shape[-1] <= 0:
        raise KLPreservationError("model logits have an empty vocabulary dimension")
    if not logits.is_floating_point():
        raise KLPreservationError("model logits must use a floating-point dtype")

    mask = batch.prediction_mask.to(logits.device)
    rows = logits[:, :-1, :][mask].detach().to(device="cpu", dtype=torch.float32)
    if rows.shape[0] <= 0:
        raise KLPreservationError("response-token masking selected no causal logits")
    if not bool(torch.isfinite(rows).all().item()):
        raise KLPreservationError("selected response-token logits contain non-finite values")
    return rows


@contextmanager
def _evaluation_mode(*models: nn.Module):
    unique_modules: dict[int, nn.Module] = {}
    for model in models:
        if not isinstance(model, nn.Module):
            raise TypeError("baseline and candidate models must be torch.nn.Module instances")
        for module in model.modules():
            unique_modules.setdefault(id(module), module)

    training_states = [(module, module.training) for module in unique_modules.values()]
    try:
        for model in models:
            model.eval()
        with torch.no_grad():
            yield
    finally:
        # Restore heterogeneous submodule states exactly without recursively
        # overwriting child state through Module.train().
        for module, was_training in training_states:
            module.training = was_training


def _token_kl(
    baseline_rows: torch.Tensor,
    candidate_rows: torch.Tensor,
    direction: KLDirection,
) -> torch.Tensor:
    if baseline_rows.shape != candidate_rows.shape:
        raise KLPreservationError(
            "baseline and candidate selected logits must have identical shapes"
        )
    if baseline_rows.ndim != 2 or baseline_rows.shape[-1] <= 0:
        raise KLPreservationError("selected logits must have shape [tokens, vocabulary]")

    baseline_log_prob = F.log_softmax(baseline_rows, dim=-1)
    candidate_log_prob = F.log_softmax(candidate_rows, dim=-1)
    forward = (baseline_log_prob.exp() * (baseline_log_prob - candidate_log_prob)).sum(dim=-1)
    if direction == "forward":
        values = forward
    else:
        reverse = (candidate_log_prob.exp() * (candidate_log_prob - baseline_log_prob)).sum(dim=-1)
        values = reverse if direction == "reverse" else 0.5 * (forward + reverse)

    values = values.clamp_min(0.0)
    if not bool(torch.isfinite(values).all().item()):
        raise KLPreservationError("response-token KL contains non-finite values")
    return values


def _nearest_rank_p95(values: list[float]) -> float:
    if not values:
        raise KLPreservationError("KL evaluation produced no response-token measurements")
    ordered = sorted(values)
    rank = max(1, math.ceil(0.95 * len(ordered)))
    return ordered[rank - 1]


def _evaluate_validated_batches(
    baseline_model: nn.Module,
    candidate_model: nn.Module,
    batches: tuple[_ValidatedBatch, ...],
    direction: KLDirection,
) -> KLPreservationMetrics:
    per_token: list[float] = []
    with _evaluation_mode(baseline_model, candidate_model):
        for batch in batches:
            baseline_rows = _selected_logit_rows(baseline_model, batch)
            candidate_rows = _selected_logit_rows(candidate_model, batch)
            per_token.extend(_token_kl(baseline_rows, candidate_rows, direction).tolist())

    if not per_token or any(not math.isfinite(value) for value in per_token):
        raise KLPreservationError("KL evaluation produced empty or non-finite measurements")
    mean_kl = math.fsum(per_token) / len(per_token)
    p95_kl = _nearest_rank_p95(per_token)
    if not math.isfinite(mean_kl) or not math.isfinite(p95_kl):
        raise KLPreservationError("aggregate KL statistics are non-finite")
    return KLPreservationMetrics(
        direction=direction,
        mean_kl=mean_kl,
        p95_kl=p95_kl,
        token_count=len(per_token),
        batch_count=len(batches),
    )


def evaluate_kl_preservation(
    baseline_model: nn.Module,
    candidate_model: nn.Module,
    benign_batches: Iterable[CausalLMBatch | Mapping[str, torch.Tensor]],
    *,
    direction: KLDirection = "forward",
) -> KLPreservationMetrics:
    """Measure baseline-vs-candidate KL on explicit benign response tokens.

    The function performs both forwards under ``torch.no_grad()``, temporarily
    switches the models to evaluation mode, restores every module's original
    training flag, and never writes model parameters.  Malformed batches,
    empty response masks, incompatible logits, and non-finite selected outputs
    raise instead of producing a permissive score.
    """

    validated_direction = _validate_direction(direction)
    batches = _materialize_batches(benign_batches)
    return _evaluate_validated_batches(
        baseline_model,
        candidate_model,
        batches,
        validated_direction,
    )


def select_kl_preserving_candidate(
    baseline_model: nn.Module,
    candidates: Mapping[str, nn.Module],
    benign_batches: Iterable[CausalLMBatch | Mapping[str, torch.Tensor]],
    *,
    efficacy: Mapping[str, float],
    thresholds: KLPreservationThresholds,
    direction: KLDirection = "forward",
) -> KLCandidateSelection:
    """Gate and deterministically rank already-created candidate models.

    Only candidates satisfying both ``mean_kl <= max_mean_kl`` and
    ``p95_kl <= max_p95_kl`` are eligible.  Eligible candidates are ranked by
    lower mean KL, lower p95 KL, higher caller-supplied efficacy, then name.
    Candidate forward failures and non-finite outputs are recorded as hard
    rejections.  This function only evaluates models; it never copies, creates,
    interpolates, rolls back, or edits their weights.
    """

    if not isinstance(candidates, Mapping):
        raise TypeError("candidates must be a mapping from name to torch.nn.Module")
    if not candidates:
        raise ValueError("at least one candidate model is required")
    if not isinstance(baseline_model, nn.Module):
        raise TypeError("baseline_model must be a torch.nn.Module")
    if not isinstance(efficacy, Mapping):
        raise TypeError("efficacy must be a mapping from candidate name to score")
    if not isinstance(thresholds, KLPreservationThresholds):
        raise TypeError("thresholds must be KLPreservationThresholds")

    names = set(candidates)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("candidate names must be non-empty strings")
    efficacy_names = set(efficacy)
    if efficacy_names != names:
        missing = sorted(map(str, names - efficacy_names))
        extra = sorted(map(repr, efficacy_names - names))
        details = []
        if missing:
            details.append("missing efficacy for " + ", ".join(missing))
        if extra:
            details.append("unknown efficacy entries " + ", ".join(extra))
        raise ValueError("; ".join(details))

    validated_efficacy: dict[str, float] = {}
    for name in sorted(names):
        score = efficacy[name]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"efficacy for {name!r} must be a finite number")
        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"efficacy for {name!r} must be finite")
        validated_efficacy[name] = numeric_score

    validated_direction = _validate_direction(direction)
    batches = _materialize_batches(benign_batches)
    accepted: list[CandidateKLResult] = []
    rejected: list[CandidateKLResult] = []

    for name in sorted(names):
        candidate = candidates[name]
        score = validated_efficacy[name]
        try:
            metrics = _evaluate_validated_batches(
                baseline_model,
                candidate,
                batches,
                validated_direction,
            )
        except Exception as exc:  # noqa: BLE001 - any failed forward must reject
            # Candidate evaluation is a gate: an exception cannot promote the
            # candidate.  Keep a concise reason so callers can investigate.
            rejected.append(
                CandidateKLResult(
                    name=name,
                    efficacy=score,
                    accepted=False,
                    metrics=None,
                    rejection_reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        violations = []
        if metrics.mean_kl > float(thresholds.max_mean_kl):
            violations.append(f"mean KL {metrics.mean_kl:.6g} exceeds {thresholds.max_mean_kl:.6g}")
        if metrics.p95_kl > float(thresholds.max_p95_kl):
            violations.append(f"p95 KL {metrics.p95_kl:.6g} exceeds {thresholds.max_p95_kl:.6g}")

        result = CandidateKLResult(
            name=name,
            efficacy=score,
            accepted=not violations,
            metrics=metrics,
            rejection_reason="; ".join(violations) if violations else None,
        )
        (accepted if result.accepted else rejected).append(result)

    accepted.sort(
        key=lambda result: (
            result.metrics.mean_kl if result.metrics is not None else math.inf,
            result.metrics.p95_kl if result.metrics is not None else math.inf,
            -result.efficacy,
            result.name,
        )
    )
    rejected.sort(key=lambda result: result.name)
    return KLCandidateSelection(
        selected_name=accepted[0].name if accepted else None,
        eligible=tuple(accepted),
        rejected=tuple(rejected),
        thresholds=thresholds,
        direction=validated_direction,
    )


__all__ = [
    "CandidateKLResult",
    "CausalLMBatch",
    "KLCandidateSelection",
    "KLDirection",
    "KLPreservationError",
    "KLPreservationMetrics",
    "KLPreservationThresholds",
    "evaluate_kl_preservation",
    "select_kl_preserving_candidate",
]
