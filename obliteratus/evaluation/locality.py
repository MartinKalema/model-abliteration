"""Model-agnostic measurements of collateral distribution drift.

The functions in this module operate on logits that have already been
produced for the *same* encoded benign prompts by an untouched model and an
edited candidate.  Keeping model execution outside this module makes the
measurement logic inexpensive to unit test and reusable across pipeline
variants.

All language-model losses use the usual causal shift.  A prediction is only
counted when both its context position and target position are real tokens, so
left and right padding cannot affect the result.  KL is measured at a small,
deterministic set of real sequence positions, including the final prompt token
that predicts the first completion token.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from obliteratus.evaluation.damage_gate import (
    paired_bootstrap_upper_bound,
    weighted_paired_bootstrap_upper_bound,
)


@dataclass(frozen=True)
class PromptLocalityArtifacts:
    """Raw, serializable measurements for one paired prompt."""

    prompt_index: int
    token_count: int
    baseline_loss_sum: float
    candidate_loss_sum: float
    loss_delta_sum: float
    sampled_positions: tuple[int, ...]
    sampled_token_kl: tuple[float, ...]
    sampled_token_kl_mean: float | None
    top1_flip_count: int
    baseline_nonfinite_logit_count: int
    candidate_nonfinite_logit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_index": self.prompt_index,
            "token_count": self.token_count,
            "baseline_loss_sum": self.baseline_loss_sum,
            "candidate_loss_sum": self.candidate_loss_sum,
            "loss_delta_sum": self.loss_delta_sum,
            "sampled_positions": list(self.sampled_positions),
            "sampled_token_kl": list(self.sampled_token_kl),
            "sampled_token_kl_mean": self.sampled_token_kl_mean,
            "top1_flip_count": self.top1_flip_count,
            "baseline_nonfinite_logit_count": self.baseline_nonfinite_logit_count,
            "candidate_nonfinite_logit_count": self.candidate_nonfinite_logit_count,
        }


@dataclass(frozen=True)
class LocalityMeasurement:
    """Gate-ready summary metrics plus auditable per-prompt artifacts."""

    metrics: dict[str, float | int]
    prompts: tuple[PromptLocalityArtifacts, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "prompts": [prompt.to_dict() for prompt in self.prompts],
        }


@dataclass(frozen=True)
class BaselinePromptArtifacts:
    """Compact untouched-model state retained across an in-place edit.

    ``sampled_logits`` contains only the selected full-vocabulary rows and is
    always detached CPU FP16 storage.  It intentionally does not retain the
    full baseline ``[sequence, vocabulary]`` tensor.
    """

    prompt_index: int
    token_count: int
    baseline_loss_sum: float
    sampled_positions: tuple[int, ...]
    sampled_logits: torch.Tensor
    baseline_nonfinite_logit_count: int

    def metadata_dict(self) -> dict[str, Any]:
        """Return JSON-safe provenance without embedding large logit arrays."""

        return {
            "prompt_index": self.prompt_index,
            "token_count": self.token_count,
            "baseline_loss_sum": self.baseline_loss_sum,
            "sampled_positions": list(self.sampled_positions),
            "sampled_logit_shape": list(self.sampled_logits.shape),
            "sampled_logit_dtype": str(self.sampled_logits.dtype),
            "baseline_nonfinite_logit_count": self.baseline_nonfinite_logit_count,
        }


@dataclass(frozen=True)
class LocalityBaseline:
    """Memory-bounded baseline capture for post-edit candidate comparison."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompts: tuple[BaselinePromptArtifacts, ...]
    vocab_size: int

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "input_shape": list(self.input_ids.shape),
            "input_ids_dtype": str(self.input_ids.dtype),
            "attention_mask_dtype": str(self.attention_mask.dtype),
            "vocab_size": self.vocab_size,
            "prompts": [prompt.metadata_dict() for prompt in self.prompts],
        }


def select_evenly_spaced_positions(
    attention_mask: torch.Tensor,
    max_positions: int,
) -> tuple[int, ...]:
    """Select deterministic real-token positions, always including the last.

    The selection is based on ranks within the valid positions rather than on
    raw sequence offsets, so it also behaves correctly with left padding or a
    non-contiguous mask.  When only one position is requested, the final real
    prompt position is selected because it is the position that predicts the
    first completion token.
    """

    if attention_mask.ndim != 1:
        raise ValueError("attention_mask must be one-dimensional")
    if not isinstance(max_positions, int) or isinstance(max_positions, bool):
        raise TypeError("max_positions must be an integer")
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")

    valid_positions = torch.nonzero(attention_mask.detach().to("cpu").bool()).flatten()
    valid = [int(position) for position in valid_positions.tolist()]
    if not valid:
        return ()

    count = min(max_positions, len(valid))
    if count == 1:
        return (valid[-1],)
    if count == len(valid):
        return tuple(valid)

    # Round half up in integer arithmetic.  Since count <= len(valid), the
    # selected ranks are strictly increasing and include both endpoints.
    denominator = count - 1
    last_rank = len(valid) - 1
    ranks = [(index * last_rank + denominator // 2) // denominator for index in range(count)]
    return tuple(valid[rank] for rank in ranks)


def _validate_inputs(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    if baseline_logits.ndim != 3 or candidate_logits.ndim != 3:
        raise ValueError("baseline_logits and candidate_logits must have shape [batch, seq, vocab]")
    if baseline_logits.shape != candidate_logits.shape:
        raise ValueError("baseline_logits and candidate_logits must have identical shapes")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must have shape [batch, seq]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have identical shapes")
    if baseline_logits.shape[:2] != input_ids.shape:
        raise ValueError("logit batch and sequence dimensions must match input_ids")
    if baseline_logits.shape[-1] <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if baseline_logits.shape[0] <= 0:
        raise ValueError("at least one prompt is required")


def _causal_loss_sum(
    prompt_logits: torch.Tensor,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
) -> tuple[float, int]:
    """Return summed causal NLL and the number of real predicted tokens."""

    mask = prompt_attention_mask.detach().bool()
    prediction_mask = mask[:-1] & mask[1:]
    positions = torch.nonzero(prediction_mask).flatten()
    token_count = int(positions.numel())
    if token_count == 0:
        return 0.0, 0

    positions_on_logits = positions.to(prompt_logits.device)
    label_positions = (positions + 1).to(prompt_input_ids.device)
    labels = prompt_input_ids[label_positions].to(prompt_logits.device, dtype=torch.long)
    selected_logits = prompt_logits.index_select(0, positions_on_logits).float()
    loss = F.cross_entropy(selected_logits, labels, reduction="sum")
    return float(loss.detach().cpu()), token_count


def _forward_kl_rows(
    baseline_rows: torch.Tensor,
    candidate_rows: torch.Tensor,
) -> tuple[list[float], int]:
    """Compute FP32 forward KL(base || candidate) for paired logit rows."""

    base = baseline_rows.float()
    candidate = candidate_rows.to(baseline_rows.device).float()
    finite_rows = torch.isfinite(base).all(dim=-1) & torch.isfinite(candidate).all(dim=-1)

    values: list[float] = []
    for row_index in range(base.shape[0]):
        if not bool(finite_rows[row_index].item()):
            values.append(float("inf"))
            continue
        base_log_prob = F.log_softmax(base[row_index], dim=-1)
        candidate_log_prob = F.log_softmax(candidate[row_index], dim=-1)
        kl = torch.sum(base_log_prob.exp() * (base_log_prob - candidate_log_prob))
        # Tiny negative values can occur from FP32 round-off; mathematical KL
        # is non-negative.
        values.append(max(0.0, float(kl.detach().cpu())))

    return values, int((~finite_rows).sum().item())


def _nearest_rank_percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("at least one value is required")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def _summarize_prompt_artifacts(
    prompts: Sequence[PromptLocalityArtifacts],
    *,
    bootstrap_confidence: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> LocalityMeasurement:
    loss_prompts = [prompt for prompt in prompts if prompt.token_count > 0]
    if not loss_prompts:
        raise ValueError("the measurements contain no real causal target tokens")

    kl_prompts = [
        prompt
        for prompt in prompts
        if prompt.sampled_token_kl_mean is not None and prompt.sampled_token_kl
    ]
    if not kl_prompts:
        raise ValueError("the measurements contain no valid KL sampling positions")

    loss_delta_sums = [prompt.loss_delta_sum for prompt in loss_prompts]
    loss_token_counts = [prompt.token_count for prompt in loss_prompts]
    all_kl_values = [value for prompt in kl_prompts for value in prompt.sampled_token_kl]
    prompt_kl_means = [float(prompt.sampled_token_kl_mean) for prompt in kl_prompts]
    total_tokens = sum(loss_token_counts)
    total_baseline_loss = sum(prompt.baseline_loss_sum for prompt in loss_prompts)
    total_candidate_loss = sum(prompt.candidate_loss_sum for prompt in loss_prompts)
    baseline_nll = total_baseline_loss / total_tokens
    candidate_nll = total_candidate_loss / total_tokens
    nll_increase = candidate_nll - baseline_nll
    nll_upper_ci = weighted_paired_bootstrap_upper_bound(
        loss_delta_sums,
        loss_token_counts,
        confidence=bootstrap_confidence,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    kl_upper_ci = paired_bootstrap_upper_bound(
        prompt_kl_means,
        confidence=bootstrap_confidence,
        n_resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )

    sampled_token_count = len(all_kl_values)
    baseline_nonfinite_count = sum(prompt.baseline_nonfinite_logit_count for prompt in prompts)
    candidate_nonfinite_count = sum(prompt.candidate_nonfinite_logit_count for prompt in prompts)
    metrics: dict[str, float | int] = {
        "baseline_nll": baseline_nll,
        "candidate_nll": candidate_nll,
        "nll_increase": nll_increase,
        "nll_increase_upper_ci": nll_upper_ci,
        "perplexity_ratio": _safe_exp(nll_increase),
        "sampled_token_kl_mean": sum(all_kl_values) / sampled_token_count,
        "sampled_token_kl_upper_ci": kl_upper_ci,
        "sampled_token_kl_p95": _nearest_rank_percentile(all_kl_values, 0.95),
        "top1_flip_rate": (sum(prompt.top1_flip_count for prompt in prompts) / sampled_token_count),
        # A broken baseline makes the paired experiment invalid too, so the
        # gate-facing hard-failure metric counts non-finite values on either side.
        "nonfinite_output_count": baseline_nonfinite_count + candidate_nonfinite_count,
        "baseline_nonfinite_logit_count": baseline_nonfinite_count,
        "candidate_nonfinite_logit_count": candidate_nonfinite_count,
        "eval_prompt_count": len(loss_prompts),
        "eval_token_count": total_tokens,
        "sampled_token_count": sampled_token_count,
    }
    return LocalityMeasurement(metrics=metrics, prompts=tuple(prompts))


def capture_locality_baseline(
    baseline_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_kl_positions_per_prompt: int = 8,
) -> LocalityBaseline:
    """Capture only the untouched state needed after an in-place model edit.

    The returned object owns CPU copies of the exact encoded inputs and masks,
    per-prompt baseline loss totals, and at most
    ``max_kl_positions_per_prompt`` full-vocabulary FP16 rows per prompt.  The
    full baseline logit tensor is never retained.
    """

    if baseline_logits.ndim != 3:
        raise ValueError("baseline_logits must have shape [batch, seq, vocab]")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must have shape [batch, seq]")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have identical shapes")
    if baseline_logits.shape[:2] != input_ids.shape:
        raise ValueError("logit batch and sequence dimensions must match input_ids")
    if baseline_logits.shape[-1] <= 0:
        raise ValueError("logits must have a non-empty vocabulary dimension")
    if baseline_logits.shape[0] <= 0:
        raise ValueError("at least one prompt is required")
    if not isinstance(max_kl_positions_per_prompt, int) or isinstance(
        max_kl_positions_per_prompt, bool
    ):
        raise TypeError("max_kl_positions_per_prompt must be an integer")
    if max_kl_positions_per_prompt <= 0:
        raise ValueError("max_kl_positions_per_prompt must be positive")

    cpu_input_ids = input_ids.detach().to("cpu").clone()
    cpu_attention_mask = attention_mask.detach().to("cpu").clone()
    prompt_artifacts: list[BaselinePromptArtifacts] = []
    total_loss_tokens = 0
    total_sampled_positions = 0

    for prompt_index in range(cpu_input_ids.shape[0]):
        mask = cpu_attention_mask[prompt_index]
        baseline_loss_sum, token_count = _causal_loss_sum(
            baseline_logits[prompt_index], cpu_input_ids[prompt_index], mask
        )
        sampled_positions = select_evenly_spaced_positions(mask, max_kl_positions_per_prompt)
        sample_index = torch.tensor(
            sampled_positions, device=baseline_logits.device, dtype=torch.long
        )
        sampled_logits = (
            baseline_logits[prompt_index]
            .index_select(0, sample_index)
            .detach()
            .to(device="cpu", dtype=torch.float16)
            .clone()
        )
        valid_positions = torch.nonzero(mask.bool()).flatten().to(baseline_logits.device)
        valid_logits = baseline_logits[prompt_index].index_select(0, valid_positions)
        nonfinite_count = int((~torch.isfinite(valid_logits)).sum().item())
        prompt_artifacts.append(
            BaselinePromptArtifacts(
                prompt_index=prompt_index,
                token_count=token_count,
                baseline_loss_sum=baseline_loss_sum,
                sampled_positions=sampled_positions,
                sampled_logits=sampled_logits,
                baseline_nonfinite_logit_count=nonfinite_count,
            )
        )
        total_loss_tokens += token_count
        total_sampled_positions += len(sampled_positions)

    if total_loss_tokens == 0:
        raise ValueError("the batch contains no real causal target tokens")
    if total_sampled_positions == 0:
        raise ValueError("the batch contains no valid KL sampling positions")

    return LocalityBaseline(
        input_ids=cpu_input_ids,
        attention_mask=cpu_attention_mask,
        prompts=tuple(prompt_artifacts),
        vocab_size=int(baseline_logits.shape[-1]),
    )


def compare_locality_candidate(
    baseline: LocalityBaseline,
    candidate_logits: torch.Tensor,
    *,
    bootstrap_confidence: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 42,
) -> LocalityMeasurement:
    """Compare later candidate logits with a compact pre-edit baseline."""

    expected_shape = (
        baseline.input_ids.shape[0],
        baseline.input_ids.shape[1],
        baseline.vocab_size,
    )
    if candidate_logits.ndim != 3 or tuple(candidate_logits.shape) != expected_shape:
        raise ValueError(
            f"candidate_logits must match the baseline [batch, seq, vocab] shape {expected_shape}"
        )
    if len(baseline.prompts) != baseline.input_ids.shape[0]:
        raise ValueError("baseline prompt artifacts do not match the captured batch")

    prompt_artifacts: list[PromptLocalityArtifacts] = []

    for baseline_prompt in baseline.prompts:
        prompt_index = baseline_prompt.prompt_index
        mask = baseline.attention_mask[prompt_index]
        candidate_loss_sum, token_count = _causal_loss_sum(
            candidate_logits[prompt_index], baseline.input_ids[prompt_index], mask
        )
        if token_count != baseline_prompt.token_count:
            raise ValueError("captured baseline token count does not match its attention mask")

        loss_delta_sum = candidate_loss_sum - baseline_prompt.baseline_loss_sum

        sampled_positions = baseline_prompt.sampled_positions
        candidate_index = torch.tensor(
            sampled_positions, device=candidate_logits.device, dtype=torch.long
        )
        baseline_rows = baseline_prompt.sampled_logits
        # The compact baseline is intentionally stored as FP16.  Quantize the
        # candidate rows the same way before computing KL/top-1 so baseline
        # compression cannot create one-sided, false damage on a no-op run.
        candidate_rows = (
            candidate_logits[prompt_index]
            .index_select(0, candidate_index)
            .detach()
            .to(device="cpu", dtype=baseline_rows.dtype)
        )
        if candidate_rows.shape != baseline_rows.shape:
            raise ValueError("candidate sampled rows do not match the baseline vocabulary")
        kl_values, _ = _forward_kl_rows(baseline_rows, candidate_rows)

        baseline_argmax = baseline_rows.argmax(dim=-1)
        candidate_argmax = candidate_rows.argmax(dim=-1).detach().to("cpu")
        finite_pairs = torch.isfinite(baseline_rows).all(dim=-1) & (
            torch.isfinite(candidate_rows).all(dim=-1).detach().to("cpu")
        )
        # A non-finite row is structurally invalid and conservatively also
        # counts as a changed top prediction.
        prompt_flips = int(((baseline_argmax != candidate_argmax) | ~finite_pairs).sum())
        if not kl_values:
            prompt_kl_mean = None
        elif all(map(math.isfinite, kl_values)):
            prompt_kl_mean = sum(kl_values) / len(kl_values)
        else:
            prompt_kl_mean = float("inf")

        valid_positions = torch.nonzero(mask.bool()).flatten().to(candidate_logits.device)
        candidate_valid = candidate_logits[prompt_index].index_select(0, valid_positions)
        prompt_candidate_nonfinite = int((~torch.isfinite(candidate_valid)).sum().item())

        prompt_artifacts.append(
            PromptLocalityArtifacts(
                prompt_index=prompt_index,
                token_count=token_count,
                baseline_loss_sum=baseline_prompt.baseline_loss_sum,
                candidate_loss_sum=candidate_loss_sum,
                loss_delta_sum=loss_delta_sum,
                sampled_positions=sampled_positions,
                sampled_token_kl=tuple(kl_values),
                sampled_token_kl_mean=prompt_kl_mean,
                top1_flip_count=prompt_flips,
                baseline_nonfinite_logit_count=(baseline_prompt.baseline_nonfinite_logit_count),
                candidate_nonfinite_logit_count=prompt_candidate_nonfinite,
            )
        )

    return _summarize_prompt_artifacts(
        prompt_artifacts,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def combine_locality_measurements(
    measurements: Sequence[LocalityMeasurement],
    *,
    bootstrap_confidence: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 42,
) -> LocalityMeasurement:
    """Combine separately evaluated batches and recompute global statistics.

    This is intentionally not an average of batch-level metrics.  It pools the
    per-prompt paired loss totals and KL rows, then recomputes token weighting,
    percentiles, rates, and bootstrap bounds over the whole evaluation set.
    Prompt indices are renumbered to remain unique in the combined audit trail.
    """

    if not measurements:
        raise ValueError("at least one locality measurement is required")

    combined_prompts: list[PromptLocalityArtifacts] = []
    for measurement in measurements:
        if not isinstance(measurement, LocalityMeasurement):
            raise TypeError("measurements must contain LocalityMeasurement objects")
        for prompt in measurement.prompts:
            combined_prompts.append(replace(prompt, prompt_index=len(combined_prompts)))

    return _summarize_prompt_artifacts(
        combined_prompts,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def measure_locality(
    baseline_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_kl_positions_per_prompt: int = 8,
    bootstrap_confidence: float = 0.95,
    bootstrap_resamples: int = 2_000,
    bootstrap_seed: int = 42,
) -> LocalityMeasurement:
    """One-shot convenience wrapper around capture and later comparison."""

    _validate_inputs(baseline_logits, candidate_logits, input_ids, attention_mask)
    baseline = capture_locality_baseline(
        baseline_logits,
        input_ids,
        attention_mask,
        max_kl_positions_per_prompt=max_kl_positions_per_prompt,
    )
    return compare_locality_candidate(
        baseline,
        candidate_logits,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


__all__ = [
    "BaselinePromptArtifacts",
    "LocalityBaseline",
    "LocalityMeasurement",
    "PromptLocalityArtifacts",
    "capture_locality_baseline",
    "combine_locality_measurements",
    "compare_locality_candidate",
    "measure_locality",
    "select_evenly_spaced_positions",
]
