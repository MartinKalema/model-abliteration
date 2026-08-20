"""Teacher-forced preservation measurements for explicit reasoning traces.

This module deliberately does not discover or infer chain-of-thought.  Callers
must provide the prompt, reference reasoning, and reference final answer as
separate strings.  Their concatenation is tokenized once; fast-tokenizer
character offsets (or strict slow-tokenizer prefix stability) establish the
reasoning and answer spans without parsing generated text or treating an
activation direction as a reasoning trace.

The evaluator is read-only: it temporarily places each scored model in
evaluation mode, runs under ``torch.no_grad()``, and restores every submodule's
original training flag.  It never changes parameters, gradients, or hooks.
"""

from __future__ import annotations

import hashlib
import math
import numbers
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


class CoTPreservationError(ValueError):
    """Raised when a preservation measurement is incomplete or invalid."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoTPreservationError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class CoTPreservationExample:
    """One explicitly segmented teacher-forced reasoning reference.

    The concatenated trace is tokenized canonically as one string.  Put any
    separator before a segment in that segment (for example,
    ``reference_reasoning="\nReasoning: ..."`` and
    ``reference_answer=" 12"``) so no tokenizer token crosses a labeled
    boundary.
    """

    prompt: str
    reference_reasoning: str
    reference_answer: str
    example_id: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.prompt, "prompt")
        _require_text(self.reference_reasoning, "reference_reasoning")
        _require_text(self.reference_answer, "reference_answer")
        if self.example_id is not None:
            _require_text(self.example_id, "example_id")


DEFAULT_COT_PRESERVATION_EXAMPLES: tuple[CoTPreservationExample, ...] = (
    CoTPreservationExample(
        prompt="Question: What is 7 + 5?",
        reference_reasoning="\nReasoning: Adding five to seven gives twelve.\nFinal answer:",
        reference_answer=" 12",
        example_id="addition",
    ),
    CoTPreservationExample(
        prompt="Question: What is 9 multiplied by 6?",
        reference_reasoning="\nReasoning: Six groups of nine total fifty-four.\nFinal answer:",
        reference_answer=" 54",
        example_id="multiplication",
    ),
    CoTPreservationExample(
        prompt="Question: What is 20 minus 8?",
        reference_reasoning="\nReasoning: Removing eight from twenty leaves twelve.\nFinal answer:",
        reference_answer=" 12",
        example_id="subtraction",
    ),
    CoTPreservationExample(
        prompt="Question: What number follows 2, 4, 6 in this pattern?",
        reference_reasoning="\nReasoning: The pattern increases by two each time.\nFinal answer:",
        reference_answer=" 8",
        example_id="sequence",
    ),
    CoTPreservationExample(
        prompt="Question: All roses are flowers. Is a rose a flower?",
        reference_reasoning="\nReasoning: A rose belongs to the stated class of flowers.\nFinal answer:",
        reference_answer=" Yes",
        example_id="class_inclusion",
    ),
    CoTPreservationExample(
        prompt="Question: What is the perimeter of a triangle with sides 3, 4, and 5?",
        reference_reasoning="\nReasoning: The perimeter is 3 + 4 + 5, which is twelve.\nFinal answer:",
        reference_answer=" 12",
        example_id="perimeter",
    ),
    CoTPreservationExample(
        prompt="Question: What is 15 divided by 3?",
        reference_reasoning="\nReasoning: Fifteen split into three equal groups gives five.\nFinal answer:",
        reference_answer=" 5",
        example_id="division",
    ),
    CoTPreservationExample(
        prompt="Question: A comes before B, and B comes before C. Does A come before C?",
        reference_reasoning="\nReasoning: The ordering is transitive, so A precedes C.\nFinal answer:",
        reference_answer=" Yes",
        example_id="transitive_order",
    ),
)


@dataclass(frozen=True, slots=True)
class SegmentCrossEntropy:
    """Token-weighted cross-entropy for the two supervised response spans."""

    reasoning_ce: float
    answer_ce: float
    reasoning_token_count: int
    answer_token_count: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.reasoning_ce) or not math.isfinite(self.answer_ce):
            raise CoTPreservationError("segment cross-entropy must be finite")
        if self.reasoning_token_count <= 0 or self.answer_token_count <= 0:
            raise CoTPreservationError("each supervised segment must contain tokens")


@dataclass(frozen=True, slots=True)
class CoTScoreSnapshot:
    """Immutable per-model scores that remain valid across an in-place edit."""

    aggregate: SegmentCrossEntropy
    example_scores: tuple[SegmentCrossEntropy, ...]
    example_ids: tuple[str | None, ...]
    reference_signatures: tuple[str, ...]

    def __post_init__(self) -> None:
        size = len(self.example_scores)
        if size == 0:
            raise CoTPreservationError("a score snapshot cannot be empty")
        if len(self.example_ids) != size or len(self.reference_signatures) != size:
            raise CoTPreservationError("score snapshot fields have inconsistent lengths")
        if not all(self.reference_signatures):
            raise CoTPreservationError("score snapshot contains an empty reference signature")
        reasoning_tokens = sum(score.reasoning_token_count for score in self.example_scores)
        answer_tokens = sum(score.answer_token_count for score in self.example_scores)
        if (
            self.aggregate.reasoning_token_count != reasoning_tokens
            or self.aggregate.answer_token_count != answer_tokens
        ):
            raise CoTPreservationError("aggregate and per-example token counts do not match")


@dataclass(frozen=True, slots=True)
class CoTPreservationExampleResult:
    """Baseline/candidate comparison for a single reference example."""

    example_index: int
    example_id: str | None
    baseline: SegmentCrossEntropy
    candidate: SegmentCrossEntropy
    reasoning_ce_delta: float
    answer_ce_delta: float
    generated_final_answer: str | None = None
    exact_final_answer_match: bool | None = None
    normalized_final_answer_match: bool | None = None

    def __post_init__(self) -> None:
        if self.example_index < 0:
            raise CoTPreservationError("example_index cannot be negative")
        if not math.isfinite(self.reasoning_ce_delta) or not math.isfinite(self.answer_ce_delta):
            raise CoTPreservationError("example preservation deltas must be finite")


@dataclass(frozen=True, slots=True)
class CoTPreservationReport:
    """Immutable aggregate suitable for a candidate-acceptance gate.

    CE deltas are ``candidate - baseline``; positive values indicate damage.
    Match rates are ``None`` unless a final-answer generation callable was
    supplied.
    """

    baseline: SegmentCrossEntropy
    candidate: SegmentCrossEntropy
    reasoning_ce_delta: float
    answer_ce_delta: float
    examples: tuple[CoTPreservationExampleResult, ...]
    exact_final_answer_match_rate: float | None = None
    normalized_final_answer_match_rate: float | None = None

    def __post_init__(self) -> None:
        if not self.examples:
            raise CoTPreservationError("a preservation report cannot be empty")
        if not math.isfinite(self.reasoning_ce_delta) or not math.isfinite(self.answer_ce_delta):
            raise CoTPreservationError("report preservation deltas must be finite")
        rates = (
            self.exact_final_answer_match_rate,
            self.normalized_final_answer_match_rate,
        )
        if (rates[0] is None) != (rates[1] is None):
            raise CoTPreservationError("final-answer match rates must be present together")
        if any(
            rate is not None and (not math.isfinite(rate) or not 0.0 <= rate <= 1.0)
            for rate in rates
        ):
            raise CoTPreservationError("final-answer match rates must be finite values in [0, 1]")

    def as_gate_metrics(self) -> dict[str, float | int]:
        """Return a fresh flat metric mapping for a pipeline acceptance gate."""

        metrics: dict[str, float | int] = {
            "cot_reasoning_ce_increase": self.reasoning_ce_delta,
            "cot_answer_ce_increase": self.answer_ce_delta,
            "cot_eval_example_count": len(self.examples),
            "cot_reasoning_baseline_ce": self.baseline.reasoning_ce,
            "cot_reasoning_candidate_ce": self.candidate.reasoning_ce,
            "cot_reasoning_ce_delta": self.reasoning_ce_delta,
            "cot_answer_baseline_ce": self.baseline.answer_ce,
            "cot_answer_candidate_ce": self.candidate.answer_ce,
            "cot_answer_ce_delta": self.answer_ce_delta,
            "cot_reasoning_token_count": self.candidate.reasoning_token_count,
            "cot_answer_token_count": self.candidate.answer_token_count,
            "cot_example_count": len(self.examples),
        }
        if self.exact_final_answer_match_rate is not None:
            assert self.normalized_final_answer_match_rate is not None
            metrics["cot_exact_final_answer_match_rate"] = self.exact_final_answer_match_rate
            metrics["cot_normalized_final_answer_match_rate"] = (
                self.normalized_final_answer_match_rate
            )
        return metrics


FinalAnswerGenerator = Callable[[CoTPreservationExample], str]
AnswerNormalizer = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class _TokenizedExample:
    input_ids: tuple[int, ...]
    reasoning_start: int
    reasoning_end: int
    answer_start: int
    answer_end: int


def normalize_final_answer(answer: str) -> str:
    """Apply conservative Unicode, whitespace, and case normalization.

    This does not parse a reasoning protocol or extract an answer from a larger
    response.  The generation callable must return only the final answer.
    """

    _require_text(answer, "final answer")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", answer)).strip().casefold()


def _coerce_token_ids(raw_ids: Any, field_name: str) -> tuple[int, ...]:
    if isinstance(raw_ids, torch.Tensor):
        raw_ids = raw_ids.detach().cpu().tolist()
    if isinstance(raw_ids, tuple):
        raw_ids = list(raw_ids)
    if not isinstance(raw_ids, list):
        raise CoTPreservationError(f"tokenizer returned malformed IDs for {field_name}")
    if raw_ids and isinstance(raw_ids[0], (list, tuple)):
        if len(raw_ids) != 1:
            raise CoTPreservationError(f"tokenizer returned a batch for {field_name}")
        raw_ids = list(raw_ids[0])

    ids: list[int] = []
    for token_id in raw_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, numbers.Integral):
            raise CoTPreservationError(f"tokenizer returned a non-integer ID for {field_name}")
        token_id = int(token_id)
        if token_id < 0:
            raise CoTPreservationError(f"tokenizer returned a negative ID for {field_name}")
        ids.append(token_id)
    if not ids:
        raise CoTPreservationError(f"{field_name} produced no tokens")
    return tuple(ids)


def _encode_component(tokenizer: Any, text: str, field_name: str) -> tuple[int, ...]:
    try:
        if hasattr(tokenizer, "encode"):
            raw_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )
        else:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )
            raw_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    except Exception as exc:
        raise CoTPreservationError(f"failed to tokenize {field_name}: {exc}") from exc
    return _coerce_token_ids(raw_ids, field_name)


def _canonical_ids_and_offsets(
    tokenizer: Any,
    text: str,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None:
    """Use fast-tokenizer character offsets when the tokenizer exposes them."""

    if not callable(tokenizer):
        return None
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
    except (AttributeError, KeyError, NotImplementedError, TypeError, ValueError):
        return None
    if isinstance(encoded, Mapping):
        raw_ids = encoded.get("input_ids")
        raw_offsets = encoded.get("offset_mapping")
    else:
        raw_ids = getattr(encoded, "input_ids", None)
        raw_offsets = getattr(encoded, "offset_mapping", None)
    if raw_ids is None or raw_offsets is None:
        return None

    ids = _coerce_token_ids(raw_ids, "concatenated reference")
    if isinstance(raw_offsets, torch.Tensor):
        raw_offsets = raw_offsets.detach().cpu().tolist()
    if isinstance(raw_offsets, tuple):
        raw_offsets = list(raw_offsets)
    if (
        isinstance(raw_offsets, list)
        and len(raw_offsets) == 1
        and isinstance(raw_offsets[0], (list, tuple))
        and raw_offsets[0]
        and isinstance(raw_offsets[0][0], (list, tuple))
    ):
        raw_offsets = list(raw_offsets[0])
    if not isinstance(raw_offsets, list) or len(raw_offsets) != len(ids):
        raise CoTPreservationError(
            "tokenizer returned malformed offsets for the concatenated reference"
        )

    offsets: list[tuple[int, int]] = []
    for raw_offset in raw_offsets:
        if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 2:
            raise CoTPreservationError(
                "tokenizer returned malformed offsets for the concatenated reference"
            )
        start, end = raw_offset
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, numbers.Integral)
            or not isinstance(end, numbers.Integral)
        ):
            raise CoTPreservationError("tokenizer offsets must be integer pairs")
        start = int(start)
        end = int(end)
        if start < 0 or end <= start or end > len(text):
            raise CoTPreservationError("tokenizer returned invalid or zero-width reference offsets")
        offsets.append((start, end))
    return ids, tuple(offsets)


def _finite_context_limit(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    value = int(value)
    # Hugging Face uses very large sentinels when a tokenizer has no known cap.
    return value if 1 < value < 1_000_000_000 else None


def _context_limit(
    tokenizer: Any,
    models: Iterable[Any],
    explicit_max_length: int | None,
) -> int | None:
    if explicit_max_length is not None and (
        isinstance(explicit_max_length, bool)
        or not isinstance(explicit_max_length, numbers.Integral)
        or explicit_max_length <= 1
    ):
        raise CoTPreservationError("max_length must be an integer greater than one")

    limits = [_finite_context_limit(explicit_max_length)]
    limits.append(_finite_context_limit(getattr(tokenizer, "model_max_length", None)))
    for model in models:
        config = getattr(model, "config", None)
        for attribute in ("max_position_embeddings", "n_positions", "max_sequence_length"):
            limits.append(_finite_context_limit(getattr(config, attribute, None)))
    finite_limits = [limit for limit in limits if limit is not None]
    return min(finite_limits) if finite_limits else None


def _tokenize_example(
    tokenizer: Any,
    example: CoTPreservationExample,
    context_limit: int | None,
) -> _TokenizedExample:
    full_text = example.prompt + example.reference_reasoning + example.reference_answer
    prompt_character_end = len(example.prompt)
    reasoning_character_end = prompt_character_end + len(example.reference_reasoning)
    canonical = _canonical_ids_and_offsets(tokenizer, full_text)
    if canonical is not None:
        input_ids, offsets = canonical
        segment_labels: list[str] = []
        for start, end in offsets:
            if end <= prompt_character_end:
                segment_labels.append("prompt")
            elif start >= prompt_character_end and end <= reasoning_character_end:
                segment_labels.append("reasoning")
            elif start >= reasoning_character_end:
                segment_labels.append("answer")
            else:
                raise CoTPreservationError(
                    "a tokenizer token crosses a prompt/reasoning/answer boundary; "
                    "move boundary whitespace into the following segment"
                )
        reasoning_positions = [
            index for index, label in enumerate(segment_labels) if label == "reasoning"
        ]
        answer_positions = [
            index for index, label in enumerate(segment_labels) if label == "answer"
        ]
        if not reasoning_positions or not answer_positions:
            raise CoTPreservationError(
                "canonical tokenization produced an empty reasoning or answer span"
            )
        reasoning_start = reasoning_positions[0]
        reasoning_end = reasoning_positions[-1] + 1
        answer_start = answer_positions[0]
        answer_end = answer_positions[-1] + 1
        if (
            reasoning_positions != list(range(reasoning_start, reasoning_end))
            or answer_positions != list(range(answer_start, answer_end))
            or answer_start != reasoning_end
            or reasoning_start <= 0
        ):
            raise CoTPreservationError(
                "tokenizer offsets do not form contiguous prompt/reasoning/answer spans"
            )
    else:
        # Slow tokenizers cannot expose character offsets.  Score the one
        # canonical full-string encoding only if both prefix encodings are
        # stable prefixes; otherwise a token merged across a labeled boundary.
        prompt_ids = _encode_component(tokenizer, example.prompt, "prompt")
        prompt_reasoning_ids = _encode_component(
            tokenizer,
            example.prompt + example.reference_reasoning,
            "prompt plus reference_reasoning",
        )
        input_ids = _encode_component(tokenizer, full_text, "concatenated reference")
        if (
            input_ids[: len(prompt_ids)] != prompt_ids
            or input_ids[: len(prompt_reasoning_ids)] != prompt_reasoning_ids
        ):
            raise CoTPreservationError(
                "slow tokenizer merges a token across a labeled segment boundary; "
                "use a fast tokenizer with offsets or adjust boundary whitespace"
            )
        reasoning_start = len(prompt_ids)
        reasoning_end = len(prompt_reasoning_ids)
        answer_start = reasoning_end
        answer_end = len(input_ids)
        if reasoning_start <= 0 or reasoning_end <= reasoning_start or answer_end <= answer_start:
            raise CoTPreservationError(
                "canonical tokenization produced an empty reasoning or answer span"
            )

    if context_limit is not None and len(input_ids) > context_limit:
        raise CoTPreservationError(
            "reference exceeds the available context length; truncation would invalidate "
            "the explicit reasoning/answer spans"
        )
    return _TokenizedExample(
        input_ids=input_ids,
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        answer_start=answer_start,
        answer_end=answer_end,
    )


def _reference_signature(tokenized: _TokenizedExample) -> str:
    payload = (
        tokenized.input_ids,
        tokenized.reasoning_start,
        tokenized.reasoning_end,
        tokenized.answer_start,
        tokenized.answer_end,
    )
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def _validate_examples(
    examples: Iterable[CoTPreservationExample],
) -> tuple[CoTPreservationExample, ...]:
    try:
        example_tuple = tuple(examples)
    except (TypeError, ValueError) as exc:
        raise CoTPreservationError("examples must be an iterable of explicit examples") from exc
    if not example_tuple:
        raise CoTPreservationError("at least one reasoning example is required")
    if not all(isinstance(example, CoTPreservationExample) for example in example_tuple):
        raise CoTPreservationError(
            "every item must be a CoTPreservationExample with explicit segments"
        )
    return example_tuple


def _model_input_device(model: Any) -> torch.device:
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.device.type != "meta":
            return weight.device

    device = getattr(model, "device", None)
    if device is not None:
        device = torch.device(device)
        if device.type != "meta":
            return device

    for collection_name in ("parameters", "buffers"):
        collection = getattr(model, collection_name, None)
        if callable(collection):
            try:
                tensor = next(collection())
            except StopIteration:
                continue
            if tensor.device.type != "meta":
                return tensor.device
    return torch.device("cpu")


def _extract_logits(outputs: Any) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    if logits is None and isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    if not isinstance(logits, torch.Tensor):
        raise CoTPreservationError("model output did not contain a logits tensor")
    return logits


def _score_model(
    model: Any,
    tokenized: _TokenizedExample,
    *,
    model_name: str,
) -> SegmentCrossEntropy:
    device = _model_input_device(model)
    input_ids = torch.tensor([tokenized.input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    try:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    except Exception as exc:
        raise CoTPreservationError(f"{model_name} forward pass failed: {exc}") from exc
    logits = _extract_logits(outputs)
    sequence_length = len(tokenized.input_ids)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < sequence_length:
        raise CoTPreservationError(
            f"{model_name} returned malformed logits with shape {tuple(logits.shape)}"
        )
    if logits.shape[2] <= max(tokenized.input_ids):
        raise CoTPreservationError(f"{model_name} logits do not cover the reference token IDs")

    # Logit position p-1 predicts the target token at input position p.  The
    # index ranges below therefore contain response targets only: prompt tokens
    # and any hypothetical batch padding cannot enter either CE calculation.
    reasoning_positions = torch.arange(
        tokenized.reasoning_start - 1,
        tokenized.reasoning_end - 1,
        device=logits.device,
    )
    answer_positions = torch.arange(
        tokenized.answer_start - 1,
        tokenized.answer_end - 1,
        device=logits.device,
    )
    targets = input_ids[0, 1:].to(logits.device)
    reasoning_targets = targets.index_select(0, reasoning_positions)
    answer_targets = targets.index_select(0, answer_positions)
    shifted_logits = logits[0, : sequence_length - 1, :]
    reasoning_logits = shifted_logits.index_select(0, reasoning_positions).float()
    answer_logits = shifted_logits.index_select(0, answer_positions).float()

    if not torch.isfinite(reasoning_logits).all() or not torch.isfinite(answer_logits).all():
        raise CoTPreservationError(f"{model_name} returned non-finite response logits")
    try:
        reasoning_losses = F.cross_entropy(reasoning_logits, reasoning_targets, reduction="none")
        answer_losses = F.cross_entropy(answer_logits, answer_targets, reduction="none")
    except Exception as exc:
        raise CoTPreservationError(f"{model_name} produced invalid CE scores: {exc}") from exc
    if not torch.isfinite(reasoning_losses).all() or not torch.isfinite(answer_losses).all():
        raise CoTPreservationError(f"{model_name} produced non-finite CE scores")

    return SegmentCrossEntropy(
        reasoning_ce=float(reasoning_losses.mean().item()),
        answer_ce=float(answer_losses.mean().item()),
        reasoning_token_count=int(reasoning_losses.numel()),
        answer_token_count=int(answer_losses.numel()),
    )


def _aggregate(scores: Iterable[SegmentCrossEntropy]) -> SegmentCrossEntropy:
    score_tuple = tuple(scores)
    reasoning_count = sum(score.reasoning_token_count for score in score_tuple)
    answer_count = sum(score.answer_token_count for score in score_tuple)
    reasoning_sum = sum(score.reasoning_ce * score.reasoning_token_count for score in score_tuple)
    answer_sum = sum(score.answer_ce * score.answer_token_count for score in score_tuple)
    return SegmentCrossEntropy(
        reasoning_ce=reasoning_sum / reasoning_count,
        answer_ce=answer_sum / answer_count,
        reasoning_token_count=reasoning_count,
        answer_token_count=answer_count,
    )


def score_cot_references(
    model: Any,
    tokenizer: Any,
    examples: Iterable[CoTPreservationExample],
    *,
    max_length: int | None = None,
) -> CoTScoreSnapshot:
    """Score one model and return a snapshot safe to retain across model edits."""

    example_tuple = _validate_examples(examples)
    context_limit = _context_limit(tokenizer, (model,), max_length)
    tokenized_examples = tuple(
        _tokenize_example(tokenizer, example, context_limit) for example in example_tuple
    )
    modules_method = getattr(model, "modules", None)
    if callable(modules_method):
        modules = tuple(modules_method())
    elif isinstance(getattr(model, "training", None), bool):
        modules = (model,)
    else:
        modules = ()
    training_states = tuple(module.training for module in modules)
    try:
        eval_method = getattr(model, "eval", None)
        if callable(eval_method):
            eval_method()
        with torch.no_grad():
            scores = tuple(
                _score_model(model, tokenized, model_name="model")
                for tokenized in tokenized_examples
            )
    finally:
        for module, was_training in zip(modules, training_states, strict=True):
            module.training = was_training
    return CoTScoreSnapshot(
        aggregate=_aggregate(scores),
        example_scores=scores,
        example_ids=tuple(example.example_id for example in example_tuple),
        reference_signatures=tuple(
            _reference_signature(tokenized) for tokenized in tokenized_examples
        ),
    )


def compare_cot_score_snapshots(
    baseline: CoTScoreSnapshot,
    candidate: CoTScoreSnapshot,
) -> CoTPreservationReport:
    """Compare compatible pre-edit and post-edit score snapshots."""

    if not isinstance(baseline, CoTScoreSnapshot) or not isinstance(candidate, CoTScoreSnapshot):
        raise CoTPreservationError("baseline and candidate must be CoTScoreSnapshot values")
    if baseline.reference_signatures != candidate.reference_signatures:
        raise CoTPreservationError(
            "baseline and candidate snapshots were not scored on identical references"
        )
    if baseline.example_ids != candidate.example_ids:
        raise CoTPreservationError("baseline and candidate example IDs do not match")
    if len(baseline.example_scores) != len(candidate.example_scores):
        raise CoTPreservationError("baseline and candidate snapshot sizes do not match")

    rows: list[CoTPreservationExampleResult] = []
    for index, (baseline_score, candidate_score) in enumerate(
        zip(baseline.example_scores, candidate.example_scores, strict=True)
    ):
        if (
            baseline_score.reasoning_token_count != candidate_score.reasoning_token_count
            or baseline_score.answer_token_count != candidate_score.answer_token_count
        ):
            raise CoTPreservationError("baseline and candidate segment token counts do not match")
        reasoning_delta = candidate_score.reasoning_ce - baseline_score.reasoning_ce
        answer_delta = candidate_score.answer_ce - baseline_score.answer_ce
        if not math.isfinite(reasoning_delta) or not math.isfinite(answer_delta):
            raise CoTPreservationError("preservation deltas must be finite")
        rows.append(
            CoTPreservationExampleResult(
                example_index=index,
                example_id=baseline.example_ids[index],
                baseline=baseline_score,
                candidate=candidate_score,
                reasoning_ce_delta=reasoning_delta,
                answer_ce_delta=answer_delta,
            )
        )

    reasoning_delta = candidate.aggregate.reasoning_ce - baseline.aggregate.reasoning_ce
    answer_delta = candidate.aggregate.answer_ce - baseline.aggregate.answer_ce
    if not math.isfinite(reasoning_delta) or not math.isfinite(answer_delta):
        raise CoTPreservationError("aggregate preservation deltas must be finite")
    return CoTPreservationReport(
        baseline=baseline.aggregate,
        candidate=candidate.aggregate,
        reasoning_ce_delta=reasoning_delta,
        answer_ce_delta=answer_delta,
        examples=tuple(rows),
    )


def _add_generated_answer_matches(
    report: CoTPreservationReport,
    examples: tuple[CoTPreservationExample, ...],
    generator: FinalAnswerGenerator,
    normalizer: AnswerNormalizer,
) -> CoTPreservationReport:
    rows: list[CoTPreservationExampleResult] = []
    exact_matches: list[bool] = []
    normalized_matches: list[bool] = []
    with torch.no_grad():
        for index, (row, example) in enumerate(zip(report.examples, examples, strict=True)):
            try:
                generated_answer = generator(example)
            except Exception as exc:
                raise CoTPreservationError(
                    f"final-answer generation failed for example {index}: {exc}"
                ) from exc
            _require_text(generated_answer, "generated final answer")
            exact_match = generated_answer == example.reference_answer
            try:
                normalized_generated = normalizer(generated_answer)
                normalized_reference = normalizer(example.reference_answer)
            except Exception as exc:
                raise CoTPreservationError(
                    f"final-answer normalization failed for example {index}: {exc}"
                ) from exc
            _require_text(normalized_generated, "normalized generated final answer")
            _require_text(normalized_reference, "normalized reference final answer")
            normalized_match = normalized_generated == normalized_reference
            exact_matches.append(exact_match)
            normalized_matches.append(normalized_match)
            rows.append(
                CoTPreservationExampleResult(
                    example_index=row.example_index,
                    example_id=row.example_id,
                    baseline=row.baseline,
                    candidate=row.candidate,
                    reasoning_ce_delta=row.reasoning_ce_delta,
                    answer_ce_delta=row.answer_ce_delta,
                    generated_final_answer=generated_answer,
                    exact_final_answer_match=exact_match,
                    normalized_final_answer_match=normalized_match,
                )
            )
    return CoTPreservationReport(
        baseline=report.baseline,
        candidate=report.candidate,
        reasoning_ce_delta=report.reasoning_ce_delta,
        answer_ce_delta=report.answer_ce_delta,
        examples=tuple(rows),
        exact_final_answer_match_rate=sum(exact_matches) / len(exact_matches),
        normalized_final_answer_match_rate=sum(normalized_matches) / len(normalized_matches),
    )


def evaluate_cot_preservation(
    baseline_model: Any,
    candidate_model: Any,
    tokenizer: Any,
    examples: Iterable[CoTPreservationExample],
    *,
    candidate_answer_generator: FinalAnswerGenerator | None = None,
    answer_normalizer: AnswerNormalizer = normalize_final_answer,
    max_length: int | None = None,
) -> CoTPreservationReport:
    """Compare explicit reasoning references under baseline and candidate models.

    The optional ``candidate_answer_generator`` must return only the candidate's
    final-answer text for the supplied example.  This function never calls
    ``model.generate`` and never attempts to extract an answer from a reasoning
    trace.  For an in-place edit, call :func:`score_cot_references` before and
    after the edit and then :func:`compare_cot_score_snapshots`.
    """

    example_tuple = _validate_examples(examples)
    if candidate_answer_generator is not None and not callable(candidate_answer_generator):
        raise CoTPreservationError("candidate_answer_generator must be callable")
    if not callable(answer_normalizer):
        raise CoTPreservationError("answer_normalizer must be callable")

    baseline = score_cot_references(baseline_model, tokenizer, example_tuple, max_length=max_length)
    candidate = score_cot_references(
        candidate_model, tokenizer, example_tuple, max_length=max_length
    )
    report = compare_cot_score_snapshots(baseline, candidate)
    if candidate_answer_generator is not None:
        report = _add_generated_answer_matches(
            report, example_tuple, candidate_answer_generator, answer_normalizer
        )
    return report


__all__ = [
    "DEFAULT_COT_PRESERVATION_EXAMPLES",
    "AnswerNormalizer",
    "CoTPreservationError",
    "CoTPreservationExample",
    "CoTPreservationExampleResult",
    "CoTPreservationReport",
    "CoTScoreSnapshot",
    "FinalAnswerGenerator",
    "SegmentCrossEntropy",
    "compare_cot_score_snapshots",
    "evaluate_cot_preservation",
    "normalize_final_answer",
    "score_cot_references",
]
