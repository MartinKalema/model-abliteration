"""Paper-faithful Refusal Direction Optimization (RDO).

This module implements Algorithm 1 from Wollschlaeger et al., *The Geometry
of Refusal in Large Language Models: Concept Cones and Representational
Independence* (ICML 2025, arXiv:2502.17420).  It intentionally differs from a
cached-activation proxy: every objective is computed through an actual model
forward with differentiable residual-stream hooks.

The pipeline-facing entry point is :func:`run_rdo`.  It performs five bounded
steps:

1. validate a decoder-only, unquantized, non-offloaded model and chat template;
2. require disjoint caller-supplied training and validation prompts;
3. generate model-specific answer/refusal/retain targets using a supplied DIM
   or otherwise validated seed direction;
4. freeze all model parameters and optimize only one unit direction using
   response-token CE plus sequence retain KL; and
5. select the best direction from the final snapshot window using held-out
   validation losses.

No model parameter is placed in the optimizer or modified by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from obliteratus.analysis.interventions import (
    DirectionalIntervention,
    InterventionError,
    run_with_directional_ablation,
    run_with_directional_addition,
)

SplitName = Literal["train", "validation"]


class RDOError(RuntimeError):
    """Raised when RDO cannot produce trustworthy optimization evidence."""


@dataclass(frozen=True)
class RDOPromptSplit:
    """Paired harmful and harmless instructions for one evidence split."""

    harmful: tuple[str, ...]
    harmless: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "harmful", tuple(self.harmful))
        object.__setattr__(self, "harmless", tuple(self.harmless))


@dataclass(frozen=True)
class RDOConfig:
    """Deterministic single-direction RDO configuration.

    Defaults mirror the paper where applicable: AdamW at ``0.01``, loss
    weights ``1.0 / 0.2 / 1.0``, thirty-token answer/refusal targets, and a
    29-token retain continuation whose loss also includes the final prompt
    position.  ``steps`` is explicit because the paper converges before one
    complete epoch and selects among its final twenty snapshots.
    """

    addition_layer: int
    addition_scale: float | None = None
    steps: int = 40
    batch_size: int = 1
    learning_rate: float = 0.01
    ablation_weight: float = 1.0
    addition_weight: float = 0.2
    retain_weight: float = 1.0
    target_new_tokens: int = 30
    retain_new_tokens: int = 29
    snapshot_window: int = 20
    seed: int = 0
    max_tangent_grad_norm: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "addition_layer",
            "steps",
            "batch_size",
            "target_new_tokens",
            "retain_new_tokens",
            "snapshot_window",
            "seed",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.addition_layer < 0:
            raise ValueError("addition_layer must be non-negative")
        for name in (
            "steps",
            "batch_size",
            "target_new_tokens",
            "retain_new_tokens",
            "snapshot_window",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

        for name in (
            "learning_rate",
            "ablation_weight",
            "addition_weight",
            "retain_weight",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if min(self.ablation_weight, self.addition_weight, self.retain_weight) < 0.0:
            raise ValueError("RDO loss weights must be non-negative")
        if self.ablation_weight + self.addition_weight + self.retain_weight <= 0.0:
            raise ValueError("at least one RDO loss weight must be positive")

        for name in ("addition_scale", "max_tangent_grad_norm"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite number or None")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number or None")
        if self.addition_scale is not None and self.addition_scale <= 0.0:
            raise ValueError("addition_scale must be positive")
        if self.max_tangent_grad_norm is not None and self.max_tangent_grad_norm <= 0.0:
            raise ValueError("max_tangent_grad_norm must be positive")

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RDOTargetSequence:
    """One formatted prompt and its exact model-generated continuation tokens."""

    instruction: str
    formatted_prompt: str
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    response_text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_token_ids", tuple(self.prompt_token_ids))
        object.__setattr__(self, "response_token_ids", tuple(self.response_token_ids))
        if not self.prompt_token_ids:
            raise ValueError("RDO target prompt tokens cannot be empty")
        if not self.response_token_ids:
            raise ValueError("RDO target response tokens cannot be empty")
        for name, values in (
            ("prompt_token_ids", self.prompt_token_ids),
            ("response_token_ids", self.response_token_ids),
        ):
            if any(not isinstance(token, int) or isinstance(token, bool) for token in values):
                raise TypeError(f"{name} must contain integer token ids")
            if any(token < 0 for token in values):
                raise ValueError(f"{name} cannot contain negative token ids")


@dataclass(frozen=True)
class RDOGeneratedExample:
    """The three model-specific targets used by one RDO training item."""

    harmful_instruction: str
    harmless_instruction: str
    answer_after_ablation: RDOTargetSequence
    refusal_after_addition: RDOTargetSequence
    harmless_retain: RDOTargetSequence


@dataclass(frozen=True)
class RDOEvidence:
    """Generated targets plus immutable split/intervention provenance."""

    train: tuple[RDOGeneratedExample, ...]
    validation: tuple[RDOGeneratedExample, ...]
    addition_layer: int
    addition_scale: float
    hidden_size: int
    target_direction_digest: str
    target_new_tokens: int
    retain_new_tokens: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "train", tuple(self.train))
        object.__setattr__(self, "validation", tuple(self.validation))


@dataclass(frozen=True)
class RDOEvidenceSummary:
    """Compact generated-target and split evidence carried in the result."""

    train_examples: int
    validation_examples: int
    train_prompt_digest: str
    validation_prompt_digest: str
    train_target_digest: str
    validation_target_digest: str
    target_direction_digest: str
    addition_layer: int
    addition_scale: float
    hidden_size: int
    target_new_tokens: int
    retain_new_tokens: int


@dataclass(frozen=True)
class RDOLossEvidence:
    """One measured RDO objective with explicit component losses."""

    split: SplitName
    step: int
    total_loss: float
    ablation_ce: float
    addition_ce: float
    retain_kl: float
    tangent_grad_norm: float | None = None


@dataclass(frozen=True)
class RDOSnapshotEvidence:
    """Held-out score for a direction from the final training window."""

    step: int
    direction_digest: str
    validation_loss: RDOLossEvidence


@dataclass(frozen=True)
class RDOResult:
    """Selected RDO direction and auditable optimization evidence."""

    direction: torch.Tensor
    final_direction: torch.Tensor
    initial_direction: torch.Tensor
    selected_step: int
    train_history: tuple[RDOLossEvidence, ...]
    snapshot_evidence: tuple[RDOSnapshotEvidence, ...]
    evidence: RDOEvidence
    evidence_summary: RDOEvidenceSummary
    config: RDOConfig

    def to_metadata(self) -> dict[str, object]:
        """Return compact audit evidence without serializing prompt/target text."""

        return {
            "algorithm": "Wollschlaeger-et-al-RDO-model-forward-optimization",
            "checkpoint_intervention": (
                "RDO-trained direction plus OBLITERATUS output-writer projection"
            ),
            "selected_step": self.selected_step,
            "direction_digest": _tensor_digest(self.direction),
            "final_direction_digest": _tensor_digest(self.final_direction),
            "initial_direction_digest": _tensor_digest(self.initial_direction),
            "configuration": self.config.to_metadata(),
            "evidence_summary": asdict(self.evidence_summary),
            "train_history": [asdict(item) for item in self.train_history],
            "snapshot_evidence": [
                {
                    "step": item.step,
                    "direction_digest": item.direction_digest,
                    "validation_loss": asdict(item.validation_loss),
                }
                for item in self.snapshot_evidence
            ],
        }


@dataclass(frozen=True)
class _Runtime:
    layers: tuple[nn.Module, ...]
    device: torch.device


@dataclass(frozen=True)
class _PreparedExample:
    example: RDOGeneratedExample
    baseline_retain_logits: torch.Tensor


def _canonical_prompt(prompt: str) -> str:
    return " ".join(prompt.split()).casefold()


def _validate_instructions(name: str, prompts: Sequence[str]) -> tuple[str, ...]:
    if isinstance(prompts, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of instructions")
    values = tuple(prompts)
    if not values:
        raise ValueError(f"{name} cannot be empty")
    for prompt in values:
        if not isinstance(prompt, str):
            raise TypeError(f"{name} must contain only strings")
        if not prompt.strip():
            raise ValueError(f"{name} cannot contain blank instructions")
        if "\x00" in prompt:
            raise ValueError(f"{name} cannot contain NUL characters")
    canonical = tuple(_canonical_prompt(prompt) for prompt in values)
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"{name} contains duplicate normalized instructions")
    return values


def _validate_prompt_splits(
    train_split: RDOPromptSplit,
    validation_split: RDOPromptSplit,
) -> tuple[RDOPromptSplit, RDOPromptSplit]:
    if not isinstance(train_split, RDOPromptSplit):
        raise TypeError("train_split must be an RDOPromptSplit")
    if not isinstance(validation_split, RDOPromptSplit):
        raise TypeError("validation_split must be an RDOPromptSplit")

    train_harmful = _validate_instructions("train_split.harmful", train_split.harmful)
    train_harmless = _validate_instructions("train_split.harmless", train_split.harmless)
    validation_harmful = _validate_instructions(
        "validation_split.harmful", validation_split.harmful
    )
    validation_harmless = _validate_instructions(
        "validation_split.harmless", validation_split.harmless
    )
    if len(train_harmful) != len(train_harmless):
        raise ValueError("training harmful and harmless prompt counts must match")
    if len(validation_harmful) != len(validation_harmless):
        raise ValueError("validation harmful and harmless prompt counts must match")

    train_normalized = {
        _canonical_prompt(prompt) for prompt in (*train_harmful, *train_harmless)
    }
    validation_normalized = {
        _canonical_prompt(prompt)
        for prompt in (*validation_harmful, *validation_harmless)
    }
    overlap = train_normalized & validation_normalized
    if overlap:
        raise ValueError(
            "RDO training and validation prompts must be disjoint after normalization"
        )
    if len(train_normalized) != len(train_harmful) + len(train_harmless):
        raise ValueError("training harmful and harmless instructions must be mutually distinct")
    if len(validation_normalized) != len(validation_harmful) + len(validation_harmless):
        raise ValueError("validation harmful and harmless instructions must be mutually distinct")

    return (
        RDOPromptSplit(train_harmful, train_harmless),
        RDOPromptSplit(validation_harmful, validation_harmless),
    )


def _canonical_device(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"cuda:{value}"
    try:
        return str(torch.device(value))
    except (TypeError, RuntimeError, ValueError):
        return str(value).casefold()


def _validate_runtime(
    model: nn.Module,
    decoder_layers: Sequence[nn.Module],
    config: RDOConfig,
) -> _Runtime:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(model, "generate") or not callable(model.generate):
        raise RDOError("RDO target construction requires a model.generate method")
    model_config = getattr(model, "config", None)
    if bool(getattr(model_config, "is_encoder_decoder", False)):
        raise RDOError("RDO currently supports decoder-only causal language models")

    quantized_flags = (
        "is_loaded_in_4bit",
        "is_loaded_in_8bit",
        "is_quantized",
    )
    if any(bool(getattr(model, name, False)) for name in quantized_flags):
        raise RDOError("quantized model-forward RDO is unsupported; load floating-point weights")
    if getattr(model_config, "quantization_config", None) is not None:
        raise RDOError("quantized model-forward RDO is unsupported; load floating-point weights")

    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        if not isinstance(device_map, Mapping):
            raise RDOError("malformed hf_device_map; cannot verify RDO device ownership")
        mapped_devices = {_canonical_device(value) for value in device_map.values()}
        if any(value in {"disk", "meta"} for value in mapped_devices) or len(mapped_devices) > 1:
            raise RDOError("sharded or offloaded models are unsupported for differentiable RDO")

    parameters = tuple(model.parameters())
    if not parameters:
        raise RDOError("RDO model has no parameters")
    if any(not parameter.is_floating_point() for parameter in parameters):
        raise RDOError("RDO requires floating-point model parameters")
    if any("4bit" in type(parameter).__name__.casefold() for parameter in parameters):
        raise RDOError("quantized parameter wrappers are unsupported for RDO")
    devices = {parameter.device for parameter in parameters}
    if any(device.type == "meta" for device in devices):
        raise RDOError("meta/offloaded parameters are unsupported for RDO")
    if len(devices) != 1:
        raise RDOError("sharded or offloaded models are unsupported for differentiable RDO")
    device = next(iter(devices))

    try:
        layers = tuple(decoder_layers)
    except TypeError as exc:
        raise TypeError("decoder_layers must be a sequence of modules") from exc
    if not layers or not all(isinstance(layer, nn.Module) for layer in layers):
        raise TypeError("decoder_layers must contain at least one torch module")
    if len({id(layer) for layer in layers}) != len(layers):
        raise ValueError("decoder_layers cannot contain duplicate modules")
    if config.addition_layer >= len(layers):
        raise ValueError(f"addition_layer must be less than {len(layers)}")
    for layer in layers:
        layer_devices = {parameter.device for parameter in layer.parameters()}
        if layer_devices and layer_devices != {device}:
            raise RDOError("decoder layers must reside on the model's single parameter device")
    return _Runtime(layers=layers, device=device)


def _validate_template(tokenizer: object) -> None:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        raise RDOError("RDO requires a tokenizer with apply_chat_template")
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, str):
        if not template.strip():
            raise RDOError("tokenizer chat_template is empty")
    elif isinstance(template, Mapping):
        default = template.get("default")
        if not isinstance(default, str) or not default.strip():
            raise RDOError("tokenizer chat_template mapping has no non-empty default template")
    else:
        raise RDOError("tokenizer has no usable default chat_template")


def _format_instruction(tokenizer: object, instruction: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        without_generation_marker = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as exc:
        raise RDOError(f"chat template failed for an RDO instruction: {exc}") from exc
    if not isinstance(prompt, str) or not prompt.strip():
        raise RDOError("chat template returned an empty or non-text generation prompt")
    if instruction not in prompt:
        raise RDOError("chat template dropped or rewrote the instruction text")
    if prompt == without_generation_marker:
        raise RDOError("chat template did not add an assistant generation marker")
    if any(marker in prompt for marker in ("{{", "{%", "{instruction}")):
        raise RDOError("chat template output contains unresolved template syntax")
    return prompt


def _tokenize_prompt(tokenizer: object, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
    except Exception as exc:
        raise RDOError(f"tokenizer failed on a formatted RDO prompt: {exc}") from exc
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise RDOError("tokenizer output must be a mapping containing input_ids")
    input_ids = encoded["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        raise RDOError("tokenizer input_ids must be a torch.Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise RDOError("each formatted RDO prompt must tokenize to one non-empty row")
    if input_ids.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise RDOError("tokenizer input_ids must use an integer dtype")
    if bool((input_ids < 0).any().item()):
        raise RDOError("tokenizer produced negative token ids")

    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        raise RDOError("tokenizer attention_mask must match input_ids")
    binary = (attention_mask == 0) | (attention_mask == 1)
    if not bool(binary.all().item()) or not bool(attention_mask.bool().all().item()):
        raise RDOError("single-prompt RDO tokenization cannot contain masked padding")
    return input_ids.to(dtype=torch.long), attention_mask.to(dtype=torch.long)


def _extract_sequences(generated: object) -> torch.Tensor:
    sequences = getattr(generated, "sequences", generated)
    if not isinstance(sequences, torch.Tensor):
        raise RDOError("model.generate did not return token sequences")
    if sequences.ndim != 2 or sequences.shape[0] != 1:
        raise RDOError("RDO target generation requires exactly one returned sequence")
    if sequences.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise RDOError("model.generate returned non-integer token ids")
    return sequences


def _decode_response(tokenizer: object, response_ids: Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return ""
    try:
        text = decode(list(response_ids), skip_special_tokens=True)
    except TypeError:
        text = decode(list(response_ids))
    except Exception as exc:
        raise RDOError(f"tokenizer failed to decode a generated RDO target: {exc}") from exc
    if not isinstance(text, str):
        raise RDOError("tokenizer.decode returned a non-text RDO target")
    return text


def _generate_target(
    model: nn.Module,
    tokenizer: object,
    runtime: _Runtime,
    *,
    instruction: str,
    direction: torch.Tensor | None,
    mode: Literal["baseline", "ablate", "add"],
    addition_layer: int,
    addition_scale: float,
    max_new_tokens: int,
) -> RDOTargetSequence:
    formatted = _format_instruction(tokenizer, instruction)
    cpu_input_ids, cpu_attention_mask = _tokenize_prompt(tokenizer, formatted)
    input_ids = cpu_input_ids.to(runtime.device)
    attention_mask = cpu_attention_mask.to(runtime.device)
    generation_kwargs: dict[str, object] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if isinstance(pad_token_id, int) and not isinstance(pad_token_id, bool):
        generation_kwargs["pad_token_id"] = pad_token_id

    with torch.no_grad():
        if mode == "baseline":
            generated = model.generate(**generation_kwargs)
        elif mode == "ablate":
            assert direction is not None
            with DirectionalIntervention(runtime.layers, direction, mode="ablate") as hooks:
                generated = model.generate(**generation_kwargs)
                hooks.assert_applied()
        else:
            assert direction is not None
            with DirectionalIntervention(
                runtime.layers,
                direction,
                mode="add",
                addition_layer=addition_layer,
                addition_scale=addition_scale,
            ) as hooks:
                generated = model.generate(**generation_kwargs)
                hooks.assert_applied()

    sequences = _extract_sequences(generated)
    prompt_length = input_ids.shape[1]
    if sequences.shape[1] <= prompt_length:
        raise RDOError(f"model generated no tokens for an RDO {mode} target")
    response = sequences[0, prompt_length : prompt_length + max_new_tokens]
    response_ids = tuple(int(token) for token in response.detach().cpu().tolist())
    if not response_ids:
        raise RDOError(f"model generated an empty RDO {mode} target")
    return RDOTargetSequence(
        instruction=instruction,
        formatted_prompt=formatted,
        prompt_token_ids=tuple(int(token) for token in cpu_input_ids[0].tolist()),
        response_token_ids=response_ids,
        response_text=_decode_response(tokenizer, response_ids),
    )


@contextmanager
def _frozen_model(model: nn.Module):
    parameters = tuple(model.parameters())
    dirty_gradients = [name for name, parameter in model.named_parameters() if parameter.grad is not None]
    if dirty_gradients:
        raise RDOError(
            "RDO requires model parameter gradients to be clear before entry; found: "
            + ", ".join(dirty_gradients[:5])
        )
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    was_training = model.training
    for parameter in parameters:
        parameter.requires_grad_(False)
    model.eval()
    try:
        yield
        if any(parameter.grad is not None for parameter in parameters):
            raise RDOError("a frozen model parameter unexpectedly received an RDO gradient")
    finally:
        for parameter, required in zip(parameters, requires_grad, strict=True):
            parameter.requires_grad_(required)
        model.train(was_training)


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def _prompt_digest(examples: Sequence[RDOGeneratedExample]) -> str:
    payload = [
        {
            "harmful": _canonical_prompt(example.harmful_instruction),
            "harmless": _canonical_prompt(example.harmless_instruction),
        }
        for example in examples
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _target_digest(examples: Sequence[RDOGeneratedExample]) -> str:
    payload = [
        {
            "answer": example.answer_after_ablation.response_token_ids,
            "refusal": example.refusal_after_addition.response_token_ids,
            "retain": example.harmless_retain.response_token_ids,
        }
        for example in examples
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _summarize_evidence(evidence: RDOEvidence) -> RDOEvidenceSummary:
    return RDOEvidenceSummary(
        train_examples=len(evidence.train),
        validation_examples=len(evidence.validation),
        train_prompt_digest=_prompt_digest(evidence.train),
        validation_prompt_digest=_prompt_digest(evidence.validation),
        train_target_digest=_target_digest(evidence.train),
        validation_target_digest=_target_digest(evidence.validation),
        target_direction_digest=evidence.target_direction_digest,
        addition_layer=evidence.addition_layer,
        addition_scale=evidence.addition_scale,
        hidden_size=evidence.hidden_size,
        target_new_tokens=evidence.target_new_tokens,
        retain_new_tokens=evidence.retain_new_tokens,
    )


def _examples_to_splits(evidence: RDOEvidence) -> tuple[RDOPromptSplit, RDOPromptSplit]:
    train = RDOPromptSplit(
        harmful=tuple(example.harmful_instruction for example in evidence.train),
        harmless=tuple(example.harmless_instruction for example in evidence.train),
    )
    validation = RDOPromptSplit(
        harmful=tuple(example.harmful_instruction for example in evidence.validation),
        harmless=tuple(example.harmless_instruction for example in evidence.validation),
    )
    return _validate_prompt_splits(train, validation)


def _validate_evidence_integrity(evidence: RDOEvidence) -> None:
    _examples_to_splits(evidence)
    if not isinstance(evidence.hidden_size, int) or isinstance(evidence.hidden_size, bool):
        raise TypeError("RDO evidence hidden_size must be an integer")
    if evidence.hidden_size <= 0:
        raise ValueError("RDO evidence hidden_size must be positive")
    if not isinstance(evidence.addition_layer, int) or isinstance(evidence.addition_layer, bool):
        raise TypeError("RDO evidence addition_layer must be an integer")
    if evidence.addition_layer < 0:
        raise ValueError("RDO evidence addition_layer must be non-negative")
    if not math.isfinite(evidence.addition_scale) or evidence.addition_scale <= 0.0:
        raise ValueError("RDO evidence addition_scale must be finite and positive")
    if not evidence.train or not evidence.validation:
        raise ValueError("RDO evidence requires non-empty train and validation examples")

    for split in (evidence.train, evidence.validation):
        for example in split:
            answer = example.answer_after_ablation
            refusal = example.refusal_after_addition
            retain = example.harmless_retain
            if answer.instruction != example.harmful_instruction:
                raise ValueError("RDO answer target does not match its harmful instruction")
            if (
                refusal.instruction != example.harmless_instruction
                or retain.instruction != example.harmless_instruction
            ):
                raise ValueError("RDO harmless targets do not match their instruction")
            if (
                refusal.formatted_prompt != retain.formatted_prompt
                or refusal.prompt_token_ids != retain.prompt_token_ids
            ):
                raise ValueError("RDO addition and retain targets must share one prompt encoding")
            if len(answer.response_token_ids) > evidence.target_new_tokens:
                raise ValueError("RDO answer target exceeds its declared token budget")
            if len(refusal.response_token_ids) > evidence.target_new_tokens:
                raise ValueError("RDO refusal target exceeds its declared token budget")
            if len(retain.response_token_ids) > evidence.retain_new_tokens:
                raise ValueError("RDO retain target exceeds its declared token budget")


def generate_rdo_evidence(
    model: nn.Module,
    tokenizer: object,
    decoder_layers: Sequence[nn.Module],
    *,
    train_split: RDOPromptSplit,
    validation_split: RDOPromptSplit,
    target_direction: torch.Tensor,
    config: RDOConfig,
) -> RDOEvidence:
    """Generate paper RDO targets for disjoint train and validation prompts."""

    if not isinstance(config, RDOConfig):
        raise TypeError("config must be an RDOConfig")
    runtime = _validate_runtime(model, decoder_layers, config)
    _validate_template(tokenizer)
    train_split, validation_split = _validate_prompt_splits(train_split, validation_split)
    if not isinstance(target_direction, torch.Tensor):
        raise TypeError("target_direction must be a torch.Tensor")
    if target_direction.ndim != 1 or not target_direction.is_floating_point():
        raise ValueError("target_direction must be a floating-point hidden-size vector")
    if not bool(torch.isfinite(target_direction.detach()).all().item()):
        raise ValueError("target_direction contains NaN or infinite values")
    target_norm = float(torch.linalg.vector_norm(target_direction.detach().float()).item())
    if not math.isfinite(target_norm) or target_norm <= 0.0:
        raise ValueError("target_direction must have a finite non-zero norm")
    addition_scale = float(config.addition_scale or target_norm)
    target = target_direction.detach().to(device=runtime.device, dtype=torch.float32).clone()

    def build(split: RDOPromptSplit) -> tuple[RDOGeneratedExample, ...]:
        generated: list[RDOGeneratedExample] = []
        for harmful, harmless in zip(split.harmful, split.harmless, strict=True):
            answer = _generate_target(
                model,
                tokenizer,
                runtime,
                instruction=harmful,
                direction=target,
                mode="ablate",
                addition_layer=config.addition_layer,
                addition_scale=addition_scale,
                max_new_tokens=config.target_new_tokens,
            )
            refusal = _generate_target(
                model,
                tokenizer,
                runtime,
                instruction=harmless,
                direction=target,
                mode="add",
                addition_layer=config.addition_layer,
                addition_scale=addition_scale,
                max_new_tokens=config.target_new_tokens,
            )
            retain = _generate_target(
                model,
                tokenizer,
                runtime,
                instruction=harmless,
                direction=None,
                mode="baseline",
                addition_layer=config.addition_layer,
                addition_scale=addition_scale,
                max_new_tokens=config.retain_new_tokens,
            )
            generated.append(
                RDOGeneratedExample(
                    harmful_instruction=harmful,
                    harmless_instruction=harmless,
                    answer_after_ablation=answer,
                    refusal_after_addition=refusal,
                    harmless_retain=retain,
                )
            )
        return tuple(generated)

    with _frozen_model(model):
        train = build(train_split)
        validation = build(validation_split)
    return RDOEvidence(
        train=train,
        validation=validation,
        addition_layer=config.addition_layer,
        addition_scale=addition_scale,
        hidden_size=target.numel(),
        target_direction_digest=_tensor_digest(target),
        target_new_tokens=config.target_new_tokens,
        retain_new_tokens=config.retain_new_tokens,
    )


def _extract_logits(outputs: object) -> torch.Tensor:
    logits: object
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    elif isinstance(outputs, Mapping):
        if "logits" not in outputs:
            raise RDOError("model output mapping has no logits field")
        logits = outputs["logits"]
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    elif isinstance(outputs, (tuple, list)) and outputs:
        logits = outputs[0]
    else:
        raise RDOError("model output does not expose causal-LM logits")
    if not isinstance(logits, torch.Tensor):
        raise RDOError("model logits must be a torch.Tensor")
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 2:
        raise RDOError("RDO model logits must have shape [1, sequence, vocabulary]")
    return logits


def _sequence_inputs(
    target: RDOTargetSequence,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = (*target.prompt_token_ids, *target.response_token_ids)
    if len(tokens) < 2:
        raise RDOError("RDO target sequence must contain at least two tokens")
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def _model_forward(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )


def _validate_logits_for_target(logits: torch.Tensor, input_ids: torch.Tensor) -> None:
    if logits.shape[:2] != input_ids.shape:
        raise RDOError("model logits sequence shape does not match RDO target input_ids")
    if input_ids.max().item() >= logits.shape[-1]:
        raise RDOError("RDO target token id is outside the model vocabulary")


def _response_ce(logits: torch.Tensor, target: RDOTargetSequence) -> torch.Tensor:
    prompt_length = len(target.prompt_token_ids)
    response_length = len(target.response_token_ids)
    start = prompt_length - 1
    stop = start + response_length
    prediction_logits = logits[:, start:stop, :]
    labels = torch.tensor(
        target.response_token_ids,
        dtype=torch.long,
        device=logits.device,
    ).unsqueeze(0)
    if prediction_logits.shape[:2] != labels.shape:
        raise RDOError("response-token CE could not align generated target tokens")
    if not bool(torch.isfinite(prediction_logits).all().item()):
        raise RDOError("RDO response-token logits contain NaN or infinite values")
    return F.cross_entropy(
        prediction_logits.float().reshape(-1, prediction_logits.shape[-1]),
        labels.reshape(-1),
    )


def _retain_rows(logits: torch.Tensor, target: RDOTargetSequence) -> torch.Tensor:
    # Paper RDO includes the last instruction position plus all target-response
    # positions.  With 29 retain tokens this selects exactly 30 distributions.
    start = len(target.prompt_token_ids) - 1
    selected = logits[:, start:, :]
    if selected.shape[1] != len(target.response_token_ids) + 1:
        raise RDOError("retain sequence KL could not align prompt/target positions")
    if not bool(torch.isfinite(selected).all().item()):
        raise RDOError("RDO retain logits contain NaN or infinite values")
    return selected.float()


def _forward_ablation_logits(
    model: nn.Module,
    runtime: _Runtime,
    direction: torch.Tensor,
    target: RDOTargetSequence,
) -> torch.Tensor:
    input_ids, attention_mask = _sequence_inputs(target, runtime.device)
    outputs = run_with_directional_ablation(
        model,
        runtime.layers,
        direction,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = _extract_logits(outputs)
    _validate_logits_for_target(logits, input_ids)
    return logits


def _forward_addition_logits(
    model: nn.Module,
    runtime: _Runtime,
    direction: torch.Tensor,
    target: RDOTargetSequence,
    evidence: RDOEvidence,
) -> torch.Tensor:
    input_ids, attention_mask = _sequence_inputs(target, runtime.device)
    outputs = run_with_directional_addition(
        model,
        runtime.layers,
        direction,
        addition_layer=evidence.addition_layer,
        addition_scale=evidence.addition_scale,
        model_kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        },
    )
    logits = _extract_logits(outputs)
    _validate_logits_for_target(logits, input_ids)
    return logits


def _prepare_examples(
    model: nn.Module,
    runtime: _Runtime,
    examples: Sequence[RDOGeneratedExample],
) -> tuple[_PreparedExample, ...]:
    prepared: list[_PreparedExample] = []
    with torch.no_grad():
        for example in examples:
            input_ids, attention_mask = _sequence_inputs(example.harmless_retain, runtime.device)
            outputs = _model_forward(model, input_ids, attention_mask)
            logits = _extract_logits(outputs)
            _validate_logits_for_target(logits, input_ids)
            baseline = _retain_rows(logits, example.harmless_retain).detach().clone()
            prepared.append(_PreparedExample(example, baseline))
    return tuple(prepared)


def _loss_for_example(
    model: nn.Module,
    runtime: _Runtime,
    direction: torch.Tensor,
    prepared: _PreparedExample,
    evidence: RDOEvidence,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    example = prepared.example
    ablation_logits = _forward_ablation_logits(
        model,
        runtime,
        direction,
        example.answer_after_ablation,
    )
    ablation_ce = _response_ce(ablation_logits, example.answer_after_ablation)

    addition_logits = _forward_addition_logits(
        model,
        runtime,
        direction,
        example.refusal_after_addition,
        evidence,
    )
    addition_ce = _response_ce(addition_logits, example.refusal_after_addition)

    retain_logits = _forward_ablation_logits(
        model,
        runtime,
        direction,
        example.harmless_retain,
    )
    candidate_rows = _retain_rows(retain_logits, example.harmless_retain)
    baseline_rows = prepared.baseline_retain_logits
    baseline_log_probs = F.log_softmax(baseline_rows, dim=-1)
    candidate_log_probs = F.log_softmax(candidate_rows, dim=-1)
    baseline_probs = baseline_log_probs.exp()
    retain_kl = (
        baseline_probs * (baseline_log_probs - candidate_log_probs)
    ).sum(dim=-1).mean()
    return ablation_ce, addition_ce, retain_kl


def _aggregate_loss(
    model: nn.Module,
    runtime: _Runtime,
    direction: torch.Tensor,
    batch: Sequence[_PreparedExample],
    evidence: RDOEvidence,
    config: RDOConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not batch:
        raise RDOError("cannot compute an RDO loss over an empty batch")
    components = [
        _loss_for_example(model, runtime, direction, example, evidence) for example in batch
    ]
    ablation_ce = torch.stack([value[0] for value in components]).mean()
    addition_ce = torch.stack([value[1] for value in components]).mean()
    retain_kl = torch.stack([value[2] for value in components]).mean()
    total = (
        config.ablation_weight * ablation_ce
        + config.addition_weight * addition_ce
        + config.retain_weight * retain_kl
    )
    if not bool(torch.isfinite(total.detach()).item()):
        raise RDOError("RDO objective contains NaN or infinite values")
    return total, ablation_ce, addition_ce, retain_kl


def _loss_evidence(
    split: SplitName,
    step: int,
    values: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    tangent_grad_norm: float | None = None,
) -> RDOLossEvidence:
    total, ablation, addition, retain = values
    return RDOLossEvidence(
        split=split,
        step=step,
        total_loss=float(total.detach().item()),
        ablation_ce=float(ablation.detach().item()),
        addition_ce=float(addition.detach().item()),
        retain_kl=float(retain.detach().item()),
        tangent_grad_norm=tangent_grad_norm,
    )


def _batch_indices(
    size: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pending: deque[int] = deque()
    batches: list[tuple[int, ...]] = []
    for _ in range(steps):
        selected: list[int] = []
        while len(selected) < min(batch_size, size):
            if not pending:
                pending.extend(torch.randperm(size, generator=generator).tolist())
            selected.append(pending.popleft())
        batches.append(tuple(selected))
    return tuple(batches)


def _initial_unit_direction(
    initial_direction: torch.Tensor | None,
    *,
    hidden_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    if initial_direction is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        value = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
    else:
        if not isinstance(initial_direction, torch.Tensor):
            raise TypeError("initial_direction must be a torch.Tensor or None")
        if initial_direction.ndim != 1 or initial_direction.numel() != hidden_size:
            raise ValueError(f"initial_direction must have shape [{hidden_size}]")
        if not initial_direction.is_floating_point():
            raise TypeError("initial_direction must use a floating-point dtype")
        value = initial_direction.detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("initial_direction contains NaN or infinite values")
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
        raise ValueError("initial_direction must have a finite non-zero norm")
    return (value / norm).to(device=device)


def optimize_rdo_direction(
    model: nn.Module,
    decoder_layers: Sequence[nn.Module],
    *,
    evidence: RDOEvidence,
    config: RDOConfig,
    initial_direction: torch.Tensor | None = None,
) -> RDOResult:
    """Optimize one RDO direction and select it on held-out evidence."""

    if not isinstance(evidence, RDOEvidence):
        raise TypeError("evidence must be RDOEvidence")
    if not isinstance(config, RDOConfig):
        raise TypeError("config must be RDOConfig")
    runtime = _validate_runtime(model, decoder_layers, config)
    _validate_evidence_integrity(evidence)
    if evidence.addition_layer != config.addition_layer:
        raise ValueError("RDO evidence/config addition_layer mismatch")
    if config.addition_scale is not None and not math.isclose(
        evidence.addition_scale,
        config.addition_scale,
        rel_tol=1e-9,
        abs_tol=0.0,
    ):
        raise ValueError("RDO evidence/config addition_scale mismatch")
    if evidence.target_new_tokens != config.target_new_tokens:
        raise ValueError("RDO evidence/config target_new_tokens mismatch")
    if evidence.retain_new_tokens != config.retain_new_tokens:
        raise ValueError("RDO evidence/config retain_new_tokens mismatch")

    initial = _initial_unit_direction(
        initial_direction,
        hidden_size=evidence.hidden_size,
        device=runtime.device,
        seed=config.seed,
    )
    direction = nn.Parameter(initial.clone(), requires_grad=True)
    optimizer = torch.optim.AdamW(
        [direction],
        lr=config.learning_rate,
        betas=(0.9, 0.98),
        weight_decay=0.0,
        amsgrad=True,
    )
    batch_plan = _batch_indices(
        len(evidence.train),
        config.batch_size,
        config.steps,
        config.seed,
    )
    train_history: list[RDOLossEvidence] = []
    final_snapshots: deque[tuple[int, torch.Tensor]] = deque(
        maxlen=min(config.snapshot_window, config.steps)
    )

    try:
        with _frozen_model(model):
            prepared_train = _prepare_examples(model, runtime, evidence.train)
            prepared_validation = _prepare_examples(model, runtime, evidence.validation)

            for step, indices in enumerate(batch_plan, start=1):
                optimizer.zero_grad(set_to_none=True)
                batch = tuple(prepared_train[index] for index in indices)
                values = _aggregate_loss(
                    model,
                    runtime,
                    direction,
                    batch,
                    evidence,
                    config,
                )
                values[0].backward()
                gradient = direction.grad
                if gradient is None:
                    raise RDOError("RDO objective produced no direction gradient")
                if not bool(torch.isfinite(gradient).all().item()):
                    raise RDOError("RDO direction gradient contains NaN or infinite values")

                with torch.no_grad():
                    unit = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
                    tangent = gradient - torch.dot(gradient, unit) * unit
                    if not bool(torch.isfinite(tangent).all().item()):
                        raise RDOError("RDO tangent gradient contains NaN or infinite values")
                    tangent_norm = float(torch.linalg.vector_norm(tangent).item())
                    if config.max_tangent_grad_norm is not None:
                        limit = config.max_tangent_grad_norm
                        tangent.mul_(min(1.0, limit / max(tangent_norm, 1e-12)))
                        tangent_norm = float(torch.linalg.vector_norm(tangent).item())
                    direction.grad.copy_(tangent)

                optimizer.step()
                with torch.no_grad():
                    if not bool(torch.isfinite(direction).all().item()):
                        raise RDOError("RDO optimizer produced a non-finite direction")
                    norm = torch.linalg.vector_norm(direction)
                    if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 0.0:
                        raise RDOError("RDO optimizer produced a zero or invalid direction")
                    direction.div_(norm)
                    snapshot = direction.detach().clone()
                train_history.append(
                    _loss_evidence(
                        "train",
                        step,
                        values,
                        tangent_grad_norm=tangent_norm,
                    )
                )
                final_snapshots.append((step, snapshot))

            snapshot_evidence: list[RDOSnapshotEvidence] = []
            with torch.no_grad():
                for step, snapshot in final_snapshots:
                    validation_values = _aggregate_loss(
                        model,
                        runtime,
                        snapshot,
                        prepared_validation,
                        evidence,
                        config,
                    )
                    loss = _loss_evidence("validation", step, validation_values)
                    snapshot_evidence.append(
                        RDOSnapshotEvidence(
                            step=step,
                            direction_digest=_tensor_digest(snapshot),
                            validation_loss=loss,
                        )
                    )
    except InterventionError as exc:
        raise RDOError(f"RDO intervention failed: {exc}") from exc

    if not snapshot_evidence:
        raise RDOError("RDO produced no final snapshots for validation selection")
    selected_index = min(
        range(len(snapshot_evidence)),
        key=lambda index: (
            snapshot_evidence[index].validation_loss.total_loss,
            snapshot_evidence[index].step,
        ),
    )
    selected_step, selected_direction = tuple(final_snapshots)[selected_index]
    final_direction = tuple(final_snapshots)[-1][1]
    return RDOResult(
        direction=selected_direction.detach().clone(),
        final_direction=final_direction.detach().clone(),
        initial_direction=initial.detach().clone(),
        selected_step=selected_step,
        train_history=tuple(train_history),
        snapshot_evidence=tuple(snapshot_evidence),
        evidence=evidence,
        evidence_summary=_summarize_evidence(evidence),
        config=config,
    )


def run_rdo(
    model: nn.Module,
    tokenizer: object,
    decoder_layers: Sequence[nn.Module],
    *,
    train_split: RDOPromptSplit,
    validation_split: RDOPromptSplit,
    target_direction: torch.Tensor,
    config: RDOConfig,
    initial_direction: torch.Tensor | None = None,
) -> RDOResult:
    """Generate RDO targets, optimize, and select one held-out direction.

    ``target_direction`` is the DIM (or another caller-validated effective
    direction) used only to generate model-specific training targets.  Passing
    ``initial_direction=None`` follows Algorithm 1's random initialization;
    the random vector and batch order are deterministic under ``config.seed``.
    """

    evidence = generate_rdo_evidence(
        model,
        tokenizer,
        decoder_layers,
        train_split=train_split,
        validation_split=validation_split,
        target_direction=target_direction,
        config=config,
    )
    return optimize_rdo_direction(
        model,
        decoder_layers,
        evidence=evidence,
        config=config,
        initial_direction=initial_direction,
    )
