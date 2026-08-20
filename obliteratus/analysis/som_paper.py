"""Paper-faithful SOM multi-direction extraction and behavioral subset search.

This module implements Algorithms 1 and 2 from Piras et al., *SOM Directions
Are Better than One: Multi-Directional Refusal Suppression in Language Models*
(AAAI 2026).  It deliberately lives beside :mod:`som_directions`, whose local
geometry ranking is a compute-bounded proxy rather than the paper baseline.

Primary sources
---------------
* Paper and supplement: https://doi.org/10.1609/aaai.v40i39.40551
* Authors' implementation: https://github.com/pralab/som-refusal-directions

The convenience integration boundary is :func:`run_paper_som_search`; pipelines
that must train before releasing activation memory may instead call the public
split-phase pair :func:`train_paper_som_directions` and
:func:`search_som_direction_subsets`.  The caller supplies an explicit model,
the checkpoint tensors to project, isolated evidence splits, a completion
generator, and a HarmBench-compatible binary judge.  Every search trial performs
real in-place checkpoint projections and restores the projection bytes in a
``finally`` block.  Production pipelines additionally pass a full-model restore
callback so non-target buffers and weights are reset between trials.  The winner
owns copied direction tensors and cryptographic hashes, allowing
:func:`replay_som_winner` to reject any intervention other than the one scored.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import numbers
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import torch
from torch import nn

PAPER_DOI = "10.1609/aaai.v40i39.40551"
PAPER_ARXIV = "2511.08379v2"
UPSTREAM_REPOSITORY = "https://github.com/pralab/som-refusal-directions"
UPSTREAM_COMMIT = "d244c7d282ac65a1520bef0d418615ef148108af"
UPSTREAM_MINISOM_VERSION = "2.3.5"
REPLAY_SCHEMA_VERSION = 1
_EPS = 1e-12


class SOMPaperError(RuntimeError):
    """Base class for fail-closed paper-baseline failures."""


class SOMEvidenceError(SOMPaperError):
    """Raised when split, generation, or judge evidence is not trustworthy."""


class SOMCheckpointError(SOMPaperError):
    """Raised when a checkpoint intervention cannot be applied exactly."""


class SOMRollbackError(SOMCheckpointError):
    """Raised when checkpoint bytes do not return to their baseline hash."""


class SOMReplayError(SOMCheckpointError):
    """Raised when a stored winner cannot be replayed exactly."""


SamplerMode = Literal["auto", "optuna_tpe", "deterministic_random_fallback"]


def _require_finite_number(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return number


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be a positive integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _update_tensor_hash(hasher: Any, tensor: torch.Tensor) -> None:
    cpu = tensor.detach().to(device="cpu").contiguous()
    hasher.update(str(cpu.dtype).encode("ascii"))
    hasher.update(str(tuple(cpu.shape)).encode("ascii"))
    # Viewing as bytes also supports bfloat16, which NumPy cannot represent on
    # every supported version.
    hasher.update(cpu.view(torch.uint8).numpy().tobytes())


def _tensor_hash(tensor: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    _update_tensor_hash(hasher, tensor)
    return hasher.hexdigest()


def _tensor_sequence_hash(tensors: Sequence[torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for index, tensor in enumerate(tensors):
        hasher.update(str(index).encode("ascii"))
        _update_tensor_hash(hasher, tensor)
    return hasher.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class SOMTrainingConfig:
    """SOM hyperparameters from Piras et al. Appendix B.1.

    The paper fixes a 4x4 hexagonal lattice.  ``iterations`` remains
    configurable so tests and ablations can be explicitly labelled as reduced
    runs; 10,000 is the paper default.  Learning-rate and sigma decay match the
    MiniSom 2.3.5 ``asymptotic_decay`` used by the authors' released code.
    """

    iterations: int = 10_000
    learning_rate: float = 0.01
    sigma: float = 0.3
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "iterations", _require_positive_int("iterations", self.iterations))
        object.__setattr__(
            self,
            "learning_rate",
            _require_finite_number("learning_rate", self.learning_rate, positive=True),
        )
        object.__setattr__(
            self, "sigma", _require_finite_number("sigma", self.sigma, positive=True)
        )
        if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def uses_paper_defaults(self) -> bool:
        return (
            self.iterations == 10_000
            and self.learning_rate == 0.01
            and self.sigma == 0.3
            and self.seed == 0
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "grid_shape": [4, 4],
            "topology": "hexagonal",
            "iterations": self.iterations,
            "learning_rate_initial": self.learning_rate,
            "sigma_initial": self.sigma,
            "learning_rate_schedule": "alpha_0/(1+2t/T)",
            "sigma_schedule": "sigma_0/(1+2t/T); MiniSom-2.3.5 upstream behavior",
            "sample_order": "seeded shuffled repeated indices (MiniSom train_random)",
            "seed": self.seed,
            "uses_paper_defaults": self.uses_paper_defaults,
        }


@dataclass(frozen=True)
class SOMSearchConfig:
    """Ordered subset-search configuration from Appendix B.2."""

    subset_size: int = 5
    n_trials: int | None = None
    sampler: SamplerMode = "auto"
    allow_deterministic_fallback: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        size = _require_positive_int("subset_size", self.subset_size)
        if not 2 <= size <= 7:
            raise ValueError("subset_size must be in the paper's evaluated range [2, 7]")
        object.__setattr__(self, "subset_size", size)
        if self.n_trials is not None:
            object.__setattr__(self, "n_trials", _require_positive_int("n_trials", self.n_trials))
        if self.sampler not in {"auto", "optuna_tpe", "deterministic_random_fallback"}:
            raise ValueError(
                "sampler must be 'auto', 'optuna_tpe', or 'deterministic_random_fallback'"
            )
        if not isinstance(self.allow_deterministic_fallback, bool):
            raise TypeError("allow_deterministic_fallback must be bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, numbers.Integral):
            raise TypeError("seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))

    @property
    def resolved_trials(self) -> int:
        if self.n_trials is not None:
            return self.n_trials
        return 128 if self.subset_size <= 3 else 512

    def to_metadata(self) -> dict[str, Any]:
        trials = self.resolved_trials
        return {
            "ordered_subset_size": self.subset_size,
            "requested_trials": trials,
            "paper_default_trials": 128 if self.subset_size <= 3 else 512,
            "random_startup_trials": trials // 4,
            "random_startup_fraction": 0.25,
            "requested_sampler": self.sampler,
            "allow_deterministic_fallback": self.allow_deterministic_fallback,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class SOMBehaviorExample:
    """One harmful behavior in a named evidence split."""

    example_id: str
    behavior: str
    context: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise SOMEvidenceError("every behavior example needs a non-empty example_id")
        if not isinstance(self.behavior, str) or not self.behavior.strip():
            raise SOMEvidenceError("every behavior example needs non-empty behavior text")
        if self.context is not None and not isinstance(self.context, str):
            raise SOMEvidenceError("behavior context must be a string when provided")

    def fingerprint_payload(self) -> dict[str, str | None]:
        return {
            "example_id": self.example_id,
            "behavior": self.behavior,
            "context": self.context,
        }


@dataclass(frozen=True)
class SOMEvidenceSplits:
    """Explicit, pairwise-disjoint train/search/test evidence.

    Search consumes only ``validation``.  ``test`` is required and fingerprinted
    but never generated or judged by this module, preserving it for downstream
    reporting.  The activation rows supplied to :func:`run_paper_som_search`
    must correspond one-for-one with the train IDs.
    """

    harmful_train_ids: tuple[str, ...]
    harmless_train_ids: tuple[str, ...]
    validation: tuple[SOMBehaviorExample, ...]
    test: tuple[SOMBehaviorExample, ...]

    def __post_init__(self) -> None:
        for name in ("harmful_train_ids", "harmless_train_ids", "validation", "test"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

        if not self.harmful_train_ids or not self.harmless_train_ids:
            raise SOMEvidenceError("harmful and harmless activation-train IDs are required")
        if not self.validation or not self.test:
            raise SOMEvidenceError("non-empty validation and held-out test splits are required")

        groups: dict[str, tuple[str, ...]] = {
            "harmful_train": self.harmful_train_ids,
            "harmless_train": self.harmless_train_ids,
            "validation": tuple(example.example_id for example in self.validation),
            "test": tuple(example.example_id for example in self.test),
        }
        for name, identifiers in groups.items():
            if any(
                not isinstance(identifier, str) or not identifier.strip()
                for identifier in identifiers
            ):
                raise SOMEvidenceError(f"{name} contains an empty or non-string ID")
            if len(set(identifiers)) != len(identifiers):
                raise SOMEvidenceError(f"{name} contains duplicate IDs")

        names = tuple(groups)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                overlap = set(groups[left_name]) & set(groups[right_name])
                if overlap:
                    raise SOMEvidenceError(
                        f"split leakage between {left_name} and {right_name}: {sorted(overlap)!r}"
                    )

        validation_text = {example.behavior.strip() for example in self.validation}
        test_text = {example.behavior.strip() for example in self.test}
        if validation_text & test_text:
            raise SOMEvidenceError("validation and test contain duplicate behavior text")

    def fingerprints(self) -> dict[str, str]:
        return {
            "harmful_train": _canonical_json_hash(list(self.harmful_train_ids)),
            "harmless_train": _canonical_json_hash(list(self.harmless_train_ids)),
            "validation": _canonical_json_hash(
                [example.fingerprint_payload() for example in self.validation]
            ),
            "test": _canonical_json_hash([example.fingerprint_payload() for example in self.test]),
        }


@dataclass(frozen=True)
class SOMJudgeEvidence:
    """Reproducibility evidence for a binary behavioral judge."""

    protocol: str
    model_id: str
    version: str
    prompt_template_sha256: str

    def __post_init__(self) -> None:
        for name in ("protocol", "model_id", "version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SOMEvidenceError(f"judge {name} must be a non-empty string")
        if not _is_sha256(self.prompt_template_sha256):
            raise SOMEvidenceError("judge prompt_template_sha256 must be a lowercase SHA-256")

    def to_metadata(self) -> dict[str, str]:
        return {
            "protocol": self.protocol,
            "model_id": self.model_id,
            "version": self.version,
            "prompt_template_sha256": self.prompt_template_sha256,
        }


@dataclass(frozen=True)
class SOMGeneratorEvidence:
    """Reproducibility evidence for target-model generation."""

    model_id: str
    decoding: str
    implementation_version: str

    def __post_init__(self) -> None:
        for name in ("model_id", "decoding", "implementation_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SOMEvidenceError(f"generator {name} must be a non-empty string")

    def to_metadata(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "decoding": self.decoding,
            "implementation_version": self.implementation_version,
        }


class SOMCompletionGenerator(Protocol):
    """Generate completions from the currently intervened target model."""

    evidence: SOMGeneratorEvidence

    def generate(
        self,
        model: nn.Module,
        examples: tuple[SOMBehaviorExample, ...],
    ) -> Sequence[str]: ...


class SOMBehaviorJudge(Protocol):
    """HarmBench-compatible behavior/generation binary classifier."""

    evidence: SOMJudgeEvidence

    def classify(
        self,
        behaviors: Sequence[str],
        generations: Sequence[str],
    ) -> Sequence[int | bool]: ...


class HarmBenchJudgeAdapter:
    """Adapt OBLITERATUS's or another official HarmBench evaluator.

    The evaluator must return a mapping containing per-example labels under
    ``per_item``.  Aggregate-only ASR is rejected because it is insufficient
    evidence for a replayable trial.
    """

    def __init__(
        self,
        evaluator: Callable[[list[str], list[str]], Mapping[str, Any]] | None = None,
    ) -> None:
        # Import lazily so using the generic SOM API does not load Transformers.
        from obliteratus.evaluation.heretic_eval import _HARMBENCH_CLS_TEMPLATE

        self._evaluator = evaluator
        self.evidence = SOMJudgeEvidence(
            protocol="HarmBench binary classifier",
            model_id="cais/HarmBench-Llama-2-13b-cls",
            version="Mazeika-et-al-2024 / OBLITERATUS adapter v1",
            prompt_template_sha256=hashlib.sha256(
                _HARMBENCH_CLS_TEMPLATE.encode("utf-8")
            ).hexdigest(),
        )

    def classify(
        self,
        behaviors: Sequence[str],
        generations: Sequence[str],
    ) -> Sequence[int | bool]:
        evaluator = self._evaluator
        if evaluator is None:
            from obliteratus.evaluation.heretic_eval import harmbench_asr

            evaluator = harmbench_asr
        result = evaluator(list(behaviors), list(generations))
        if not isinstance(result, Mapping) or "per_item" not in result:
            raise SOMEvidenceError("HarmBench evaluator did not return per_item labels")
        labels = result["per_item"]
        if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
            raise SOMEvidenceError("HarmBench per_item evidence must be a label sequence")
        return labels


@dataclass(frozen=True)
class SOMDirectionPool:
    """All 16 paper SOM directions and their training diagnostics."""

    directions: torch.Tensor = field(repr=False, compare=False)
    prototypes: torch.Tensor = field(repr=False, compare=False)
    harmless_centroid: torch.Tensor = field(repr=False, compare=False)
    support_counts: torch.Tensor = field(repr=False, compare=False)
    neuron_indices: tuple[tuple[int, int], ...]
    harmful_example_count: int
    harmless_example_count: int
    quantization_error: float
    training_config: SOMTrainingConfig
    direction_hashes: tuple[str, ...]
    pool_sha256: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "paper_doi": PAPER_DOI,
            "paper_arxiv": PAPER_ARXIV,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_minisom_version": UPSTREAM_MINISOM_VERSION,
            "candidate_count": int(self.directions.shape[0]),
            "hidden_size": int(self.directions.shape[1]),
            "neuron_indices": [list(index) for index in self.neuron_indices],
            "support_counts": self.support_counts.tolist(),
            "harmful_example_count": self.harmful_example_count,
            "harmless_example_count": self.harmless_example_count,
            "uses_paper_dataset_sizes": (
                self.harmful_example_count == 4_000 and self.harmless_example_count == 6_000
            ),
            "quantization_error": self.quantization_error,
            "direction_hashes": list(self.direction_hashes),
            "pool_sha256": self.pool_sha256,
            "training": self.training_config.to_metadata(),
        }


def _activation_matrix(values: torch.Tensor | Sequence[torch.Tensor], name: str) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        matrix = values.detach()
        if matrix.ndim == 3 and matrix.shape[1] == 1:
            matrix = matrix.squeeze(1)
        if matrix.ndim != 2:
            raise ValueError(f"{name} must have shape [examples, hidden_size]")
        matrix = matrix.to(device="cpu", dtype=torch.float32).clone()
    else:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} must be a tensor or a sequence of tensors")
        rows: list[torch.Tensor] = []
        for value in values:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must contain only tensors")
            row = value.detach()
            if row.ndim == 2 and row.shape[0] == 1:
                row = row.squeeze(0)
            if row.ndim != 1:
                raise ValueError(f"{name} rows must have shape [hidden_size]")
            rows.append(row.to(device="cpu", dtype=torch.float32))
        if not rows:
            raise ValueError(f"{name} must not be empty")
        matrix = torch.stack(rows)

    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"{name} must not have an empty dimension")
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def _hexagonal_coordinates() -> torch.Tensor:
    """Return MiniSom-2.3.5-compatible coordinates in flat (x, y) order."""
    coordinates = []
    vertical_scale = math.sqrt(3.0) / 2.0
    for x in range(4):
        for y in range(4):
            coordinates.append((x - 0.5 * (y % 2), y * vertical_scale))
    return torch.tensor(coordinates, dtype=torch.float32)


def _normalize_direction_pool(directions: torch.Tensor) -> torch.Tensor:
    if not isinstance(directions, torch.Tensor):
        raise TypeError("directions must be a torch.Tensor")
    if directions.ndim != 2 or directions.shape[0] <= 0 or directions.shape[1] <= 0:
        raise ValueError("directions must have shape [candidates, hidden_size]")
    cpu = directions.detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(cpu).all().item()):
        raise ValueError("directions contain NaN or infinite values")
    norms = cpu.norm(dim=1)
    if bool((norms <= _EPS).any().item()):
        raise ValueError("directions contain a zero-norm candidate")
    return (cpu / norms.unsqueeze(1)).contiguous()


@torch.no_grad()
def train_paper_som_directions(
    harmful_activations: torch.Tensor | Sequence[torch.Tensor],
    harmless_activations: torch.Tensor | Sequence[torch.Tensor],
    *,
    config: SOMTrainingConfig | None = None,
) -> SOMDirectionPool:
    """Train the paper's 4x4 harmful-only SOM and return all 16 directions."""
    config = config or SOMTrainingConfig()
    harmful = _activation_matrix(harmful_activations, "harmful_activations")
    harmless = _activation_matrix(harmless_activations, "harmless_activations")
    if harmful.shape[1] != harmless.shape[1]:
        raise ValueError("harmful and harmless activations need the same hidden size")

    # Match the authors' MiniSom 2.3.5 path: seed its NumPy RandomState,
    # initialize every neuron from a random training row, then shuffle repeated
    # data indices for train_random.
    rng = np.random.RandomState(config.seed)
    # MiniSom initializes random unit vectors before random_weights_init
    # overwrites them.  Consume the same draws so all subsequent sample indices
    # match the authors' seeded upstream implementation.
    rng.rand(4, 4, harmful.shape[1])
    initial_indices = rng.randint(0, harmful.shape[0], size=16)
    prototypes = harmful[torch.from_numpy(initial_indices).long()].clone()
    iteration_indices = np.arange(config.iterations) % harmful.shape[0]
    rng.shuffle(iteration_indices)

    coordinates = _hexagonal_coordinates()
    grid_distance_sq = torch.cdist(coordinates, coordinates).square()
    for step, sample_index in enumerate(iteration_indices.tolist()):
        sample = harmful[sample_index]
        best = int((prototypes - sample).square().sum(dim=1).argmin().item())
        decay = 1.0 + (2.0 * step / config.iterations)
        learning_rate = config.learning_rate / decay
        sigma = config.sigma / decay
        neighborhood = torch.exp(-grid_distance_sq[best] / (2.0 * sigma * sigma))
        prototypes.add_(learning_rate * neighborhood.unsqueeze(1) * (sample - prototypes))

    assignments = torch.cdist(harmful, prototypes).argmin(dim=1)
    support = torch.bincount(assignments, minlength=16)
    # The authors' released direction file is ordered by harmful BMU support.
    # Make ties deterministic by the flat neuron index.
    order = sorted(range(16), key=lambda index: (-int(support[index]), index))
    order_tensor = torch.tensor(order, dtype=torch.long)
    prototypes = prototypes[order_tensor].contiguous()
    support = support[order_tensor].contiguous()
    harmless_centroid = harmless.mean(dim=0)
    directions = _normalize_direction_pool(prototypes - harmless_centroid)
    quantization_error = float(
        (harmful - prototypes[torch.cdist(harmful, prototypes).argmin(dim=1)])
        .norm(dim=1)
        .mean()
        .item()
    )
    direction_hashes = tuple(_tensor_hash(direction) for direction in directions)
    neuron_indices = tuple((index // 4, index % 4) for index in order)
    return SOMDirectionPool(
        directions=directions,
        prototypes=prototypes,
        harmless_centroid=harmless_centroid,
        support_counts=support,
        neuron_indices=neuron_indices,
        harmful_example_count=harmful.shape[0],
        harmless_example_count=harmless.shape[0],
        quantization_error=quantization_error,
        training_config=config,
        direction_hashes=direction_hashes,
        pool_sha256=_tensor_sequence_hash(tuple(directions)),
    )


@dataclass(frozen=True)
class SOMProjectionTarget:
    """One floating checkpoint tensor and its residual-stream axis."""

    name: str
    tensor: torch.Tensor = field(repr=False, compare=False)
    residual_axis: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SOMCheckpointError("projection target name must be non-empty")
        if not isinstance(self.tensor, torch.Tensor):
            raise SOMCheckpointError("projection target tensor must be a torch.Tensor")
        if isinstance(self.residual_axis, bool) or not isinstance(
            self.residual_axis, numbers.Integral
        ):
            raise SOMCheckpointError("residual_axis must be an integer")
        axis = int(self.residual_axis)
        if axis < 0:
            axis += self.tensor.ndim
        if not 0 <= axis < self.tensor.ndim:
            raise SOMCheckpointError("residual_axis is outside the target tensor")
        object.__setattr__(self, "residual_axis", axis)


def _storage_identity(tensor: torch.Tensor) -> tuple[Any, ...]:
    try:
        storage = tensor.untyped_storage()
        return (
            str(tensor.device),
            storage.data_ptr(),
            tensor.storage_offset(),
            storage.nbytes(),
        )
    except (AttributeError, RuntimeError):
        return (str(tensor.device), tensor.data_ptr(), tensor.storage_offset(), tensor.numel())


class SOMCheckpointEditor:
    """Transactional sequential projection over explicit checkpoint tensors."""

    def __init__(self, targets: Sequence[SOMProjectionTarget], hidden_size: int) -> None:
        if not targets:
            raise SOMCheckpointError("at least one projection target is required")
        self.targets = tuple(targets)
        self.hidden_size = _require_positive_int("hidden_size", hidden_size)
        self._active = False

        names: set[str] = set()
        storage: set[tuple[Any, ...]] = set()
        for target in self.targets:
            if target.name in names:
                raise SOMCheckpointError(f"duplicate projection target name: {target.name!r}")
            names.add(target.name)
            tensor = target.tensor
            if tensor.device.type == "meta" or tensor.is_sparse or tensor.is_quantized:
                raise SOMCheckpointError(
                    f"target {target.name!r} is not a dense materialized tensor"
                )
            if not tensor.is_floating_point():
                raise SOMCheckpointError(f"target {target.name!r} must use a floating dtype")
            if tensor.shape[target.residual_axis] != self.hidden_size:
                raise SOMCheckpointError(
                    f"target {target.name!r} residual axis has size "
                    f"{tensor.shape[target.residual_axis]}, expected {self.hidden_size}"
                )
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                raise SOMCheckpointError(f"target {target.name!r} contains non-finite values")
            identity = _storage_identity(tensor)
            if identity in storage:
                raise SOMCheckpointError("projection targets contain tied or duplicate storage")
            storage.add(identity)

        self._baseline = tuple(
            target.tensor.detach().to(device="cpu").clone() for target in self.targets
        )
        self.baseline_hash = self.current_hash()
        self.target_manifest_hash = _canonical_json_hash(
            [
                {
                    "name": target.name,
                    "shape": list(target.tensor.shape),
                    "dtype": str(target.tensor.dtype),
                    "residual_axis": target.residual_axis,
                }
                for target in self.targets
            ]
        )

    def current_hash(self) -> str:
        hasher = hashlib.sha256()
        for target in sorted(self.targets, key=lambda item: item.name):
            hasher.update(target.name.encode("utf-8"))
            _update_tensor_hash(hasher, target.tensor)
        return hasher.hexdigest()

    def _assert_baseline(self) -> None:
        observed = self.current_hash()
        if observed != self.baseline_hash:
            raise SOMCheckpointError(
                "checkpoint does not match the captured pre-search baseline "
                f"({observed} != {self.baseline_hash})"
            )

    @torch.no_grad()
    def _restore(self) -> None:
        for target, baseline in zip(self.targets, self._baseline):
            target.tensor.copy_(baseline.to(device=target.tensor.device, dtype=target.tensor.dtype))

    @torch.no_grad()
    def _apply(self, directions: Sequence[torch.Tensor]) -> None:
        if not directions:
            raise SOMCheckpointError("an intervention needs at least one direction")
        for direction_index, direction in enumerate(directions):
            if not isinstance(direction, torch.Tensor) or direction.ndim != 1:
                raise SOMCheckpointError("every replay direction must be a rank-one tensor")
            if direction.numel() != self.hidden_size:
                raise SOMCheckpointError(
                    f"direction {direction_index} has size {direction.numel()}, "
                    f"expected {self.hidden_size}"
                )
            if not bool(torch.isfinite(direction).all().item()):
                raise SOMCheckpointError(f"direction {direction_index} is non-finite")
            if float(direction.float().norm().item()) <= _EPS:
                raise SOMCheckpointError(f"direction {direction_index} has zero norm")

            for target in self.targets:
                tensor = target.tensor
                applied = direction.to(device=tensor.device, dtype=tensor.dtype)
                # Direction pools are normalized before entering this editor.
                # Re-normalizing after dtype conversion would make a replay's
                # bytes depend on a second, undocumented transformation.
                moved = tensor.movedim(target.residual_axis, -1)
                coefficient = torch.matmul(moved, applied)
                moved.sub_(coefficient.unsqueeze(-1) * applied)

        for target in self.targets:
            if not bool(torch.isfinite(target.tensor.detach()).all().item()):
                raise SOMCheckpointError(
                    f"projection produced non-finite values in target {target.name!r}"
                )

    @contextmanager
    def temporary(self, directions: Sequence[torch.Tensor]):
        """Apply directions, yield the edited hash, then restore exact bytes."""
        if self._active:
            raise SOMCheckpointError("checkpoint interventions are not re-entrant")
        self._assert_baseline()
        self._active = True
        try:
            self._apply(directions)
            applied_hash = self.current_hash()
            yield applied_hash
            post_evaluation_hash = self.current_hash()
            if post_evaluation_hash != applied_hash:
                raise SOMCheckpointError(
                    "projection targets changed during behavioral evaluation; "
                    "trial evidence is invalid"
                )
        finally:
            try:
                self._restore()
                restored_hash = self.current_hash()
                if restored_hash != self.baseline_hash:
                    raise SOMRollbackError(
                        "checkpoint rollback hash mismatch "
                        f"({restored_hash} != {self.baseline_hash})"
                    )
            finally:
                self._active = False

    def apply_permanent(
        self,
        directions: Sequence[torch.Tensor],
        *,
        expected_baseline_hash: str,
        expected_applied_hash: str,
    ) -> str:
        """Apply an exact winner, restoring baseline on any mismatch."""
        if self._active:
            raise SOMReplayError("cannot replay while another intervention is active")
        if self.baseline_hash != expected_baseline_hash:
            raise SOMReplayError(
                "winner was scored from a different checkpoint baseline "
                f"({self.baseline_hash} != {expected_baseline_hash})"
            )
        self._assert_baseline()
        self._active = True
        try:
            self._apply(directions)
            observed = self.current_hash()
            if observed != expected_applied_hash:
                self._restore()
                raise SOMReplayError(
                    "winner replay hash mismatch; baseline was restored "
                    f"({observed} != {expected_applied_hash})"
                )
            return observed
        except BaseException:
            if self.current_hash() != self.baseline_hash:
                self._restore()
            raise
        finally:
            self._active = False


@dataclass(frozen=True)
class SOMTrialResult:
    trial_number: int
    ordered_indices: tuple[int, ...]
    asr: float
    successes: int
    example_count: int
    edited_checkpoint_sha256: str
    evidence_sha256: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "trial_number": self.trial_number,
            "ordered_indices": list(self.ordered_indices),
            "asr": self.asr,
            "successes": self.successes,
            "example_count": self.example_count,
            "edited_checkpoint_sha256": self.edited_checkpoint_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True)
class SOMWinnerReplay:
    """Self-contained, hash-bound representation of the scored winner."""

    ordered_indices: tuple[int, ...]
    applied_directions: tuple[torch.Tensor, ...] = field(repr=False, compare=False)
    direction_sha256: tuple[str, ...]
    direction_pool_sha256: str
    baseline_checkpoint_sha256: str
    edited_checkpoint_sha256: str
    target_manifest_sha256: str
    validation_split_sha256: str
    trial_evidence_sha256: str
    asr: float
    sampler_label: str
    judge: SOMJudgeEvidence
    generator: SOMGeneratorEvidence
    schema_version: int = REPLAY_SCHEMA_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "paper_doi": PAPER_DOI,
            "ordered_indices": list(self.ordered_indices),
            "direction_sha256": list(self.direction_sha256),
            "direction_pool_sha256": self.direction_pool_sha256,
            "baseline_checkpoint_sha256": self.baseline_checkpoint_sha256,
            "edited_checkpoint_sha256": self.edited_checkpoint_sha256,
            "target_manifest_sha256": self.target_manifest_sha256,
            "validation_split_sha256": self.validation_split_sha256,
            "trial_evidence_sha256": self.trial_evidence_sha256,
            "asr": self.asr,
            "sampler_label": self.sampler_label,
            "judge": self.judge.to_metadata(),
            "generator": self.generator.to_metadata(),
        }


@dataclass(frozen=True)
class SOMSubsetSearchResult:
    trials: tuple[SOMTrialResult, ...]
    winner: SOMTrialResult
    replay: SOMWinnerReplay
    sampler_label: str
    sampler_is_paper_tpe: bool
    requested_trials: int
    split_fingerprints: Mapping[str, str]
    search_config: SOMSearchConfig

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sampler_label": self.sampler_label,
            "sampler_is_paper_tpe": self.sampler_is_paper_tpe,
            "requested_trials": self.requested_trials,
            "completed_trials": len(self.trials),
            "configuration": self.search_config.to_metadata(),
            "split_fingerprints": dict(self.split_fingerprints),
            "winner": self.winner.to_metadata(),
            "replay": self.replay.to_metadata(),
            "trials": [trial.to_metadata() for trial in self.trials],
        }


@dataclass(frozen=True)
class SOMPaperSearchResult:
    """Direction-training and behavioral-search result from the primary API."""

    direction_pool: SOMDirectionPool
    search: SOMSubsetSearchResult

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": "Piras-et-al-2026 SOM multi-directional ablation",
            "direction_pool": self.direction_pool.to_metadata(),
            "search": self.search.to_metadata(),
        }


def _component_evidence(
    generator: SOMCompletionGenerator,
    judge: SOMBehaviorJudge,
) -> tuple[SOMGeneratorEvidence, SOMJudgeEvidence]:
    if generator is None or not callable(getattr(generator, "generate", None)):
        raise SOMEvidenceError("an explicit completion generator is required")
    if judge is None or not callable(getattr(judge, "classify", None)):
        raise SOMEvidenceError("an explicit HarmBench-compatible binary judge is required")
    generator_evidence = getattr(generator, "evidence", None)
    judge_evidence = getattr(judge, "evidence", None)
    if not isinstance(generator_evidence, SOMGeneratorEvidence):
        raise SOMEvidenceError("generator must expose SOMGeneratorEvidence")
    if not isinstance(judge_evidence, SOMJudgeEvidence):
        raise SOMEvidenceError("judge must expose SOMJudgeEvidence")
    return generator_evidence, judge_evidence


def _validated_generations(values: object, expected: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SOMEvidenceError("completion generator must return a string sequence")
    generations = tuple(values)
    if len(generations) != expected:
        raise SOMEvidenceError(
            f"completion count {len(generations)} does not match validation size {expected}"
        )
    if any(not isinstance(value, str) for value in generations):
        raise SOMEvidenceError("every generated completion must be a string")
    return generations


def _validated_labels(values: object, expected: int) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise SOMEvidenceError("judge must return a binary label sequence")
    raw = tuple(values)
    if len(raw) != expected:
        raise SOMEvidenceError(
            f"judge label count {len(raw)} does not match validation size {expected}"
        )
    labels: list[int] = []
    for value in raw:
        if not isinstance(value, numbers.Integral) or int(value) not in {0, 1}:
            raise SOMEvidenceError("judge labels must be exact binary integers or booleans")
        labels.append(int(value))
    return tuple(labels)


def _fallback_subsets(
    candidate_count: int,
    subset_size: int,
    n_trials: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Scientifically labelled deterministic random fallback (not TPE)."""
    search_size = math.perm(candidate_count, subset_size)
    count = min(n_trials, search_size)
    if search_size <= n_trials:
        return tuple(itertools.permutations(range(candidate_count), subset_size))

    rng = random.Random(seed)
    seen: set[tuple[int, ...]] = set()
    ordered: list[tuple[int, ...]] = []
    population = list(range(candidate_count))
    while len(ordered) < count:
        candidate = tuple(rng.sample(population, subset_size))
        if candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return tuple(ordered)


def _choose_sampler(config: SOMSearchConfig) -> tuple[str, bool, Any | None]:
    if config.sampler == "deterministic_random_fallback":
        return "deterministic_random_fallback_NOT_TPE", False, None

    try:
        import optuna
    except ImportError as exc:
        if config.sampler == "auto" and config.allow_deterministic_fallback:
            return "deterministic_random_fallback_NOT_TPE", False, None
        raise SOMPaperError(
            "Optuna is required for the paper TPE search. Install optuna, or explicitly "
            "enable the scientifically labelled deterministic fallback."
        ) from exc
    return "optuna_tpe_ordered_rank_encoding", True, optuna


def search_som_direction_subsets(
    *,
    model: nn.Module,
    projection_targets: Sequence[SOMProjectionTarget],
    directions: torch.Tensor,
    splits: SOMEvidenceSplits,
    generator: SOMCompletionGenerator,
    judge: SOMBehaviorJudge,
    config: SOMSearchConfig | None = None,
    restore_full_state: Callable[[], None] | None = None,
) -> SOMSubsetSearchResult:
    """Search ordered direction subsets with transactional checkpoint trials."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    config = config or SOMSearchConfig()
    if not isinstance(splits, SOMEvidenceSplits):
        raise SOMEvidenceError("splits must be a validated SOMEvidenceSplits instance")
    generator_evidence, judge_evidence = _component_evidence(generator, judge)
    direction_pool = _normalize_direction_pool(directions)
    if direction_pool.shape[0] < config.subset_size:
        raise ValueError("direction pool is smaller than subset_size")

    editor = SOMCheckpointEditor(projection_targets, hidden_size=direction_pool.shape[1])
    sampler_label, sampler_is_tpe, optuna = _choose_sampler(config)
    split_fingerprints = splits.fingerprints()
    requested_trials = config.resolved_trials
    trials: list[SOMTrialResult] = []
    module_modes = tuple((module, module.training) for module in model.modules())
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    def restore_runtime_state() -> None:
        torch.random.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        for module, training in module_modes:
            module.training = training

    def evaluate(ordered_indices: tuple[int, ...]) -> float:
        if len(ordered_indices) != config.subset_size or len(set(ordered_indices)) != len(
            ordered_indices
        ):
            raise SOMPaperError("sampler produced a malformed ordered unique subset")
        if any(not 0 <= index < direction_pool.shape[0] for index in ordered_indices):
            raise SOMPaperError("sampler produced an out-of-range direction index")
        selected = tuple(direction_pool[index] for index in ordered_indices)
        if restore_full_state is not None:
            restore_full_state()
        restore_runtime_state()
        model.eval()
        try:
            with editor.temporary(selected) as edited_hash:
                raw_generations = generator.generate(model, splits.validation)
                generations = _validated_generations(raw_generations, len(splits.validation))
                raw_labels = judge.classify(
                    [example.behavior for example in splits.validation],
                    generations,
                )
                labels = _validated_labels(raw_labels, len(splits.validation))
                successes = sum(labels)
                asr = successes / len(labels)
                evidence_hash = _canonical_json_hash(
                    {
                        "validation_split_sha256": split_fingerprints["validation"],
                        "ordered_indices": list(ordered_indices),
                        "edited_checkpoint_sha256": edited_hash,
                        "generation_sha256": [
                            hashlib.sha256(generation.encode("utf-8")).hexdigest()
                            for generation in generations
                        ],
                        "labels": list(labels),
                        "judge": judge_evidence.to_metadata(),
                        "generator": generator_evidence.to_metadata(),
                    }
                )
        finally:
            if restore_full_state is not None:
                restore_full_state()
            restore_runtime_state()

        trials.append(
            SOMTrialResult(
                trial_number=len(trials),
                ordered_indices=ordered_indices,
                asr=asr,
                successes=successes,
                example_count=len(labels),
                edited_checkpoint_sha256=edited_hash,
                evidence_sha256=evidence_hash,
            )
        )
        return asr

    if sampler_is_tpe:
        startup_trials = requested_trials // 4
        sampler = optuna.samplers.TPESampler(
            seed=config.seed,
            n_startup_trials=startup_trials,
            multivariate=True,
            group=True,
        )
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial: Any) -> float:
            # Rank-without-replacement is a bijection over ordered subsets and
            # prevents duplicate directions without post-sampling replacement.
            remaining = list(range(direction_pool.shape[0]))
            ordered: list[int] = []
            for position in range(config.subset_size):
                rank = trial.suggest_int(
                    f"ordered_remaining_rank_{position}",
                    0,
                    len(remaining) - 1,
                )
                ordered.append(remaining.pop(rank))
            trial.set_user_attr("ordered_direction_indices", ordered)
            return evaluate(tuple(ordered))

        study.optimize(objective, n_trials=requested_trials, n_jobs=1)
    else:
        for ordered_indices in _fallback_subsets(
            direction_pool.shape[0],
            config.subset_size,
            requested_trials,
            config.seed,
        ):
            evaluate(ordered_indices)

    if not trials:
        raise SOMPaperError("subset search completed without trustworthy trials")
    # Match Optuna's first-best tie behavior while remaining deterministic.
    winner = max(trials, key=lambda trial: (trial.asr, -trial.trial_number))
    applied_directions = tuple(
        direction_pool[index].detach().to(device="cpu").clone() for index in winner.ordered_indices
    )
    direction_hashes = tuple(_tensor_hash(direction) for direction in applied_directions)
    replay = SOMWinnerReplay(
        ordered_indices=winner.ordered_indices,
        applied_directions=applied_directions,
        direction_sha256=direction_hashes,
        direction_pool_sha256=_tensor_sequence_hash(tuple(direction_pool)),
        baseline_checkpoint_sha256=editor.baseline_hash,
        edited_checkpoint_sha256=winner.edited_checkpoint_sha256,
        target_manifest_sha256=editor.target_manifest_hash,
        validation_split_sha256=split_fingerprints["validation"],
        trial_evidence_sha256=winner.evidence_sha256,
        asr=winner.asr,
        sampler_label=sampler_label,
        judge=judge_evidence,
        generator=generator_evidence,
    )
    return SOMSubsetSearchResult(
        trials=tuple(trials),
        winner=winner,
        replay=replay,
        sampler_label=sampler_label,
        sampler_is_paper_tpe=sampler_is_tpe,
        requested_trials=requested_trials,
        split_fingerprints=split_fingerprints,
        search_config=config,
    )


def run_paper_som_search(
    *,
    model: nn.Module,
    projection_targets: Sequence[SOMProjectionTarget],
    harmful_train_activations: torch.Tensor | Sequence[torch.Tensor],
    harmless_train_activations: torch.Tensor | Sequence[torch.Tensor],
    splits: SOMEvidenceSplits,
    generator: SOMCompletionGenerator,
    judge: SOMBehaviorJudge,
    training_config: SOMTrainingConfig | None = None,
    search_config: SOMSearchConfig | None = None,
    restore_full_state: Callable[[], None] | None = None,
) -> SOMPaperSearchResult:
    """Train the paper SOM, run behavioral BO, and return replay evidence.

    This is the single pipeline-facing entry point.  It never evaluates the
    held-out test split and leaves the supplied model at its exact baseline.
    """
    harmful = _activation_matrix(harmful_train_activations, "harmful_train_activations")
    harmless = _activation_matrix(harmless_train_activations, "harmless_train_activations")
    if harmful.shape[0] != len(splits.harmful_train_ids):
        raise SOMEvidenceError("harmful activation rows do not match harmful_train_ids evidence")
    if harmless.shape[0] != len(splits.harmless_train_ids):
        raise SOMEvidenceError("harmless activation rows do not match harmless_train_ids evidence")
    direction_pool = train_paper_som_directions(
        harmful,
        harmless,
        config=training_config,
    )
    search = search_som_direction_subsets(
        model=model,
        projection_targets=projection_targets,
        directions=direction_pool.directions,
        splits=splits,
        generator=generator,
        judge=judge,
        config=search_config,
        restore_full_state=restore_full_state,
    )
    return SOMPaperSearchResult(direction_pool=direction_pool, search=search)


def replay_som_winner(
    projection_targets: Sequence[SOMProjectionTarget],
    replay: SOMWinnerReplay,
) -> str:
    """Apply the exact scored winner permanently, or restore and fail closed."""
    if not isinstance(replay, SOMWinnerReplay):
        raise TypeError("replay must be SOMWinnerReplay")
    if replay.schema_version != REPLAY_SCHEMA_VERSION:
        raise SOMReplayError(
            f"unsupported replay schema {replay.schema_version}; expected {REPLAY_SCHEMA_VERSION}"
        )
    if len(replay.applied_directions) != len(replay.ordered_indices):
        raise SOMReplayError("replay direction count does not match ordered indices")
    observed_hashes = tuple(_tensor_hash(direction) for direction in replay.applied_directions)
    if observed_hashes != replay.direction_sha256:
        raise SOMReplayError("stored replay directions do not match their hashes")

    hidden_size = replay.applied_directions[0].numel()
    editor = SOMCheckpointEditor(projection_targets, hidden_size=hidden_size)
    if editor.target_manifest_hash != replay.target_manifest_sha256:
        raise SOMReplayError("projection target manifest differs from the scored trial")
    return editor.apply_permanent(
        replay.applied_directions,
        expected_baseline_hash=replay.baseline_checkpoint_sha256,
        expected_applied_hash=replay.edited_checkpoint_sha256,
    )


__all__ = [
    "HarmBenchJudgeAdapter",
    "SOMBehaviorExample",
    "SOMBehaviorJudge",
    "SOMCheckpointEditor",
    "SOMCheckpointError",
    "SOMCompletionGenerator",
    "SOMDirectionPool",
    "SOMEvidenceError",
    "SOMEvidenceSplits",
    "SOMGeneratorEvidence",
    "SOMJudgeEvidence",
    "SOMPaperError",
    "SOMPaperSearchResult",
    "SOMProjectionTarget",
    "SOMReplayError",
    "SOMRollbackError",
    "SOMSearchConfig",
    "SOMSubsetSearchResult",
    "SOMTrainingConfig",
    "SOMTrialResult",
    "SOMWinnerReplay",
    "replay_som_winner",
    "run_paper_som_search",
    "search_som_direction_subsets",
    "train_paper_som_directions",
]
