"""Focused tests for the paper-faithful SOM behavioral-search pipeline."""

from __future__ import annotations

import hashlib
import importlib.util
from dataclasses import replace

import pytest
import torch
from torch import nn

from obliteratus.analysis.som_paper import (
    SOMBehaviorExample,
    SOMCheckpointEditor,
    SOMCheckpointError,
    SOMEvidenceError,
    SOMEvidenceSplits,
    SOMGeneratorEvidence,
    SOMJudgeEvidence,
    SOMPaperError,
    SOMProjectionTarget,
    SOMReplayError,
    SOMSearchConfig,
    SOMTrainingConfig,
    replay_som_winner,
    run_paper_som_search,
    search_som_direction_subsets,
    train_paper_som_directions,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor([[1.0, 2.0], [-0.5, 3.0]]),
            requires_grad=False,
        )


def _target(model: ToyModel) -> SOMProjectionTarget:
    return SOMProjectionTarget("toy.weight", model.weight, residual_axis=1)


def _splits() -> SOMEvidenceSplits:
    return SOMEvidenceSplits(
        harmful_train_ids=("sorrybench:h0", "sorrybench:h1", "sorrybench:h2", "sorrybench:h3"),
        harmless_train_ids=("alpaca:b0", "alpaca:b1", "alpaca:b2", "alpaca:b3"),
        validation=(
            SOMBehaviorExample("harmbench-val:0", "validation behavior zero"),
            SOMBehaviorExample("harmbench-val:1", "validation behavior one"),
            SOMBehaviorExample("harmbench-val:2", "validation behavior two"),
        ),
        test=(SOMBehaviorExample("harmbench-test:0", "held-out behavior"),),
    )


class BinaryJudge:
    evidence = SOMJudgeEvidence(
        protocol="HarmBench-compatible toy binary classifier",
        model_id="toy/harmbench-judge",
        version="1",
        prompt_template_sha256=_sha("toy-harmbench-template"),
    )

    def classify(self, behaviors, generations):
        assert len(behaviors) == len(generations)
        return [generation == "unsafe" for generation in generations]


def _normalized(directions: torch.Tensor) -> torch.Tensor:
    return directions.float() / directions.float().norm(dim=1, keepdim=True)


def _projected(weight: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    result = weight.detach().clone()
    for direction in directions:
        result.sub_((result @ direction).unsqueeze(-1) * direction)
    return result


class TargetStateGenerator:
    evidence = SOMGeneratorEvidence(
        model_id="toy/target",
        decoding="deterministic state-dependent toy generations",
        implementation_version="1",
    )

    def __init__(self, target_state: torch.Tensor) -> None:
        self.target_state = target_state
        self.seen_ids: list[tuple[str, ...]] = []

    def generate(self, model, examples):
        self.seen_ids.append(tuple(example.example_id for example in examples))
        if torch.allclose(model.weight.detach(), self.target_state, atol=1e-7, rtol=0.0):
            return ["unsafe"] * len(examples)
        return ["unsafe", *("safe" for _ in examples[1:])]


def _search_fixture():
    model = ToyModel()
    directions = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, -2.0],
        ]
    )
    unit = _normalized(directions)
    desired_order = (1, 2)
    desired_state = _projected(model.weight, unit[list(desired_order)])
    generator = TargetStateGenerator(desired_state)
    result = search_som_direction_subsets(
        model=model,
        projection_targets=(_target(model),),
        directions=directions,
        splits=_splits(),
        generator=generator,
        judge=BinaryJudge(),
        config=SOMSearchConfig(
            subset_size=2,
            n_trials=6,
            sampler="deterministic_random_fallback",
            seed=17,
        ),
    )
    return model, directions, desired_order, desired_state, generator, result


def test_paper_training_uses_4x4_som_and_declared_defaults():
    defaults = SOMTrainingConfig()
    assert defaults.iterations == 10_000
    assert defaults.learning_rate == pytest.approx(0.01)
    assert defaults.sigma == pytest.approx(0.3)
    assert defaults.uses_paper_defaults

    harmful = torch.tensor([[3.0, 1.0], [3.2, 1.1], [1.0, 3.0], [1.1, 3.2], [2.5, 2.4]])
    harmless = torch.tensor([[-1.0, -1.0], [-0.8, -1.1], [-1.2, -0.9], [-0.9, -0.8]])
    result = train_paper_som_directions(
        harmful,
        harmless,
        config=SOMTrainingConfig(iterations=40, seed=9),
    )

    assert result.directions.shape == (16, 2)
    assert torch.allclose(result.directions.norm(dim=1), torch.ones(16), atol=1e-6)
    assert len(result.neuron_indices) == 16
    assert sorted(result.neuron_indices) == [(x, y) for x in range(4) for y in range(4)]
    assert result.support_counts.sum().item() == harmful.shape[0]
    assert len(result.direction_hashes) == 16
    assert len(result.pool_sha256) == 64
    assert result.to_metadata()["training"]["topology"] == "hexagonal"
    assert not result.training_config.uses_paper_defaults


def test_ordered_subset_search_intervenes_and_restores_every_trial():
    model, _, desired_order, _, generator, result = _search_fixture()

    assert result.winner.ordered_indices == desired_order
    assert result.winner.asr == pytest.approx(1.0)
    assert len(result.trials) == 6
    assert {trial.ordered_indices for trial in result.trials} == {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    }
    assert all(len(set(trial.ordered_indices)) == 2 for trial in result.trials)
    assert result.sampler_label == "deterministic_random_fallback_NOT_TPE"
    assert not result.sampler_is_paper_tpe
    assert torch.equal(model.weight.detach(), torch.tensor([[1.0, 2.0], [-0.5, 3.0]]))
    assert all(
        ids == ("harmbench-val:0", "harmbench-val:1", "harmbench-val:2")
        for ids in generator.seen_ids
    )
    assert all(
        "harmbench-test" not in identifier for ids in generator.seen_ids for identifier in ids
    )


def test_winner_replay_is_byte_exact_and_independent_of_mutated_input_pool():
    model, directions, _, desired_state, _, result = _search_fixture()
    directions.zero_()  # replay owns copied, hash-bound direction tensors

    observed = replay_som_winner((_target(model),), result.replay)

    assert observed == result.winner.edited_checkpoint_sha256
    assert observed == result.replay.edited_checkpoint_sha256
    assert torch.equal(model.weight.detach(), desired_state)
    assert result.replay.to_metadata()["trial_evidence_sha256"] == result.winner.evidence_sha256


def test_replay_rejects_changed_baseline_before_editing():
    model, _, _, _, _, result = _search_fixture()
    with torch.no_grad():
        model.weight[0, 0] += 0.125
    changed = model.weight.detach().clone()

    with pytest.raises(SOMReplayError, match="different checkpoint baseline"):
        replay_som_winner((_target(model),), result.replay)

    assert torch.equal(model.weight.detach(), changed)


def test_editor_rolls_back_on_evaluation_failure_and_detects_evaluator_mutation():
    model = ToyModel()
    baseline = model.weight.detach().clone()
    editor = SOMCheckpointEditor((_target(model),), hidden_size=2)

    with (
        pytest.raises(RuntimeError, match="judge crashed"),
        editor.temporary((torch.tensor([1.0, 0.0]),)),
    ):
        raise RuntimeError("judge crashed")
    assert torch.equal(model.weight.detach(), baseline)
    assert editor.current_hash() == editor.baseline_hash

    with (
        pytest.raises(SOMCheckpointError, match="changed during behavioral evaluation"),
        editor.temporary((torch.tensor([1.0, 0.0]),)),
        torch.no_grad(),
    ):
        model.weight.add_(1.0)
    assert torch.equal(model.weight.detach(), baseline)
    assert editor.current_hash() == editor.baseline_hash


def test_search_full_restore_callback_resets_non_targets_modes_and_rng():
    class StatefulToy(ToyModel):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("evaluation_counter", torch.tensor(0))

    class MutatingGenerator(TargetStateGenerator):
        def __init__(self, target_state: torch.Tensor) -> None:
            super().__init__(target_state)
            self.training_flags: list[bool] = []
            self.random_draws: list[torch.Tensor] = []

        def generate(self, model, examples):
            self.training_flags.append(model.training)
            self.random_draws.append(torch.rand(3))
            model.evaluation_counter.add_(1)
            return super().generate(model, examples)

    model = StatefulToy()
    model.train()
    baseline = {name: value.detach().clone() for name, value in model.state_dict().items()}
    restore_calls = 0

    def restore_full_state() -> None:
        nonlocal restore_calls
        restore_calls += 1
        model.load_state_dict(baseline, strict=True)

    generator = MutatingGenerator(torch.full_like(model.weight, 99.0))
    search_som_direction_subsets(
        model=model,
        projection_targets=(_target(model),),
        directions=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        splits=_splits(),
        generator=generator,
        judge=BinaryJudge(),
        config=SOMSearchConfig(
            subset_size=2,
            n_trials=2,
            sampler="deterministic_random_fallback",
        ),
        restore_full_state=restore_full_state,
    )

    assert restore_calls == 4
    assert model.training is True
    assert model.evaluation_counter.item() == 0
    assert generator.training_flags == [False, False]
    assert torch.equal(generator.random_draws[0], generator.random_draws[1])

def test_split_overlap_and_missing_or_invalid_judge_fail_closed():
    with pytest.raises(SOMEvidenceError, match="split leakage"):
        SOMEvidenceSplits(
            harmful_train_ids=("duplicate",),
            harmless_train_ids=("benign",),
            validation=(SOMBehaviorExample("duplicate", "v"),),
            test=(SOMBehaviorExample("test", "t"),),
        )

    model = ToyModel()
    generator = TargetStateGenerator(model.weight.detach().clone())
    kwargs = {
        "model": model,
        "projection_targets": (_target(model),),
        "directions": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "splits": _splits(),
        "generator": generator,
        "config": SOMSearchConfig(
            subset_size=2,
            n_trials=1,
            sampler="deterministic_random_fallback",
        ),
    }
    with pytest.raises(SOMEvidenceError, match="explicit HarmBench-compatible"):
        search_som_direction_subsets(judge=None, **kwargs)

    class InvalidJudge(BinaryJudge):
        def classify(self, behaviors, generations):
            return [1, "yes", 0]

    baseline = model.weight.detach().clone()
    with pytest.raises(SOMEvidenceError, match="exact binary"):
        search_som_direction_subsets(judge=InvalidJudge(), **kwargs)
    assert torch.equal(model.weight.detach(), baseline)


def test_primary_api_binds_activation_rows_and_returns_complete_metadata():
    model = ToyModel()
    harmful = torch.tensor([[3.0, 0.5], [3.1, 0.6], [0.5, 3.0], [0.6, 3.1]])
    harmless = torch.tensor([[-1.0, -1.0], [-0.9, -1.1], [-1.1, -0.9], [-0.8, -1.0]])
    splits = _splits()
    generator = TargetStateGenerator(torch.full_like(model.weight, 99.0))

    result = run_paper_som_search(
        model=model,
        projection_targets=(_target(model),),
        harmful_train_activations=harmful,
        harmless_train_activations=harmless,
        splits=splits,
        generator=generator,
        judge=BinaryJudge(),
        training_config=SOMTrainingConfig(iterations=20, seed=3),
        search_config=SOMSearchConfig(
            subset_size=2,
            n_trials=2,
            sampler="deterministic_random_fallback",
            seed=3,
        ),
    )

    metadata = result.to_metadata()
    assert metadata["direction_pool"]["candidate_count"] == 16
    assert metadata["search"]["completed_trials"] == 2
    assert metadata["search"]["split_fingerprints"]["test"] == splits.fingerprints()["test"]
    assert torch.equal(model.weight.detach(), torch.tensor([[1.0, 2.0], [-0.5, 3.0]]))

    with pytest.raises(SOMEvidenceError, match="harmful activation rows"):
        run_paper_som_search(
            model=model,
            projection_targets=(_target(model),),
            harmful_train_activations=harmful[:3],
            harmless_train_activations=harmless,
            splits=splits,
            generator=generator,
            judge=BinaryJudge(),
            training_config=SOMTrainingConfig(iterations=1),
            search_config=SOMSearchConfig(
                subset_size=2,
                n_trials=1,
                sampler="deterministic_random_fallback",
            ),
        )


def test_optuna_absence_requires_explicit_scientific_fallback():
    if importlib.util.find_spec("optuna") is not None:
        pytest.skip("environment has Optuna")

    model = ToyModel()
    generator = TargetStateGenerator(model.weight.detach().clone())
    with pytest.raises(SOMPaperError, match="Optuna is required"):
        search_som_direction_subsets(
            model=model,
            projection_targets=(_target(model),),
            directions=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            splits=_splits(),
            generator=generator,
            judge=BinaryJudge(),
            config=SOMSearchConfig(subset_size=2, n_trials=1, sampler="optuna_tpe"),
        )

    fallback = search_som_direction_subsets(
        model=model,
        projection_targets=(_target(model),),
        directions=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        splits=_splits(),
        generator=generator,
        judge=BinaryJudge(),
        config=SOMSearchConfig(
            subset_size=2,
            n_trials=1,
            sampler="auto",
            allow_deterministic_fallback=True,
        ),
    )
    assert fallback.sampler_label == "deterministic_random_fallback_NOT_TPE"
    assert not fallback.sampler_is_paper_tpe


def test_optuna_tpe_search_preserves_order_and_uniqueness_when_installed():
    if importlib.util.find_spec("optuna") is None:
        pytest.skip("Optuna is an optional runtime dependency")

    model = ToyModel()
    generator = TargetStateGenerator(model.weight.detach().clone())
    result = search_som_direction_subsets(
        model=model,
        projection_targets=(_target(model),),
        directions=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        splits=_splits(),
        generator=generator,
        judge=BinaryJudge(),
        config=SOMSearchConfig(subset_size=2, n_trials=4, sampler="optuna_tpe", seed=2),
    )

    assert result.sampler_label == "optuna_tpe_ordered_rank_encoding"
    assert result.sampler_is_paper_tpe
    assert len(result.trials) == 4
    assert all(len(trial.ordered_indices) == 2 for trial in result.trials)
    assert all(len(set(trial.ordered_indices)) == 2 for trial in result.trials)


def test_tampered_replay_direction_hash_fails_before_checkpoint_edit():
    model, _, _, _, _, result = _search_fixture()
    tampered = replace(
        result.replay,
        direction_sha256=("0" * 64, *result.replay.direction_sha256[1:]),
    )
    baseline = model.weight.detach().clone()

    with pytest.raises(SOMReplayError, match="do not match their hashes"):
        replay_som_winner((_target(model),), tampered)

    assert torch.equal(model.weight.detach(), baseline)
