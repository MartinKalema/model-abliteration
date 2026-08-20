"""Fail-closed data-separation tests for Bayesian candidate tuning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.architecture_manifest import ProjectionManifest, ProjectionManifestEntry
from obliteratus.bayesian_optimizer import (
    ExactTPESampler,
    _measure_kl_divergence,
    _measure_refusal_rate,
    apply_exact_projection_candidate,
    build_exact_projection_candidate,
    run_bayesian_optimization,
)


class _Tokenizer:
    def __call__(self, _prompt, **_kwargs):
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def decode(self, _tokens, **_kwargs):
        return ""


class _Model:
    def __init__(self, *, fail: bool = False):
        self.fail = fail

    def generate(self, **_kwargs):
        if self.fail:
            raise RuntimeError("generation failed")
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def __call__(self, **_kwargs):
        if self.fail:
            raise RuntimeError("forward failed")
        return SimpleNamespace(logits=torch.zeros((1, 2, 4)))


class _Pipeline:
    def __init__(self, *, fail: bool = False):
        self._discovery_harmful = ["discovery-only"]
        self.harmful_prompts = ["must-not-be-read"]
        self.use_chat_template = False
        self.formatted: list[str] = []
        self.handle = SimpleNamespace(
            model=_Model(fail=fail),
            tokenizer=_Tokenizer(),
        )

    def _get_model_device(self, _model):
        return torch.device("cpu")

    def _maybe_apply_chat_template(self, prompts):
        self.formatted = list(prompts)
        return list(prompts)

    def _free_gpu_memory(self):
        return None


def test_bayesian_refusal_tuning_reads_discovery_only():
    pipeline = _Pipeline()

    # The tokenizer emits a blank completion, which is a target failure rather
    # than a falsely successful non-refusal.
    assert _measure_refusal_rate(pipeline, n_prompts=10) == 1.0
    assert pipeline.formatted == ["discovery-only"]


def test_bayesian_refusal_tuning_treats_generation_error_as_failure():
    pipeline = _Pipeline(fail=True)

    assert _measure_refusal_rate(pipeline) == 1.0


def test_bayesian_refusal_tuning_requires_an_explicit_discovery_set():
    pipeline = _Pipeline()
    del pipeline._discovery_harmful

    with pytest.raises(RuntimeError, match="direction-discovery"):
        _measure_refusal_rate(pipeline)


def test_bayesian_kl_measurement_fails_closed_when_forward_fails():
    pipeline = _Pipeline(fail=True)

    result = _measure_kl_divergence(
        pipeline,
        [torch.zeros(4)],
        ["benign"],
    )

    assert result == float("inf")


def test_bayesian_optimizer_rejects_non_bayesian_pipeline_before_editing_weights():
    pipeline = _Pipeline()
    pipeline.method = "basic"
    sentinel = torch.tensor([1.0, 2.0])
    pipeline.sentinel_weight = sentinel

    with pytest.raises(ValueError, match="optimized/heretic"):
        run_bayesian_optimization(pipeline, n_trials=1)

    assert torch.equal(sentinel, torch.tensor([1.0, 2.0]))


def test_bayesian_presets_advertise_exact_replay_and_construct():
    from obliteratus.abliterate import METHODS, available_method_names

    for method in ("optimized", "heretic"):
        assert "Exact" in METHODS[method]["label"]
        assert "replay" in METHODS[method]["description"].lower()
        assert method in available_method_names()
        pipeline = AbliterationPipeline(model_name="must-not-load", method=method)
        assert pipeline.method == method


class _ProjectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        self.attention.o_proj = nn.Linear(3, 3, bias=False)
        self.ffn = nn.Module()
        self.ffn.down_proj = nn.Linear(5, 3, bias=False)


def _projection_entry(
    layer: _ProjectionLayer,
    *,
    path: str,
    branch_kind: str,
) -> ProjectionManifestEntry:
    module = layer
    for part in path.split("."):
        module = getattr(module, part)
    parameter = module.weight
    return ProjectionManifestEntry(
        qualified_name=f"model.layers.0.{path}.weight",
        aliases=(f"model.layers.0.{path}.weight",),
        layer_indices=(0,),
        branch_kind=branch_kind,
        branch_paths=(path.rsplit(".", 1)[0],),
        component=f"{branch_kind}_output",
        role="writer",
        orientation="output",
        shape=tuple(parameter.shape),
        dtype=str(parameter.dtype),
        storage_identity=f"storage:{path}",
        residual_axis=0,
        expert_axis=None,
        projection_kind="module_weight",
        owner=layer,
        attribute_path=path,
        parameter=parameter,
    )


def _exact_pipeline_fixture() -> tuple[AbliterationPipeline, _ProjectionLayer]:
    layer = _ProjectionLayer()
    entries = (
        _projection_entry(layer, path="attention.o_proj", branch_kind="attention"),
        _projection_entry(layer, path="ffn.down_proj", branch_kind="ffn"),
    )
    manifest = ProjectionManifest(
        architecture="toy",
        target="output",
        layer_path="model.layers",
        hidden_size=3,
        num_layers=1,
        entries=entries,
        branch_coverage=(),
    )
    pipeline = AbliterationPipeline.__new__(AbliterationPipeline)
    pipeline.method = "optimized"
    pipeline._strong_layers = [0]
    pipeline.refusal_subspaces = {
        0: torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    }
    pipeline.refusal_directions = {0: torch.tensor([1.0, 0.0, 0.0])}
    pipeline._projection_manifests = {"output": manifest}
    pipeline.projection_target = "output"
    pipeline.norm_preserve = False
    pipeline.project_biases = False
    pipeline.projection_row_fraction = 1.0
    pipeline.per_expert_directions = False
    pipeline.invert_refusal = False
    pipeline._expert_directions = {}
    return pipeline, layer


def _candidate_parameters() -> dict[str, float]:
    return {
        "attention_max_weight": 0.7,
        "attention_peak_position": 0.5,
        "attention_min_weight": 0.1,
        "attention_min_weight_distance": 0.5,
        "ffn_max_weight": 0.4,
        "ffn_peak_position": 0.5,
        "ffn_min_weight": 0.1,
        "ffn_min_weight_distance": 0.5,
        "direction_index": 0.25,
    }


def test_exact_candidate_replay_matches_scored_tensor_hash():
    pipeline, layer = _exact_pipeline_fixture()
    candidate = build_exact_projection_candidate(
        pipeline,
        trial_index=0,
        parameters=_candidate_parameters(),
    )
    originals = {name: value.detach().clone() for name, value in layer.state_dict().items()}

    first = apply_exact_projection_candidate(pipeline, candidate)
    first_state = {name: value.detach().clone() for name, value in layer.state_dict().items()}
    layer.load_state_dict(originals)
    replay = apply_exact_projection_candidate(
        pipeline,
        candidate,
        expected_state_hash=first.target_state_hash,
    )

    assert replay.target_state_hash == first.target_state_hash
    assert all(
        torch.equal(value, first_state[name])
        for name, value in layer.state_dict().items()
    )


def test_exact_candidate_rejects_mutated_direction():
    pipeline, _ = _exact_pipeline_fixture()
    candidate = build_exact_projection_candidate(
        pipeline,
        trial_index=0,
        parameters=_candidate_parameters(),
    )
    candidate.directions[0][1][0] += 1.0

    with pytest.raises(RuntimeError, match="direction tensor changed"):
        apply_exact_projection_candidate(pipeline, candidate)


def test_tpe_sampler_is_deterministic_and_observes_complete_parameter_sets():
    left = ExactTPESampler(direction_upper=3.0, seed=7)
    right = ExactTPESampler(direction_upper=3.0, seed=7)
    for index in range(12):
        left_params = left.suggest()
        right_params = right.suggest()
        assert left_params == right_params
        objective = abs(left_params["direction_index"] - 1.25) + index * 1e-4
        left.observe(left_params, objective)
        right.observe(right_params, objective)

    assert len(left.observations) == 12


def test_bayesian_cancellation_restores_the_pretrial_weights():
    pipeline, layer = _exact_pipeline_fixture()
    original = {
        name: value.detach().clone() for name, value in layer.state_dict().items()
    }
    restores: list[int] = []
    pipeline._layer_excise_weights = {}
    pipeline.damage_eval_seed = 0
    pipeline._assert_auto_projection_prerequisites = lambda _purpose: None
    pipeline.log = lambda _message: None

    def restore(_weights, *, purpose):
        assert "exact Bayesian search" in purpose
        layer.load_state_dict(original)
        restores.append(1)

    pipeline._restore_auto_projection_baseline = restore
    pipeline._verify = lambda: (_ for _ in ()).throw(KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        run_bayesian_optimization(pipeline, n_trials=1)

    assert len(restores) >= 2
    assert all(
        torch.equal(value, original[name])
        for name, value in layer.state_dict().items()
    )
