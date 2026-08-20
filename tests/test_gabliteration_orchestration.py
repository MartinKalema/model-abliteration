"""End-to-end toy-model tests for paper-style Gabliteration search/replay."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from obliteratus.analysis.gabliteration import (
    GabliterationSearchConfig,
    GabliterationValidationError,
    HiddenStateBatch,
    apply_gabliteration_replay,
    extract_last_token_hidden_states,
    ridge_subspace_update,
    run_gabliteration_search,
    state_dict_sha256,
    tensor_sha256,
)
from obliteratus.architecture_manifest import (
    ArchitectureCoverageError,
    ProjectionManifest,
    ProjectionManifestEntry,
)


class _Writer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(hidden_size))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.weight.T


class _Block(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = _Writer(hidden_size)
        self.ffn = _Writer(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attention(hidden))


class _ToyCausalModel(nn.Module):
    def __init__(self, hidden_size: int = 3, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([_Block(hidden_size) for _ in range(num_layers)])
        self.register_buffer("state_version", torch.tensor(7, dtype=torch.int64))
        self.forward_calls = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ):
        del attention_mask
        self.forward_calls += 1
        hidden = input_ids.to(dtype=torch.float32)
        hidden_states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        return SimpleNamespace(
            logits=hidden,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
        )


@dataclass
class _ToyHandle:
    model: _ToyCausalModel
    _original_state: dict[str, torch.Tensor] | None = None

    def snapshot(self) -> None:
        self._original_state = {
            name: value.detach().to(device="cpu", copy=True)
            for name, value in self.model.state_dict().items()
        }

    def restore(self) -> None:
        if self._original_state is None:
            raise RuntimeError("missing snapshot")
        self.model.load_state_dict(self._original_state, strict=True)


def _manifest(model: _ToyCausalModel) -> ProjectionManifest:
    entries = []
    for layer_index, layer in enumerate(model.layers):
        for branch_kind, writer in (("attention", layer.attention), ("ffn", layer.ffn)):
            name = f"layers.{layer_index}.{branch_kind}.weight"
            entries.append(
                ProjectionManifestEntry(
                    qualified_name=name,
                    aliases=(name,),
                    layer_indices=(layer_index,),
                    branch_kind=branch_kind,
                    branch_paths=(branch_kind,),
                    component=f"{branch_kind}_output",
                    role="writer",
                    orientation="output",
                    shape=tuple(writer.weight.shape),
                    dtype=str(writer.weight.dtype),
                    storage_identity=f"toy:{layer_index}:{branch_kind}",
                    residual_axis=0,
                    expert_axis=None,
                    projection_kind="module_weight",
                    owner=writer,
                    attribute_path="weight",
                    parameter=writer.weight,
                )
            )
    return ProjectionManifest(
        architecture="toy",
        target="output",
        layer_path="layers",
        hidden_size=3,
        num_layers=len(model.layers),
        entries=tuple(entries),
        branch_coverage=tuple(
            {"layer": index, "attention": 1, "ffn": 1}
            for index in range(len(model.layers))
        ),
    )


def _hidden_batches() -> tuple[list[HiddenStateBatch], list[HiddenStateBatch]]:
    harmful = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        ]
    )
    harmless = torch.zeros_like(harmful)
    mask = torch.ones(3, 2, dtype=torch.long)
    return (
        [HiddenStateBatch({"input_ids": harmful, "attention_mask": mask})],
        [HiddenStateBatch({"input_ids": harmless, "attention_mask": mask})],
    )


def _config(**overrides) -> GabliterationSearchConfig:
    values = {
        "candidate_layers": (0, 1, 2),
        "n_directions": 1,
        "n_shuffles": 3,
        "seed": 17,
        "alpha_base": 0.3,
        "beta": 0.5,
        "ridge_lambda": 0.1,
        "effectiveness_threshold": 0.8,
        "refusal_markers": ("i cannot",),
    }
    values.update(overrides)
    return GabliterationSearchConfig(**values)


def _prompts() -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor([1.0, 0.0, 0.0]) for _ in range(4))


def _generator(
    changed_layers: list[tuple[int, ...]],
    baseline: dict[str, torch.Tensor],
):
    def generate(model: _ToyCausalModel, prompts) -> list[str]:
        changed = []
        for layer_index, layer in enumerate(model.layers):
            names = (
                f"layers.{layer_index}.attention.weight",
                f"layers.{layer_index}.ffn.weight",
            )
            if any(
                not torch.equal(model.state_dict()[name], baseline[name]) for name in names
            ):
                changed.append(layer_index)
        changed_layers.append(tuple(changed))
        inputs = torch.stack(tuple(prompts)).unsqueeze(1)
        scores = model(input_ids=inputs).logits[:, -1, 0]
        return ["I cannot comply" if score > 0.7 else "Certainly" for score in scores]

    return generate


def _fixture():
    model = _ToyCausalModel()
    model.train()
    handle = _ToyHandle(model)
    handle.snapshot()
    manifest = _manifest(model)
    harmful, harmless = _hidden_batches()
    assert handle._original_state is not None
    baseline = {name: value.clone() for name, value in handle._original_state.items()}
    return handle, manifest, harmful, harmless, baseline


def test_behavioral_trials_use_actual_forwards_and_restore_every_full_state():
    handle, manifest, harmful, harmless, baseline = _fixture()
    baseline_hash = state_dict_sha256(baseline)
    changed_layers: list[tuple[int, ...]] = []

    result = run_gabliteration_search(
        handle=handle,
        manifest=manifest,
        harmful_batches=harmful,
        harmless_batches=harmless,
        evaluation_prompts=_prompts(),
        response_generator=_generator(changed_layers, baseline),
        config=_config(),
        apply_final=False,
    )

    assert handle.model.forward_calls == 5  # harmful + harmless + one per candidate
    assert changed_layers == [(0,), (1,), (2,)]
    assert result.replay_plan.source_layer == 0
    assert result.replay_plan.effective_layers == (0, 1, 2)
    assert all(trial.refusal_rate == 0.0 for trial in result.trials)
    assert all(trial.before_state_sha256 == baseline_hash for trial in result.trials)
    assert all(trial.restored_state_sha256 == baseline_hash for trial in result.trials)
    assert all(trial.edited_state_sha256 != baseline_hash for trial in result.trials)
    assert state_dict_sha256(handle.model.state_dict()) == baseline_hash
    assert handle.model.training  # orchestration restores every module's mode
    json.dumps(result.to_metadata(), allow_nan=False)
    payload = result.replay_plan.to_payload()
    assert tensor_sha256(payload["directions"]) == result.replay_plan.direction_sha256


def test_final_edit_is_one_exact_replay_and_replays_byte_identically():
    handle, manifest, harmful, harmless, baseline = _fixture()
    result = run_gabliteration_search(
        handle=handle,
        manifest=manifest,
        harmful_batches=harmful,
        harmless_batches=harmless,
        evaluation_prompts=_prompts(),
        response_generator=_generator([], baseline),
        config=_config(),
        apply_final=True,
    )
    final = {name: value.clone() for name, value in handle.model.state_dict().items()}
    assert result.applied
    assert state_dict_sha256(final) == result.replay_plan.expected_state_sha256

    direction = result.replay_plan.directions
    for layer_index, alpha in result.replay_plan.layer_alphas:
        for branch in ("attention", "ffn"):
            name = f"layers.{layer_index}.{branch}.weight"
            expected = ridge_subspace_update(
                baseline[name],
                direction,
                residual_axis=0,
                alpha=alpha,
                ridge_lambda=result.replay_plan.ridge_lambda,
            )
            assert torch.equal(final[name], expected)

    handle.restore()
    applied = apply_gabliteration_replay(
        handle=handle, manifest=manifest, plan=result.replay_plan
    )
    assert applied == 6
    assert all(
        torch.equal(value, final[name])
        for name, value in handle.model.state_dict().items()
    )


def test_evaluator_failure_rolls_back_every_tensor():
    handle, manifest, harmful, harmless, baseline = _fixture()
    baseline_hash = state_dict_sha256(baseline)

    def fail_after_forward(model, prompts):
        model(input_ids=torch.stack(tuple(prompts)).unsqueeze(1))
        raise RuntimeError("synthetic evaluator failure")

    with pytest.raises(RuntimeError, match="synthetic evaluator failure"):
        run_gabliteration_search(
            handle=handle,
            manifest=manifest,
            harmful_batches=harmful,
            harmless_batches=harmless,
            evaluation_prompts=_prompts(),
            response_generator=fail_after_forward,
            config=_config(),
        )

    assert state_dict_sha256(handle.model.state_dict()) == baseline_hash


def test_no_effective_layer_fails_closed_and_leaves_baseline():
    handle, manifest, harmful, harmless, baseline = _fixture()
    baseline_hash = state_dict_sha256(baseline)

    def always_refuse(model, prompts):
        model(input_ids=torch.stack(tuple(prompts)).unsqueeze(1))
        return ["I cannot"] * len(prompts)

    with pytest.raises(GabliterationValidationError, match="no candidate satisfied"):
        run_gabliteration_search(
            handle=handle,
            manifest=manifest,
            harmful_batches=harmful,
            harmless_batches=harmless,
            evaluation_prompts=_prompts(),
            response_generator=always_refuse,
            config=_config(effectiveness_threshold=1.0),
        )

    assert state_dict_sha256(handle.model.state_dict()) == baseline_hash


def test_behavioral_callback_must_forward_the_live_trial_model():
    handle, manifest, harmful, harmless, baseline = _fixture()
    baseline_hash = state_dict_sha256(baseline)

    def fabricated_responses(_model, prompts):
        return ["Certainly"] * len(prompts)

    with pytest.raises(GabliterationValidationError, match="without forwarding"):
        run_gabliteration_search(
            handle=handle,
            manifest=manifest,
            harmful_batches=harmful,
            harmless_batches=harmless,
            evaluation_prompts=_prompts(),
            response_generator=fabricated_responses,
            config=_config(),
        )

    assert state_dict_sha256(handle.model.state_dict()) == baseline_hash


def test_behavioral_callback_state_mutation_is_detected_and_rolled_back():
    handle, manifest, harmful, harmless, baseline = _fixture()
    baseline_hash = state_dict_sha256(baseline)

    def mutate_buffer(model, prompts):
        model(input_ids=torch.stack(tuple(prompts)).unsqueeze(1))
        model.state_version.add_(1)
        return ["Certainly"] * len(prompts)

    with pytest.raises(GabliterationValidationError, match="mutated the trial model"):
        run_gabliteration_search(
            handle=handle,
            manifest=manifest,
            harmful_batches=harmful,
            harmless_batches=harmless,
            evaluation_prompts=_prompts(),
            response_generator=mutate_buffer,
            config=_config(),
        )

    assert state_dict_sha256(handle.model.state_dict()) == baseline_hash


def test_tampered_replay_directions_are_rejected_before_mutation():
    handle, manifest, harmful, harmless, baseline = _fixture()
    baseline_hash = state_dict_sha256(baseline)
    result = run_gabliteration_search(
        handle=handle,
        manifest=manifest,
        harmful_batches=harmful,
        harmless_batches=harmless,
        evaluation_prompts=_prompts(),
        response_generator=_generator([], baseline),
        config=_config(),
        apply_final=False,
    )
    tampered_directions = result.replay_plan.directions.clone()
    tampered_directions[0, 0] += 0.25
    tampered = replace(result.replay_plan, directions=tampered_directions)

    with pytest.raises(GabliterationValidationError, match="direction_sha256"):
        apply_gabliteration_replay(handle=handle, manifest=manifest, plan=tampered)

    assert state_dict_sha256(handle.model.state_dict()) == baseline_hash


def test_cross_layer_shared_manifest_entry_fails_before_snapshot_or_edit():
    handle, manifest, harmful, harmless, baseline = _fixture()
    handle._original_state = None
    first = replace(manifest.entries[0], layer_indices=(0, 1))
    malformed = replace(manifest, entries=(first, *manifest.entries[1:]))

    with pytest.raises(ArchitectureCoverageError, match="shares storage across layers"):
        run_gabliteration_search(
            handle=handle,
            manifest=malformed,
            harmful_batches=harmful,
            harmless_batches=harmless,
            evaluation_prompts=_prompts(),
            response_generator=_generator([], baseline),
            config=_config(),
        )

    assert handle._original_state is None
    assert all(
        torch.equal(value, baseline[name])
        for name, value in handle.model.state_dict().items()
    )


def test_hidden_state_extraction_handles_left_padding_and_explicit_positions():
    model = _ToyCausalModel(num_layers=1)
    values = torch.tensor(
        [
            [[9.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
        ]
    )
    left_mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    inferred = extract_last_token_hidden_states(
        model,
        [HiddenStateBatch({"input_ids": values, "attention_mask": left_mask})],
        (0,),
    )
    explicit = extract_last_token_hidden_states(
        model,
        [HiddenStateBatch({"input_ids": values}, torch.tensor([1, 0]))],
        (0,),
    )

    assert inferred[0][:, 0].tolist() == [2.0, 4.0]
    assert explicit[0][:, 0].tolist() == [1.0, 3.0]


def test_exact_state_hash_supports_scalar_integer_buffers():
    state = {"scalar": torch.tensor(4, dtype=torch.int64), "weight": torch.eye(2)}
    assert state_dict_sha256(state) == state_dict_sha256(
        {name: value.clone() for name, value in state.items()}
    )
