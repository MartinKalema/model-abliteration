"""Tests for self-organizing-map refusal direction extraction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.analysis.som_directions import SOMDirectionExtractor


def _activations(rows: torch.Tensor) -> list[torch.Tensor]:
    """Match the pipeline's usual per-prompt ``(1, hidden_dim)`` shape."""
    return [row.unsqueeze(0) for row in rows]


def test_one_neuron_matches_difference_of_means():
    harmful = torch.tensor([[3.0, 1.0], [5.0, 1.0]])
    harmless = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    extractor = SOMDirectionExtractor(
        n_iterations=1,
        candidate_count=1,
        distortion_aware=False,
    )

    result = extractor.extract(
        _activations(harmful),
        _activations(harmless),
        n_directions=1,
        layer_idx=7,
    )

    expected = harmful.mean(dim=0) - harmless.mean(dim=0)
    expected = expected / expected.norm()
    assert result.layer_idx == 7
    assert torch.allclose(result.directions[0], expected)
    assert result.coverage_score == pytest.approx(1.0)
    assert result.grid_shape == (1, 1)


def test_two_harmful_modes_produce_distinct_deterministic_directions():
    harmful = torch.tensor(
        [
            [4.0, -0.10],
            [4.2, 0.05],
            [3.8, 0.10],
            [4.1, -0.05],
            [-0.10, 4.0],
            [0.05, 4.2],
            [0.10, 3.8],
            [-0.05, 4.1],
        ]
    )
    harmless = torch.tensor(
        [[-0.1, 0.0], [0.1, 0.0], [0.0, -0.1], [0.0, 0.1]]
    )
    kwargs = {
        "n_iterations": 500,
        "learning_rate": 0.3,
        "sigma": 0.25,
        "candidate_count": 2,
        "seed": 19,
    }

    first = SOMDirectionExtractor(**kwargs).extract(
        _activations(harmful),
        _activations(harmless),
        n_directions=2,
    )
    second = SOMDirectionExtractor(**kwargs).extract(
        _activations(harmful),
        _activations(harmless),
        n_directions=2,
    )

    assert torch.equal(first.directions, second.directions)
    assert torch.equal(first.direction_scores, second.direction_scores)
    assert first.coverage_score == second.coverage_score
    assert first.quantization_error == second.quantization_error
    assert torch.allclose(first.directions.norm(dim=1), torch.ones(2), atol=1e-6)
    assert first.coverage_score == pytest.approx(1.0)
    assert torch.isfinite(first.direction_scores).all()
    assert torch.isfinite(first.signal_to_noise).all()

    axis_similarity = first.directions.abs() @ torch.eye(2)
    assert axis_similarity[:, 0].max() > 0.95
    assert axis_similarity[:, 1].max() > 0.95


def test_harmless_pc_removal_suppresses_benign_variance_axis():
    harmless = torch.tensor(
        [[-10.0, 0.0], [-5.0, 0.0], [5.0, 0.0], [10.0, 0.0]]
    )
    harmful = torch.tensor([[6.0, 4.0], [7.0, 4.0], [8.0, 4.0]])
    extractor = SOMDirectionExtractor(
        n_iterations=1,
        candidate_count=1,
        harmless_pc_count=1,
        distortion_aware=False,
    )

    result = extractor.extract(
        _activations(harmful),
        _activations(harmless),
        n_directions=1,
    )

    assert abs(result.directions[0, 0]) < 1e-5
    assert result.directions[0, 1] > 0.999


def test_harmless_pc_removal_ignores_zero_rank_tiny_dataset():
    """Zero singular vectors must not erase arbitrary feature dimensions."""
    harmful = [torch.tensor([3.0, 2.0, 1.0])]
    harmless = [torch.tensor([1.0, 1.0, 1.0])]
    expected = harmful[0] - harmless[0]
    expected = expected / expected.norm()

    result = SOMDirectionExtractor(
        candidate_count=1,
        harmless_pc_count=2,
        n_iterations=1,
    ).extract(harmful, harmless, n_directions=1)

    assert torch.allclose(result.directions[0], expected, atol=1e-6)


def test_invalid_inputs_raise_clear_errors():
    extractor = SOMDirectionExtractor(n_iterations=1, candidate_count=2)
    valid = _activations(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))

    with pytest.raises(ValueError, match="at least one activation"):
        extractor.extract([], valid, n_directions=1)
    with pytest.raises(ValueError, match="same hidden dimension"):
        extractor.extract(valid, _activations(torch.ones(2, 3)), n_directions=1)
    with pytest.raises(ValueError, match="NaN or infinite"):
        extractor.extract(
            _activations(torch.tensor([[float("nan"), 0.0], [1.0, 0.0]])),
            valid,
            n_directions=1,
        )
    with pytest.raises(ValueError, match="at least n_directions"):
        extractor.extract(valid, valid, n_directions=3)


def test_signal_threshold_rejects_all_candidates():
    extractor = SOMDirectionExtractor(
        n_iterations=1,
        candidate_count=1,
        min_signal_to_noise=1_000_000.0,
    )
    harmful = _activations(torch.tensor([[1.0, 0.0], [1.1, 0.0]]))
    harmless = _activations(torch.tensor([[-100.0, 0.0], [100.0, 0.0]]))

    with pytest.raises(ValueError, match="no finite directions"):
        extractor.extract(harmful, harmless, n_directions=1)


def test_iterative_distillation_keeps_using_som(monkeypatch):
    pipeline = AbliterationPipeline(
        model_name="fake/model",
        method="som",
        harmful_prompts=["harmful one", "harmful two"],
        harmless_prompts=["harmless one", "harmless two"],
    )
    pipeline.attention_head_surgery = False
    pipeline.handle = SimpleNamespace(hidden_size=4096, total_params=3_000_000_000)
    pipeline._harmful_acts = {
        layer: _activations(torch.tensor([[2.0, 0.0, 1.0, 0.0], [2.2, 0.1, 1.0, 0.0]]))
        for layer in range(2)
    }
    pipeline._harmless_acts = {
        layer: _activations(torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.0, 0.0]]))
        for layer in range(2)
    }
    pipeline._harmful_means = {
        layer: torch.stack(values).mean(dim=0)
        for layer, values in pipeline._harmful_acts.items()
    }
    pipeline._harmless_means = {
        layer: torch.stack(values).mean(dim=0)
        for layer, values in pipeline._harmless_acts.items()
    }

    calls = []

    class FakeExtractor:
        def extract(self, harmful, harmless, n_directions, layer_idx):
            calls.append((layer_idx, n_directions))
            directions = torch.eye(4)[:n_directions]
            return SimpleNamespace(
                directions=directions,
                direction_scores=torch.ones(n_directions),
                coverage_score=1.0,
            )

    monkeypatch.setattr(pipeline, "_make_som_extractor", lambda: FakeExtractor())
    pipeline._distill_inner()

    assert calls == [(0, 2), (1, 2)]
    assert torch.equal(pipeline.refusal_subspaces[0], torch.eye(4)[:2])
    assert torch.equal(pipeline.refusal_subspaces[1], torch.eye(4)[:2])
