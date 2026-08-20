"""Semantic regression tests for descriptive category-direction dispersion."""

from __future__ import annotations

import math

import pytest
import torch

from obliteratus.analysis.concept_geometry import (
    CategoryDirectionDispersionAnalyzer,
    ConceptConeAnalyzer,
    ConeConeResult,
)


def _activation_pairs(
    directions: list[torch.Tensor],
    *,
    prompts_per_category: int = 2,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[int, str]]:
    harmful: list[torch.Tensor] = []
    harmless: list[torch.Tensor] = []
    labels: dict[int, str] = {}
    for category_index, direction in enumerate(directions):
        for _ in range(prompts_per_category):
            prompt_index = len(harmful)
            harmful.append(direction.clone())
            harmless.append(torch.zeros_like(direction))
            labels[prompt_index] = f"category_{category_index}"
    return harmful, harmless, labels


def test_opposite_rays_retain_negative_cosine_and_are_not_coherent():
    positive = torch.tensor([1.0, 0.0, 0.0])
    negative = -positive
    harmful, harmless, labels = _activation_pairs([positive, negative])

    result = CategoryDirectionDispersionAnalyzer(labels).analyze_layer(
        harmful,
        harmless,
    )

    assert result.pairwise_cosines[("category_0", "category_1")] == pytest.approx(-1.0)
    assert result.pairwise_angles_degrees[("category_0", "category_1")] == pytest.approx(
        180.0
    )
    assert result.angular_dispersion == pytest.approx(1.0)
    assert result.is_directionally_coherent is False
    assert result.is_directionally_disperse is True
    assert result.category_directions[0].directional_distinctiveness == pytest.approx(1.0)
    assert result.category_directions[1].directional_distinctiveness == pytest.approx(1.0)


def test_effective_rank_uses_squared_singular_value_weights():
    # Rows [e1, e1, e2] have squared singular values [2, 1].  The expected
    # entropy rank therefore uses probabilities [2/3, 1/3], not normalized
    # singular values [sqrt(2), 1].
    e1 = torch.tensor([1.0, 0.0])
    e2 = torch.tensor([0.0, 1.0])
    harmful, harmless, labels = _activation_pairs([e1, e1, e2])

    result = CategoryDirectionDispersionAnalyzer(labels).analyze_layer(
        harmful,
        harmless,
    )

    probabilities = torch.tensor([2.0 / 3.0, 1.0 / 3.0])
    expected = torch.exp(-(probabilities * probabilities.log()).sum()).item()
    assert result.effective_rank == pytest.approx(expected, rel=1e-6)


def test_angular_diagnostics_are_dimension_agnostic_and_not_solid_angles():
    angle_radians = math.radians(60.0)
    low_dimensional = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([math.cos(angle_radians), math.sin(angle_radians)]),
    ]
    high_dimensional = [torch.nn.functional.pad(vector, (0, 62)) for vector in low_dimensional]

    low_data = _activation_pairs(low_dimensional)
    high_data = _activation_pairs(high_dimensional)
    low_result = CategoryDirectionDispersionAnalyzer(low_data[2]).analyze_layer(
        low_data[0], low_data[1]
    )
    high_result = CategoryDirectionDispersionAnalyzer(high_data[2]).analyze_layer(
        high_data[0], high_data[1]
    )

    assert low_result.mean_pairwise_angle_degrees == pytest.approx(60.0, abs=1e-5)
    assert high_result.mean_pairwise_angle_degrees == pytest.approx(60.0, abs=1e-5)
    assert "cone_solid_angle" not in low_result.__dict__
    report = CategoryDirectionDispersionAnalyzer.format_report(low_result)
    assert "NOT CAUSALLY VALIDATED" in report
    assert "Solid angle" not in report
    assert "steradian" not in report.lower()


def test_nonempty_analysis_requires_an_explicit_label_for_every_pair():
    harmful = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    harmless = [torch.zeros(2), torch.zeros(2)]

    with pytest.raises(TypeError):
        CategoryDirectionDispersionAnalyzer()  # type: ignore[call-arg]

    analyzer = CategoryDirectionDispersionAnalyzer({0: "labeled"}, min_category_size=1)
    with pytest.raises(ValueError, match="missing prompt indices: 1"):
        analyzer.analyze_layer(harmful, harmless)

    with pytest.raises(ValueError, match="non-empty strings"):
        CategoryDirectionDispersionAnalyzer({0: ""})


def test_results_explicitly_disclaim_causal_validation():
    directions = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    harmful, harmless, labels = _activation_pairs(directions)
    analyzer = CategoryDirectionDispersionAnalyzer(labels)

    layer_result = analyzer.analyze_layer(harmful, harmless, layer_idx=4)
    multi_result = analyzer.analyze_all_layers(
        {4: harmful, 7: harmful},
        {4: harmless, 7: harmless},
    )

    assert layer_result.causally_validated is False
    assert multi_result.causally_validated is False
    assert layer_result.analysis_kind == "descriptive_category_direction_dispersion"
    assert multi_result.analysis_kind == "descriptive_category_direction_dispersion"


def test_legacy_cone_names_are_warning_emitting_compatibility_shims():
    directions = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
    harmful, harmless, labels = _activation_pairs(directions)

    with pytest.deprecated_call(match="ConceptConeAnalyzer"):
        analyzer = ConceptConeAnalyzer(category_map=labels)
    result = analyzer.analyze_layer(harmful, harmless)

    assert isinstance(result, ConeConeResult)
    with pytest.deprecated_call(match="cone_dimensionality"):
        assert result.cone_dimensionality == result.effective_rank
    with pytest.deprecated_call(match="cone_solid_angle"):
        legacy_value = result.cone_solid_angle
    assert legacy_value == pytest.approx(math.radians(90.0))
    with pytest.deprecated_call(match="is_polyhedral"):
        assert result.is_polyhedral == result.is_directionally_disperse


def test_legacy_analyzer_no_longer_fabricates_unknown_or_default_labels():
    harmful = [torch.tensor([1.0, 0.0])]
    harmless = [torch.zeros(2)]

    with pytest.deprecated_call():
        analyzer = ConceptConeAnalyzer()
    with pytest.raises(ValueError, match="Explicit category labels"):
        analyzer.analyze_layer(harmful, harmless)
