"""Tests for the Analysis-Informed Abliteration Pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from obliteratus.abliterate import METHODS, AbliterationPipeline
from obliteratus.informed_pipeline import (
    INFORMED_METHOD,
    AnalysisInsights,
    InformedAbliterationPipeline,
    InformedPipelineReport,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def insights():
    """Default AnalysisInsights for testing."""
    return AnalysisInsights()


@pytest.fixture
def pipeline(tmp_path):
    """An InformedAbliterationPipeline with no model loaded."""
    return InformedAbliterationPipeline(
        model_name="test-model",
        output_dir=str(tmp_path / "test_informed"),
    )


# ---------------------------------------------------------------------------
# AnalysisInsights
# ---------------------------------------------------------------------------

class TestAnalysisInsights:
    def test_default_values(self, insights):
        assert insights.detected_alignment_method == "unknown"
        assert insights.alignment_confidence == 0.0
        assert insights.cone_is_polyhedral is False
        assert insights.cone_dimensionality == 1.0
        assert insights.mean_pairwise_cosine == 1.0
        assert insights.per_category_directions == {}
        assert insights.direction_specificity == {}
        assert insights.cluster_count == 0
        assert insights.direction_persistence == 0.0
        assert insights.use_sparse_surgery is False
        assert insights.recommended_n_directions == 1
        assert insights.recommended_direction_method == "diff_means"
        assert insights.recommended_regularization == 0.0
        assert insights.recommended_refinement_passes == 2
        assert insights.recommended_layers == []
        assert insights.skip_layers == []

    def test_default_robustness(self, insights):
        assert insights.estimated_robustness == "unknown"
        assert insights.self_repair_estimate == 0.0
        assert insights.entanglement_score == 0.0
        assert insights.entangled_layers == []
        assert insights.clean_layers == []


class TestInformedPipelineReport:
    def test_default_report(self):
        insights = AnalysisInsights()
        report = InformedPipelineReport(insights=insights)
        assert report.analysis_duration == 0.0
        assert report.total_duration == 0.0
        assert report.ouroboros_passes == 0
        assert report.final_refusal_rate is None
        assert report.stages == []


# ---------------------------------------------------------------------------
# Method preset
# ---------------------------------------------------------------------------

class TestInformedMethod:
    def test_informed_method_in_abliterate_methods(self):
        assert "informed" in METHODS
        cfg = METHODS["informed"]
        assert cfg["norm_preserve"] is True
        assert cfg["project_biases"] is True
        assert cfg["use_chat_template"] is True
        assert cfg["n_directions"] == 1
        assert cfg["direction_method"] == "diff_means"
        assert cfg["use_whitened_svd"] is False
        assert cfg["true_iterative_refinement"] is True

    def test_informed_method_standalone(self):
        assert INFORMED_METHOD["label"] == "Informed (Analysis-Guided)"
        assert INFORMED_METHOD["n_directions"] == 1
        assert INFORMED_METHOD["direction_method"] == "diff_means"
        assert INFORMED_METHOD["norm_preserve"] is True


# ---------------------------------------------------------------------------
# Pipeline initialization
# ---------------------------------------------------------------------------

class TestPipelineInit:
    def test_method_set_to_informed(self, pipeline):
        assert pipeline.method == "informed"

    def test_default_analysis_flags(self, pipeline):
        assert pipeline._run_cone is True
        assert pipeline._run_alignment is True
        assert pipeline._run_cross_layer is True
        assert pipeline._run_sparse is True
        assert pipeline._run_defense is True

    def test_ouroboros_defaults(self, pipeline):
        assert pipeline._ouroboros_threshold == 0.5
        assert pipeline._max_ouroboros_passes == 3

    def test_informed_keeps_deterministic_path_when_bayesian_is_disabled(self, pipeline):
        pipeline._configure_bayesian_warm_start()

        assert pipeline._bayesian_trials == 0
        assert pipeline.layer_adaptive_strength is True
        assert pipeline.float_layer_interpolation is True
        assert pipeline.use_kl_optimization is False

    def test_entanglement_gate(self, pipeline):
        assert pipeline._entanglement_gate == 0.8

    def test_inherits_base_pipeline(self, pipeline):
        assert pipeline.norm_preserve is True
        assert pipeline.project_biases is True
        assert pipeline.use_chat_template is True
        assert pipeline.n_directions == 1
        assert pipeline.direction_method == "diff_means"
        assert pipeline.use_whitened_svd is False
        assert pipeline.true_iterative_refinement is True

    def test_projection_target_is_forwarded_to_base_pipeline(self):
        pipeline = InformedAbliterationPipeline(
            model_name="test",
            projection_target="all",
        )

        assert pipeline.projection_target == "all"

    def test_custom_flags(self):
        p = InformedAbliterationPipeline(
            model_name="test",
            run_cone_analysis=False,
            run_alignment_detection=False,
            ouroboros_threshold=0.3,
            max_ouroboros_passes=5,
            entanglement_gate=0.9,
        )
        assert p._run_cone is False
        assert p._run_alignment is False
        assert p._ouroboros_threshold == 0.3
        assert p._max_ouroboros_passes == 5
        assert p._entanglement_gate == 0.9


# ---------------------------------------------------------------------------
# Configuration derivation
# ---------------------------------------------------------------------------

class TestConfigurationDerivation:
    """Test the _derive_configuration logic with various insights."""

    def _make_pipeline_with_insights(self, **kwargs):
        p = InformedAbliterationPipeline(
            model_name="test",
            on_log=lambda m: None,
        )
        for k, v in kwargs.items():
            setattr(p._insights, k, v)
        return p

    def test_polyhedral_cone_more_directions(self):
        p = self._make_pipeline_with_insights(
            cone_is_polyhedral=True,
            cone_dimensionality=3.5,
        )
        p._derive_configuration()
        # Polyhedral with dim 3.5 → n_dirs = max(4, min(8, int(3.5*2))) = 7
        assert p.n_directions == 7

    def test_linear_cone_uses_single_difference_of_means(self):
        p = self._make_pipeline_with_insights(
            cone_is_polyhedral=False,
            cone_dimensionality=1.0,
        )
        p._derive_configuration()
        # Linear geometry does not justify a higher-dimensional subspace.
        assert p.n_directions == 1
        assert p.direction_method == "diff_means"
        assert p.use_whitened_svd is False

    def test_dpo_zero_regularization(self):
        p = self._make_pipeline_with_insights(
            detected_alignment_method="dpo",
            entanglement_score=0.1,
        )
        p._derive_configuration()
        assert p.regularization == 0.0

    def test_rlhf_moderate_regularization(self):
        p = self._make_pipeline_with_insights(
            detected_alignment_method="rlhf",
            entanglement_score=0.2,
        )
        p._derive_configuration()
        assert p.regularization == 0.15

    def test_cai_regularization(self):
        p = self._make_pipeline_with_insights(
            detected_alignment_method="cai",
            entanglement_score=0.2,
        )
        p._derive_configuration()
        assert p.regularization == 0.2

    def test_sft_low_regularization(self):
        p = self._make_pipeline_with_insights(
            detected_alignment_method="sft",
            entanglement_score=0.1,
        )
        p._derive_configuration()
        assert p.regularization == 0.05

    def test_high_entanglement_increases_regularization(self):
        p = self._make_pipeline_with_insights(
            detected_alignment_method="dpo",
            entanglement_score=0.7,
        )
        p._derive_configuration()
        # DPO base = 0.0, + 0.15 for high entanglement = 0.15
        assert p.regularization == 0.15

    def test_high_self_repair_more_passes(self):
        p = self._make_pipeline_with_insights(
            self_repair_estimate=0.8,
        )
        p._derive_configuration()
        assert p.refinement_passes == 3

    def test_moderate_self_repair_two_passes(self):
        p = self._make_pipeline_with_insights(
            self_repair_estimate=0.5,
        )
        p._derive_configuration()
        assert p.refinement_passes == 2

    def test_low_self_repair_one_pass(self):
        p = self._make_pipeline_with_insights(
            self_repair_estimate=0.2,
        )
        p._derive_configuration()
        assert p.refinement_passes == 1

    def test_cluster_layers_used(self):
        p = self._make_pipeline_with_insights(
            cluster_representative_layers=[5, 10, 15],
            direction_clusters=[[3, 4, 5], [9, 10, 11], [14, 15, 16]],
        )
        p.refusal_directions = {i: torch.randn(64) for i in range(20)}
        p._derive_configuration()
        # Should include all cluster layers
        assert 5 in p._insights.recommended_layers
        assert 10 in p._insights.recommended_layers

    def test_entangled_layers_skipped(self):
        p = self._make_pipeline_with_insights(
            cluster_representative_layers=[5, 10, 15],
            direction_clusters=[[3, 4, 5], [9, 10, 11], [14, 15, 16]],
            entangled_layers=[10],
        )
        p._derive_configuration()
        # Layer 10 should be skipped
        assert 10 not in p._insights.recommended_layers
        assert 10 in p._insights.skip_layers

    def test_sparse_surgery_enabled_when_rsi_high(self):
        p = self._make_pipeline_with_insights(
            mean_refusal_sparsity_index=0.7,
        )
        p._sparse_threshold = 0.5
        p._derive_configuration()
        assert p._insights.use_sparse_surgery is True

    def test_sparse_surgery_disabled_when_rsi_low(self):
        p = self._make_pipeline_with_insights(
            mean_refusal_sparsity_index=0.3,
        )
        p._sparse_threshold = 0.5
        p._derive_configuration()
        assert p._insights.use_sparse_surgery is False

    def test_whitened_svd_for_multi_direction(self):
        p = self._make_pipeline_with_insights(
            cone_is_polyhedral=True,
            cone_dimensionality=2.5,
        )
        p._derive_configuration()
        assert p.n_directions > 1
        assert p.use_whitened_svd is True

    def test_no_whitened_svd_for_single_direction(self):
        p = self._make_pipeline_with_insights(
            cone_is_polyhedral=False,
            cone_dimensionality=0.5,
        )
        p._derive_configuration()
        # dim 0.5 → max(1, min(4, int(0.5+1))) = 1
        assert p.n_directions == 1
        assert p.use_whitened_svd is False


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------

class TestFormatInsights:
    def test_format_default(self, insights):
        text = InformedAbliterationPipeline.format_insights(insights)
        assert "Analysis-Informed Pipeline" in text
        assert "UNKNOWN" in text  # detected method
        assert "LINEAR" in text  # cone type

    def test_format_polyhedral(self):
        insights = AnalysisInsights(
            detected_alignment_method="dpo",
            alignment_confidence=0.85,
            cone_is_polyhedral=True,
            cone_dimensionality=3.5,
            cluster_count=4,
        )
        text = InformedAbliterationPipeline.format_insights(insights)
        assert "DPO" in text
        assert "POLYHEDRAL" in text
        assert "3.50" in text

    def test_format_includes_derived_config(self, insights):
        insights.recommended_n_directions = 6
        insights.recommended_regularization = 0.2
        insights.recommended_refinement_passes = 3
        text = InformedAbliterationPipeline.format_insights(insights)
        assert "n_directions: 6" in text
        assert "regularization: 0.2" in text
        assert "refinement_passes: 3" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_cluster_layers_falls_back(self):
        p = InformedAbliterationPipeline(
            model_name="test",
            on_log=lambda m: None,
        )
        p._insights.cluster_representative_layers = []
        p._derive_configuration()
        assert p._insights.recommended_layers == []

    def test_regularization_capped(self):
        p = InformedAbliterationPipeline(
            model_name="test",
            on_log=lambda m: None,
        )
        p._insights.detected_alignment_method = "cai"
        p._insights.entanglement_score = 0.9
        p._derive_configuration()
        # CAI base = 0.2, + 0.15 = 0.35, capped at 0.5
        assert p.regularization <= 0.5

    def test_all_layers_entangled_keeps_some(self):
        """If all cluster layers are entangled, don't skip all of them."""
        p = InformedAbliterationPipeline(
            model_name="test",
            on_log=lambda m: None,
        )
        p._insights.cluster_representative_layers = [5]
        p._insights.direction_clusters = [[5]]
        p._insights.entangled_layers = [5]
        p._derive_configuration()
        # Should NOT skip the only layer
        assert 5 in p._insights.recommended_layers

    def test_cone_dimensionality_bounds(self):
        """Extreme cone dimensionality values are handled."""
        p = InformedAbliterationPipeline(
            model_name="test",
            on_log=lambda m: None,
        )
        # Very high dimensionality
        p._insights.cone_is_polyhedral = True
        p._insights.cone_dimensionality = 10.0
        p._derive_configuration()
        assert p.n_directions <= 8  # capped

        # Very low dimensionality
        p._insights.cone_is_polyhedral = False
        p._insights.cone_dimensionality = 0.1
        p._derive_configuration()
        assert p.n_directions >= 1  # at least 1


# ---------------------------------------------------------------------------
# Fail-closed informed orchestration
# ---------------------------------------------------------------------------

class TestFailClosedInformedOrchestration:
    @staticmethod
    def _assessment(*, damage_accepted=True, accepted=True):
        return SimpleNamespace(
            damage_accepted=damage_accepted,
            efficacy_accepted=accepted,
            accepted=accepted,
        )

    @pytest.mark.parametrize("sparse", [False, True])
    def test_each_excise_call_applies_one_persistent_pass_and_removes_hooks(
        self,
        pipeline,
        monkeypatch,
        sparse,
    ):
        class FakeHook:
            def __init__(self):
                self.removed = False

            def remove(self):
                self.removed = True

        hook = FakeHook()
        observed_pass_counts = []
        pipeline.refinement_passes = 3
        pipeline._insights.use_sparse_surgery = sparse
        monkeypatch.setattr(
            pipeline,
            "_configure_bayesian_warm_start",
            lambda: None,
        )

        def edit_once():
            observed_pass_counts.append(pipeline.refinement_passes)
            pipeline._steering_hooks.append(hook)

        target = "_excise_sparse" if sparse else "_excise"
        monkeypatch.setattr(pipeline, target, edit_once)

        pipeline._excise_informed()

        assert observed_pass_counts == [1]
        assert pipeline.refinement_passes == 3
        assert hook.removed is True
        assert pipeline._steering_hooks == []

    def test_excise_failure_still_restores_configuration_and_removes_hooks(
        self,
        pipeline,
        monkeypatch,
    ):
        class FakeHook:
            removed = False

            def remove(self):
                self.removed = True

        hook = FakeHook()
        pipeline.refinement_passes = 3
        pipeline._insights.use_sparse_surgery = False
        monkeypatch.setattr(
            pipeline,
            "_configure_bayesian_warm_start",
            lambda: None,
        )

        def fail_during_edit():
            pipeline._steering_hooks.append(hook)
            raise RuntimeError("edit failed")

        monkeypatch.setattr(pipeline, "_excise", fail_during_edit)

        with pytest.raises(RuntimeError, match="edit failed"):
            pipeline._excise_informed()

        assert pipeline.refinement_passes == 3
        assert hook.removed is True
        assert pipeline._steering_hooks == []

    def test_every_compensation_edit_is_followed_by_verification(
        self,
        pipeline,
        monkeypatch,
    ):
        rates = iter([0.8, 0.4, 0.1])
        events = []
        pipeline._strong_layers = [1]
        pipeline._max_ouroboros_passes = 2

        def verify():
            rate = next(rates)
            pipeline._quality_metrics.update(
                {
                    "refusal_rate": rate,
                    "refusal_eval_count": 30,
                    # A missing compatibility alias must not crash logging.
                    "kl_divergence": None,
                }
            )
            events.append(("verify", rate))
            return self._assessment(accepted=rate <= 0.2)

        monkeypatch.setattr(pipeline, "_verify", verify)
        monkeypatch.setattr(pipeline, "_probe", lambda: events.append(("probe", None)))
        monkeypatch.setattr(
            pipeline,
            "_distill_inner",
            lambda: events.append(("distill", None)),
        )
        monkeypatch.setattr(
            pipeline,
            "_excise_informed",
            lambda: events.append(("edit", None)),
        )
        monkeypatch.setattr(pipeline, "_remove_activation_steering", lambda: 0)

        assessment = pipeline._verify_and_compensate()

        assert assessment.accepted is True
        assert [event for event, _ in events].count("edit") == 2
        assert [event for event, _ in events].count("verify") == 3
        assert events[-1] == ("verify", 0.1)
        assert pipeline._report.ouroboros_passes == 2
        assert pipeline._report.final_refusal_rate == pytest.approx(0.1)

    def test_missing_refusal_evidence_rejects_without_another_edit(
        self,
        pipeline,
        monkeypatch,
    ):
        class Rejected(RuntimeError):
            pass

        edits = []
        assessment = self._assessment(accepted=False)
        pipeline._quality_metrics.update(
            {"refusal_rate": None, "refusal_eval_count": None}
        )
        monkeypatch.setattr(pipeline, "_verify", lambda: assessment)
        monkeypatch.setattr(
            pipeline,
            "_excise_informed",
            lambda: edits.append("edit"),
        )

        def reject(candidate):
            assert candidate is assessment
            raise Rejected("rejected")

        monkeypatch.setattr(pipeline, "_reject_and_restore", reject)

        with pytest.raises(Rejected, match="rejected"):
            pipeline._verify_and_compensate()

        assert edits == []

    def test_damage_failure_rejects_before_compensation(self, pipeline, monkeypatch):
        class Rejected(RuntimeError):
            pass

        edits = []
        assessment = self._assessment(damage_accepted=False, accepted=False)
        monkeypatch.setattr(pipeline, "_verify", lambda: assessment)
        monkeypatch.setattr(
            pipeline,
            "_excise_informed",
            lambda: edits.append("edit"),
        )
        monkeypatch.setattr(
            pipeline,
            "_reject_and_restore",
            lambda candidate: (_ for _ in ()).throw(Rejected("damage")),
        )

        with pytest.raises(Rejected, match="damage"):
            pipeline._verify_and_compensate()

        assert edits == []

    def test_run_captures_baseline_before_edit_and_does_not_save_on_rejection(
        self,
        pipeline,
        monkeypatch,
    ):
        events = []
        monkeypatch.setattr(pipeline, "_remove_activation_steering", lambda: 0)
        for name in ("_summon", "_probe", "_analyze", "_distill_informed"):
            monkeypatch.setattr(
                pipeline,
                name,
                lambda stage=name: events.append(stage),
            )
        monkeypatch.setattr(
            pipeline,
            "_capture_damage_baseline",
            lambda: events.append("baseline"),
        )
        monkeypatch.setattr(
            pipeline,
            "_excise_informed",
            lambda: events.append("edit"),
        )

        def reject():
            events.append("verify")
            raise RuntimeError("gate rejected")

        monkeypatch.setattr(pipeline, "_verify_and_compensate", reject)
        monkeypatch.setattr(
            pipeline,
            "_rebirth_informed",
            lambda: events.append("save"),
        )

        with pytest.raises(RuntimeError, match="gate rejected"):
            pipeline.run_informed()

        assert events.index("baseline") < events.index("edit")
        assert events.index("edit") < events.index("verify")
        assert "save" not in events

    def test_rebirth_delegates_to_transactional_base_path(
        self,
        pipeline,
        monkeypatch,
        tmp_path,
    ):
        expected = tmp_path / "published"
        calls = []

        def base_rebirth(instance):
            calls.append(instance)
            return expected

        monkeypatch.setattr(AbliterationPipeline, "_rebirth", base_rebirth)

        assert pipeline._rebirth_informed() == expected
        assert calls == [pipeline]
