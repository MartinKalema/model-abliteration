"""Tests for the opt-in telemetry module."""

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from obliteratus.telemetry import (
    _ALLOWED_METHOD_CONFIG_KEYS,
    _direction_stats,
    _extract_excise_details,
    _extract_prompt_counts,
    _extract_analysis_insights,
    _is_mount_point,
    _test_writable,
    build_report,
    disable_telemetry,
    enable_telemetry,
    is_enabled,
    maybe_send_informed_report,
    maybe_send_pipeline_report,
    restore_from_hub,
    send_report,
    storage_diagnostic,
)


def _reset_telemetry():
    import obliteratus.telemetry as t
    t._enabled = None


# ── Enable / disable ────────────────────────────────────────────────────


class TestTelemetryConfig:
    """Test telemetry enable/disable logic."""

    def setup_method(self):
        _reset_telemetry()

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            _reset_telemetry()
            assert not is_enabled()

    def test_enabled_by_default_on_hf_spaces(self):
        with patch.dict(os.environ, {"SPACE_ID": "user/space"}, clear=True):
            import obliteratus.telemetry as t
            old_val = t._ON_HF_SPACES
            t._ON_HF_SPACES = True
            _reset_telemetry()
            assert is_enabled()
            t._ON_HF_SPACES = old_val

    def test_disable_via_env_zero(self):
        with patch.dict(os.environ, {"OBLITERATUS_TELEMETRY": "0"}):
            _reset_telemetry()
            assert not is_enabled()

    def test_disable_via_env_false(self):
        with patch.dict(os.environ, {"OBLITERATUS_TELEMETRY": "false"}):
            _reset_telemetry()
            assert not is_enabled()

    def test_enable_via_env_explicit(self):
        with patch.dict(os.environ, {"OBLITERATUS_TELEMETRY": "1"}):
            _reset_telemetry()
            assert is_enabled()

    def test_enable_programmatically(self):
        enable_telemetry()
        assert is_enabled()

    def test_disable_programmatically(self):
        enable_telemetry()
        assert is_enabled()
        disable_telemetry()
        assert not is_enabled()

    def test_programmatic_overrides_env(self):
        with patch.dict(os.environ, {"OBLITERATUS_TELEMETRY": "1"}):
            disable_telemetry()
            assert not is_enabled()


# ── Report building ─────────────────────────────────────────────────────


class TestBuildReport:
    """Test report payload construction."""

    def _base_kwargs(self, **overrides):
        defaults = dict(
            architecture="LlamaForCausalLM",
            num_layers=32,
            num_heads=32,
            hidden_size=4096,
            total_params=8_000_000_000,
            method="advanced",
            method_config={"n_directions": 4, "norm_preserve": True},
            quality_metrics={"perplexity": 5.2, "refusal_rate": 0.05},
        )
        defaults.update(overrides)
        return defaults

    def test_schema_version_2(self):
        report = build_report(**self._base_kwargs())
        assert report["schema_version"] == 2

    def test_basic_fields(self):
        report = build_report(**self._base_kwargs())
        assert report["model"]["architecture"] == "LlamaForCausalLM"
        assert report["model"]["num_layers"] == 32
        assert report["model"]["total_params"] == 8_000_000_000
        assert report["method"] == "advanced"
        assert report["quality_metrics"]["refusal_rate"] == 0.05
        assert len(report["session_id"]) == 32

    def test_filters_unknown_config_keys(self):
        report = build_report(**self._base_kwargs(
            method_config={"n_directions": 1, "secret_flag": True, "nuke": "boom"},
        ))
        assert "n_directions" in report["method_config"]
        assert "secret_flag" not in report["method_config"]
        assert "nuke" not in report["method_config"]

    def test_allows_all_valid_config_keys(self):
        """Every key in the allowlist should pass through."""
        config = {k: True for k in _ALLOWED_METHOD_CONFIG_KEYS}
        report = build_report(**self._base_kwargs(method_config=config))
        for k in _ALLOWED_METHOD_CONFIG_KEYS:
            assert k in report["method_config"], f"Missing allowlisted key: {k}"

    def test_no_model_name_in_report(self):
        report = build_report(**self._base_kwargs())
        report_str = json.dumps(report)
        assert "meta-llama" not in report_str
        assert "Llama-3" not in report_str

    def test_environment_info(self):
        report = build_report(**self._base_kwargs())
        env = report["environment"]
        assert "python_version" in env
        assert "os" in env
        assert "arch" in env

    def test_stage_durations(self):
        durations = {"summon": 2.5, "probe": 10.1, "distill": 3.2}
        report = build_report(**self._base_kwargs(stage_durations=durations))
        assert report["stage_durations"] == durations

    def test_direction_stats(self):
        stats = {"direction_norms": {"10": 0.95}, "mean_direction_persistence": 0.87}
        report = build_report(**self._base_kwargs(direction_stats=stats))
        assert report["direction_stats"]["mean_direction_persistence"] == 0.87

    def test_excise_details(self):
        details = {"modified_count": 128, "used_techniques": ["head_surgery"]}
        report = build_report(**self._base_kwargs(excise_details=details))
        assert report["excise_details"]["modified_count"] == 128

    def test_prompt_counts(self):
        counts = {"harmful": 33, "harmless": 33, "jailbreak": 15}
        report = build_report(**self._base_kwargs(prompt_counts=counts))
        assert report["prompt_counts"]["harmful"] == 33
        assert report["prompt_counts"]["jailbreak"] == 15

    def test_gpu_memory(self):
        mem = {"peak_allocated_gb": 7.2, "peak_reserved_gb": 8.0}
        report = build_report(**self._base_kwargs(gpu_memory=mem))
        assert report["gpu_memory"]["peak_allocated_gb"] == 7.2

    def test_analysis_insights_filtered(self):
        """Only allowlisted analysis keys should pass through."""
        insights = {
            "detected_alignment_method": "DPO",
            "alignment_confidence": 0.92,
            "secret_internal_data": "should not appear",
        }
        report = build_report(**self._base_kwargs(analysis_insights=insights))
        assert report["analysis_insights"]["detected_alignment_method"] == "DPO"
        assert "secret_internal_data" not in report["analysis_insights"]

    def test_informed_extras(self):
        extras = {"ouroboros_passes": 3, "final_refusal_rate": 0.02, "total_duration": 120.5}
        report = build_report(**self._base_kwargs(informed_extras=extras))
        assert report["informed"]["ouroboros_passes"] == 3

    def test_optional_fields_omitted_when_empty(self):
        """Optional fields should not appear when not provided."""
        report = build_report(**self._base_kwargs())
        assert "stage_durations" not in report
        assert "direction_stats" not in report
        assert "excise_details" not in report
        assert "prompt_counts" not in report
        assert "gpu_memory" not in report
        assert "analysis_insights" not in report
        assert "informed" not in report


# ── Direction stats extraction ──────────────────────────────────────────


class TestDirectionStats:
    """Test direction quality metric extraction."""

    def test_direction_norms(self):
        pipeline = MagicMock()
        pipeline.refusal_directions = {
            0: torch.randn(128),
            1: torch.randn(128),
        }
        pipeline.refusal_subspaces = {}
        stats = _direction_stats(pipeline)
        assert "direction_norms" in stats
        assert "0" in stats["direction_norms"]
        assert "1" in stats["direction_norms"]

    def test_direction_persistence(self):
        """Adjacent layers with similar directions should have high persistence."""
        d = torch.randn(128)
        d = d / d.norm()
        pipeline = MagicMock()
        pipeline.refusal_directions = {0: d, 1: d + 0.01 * torch.randn(128)}
        pipeline.refusal_subspaces = {}
        stats = _direction_stats(pipeline)
        assert "mean_direction_persistence" in stats
        assert stats["mean_direction_persistence"] > 0.9

    def test_effective_rank(self):
        """Multi-direction subspace should yield effective rank > 1."""
        pipeline = MagicMock()
        pipeline.refusal_directions = {0: torch.randn(128)}
        # 4-direction subspace with distinct directions
        sub = torch.randn(4, 128)
        pipeline.refusal_subspaces = {0: sub}
        stats = _direction_stats(pipeline)
        assert "effective_ranks" in stats
        assert float(stats["effective_ranks"]["0"]) > 1.0

    def test_empty_directions(self):
        pipeline = MagicMock()
        pipeline.refusal_directions = {}
        pipeline.refusal_subspaces = {}
        stats = _direction_stats(pipeline)
        assert stats == {}


# ── Excise details extraction ───────────────────────────────────────────


class TestExciseDetails:
    def test_basic_excise_details(self):
        pipeline = MagicMock()
        pipeline._excise_modified_count = 64
        pipeline._refusal_heads = {10: [(0, 0.9), (3, 0.8)], 11: [(1, 0.7)]}
        pipeline._sae_directions = {}
        pipeline._expert_safety_scores = {}
        pipeline._layer_excise_weights = {}
        pipeline._expert_directions = {}
        pipeline._steering_hooks = []
        pipeline.invert_refusal = False
        pipeline.project_embeddings = False
        pipeline.activation_steering = False
        pipeline.expert_transplant = False

        details = _extract_excise_details(pipeline)
        assert details["modified_count"] == 64
        assert details["head_surgery_layers"] == 2
        assert details["total_heads_projected"] == 3
        assert "head_surgery" in details["used_techniques"]

    def test_adaptive_weights(self):
        pipeline = MagicMock()
        pipeline._excise_modified_count = None
        pipeline._refusal_heads = {}
        pipeline._sae_directions = {}
        pipeline._expert_safety_scores = {}
        pipeline._layer_excise_weights = {0: 0.2, 1: 0.8, 2: 0.5}
        pipeline._expert_directions = {}
        pipeline._steering_hooks = []
        pipeline.invert_refusal = False
        pipeline.project_embeddings = False
        pipeline.activation_steering = False
        pipeline.expert_transplant = False

        details = _extract_excise_details(pipeline)
        assert details["adaptive_weight_min"] == 0.2
        assert details["adaptive_weight_max"] == 0.8
        assert "layer_adaptive" in details["used_techniques"]


# ── Prompt counts extraction ────────────────────────────────────────────


class TestPromptCounts:
    def test_basic_counts(self):
        pipeline = MagicMock()
        pipeline.harmful_prompts = ["a"] * 33
        pipeline.harmless_prompts = ["b"] * 33
        pipeline.jailbreak_prompts = None
        counts = _extract_prompt_counts(pipeline)
        assert counts["harmful"] == 33
        assert counts["harmless"] == 33
        assert "jailbreak" not in counts

    def test_with_jailbreak(self):
        pipeline = MagicMock()
        pipeline.harmful_prompts = ["a"] * 33
        pipeline.harmless_prompts = ["b"] * 33
        pipeline.jailbreak_prompts = ["c"] * 10
        counts = _extract_prompt_counts(pipeline)
        assert counts["jailbreak"] == 10


# ── Send behavior ───────────────────────────────────────────────────────


class TestSendReport:
    def setup_method(self):
        _reset_telemetry()

    def test_does_not_send_when_disabled(self):
        disable_telemetry()
        with patch("obliteratus.telemetry._send_sync") as mock_send:
            send_report({"test": True})
            mock_send.assert_not_called()

    def test_sends_when_enabled(self):
        enable_telemetry()
        with patch("obliteratus.telemetry._send_sync") as mock_send:
            send_report({"test": True})
            import time
            time.sleep(0.1)
            mock_send.assert_called_once_with({"test": True})

    def test_send_failure_is_silent(self):
        enable_telemetry()
        with patch("obliteratus.telemetry._send_sync", side_effect=Exception("network down")) as mock_send:
            # send_report should not propagate the exception to the caller
            send_report({"test": True})
            import time
            time.sleep(0.1)  # Allow background thread to execute
            mock_send.assert_called_once_with({"test": True})


# ── Pipeline integration ────────────────────────────────────────────────


def _make_mock_pipeline():
    """Build a mock pipeline with all fields the telemetry module reads."""
    p = MagicMock()
    p.handle.summary.return_value = {
        "architecture": "LlamaForCausalLM",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 4096,
        "total_params": 8_000_000_000,
    }
    p.method = "advanced"
    p.n_directions = 4
    p.norm_preserve = True
    p.regularization = 0.1
    p.refinement_passes = 2
    p.project_biases = True
    p.use_chat_template = True
    p.use_whitened_svd = True
    p.true_iterative_refinement = False
    p.use_jailbreak_contrast = False
    p.layer_adaptive_strength = False
    p.attention_head_surgery = True
    p.safety_neuron_masking = False
    p.per_expert_directions = False
    p.use_sae_features = False
    p.invert_refusal = False
    p.project_embeddings = False
    p.embed_regularization = 0.5
    p.activation_steering = False
    p.steering_strength = 0.3
    p.expert_transplant = False
    p.transplant_blend = 0.3
    p.reflection_strength = 2.0
    p.quantization = None

    p._quality_metrics = {"perplexity": 5.2, "coherence": 0.8, "refusal_rate": 0.05}
    p._strong_layers = [10, 11, 12, 13]
    p._stage_durations = {"summon": 3.0, "probe": 12.5, "distill": 4.1, "excise": 2.0, "verify": 8.3, "rebirth": 5.0}
    p._excise_modified_count = 128

    # Direction data
    d = torch.randn(4096)
    d = d / d.norm()
    p.refusal_directions = {10: d, 11: d + 0.01 * torch.randn(4096), 12: d, 13: d}
    p.refusal_subspaces = {10: torch.randn(4, 4096)}

    # Excise details
    p._refusal_heads = {10: [(0, 0.9), (3, 0.8)]}
    p._sae_directions = {}
    p._expert_safety_scores = {}
    p._layer_excise_weights = {}
    p._expert_directions = {}
    p._steering_hooks = []

    # Prompts
    p.harmful_prompts = ["x"] * 33
    p.harmless_prompts = ["y"] * 33
    p.jailbreak_prompts = None

    return p


class TestPipelineIntegration:
    def setup_method(self):
        _reset_telemetry()

    def test_does_nothing_when_disabled(self):
        disable_telemetry()
        with patch("obliteratus.telemetry.send_report") as mock_send:
            maybe_send_pipeline_report(_make_mock_pipeline())
            mock_send.assert_not_called()

    def test_comprehensive_report(self):
        """Verify that all data points are extracted from the pipeline."""
        enable_telemetry()
        p = _make_mock_pipeline()
        with patch("obliteratus.telemetry.send_report") as mock_send:
            maybe_send_pipeline_report(p)
            mock_send.assert_called_once()
            report = mock_send.call_args[0][0]

            # Core fields
            assert report["schema_version"] == 2
            assert report["model"]["architecture"] == "LlamaForCausalLM"
            assert report["method"] == "advanced"

            # Method config — check all keys passed through
            cfg = report["method_config"]
            assert cfg["n_directions"] == 4
            assert cfg["norm_preserve"] is True
            assert cfg["use_whitened_svd"] is True
            assert cfg["attention_head_surgery"] is True

            # Quality metrics
            assert report["quality_metrics"]["perplexity"] == 5.2
            assert report["quality_metrics"]["refusal_rate"] == 0.05

            # Stage durations
            assert "stage_durations" in report
            assert report["stage_durations"]["summon"] == 3.0
            assert report["stage_durations"]["verify"] == 8.3

            # Strong layers
            assert report["strong_layers"] == [10, 11, 12, 13]

            # Direction stats
            assert "direction_stats" in report
            assert "direction_norms" in report["direction_stats"]
            assert "mean_direction_persistence" in report["direction_stats"]

            # Excise details
            assert "excise_details" in report
            assert report["excise_details"]["modified_count"] == 128
            assert "head_surgery" in report["excise_details"]["used_techniques"]

            # Prompt counts
            assert report["prompt_counts"]["harmful"] == 33
            assert report["prompt_counts"]["harmless"] == 33

            # Environment
            assert "os" in report["environment"]
            assert "python_version" in report["environment"]


# ── Informed pipeline integration ────────────────────────────────────────


@dataclass
class _MockInsights:
    detected_alignment_method: str = "DPO"
    alignment_confidence: float = 0.92
    alignment_probabilities: dict = field(default_factory=lambda: {"DPO": 0.92, "RLHF": 0.05})
    cone_is_polyhedral: bool = True
    cone_dimensionality: float = 3.2
    mean_pairwise_cosine: float = 0.45
    direction_specificity: dict = field(default_factory=lambda: {"violence": 0.8})
    cluster_count: int = 3
    direction_persistence: float = 0.87
    mean_refusal_sparsity_index: float = 0.15
    recommended_sparsity: float = 0.1
    use_sparse_surgery: bool = True
    estimated_robustness: str = "medium"
    self_repair_estimate: float = 0.3
    entanglement_score: float = 0.2
    entangled_layers: list = field(default_factory=lambda: [15, 16])
    clean_layers: list = field(default_factory=lambda: [10, 11, 12])
    recommended_n_directions: int = 6
    recommended_regularization: float = 0.05
    recommended_refinement_passes: int = 3
    recommended_layers: list = field(default_factory=lambda: [10, 11, 12, 13])
    skip_layers: list = field(default_factory=lambda: [15])


@dataclass
class _MockInformedReport:
    insights: _MockInsights = field(default_factory=_MockInsights)
    ouroboros_passes: int = 2
    final_refusal_rate: float = 0.02
    analysis_duration: float = 15.3
    total_duration: float = 85.7


class TestInformedPipelineIntegration:
    def setup_method(self):
        _reset_telemetry()

    def test_does_nothing_when_disabled(self):
        disable_telemetry()
        with patch("obliteratus.telemetry.send_report") as mock_send:
            maybe_send_informed_report(_make_mock_pipeline(), _MockInformedReport())
            mock_send.assert_not_called()

    def test_comprehensive_informed_report(self):
        enable_telemetry()
        p = _make_mock_pipeline()
        report_obj = _MockInformedReport()

        with patch("obliteratus.telemetry.send_report") as mock_send:
            maybe_send_informed_report(p, report_obj)
            mock_send.assert_called_once()
            report = mock_send.call_args[0][0]

            # All base fields present
            assert report["schema_version"] == 2
            assert report["model"]["architecture"] == "LlamaForCausalLM"
            assert "direction_stats" in report
            assert "excise_details" in report

            # Analysis insights
            ai = report["analysis_insights"]
            assert ai["detected_alignment_method"] == "DPO"
            assert ai["alignment_confidence"] == 0.92
            assert ai["cone_is_polyhedral"] is True
            assert ai["cone_dimensionality"] == 3.2
            assert ai["cluster_count"] == 3
            assert ai["self_repair_estimate"] == 0.3
            assert ai["entanglement_score"] == 0.2
            assert ai["recommended_n_directions"] == 6

            # Informed extras
            inf = report["informed"]
            assert inf["ouroboros_passes"] == 2
            assert inf["final_refusal_rate"] == 0.02
            assert inf["analysis_duration"] == 15.3
            assert inf["total_duration"] == 85.7

    def test_analysis_insights_filter_unknown_keys(self):
        enable_telemetry()
        _make_mock_pipeline()

        @dataclass
        class _BadInsights(_MockInsights):
            secret_sauce: str = "should not appear"

        report_obj = _MockInformedReport(insights=_BadInsights())
        insights = _extract_analysis_insights(report_obj)
        assert "detected_alignment_method" in insights
        assert "secret_sauce" not in insights


# ── Stage duration tracking on pipeline ──────────────────────────────────


class TestStageDurationTracking:
    def test_emit_records_durations(self):
        """Verify _emit stores durations in _stage_durations dict."""
        from obliteratus.abliterate import AbliterationPipeline

        p = AbliterationPipeline.__new__(AbliterationPipeline)
        p._stage_durations = {}
        p._excise_modified_count = None
        p._on_stage = lambda r: None

        p._emit("summon", "done", "loaded", duration=3.5)
        p._emit("probe", "done", "probed", duration=10.2)
        p._emit("excise", "done", "excised", duration=2.1, modified_count=64)

        assert p._stage_durations == {"summon": 3.5, "probe": 10.2, "excise": 2.1}
        assert p._excise_modified_count == 64

    def test_running_status_does_not_record(self):
        """Only 'done' status should record durations."""
        from obliteratus.abliterate import AbliterationPipeline

        p = AbliterationPipeline.__new__(AbliterationPipeline)
        p._stage_durations = {}
        p._excise_modified_count = None
        p._on_stage = lambda r: None

        p._emit("summon", "running", "loading...", duration=0)
        assert p._stage_durations == {}


# ── Storage helpers ──────────────────────────────────────────────────────


class TestStorageHelpers:
    """Test persistent storage helper functions."""

    def test_test_writable_valid_dir(self):
        with tempfile.TemporaryDirectory() as d:
            assert _test_writable(Path(d) / "subdir")

    def test_test_writable_unwritable(self):
        # /proc is never writable for arbitrary files
        assert not _test_writable(Path("/proc/obliteratus_test"))

    def test_is_mount_point_existing_path(self):
        # Should return a bool without raising for any existing path
        result = _is_mount_point(Path("/"))
        assert isinstance(result, bool)

    def test_is_mount_point_nonexistent(self):
        assert not _is_mount_point(Path("/nonexistent_dir_12345"))

    def test_storage_diagnostic_returns_dict(self):
        diag = storage_diagnostic()
        assert isinstance(diag, dict)
        assert "telemetry_dir" in diag
        assert "is_persistent" in diag
        assert "on_hf_spaces" in diag
        assert "telemetry_enabled" in diag
        assert "data_dir_exists" in diag


# ── Hub restore ──────────────────────────────────────────────────────────


class TestHubRestore:
    """Test Hub-to-local restore functionality."""

    def setup_method(self):
        _reset_telemetry()
        # Reset restore state so each test can trigger it
        import obliteratus.telemetry as t
        t._restore_done = False

    def test_restore_skips_when_no_repo(self):
        with patch("obliteratus.telemetry._TELEMETRY_REPO", ""):
            assert restore_from_hub() == 0

    def test_restore_deduplicates(self):
        """Records already in local JSONL should not be re-added."""
        import obliteratus.telemetry as t

        with tempfile.TemporaryDirectory() as d:
            test_file = Path(d) / "telemetry.jsonl"
            existing = {"session_id": "abc", "timestamp": "2025-01-01T00:00:00"}
            test_file.write_text(json.dumps(existing) + "\n")

            old_file = t.TELEMETRY_FILE
            old_repo = t._TELEMETRY_REPO
            t.TELEMETRY_FILE = test_file
            t._TELEMETRY_REPO = "test/repo"
            t._restore_done = False

            try:
                hub_records = [
                    {"session_id": "abc", "timestamp": "2025-01-01T00:00:00"},  # duplicate
                    {"session_id": "def", "timestamp": "2025-01-02T00:00:00"},  # new
                ]
                with patch("obliteratus.telemetry.fetch_hub_records", return_value=hub_records):
                    count = restore_from_hub()
                    assert count == 1  # Only the new record

                # Verify file contents
                lines = test_file.read_text().strip().split("\n")
                assert len(lines) == 2  # original + 1 new
            finally:
                t.TELEMETRY_FILE = old_file
                t._TELEMETRY_REPO = old_repo

    def test_restore_only_runs_once(self):
        """Calling restore_from_hub() twice should be a no-op the second time."""
        import obliteratus.telemetry as t
        t._restore_done = False

        with patch("obliteratus.telemetry._TELEMETRY_REPO", "test/repo"):
            with patch("obliteratus.telemetry.fetch_hub_records", return_value=[]):
                restore_from_hub()
                # Second call should return 0 immediately
                assert restore_from_hub() == 0
