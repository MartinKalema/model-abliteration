"""CLI dispatch tests for obliteratus.cli.main().

These tests verify argument parsing and subcommand routing without
downloading real models or running any pipeline.  They use
``unittest.mock.patch`` to capture stdout/stderr and
``pytest.raises(SystemExit)`` for argparse exits.
"""

from __future__ import annotations

import math
import sys
from io import StringIO
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from obliteratus.cli import _cmd_self_improve, _damage_pipeline_kwargs, main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _capture_exit(argv: list[str] | None, *, expect_code: int | None = None):
    """Call main(argv), expecting SystemExit; return captured stderr text."""
    buf = StringIO()
    with pytest.raises(SystemExit) as exc_info, patch("sys.stderr", buf):
        main(argv)
    if expect_code is not None:
        assert exc_info.value.code == expect_code
    return buf.getvalue()


def _fake_pipeline_modules():
    """Build lightweight pipeline modules for offline CLI constructor tests."""
    pipeline_cls = MagicMock(name="AbliterationPipeline")
    informed_cls = MagicMock(name="InformedAbliterationPipeline")

    pipeline_module = ModuleType("obliteratus.abliterate")
    pipeline_module.AbliterationPipeline = pipeline_cls
    pipeline_module.METHODS = {
        "advanced": {"label": "Advanced"},
        "informed": {"label": "Informed"},
    }
    pipeline_module.STAGES = [SimpleNamespace(key="rebirth", name="Rebirth")]

    informed_module = ModuleType("obliteratus.informed_pipeline")
    informed_module.InformedAbliterationPipeline = informed_cls

    telemetry_module = ModuleType("obliteratus.telemetry")
    telemetry_module.maybe_send_pipeline_report = MagicMock()
    return pipeline_cls, informed_cls, {
        "obliteratus.abliterate": pipeline_module,
        "obliteratus.informed_pipeline": informed_module,
        "obliteratus.telemetry": telemetry_module,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCLIDispatch:
    """Test suite for CLI argument parsing and subcommand dispatch."""

    # 1. No args -> prints help / exits with error
    def test_main_no_args_prints_help(self):
        """Calling main() with no args should exit (subcommand is required)."""
        stderr_text = _capture_exit([], expect_code=2)
        # argparse prints usage info to stderr on error
        assert "usage" in stderr_text.lower() or "required" in stderr_text.lower()

    # 2. models command lists models without error
    def test_models_command(self):
        """Calling main(['models']) should list models without raising."""
        with patch("obliteratus.cli.console") as mock_console:
            main(["models"])
        # console.print is called at least once to render the table
        assert mock_console.print.call_count >= 1

    # 3. obliterate without model arg -> error
    def test_obliterate_requires_model(self):
        """Calling main(['obliterate']) without a model arg should error."""
        stderr_text = _capture_exit(["obliterate"], expect_code=2)
        assert "model" in stderr_text.lower() or "required" in stderr_text.lower()

    # 4. obliterate --method accepts valid methods
    def test_obliterate_valid_methods(self):
        """Test that --method accepts every CLI-advertised pipeline method."""
        valid_methods = [
            "basic", "advanced", "aggressive", "spectral_cascade",
            "informed", "surgical", "optimized", "som", "inverted", "nuclear",
        ]
        for method in valid_methods:
            # Patch the actual pipeline execution so nothing runs
            with patch("obliteratus.cli._cmd_abliterate") as mock_cmd:
                main(["obliterate", "fake/model", "--method", method])
                mock_cmd.assert_called_once()
                args_passed = mock_cmd.call_args[0][0]
                assert args_passed.method == method

    # 4b. invalid methods are rejected
    def test_obliterate_rejects_invalid_method(self):
        """The CLI --method flag rejects unknown method names."""
        stderr_text = _capture_exit(
            ["obliterate", "fake/model", "--method", "nonexistent"],
            expect_code=2,
        )
        assert "invalid choice" in stderr_text.lower()

    @pytest.mark.parametrize("command", ["obliterate", "abliterate"])
    def test_gguf_flags_parse_on_primary_command_and_alias(self, command):
        """Both public spellings expose the complete GGUF import/export surface."""
        with patch("obliteratus.cli._cmd_abliterate") as mock_cmd:
            main([
                command,
                "openai/gpt-oss-20b",
                "--gguf-file", "gpt-oss-20b-Q4_K_M.gguf",
                "--base-model-id", "openai/gpt-oss-20b",
                "--tokenizer-path", "/models/tokenizer",
                "--output-format", "both",
                "--gguf-quant", "Q5_K_M",
                "--llama-cpp-dir", "/opt/llama.cpp",
                "--llama-cpp-python", "/opt/venv/bin/python",
                "--gguf-imatrix", "/models/calibration.imatrix",
                "--keep-dense-intermediate",
                "--no-post-quant-verify",
            ])

        args_passed = mock_cmd.call_args.args[0]
        assert args_passed.gguf_file == "gpt-oss-20b-Q4_K_M.gguf"
        assert args_passed.base_model_id == "openai/gpt-oss-20b"
        assert args_passed.tokenizer_path == "/models/tokenizer"
        assert args_passed.output_format == "both"
        assert args_passed.gguf_quant == "Q5_K_M"
        assert args_passed.llama_cpp_dir == "/opt/llama.cpp"
        assert args_passed.llama_cpp_python == "/opt/venv/bin/python"
        assert args_passed.gguf_imatrix == "/models/calibration.imatrix"
        assert args_passed.keep_dense_intermediate is True
        assert args_passed.post_quant_verify is False

    def test_gguf_defaults_are_backward_compatible(self):
        """HF remains the default output and post-quant verification is fail-closed."""
        with patch("obliteratus.cli._cmd_abliterate") as mock_cmd:
            main(["obliterate", "fake/model"])

        args_passed = mock_cmd.call_args.args[0]
        assert args_passed.gguf_file is None
        assert args_passed.output_format == "hf"
        assert args_passed.gguf_quant == "Q4_K_M"
        assert args_passed.keep_dense_intermediate is False
        assert args_passed.post_quant_verify is True

    def test_info_profiles_local_gguf_without_loading_weights(self):
        """Architecture inspection must not dequantize a multi-gigabyte GGUF."""
        profile = MagicMock()
        profile.to_json.return_value = {
            "model_type": "gpt-oss",
            "total_params": 20_914_757_184,
        }
        with (
            patch("obliteratus.model_profile.profile_model", return_value=profile) as profiler,
            patch("obliteratus.models.loader.load_model") as loader,
            patch("obliteratus.cli.console"),
        ):
            main(["info", "/models/gpt-oss-20b-Q4_K_M.gguf"])

        profiler.assert_called_once_with("/models/gpt-oss-20b-Q4_K_M.gguf")
        loader.assert_not_called()

    def test_remote_gguf_fails_before_starting_ssh_runner(self):
        """Large local GGUF artifacts are never silently assumed to exist remotely."""
        with (
            patch("obliteratus.cli._make_remote_runner") as make_runner,
            pytest.raises(SystemExit) as exc_info,
        ):
            main([
                "obliterate",
                "/models/gpt-oss-20b-Q4_K_M.gguf",
                "--base-model-id", "openai/gpt-oss-20b",
                "--remote", "gpu.example.test",
            ])

        assert exc_info.value.code == 2
        make_runner.assert_not_called()

    def test_positional_gguf_is_inferred_and_forwarded_to_pipeline(self):
        """A local GGUF positional path needs no duplicate --gguf-file flag."""
        gguf_path = "/models/gpt-oss-20b/gpt-oss-20b-Q4_K_M.gguf"
        pipeline_cls, _, fake_modules = _fake_pipeline_modules()
        with (
            patch.dict(sys.modules, fake_modules),
            patch("obliteratus.cli._damage_pipeline_kwargs", return_value={}),
            patch("rich.live.Live"),
            patch("obliteratus.cli.console"),
        ):
            pipeline_cls.return_value.run.return_value = "/tmp/output-bundle"
            main([
                "obliterate",
                gguf_path,
                "--base-model-id", "openai/gpt-oss-20b",
                "--tokenizer-path", "/models/gpt-oss-tokenizer",
                "--output-format", "both",
                "--gguf-quant", "Q5_K_M",
                "--llama-cpp-dir", "/opt/llama.cpp",
                "--llama-cpp-python", "/opt/venv/bin/python",
                "--gguf-imatrix", "/models/calibration.imatrix",
                "--keep-dense-intermediate",
                "--no-post-quant-verify",
            ])

        forwarded = pipeline_cls.call_args.kwargs
        assert forwarded["model_name"] == gguf_path
        assert forwarded["output_dir"] == "abliterated/gpt-oss-20b-Q4_K_M"
        assert forwarded["gguf_file"] == gguf_path
        assert forwarded["base_model_id"] == "openai/gpt-oss-20b"
        assert forwarded["tokenizer_path"] == "/models/gpt-oss-tokenizer"
        assert forwarded["output_format"] == "both"
        assert forwarded["gguf_quant"] == "Q5_K_M"
        assert forwarded["llama_cpp_dir"] == "/opt/llama.cpp"
        assert forwarded["llama_cpp_python"] == "/opt/venv/bin/python"
        assert forwarded["gguf_imatrix"] == "/models/calibration.imatrix"
        assert forwarded["keep_dense_intermediate"] is True
        assert forwarded["post_quant_verify"] is False

    def test_informed_pipeline_receives_gguf_constructor_options(self):
        """The informed analysis pipeline supports the same GGUF lifecycle."""
        base_cls, informed_cls, fake_modules = _fake_pipeline_modules()
        with (
            patch.dict(sys.modules, fake_modules),
            patch("obliteratus.cli._damage_pipeline_kwargs", return_value={}),
            patch("rich.live.Live"),
            patch("obliteratus.cli.console"),
        ):
            informed_cls.return_value.run_informed.return_value = (
                "/tmp/output-bundle",
                object(),
            )
            main([
                "obliterate",
                "google/gemma-4-26B-A4B-it",
                "--method", "informed",
                "--gguf-file", "gemma-4-26B-A4B-it-Q4_K_M.gguf",
                "--base-model-id", "google/gemma-4-26B-A4B-it",
                "--output-format", "gguf",
                "--llama-cpp-dir", "/opt/llama.cpp",
            ])

        base_cls.assert_not_called()
        forwarded = informed_cls.call_args.kwargs
        assert forwarded["gguf_file"] == "gemma-4-26B-A4B-it-Q4_K_M.gguf"
        assert forwarded["base_model_id"] == "google/gemma-4-26B-A4B-it"
        assert forwarded["output_format"] == "gguf"
        assert forwarded["gguf_quant"] == "Q4_K_M"
        assert forwarded["llama_cpp_dir"] == "/opt/llama.cpp"
        assert forwarded["post_quant_verify"] is True

    def test_residue_metadata_stays_inside_gguf_bundle_directory(self, tmp_path):
        """Ancillary metadata is written beside, not beneath, the GGUF artifact."""
        result_bundle = tmp_path / "bundle.gguf"
        result_bundle.mkdir()
        pipeline_cls, _, fake_modules = _fake_pipeline_modules()
        pipeline_cls.return_value.run.return_value = str(result_bundle)

        hard_negative_module = ModuleType("obliteratus.hard_negative")
        hard_negative_module.build_weighted_prompt_pairs = MagicMock(
            return_value=(
                ["harmful"],
                ["harmless"],
                {
                    "residue_examples": 1,
                    "residue_added_pairs": 5,
                    "total_pairs": 6,
                },
            )
        )
        fake_modules["obliteratus.hard_negative"] = hard_negative_module

        with (
            patch.dict(sys.modules, fake_modules),
            patch("obliteratus.cli._damage_pipeline_kwargs", return_value={}),
            patch("rich.live.Live"),
            patch("obliteratus.cli.console"),
        ):
            main([
                "obliterate",
                "/models/input.gguf",
                "--output-dir", str(result_bundle),
                "--output-format", "gguf",
                "--residue-file", "/tmp/residue.json",
            ])

        assert (result_bundle / "hard_negative_residue.json").is_file()

    def test_self_improve_som_keeps_preset_direction_method(self):
        """An omitted low-level override must not replace the SOM preset."""
        with patch("obliteratus.cli._cmd_self_improve") as mock_cmd:
            main([
                "self-improve",
                "fake/model",
                "--audit", "audit.json",
                "--output-dir", "candidate",
                "--method", "som",
            ])
        args_passed = mock_cmd.call_args[0][0]
        assert args_passed.method == "som"
        assert args_passed.direction_method is None

    @pytest.mark.parametrize("command", ["obliterate", "self-improve"])
    def test_projection_target_auto_is_advertised(self, command):
        """Both checkpoint-producing CLI surfaces expose gated target search."""
        argv = [command, "fake/model"]
        target = "_cmd_abliterate"
        if command == "self-improve":
            argv.extend(["--audit", "audit.json", "--output-dir", "candidate"])
            target = "_cmd_self_improve"
        argv.extend(["--projection-target", "auto"])

        with patch(f"obliteratus.cli.{target}") as mock_cmd:
            main(argv)

        assert mock_cmd.call_args.args[0].projection_target == "auto"

    def test_obliterate_damage_safety_flags_parse(self):
        """Damage budgets and explicit risky edit controls are exposed together."""
        with patch("obliteratus.cli._cmd_abliterate") as mock_cmd:
            main([
                "obliterate",
                "fake/model",
                "--unsafe-disable-damage-gate",
                "--damage-eval-size", "96",
                "--max-ppl-ratio", "1.03",
                "--max-sampled-token-kl", "0.02",
                "--max-p95-sampled-token-kl", "0.08",
                "--max-top1-flip-rate", "0.01",
                "--max-coherence-drop", "0.04",
                "--max-refusal-rate", "0.15",
                "--project-lm-head",
                "--no-project-embeddings",
                "--overwrite-output",
            ])

        args_passed = mock_cmd.call_args[0][0]
        assert args_passed.damage_gate_enabled is False
        assert args_passed.damage_eval_size == 96
        assert args_passed.max_ppl_ratio == pytest.approx(1.03)
        assert args_passed.max_sampled_token_kl == pytest.approx(0.02)
        assert args_passed.max_p95_sampled_token_kl == pytest.approx(0.08)
        assert args_passed.max_top1_flip_rate == pytest.approx(0.01)
        assert args_passed.max_coherence_drop == pytest.approx(0.04)
        assert args_passed.max_refusal_rate == pytest.approx(0.15)
        assert args_passed.project_lm_head is True
        assert args_passed.project_embeddings is False
        assert args_passed.overwrite_output is True

    def test_damage_flags_default_to_fail_closed_and_preserve_method_edits(self):
        """Omitted edit flags defer to presets, while checkpoint gating stays on."""
        with patch("obliteratus.cli._cmd_abliterate") as mock_cmd:
            main(["obliterate", "fake/model", "--method", "nuclear"])

        args_passed = mock_cmd.call_args[0][0]
        assert args_passed.damage_gate_enabled is True
        assert args_passed.damage_eval_size == 64
        assert args_passed.project_lm_head is None
        assert args_passed.project_embeddings is None
        assert args_passed.overwrite_output is False

    @pytest.mark.parametrize(
        "flag,value",
        [
            ("--damage-eval-size", "31"),
            ("--max-ppl-ratio", "0.99"),
            ("--max-sampled-token-kl", "-0.1"),
            ("--max-p95-sampled-token-kl", "nan"),
            ("--max-top1-flip-rate", "1.1"),
            ("--max-coherence-drop", "-0.1"),
            ("--max-refusal-rate", "inf"),
        ],
    )
    def test_obliterate_rejects_invalid_damage_budgets(self, flag, value):
        stderr_text = _capture_exit(
            ["obliterate", "fake/model", flag, value],
            expect_code=2,
        )
        assert "must be" in stderr_text.lower()

    def test_damage_pipeline_kwargs_convert_and_forward_policy(self):
        """The CLI PPL ratio is converted to the gate's paired NLL budget."""
        args = SimpleNamespace(
            damage_gate_enabled=True,
            damage_eval_size=80,
            max_ppl_ratio=1.04,
            max_sampled_token_kl=0.03,
            max_p95_sampled_token_kl=0.11,
            max_top1_flip_rate=0.015,
            max_coherence_drop=0.06,
            max_refusal_rate=0.12,
            project_lm_head=False,
            project_embeddings=True,
            overwrite_output=True,
        )

        kwargs = _damage_pipeline_kwargs(args)

        assert kwargs["damage_gate_enabled"] is True
        assert kwargs["damage_eval_max_samples"] == 80
        assert kwargs["project_lm_head"] is False
        assert kwargs["project_embeddings"] is True
        assert kwargs["overwrite_output"] is True
        budget = kwargs["damage_budget"]
        assert budget.damage.max_nll_increase_upper_ci == pytest.approx(math.log(1.04))
        assert budget.damage.max_sampled_token_kl_upper_ci == pytest.approx(0.03)
        assert budget.damage.max_p95_sampled_token_kl == pytest.approx(0.11)
        assert budget.damage.max_top1_flip_rate == pytest.approx(0.015)
        assert budget.damage.max_coherence_drop == pytest.approx(0.06)
        assert budget.efficacy.max_refusal_rate == pytest.approx(0.12)

    def test_obliterate_forwards_damage_policy_to_pipeline(self):
        """The parsed policy reaches the pipeline constructor unchanged."""
        with (
            patch("obliteratus.abliterate.AbliterationPipeline") as pipeline_cls,
            patch("obliteratus.telemetry.maybe_send_pipeline_report"),
            patch("rich.live.Live"),
            patch("obliteratus.cli.console"),
        ):
            pipeline_cls.return_value.run.return_value = "/tmp/accepted-candidate"
            main([
                "obliterate",
                "fake/model",
                "--damage-eval-size", "72",
                "--max-ppl-ratio", "1.02",
                "--max-refusal-rate", "0.08",
                "--no-project-lm-head",
                "--project-embeddings",
                "--overwrite-output",
            ])

        forwarded = pipeline_cls.call_args.kwargs
        assert forwarded["damage_gate_enabled"] is True
        assert forwarded["damage_eval_max_samples"] == 72
        assert forwarded["project_lm_head"] is False
        assert forwarded["project_embeddings"] is True
        assert forwarded["overwrite_output"] is True
        assert forwarded[
            "damage_budget"
        ].damage.max_nll_increase_upper_ci == pytest.approx(math.log(1.02))
        assert forwarded[
            "damage_budget"
        ].efficacy.max_refusal_rate == pytest.approx(0.08)

    def test_informed_obliterate_uses_analysis_pipeline_with_same_gate(self):
        """The advertised informed method uses its feedback loop and safety policy."""
        with (
            patch("obliteratus.abliterate.AbliterationPipeline") as base_cls,
            patch(
                "obliteratus.informed_pipeline.InformedAbliterationPipeline"
            ) as informed_cls,
            patch("obliteratus.telemetry.maybe_send_pipeline_report"),
            patch("rich.live.Live"),
            patch("obliteratus.cli.console"),
        ):
            informed_cls.return_value.run_informed.return_value = (
                "/tmp/accepted-informed",
                object(),
            )
            main([
                "obliterate",
                "fake/model",
                "--method", "informed",
                "--projection-target", "all",
                "--damage-eval-size", "80",
                "--max-refusal-rate", "0.09",
                "--no-project-embeddings",
            ])

        base_cls.assert_not_called()
        informed_cls.return_value.run_informed.assert_called_once_with()
        forwarded = informed_cls.call_args.kwargs
        assert forwarded["damage_gate_enabled"] is True
        assert forwarded["damage_eval_max_samples"] == 80
        assert forwarded["project_embeddings"] is False
        assert forwarded["projection_target"] == "all"
        assert forwarded[
            "damage_budget"
        ].efficacy.max_refusal_rate == pytest.approx(0.09)

    def test_remote_obliterate_forwards_declared_damage_policy(self):
        """Remote execution must not silently fall back to different budgets."""
        with (
            patch("obliteratus.cli._make_remote_runner") as make_runner,
            patch("obliteratus.cli.console"),
        ):
            make_runner.return_value.run_obliterate.return_value = "/tmp/result"
            main([
                "obliterate",
                "fake/model",
                "--remote", "gpu.example",
                "--unsafe-disable-damage-gate",
                "--damage-eval-size", "88",
                "--max-ppl-ratio", "1.025",
                "--max-sampled-token-kl", "0.03",
                "--max-p95-sampled-token-kl", "0.09",
                "--max-top1-flip-rate", "0.01",
                "--max-coherence-drop", "0.05",
                "--max-refusal-rate", "0.11",
                "--project-lm-head",
                "--no-project-embeddings",
                "--overwrite-output",
            ])

        forwarded = make_runner.return_value.run_obliterate.call_args.kwargs
        assert forwarded["damage_gate_enabled"] is False
        assert forwarded["damage_eval_size"] == 88
        assert forwarded["max_ppl_ratio"] == pytest.approx(1.025)
        assert forwarded["max_sampled_token_kl"] == pytest.approx(0.03)
        assert forwarded["max_p95_sampled_token_kl"] == pytest.approx(0.09)
        assert forwarded["max_top1_flip_rate"] == pytest.approx(0.01)
        assert forwarded["max_coherence_drop"] == pytest.approx(0.05)
        assert forwarded["max_refusal_rate"] == pytest.approx(0.11)
        assert forwarded["project_lm_head"] is True
        assert forwarded["project_embeddings"] is False
        assert forwarded["overwrite_output"] is True

    def test_self_improve_accepts_the_same_damage_policy_flags(self):
        """Recursive hard-negative runs use the same predeclared acceptance policy."""
        with patch("obliteratus.cli._cmd_self_improve") as mock_cmd:
            main([
                "self-improve",
                "fake/model",
                "--audit", "audit.json",
                "--output-dir", "candidate",
                "--damage-eval-size", "48",
                "--max-refusal-rate", "0.10",
                "--no-project-lm-head",
                "--project-embeddings",
            ])

        args_passed = mock_cmd.call_args[0][0]
        assert args_passed.damage_gate_enabled is True
        assert args_passed.damage_eval_size == 48
        assert args_passed.max_refusal_rate == pytest.approx(0.10)
        assert args_passed.project_lm_head is False
        assert args_passed.project_embeddings is True

    def test_self_improve_defers_artifacts_until_checkpoint_commit(self, tmp_path):
        """Planning files must not make the transactional destination non-empty."""
        from obliteratus.model_profile import ModelProfile

        audit_path = tmp_path / "audit.json"
        audit_path.write_text('{"examples": []}')
        output_path = tmp_path / "candidate"
        profile = ModelProfile(
            model="fake/model",
            source="test",
            total_params=1_000,
            total_params_b=0.000001,
            active_params_b=0.000001,
            num_layers=2,
            hidden_size=8,
            intermediate_size=16,
            vocab_size=32,
            model_type="test",
            dtype="float16",
        )
        defaults = {
            "n_directions": 1,
            "regularization": 0.3,
            "refinement_passes": 1,
            "residue_weight": 1,
            "verify_sample_size": 30,
            "note": "test defaults",
        }

        def commit_checkpoint():
            assert not output_path.exists()
            output_path.mkdir()
            (output_path / "config.json").write_text("{}")
            return str(output_path)

        with (
            patch("obliteratus.model_profile.profile_model", return_value=profile),
            patch(
                "obliteratus.model_profile.default_self_improve_params",
                return_value=defaults,
            ),
            patch(
                "obliteratus.hard_negative.build_weighted_prompt_pairs",
                return_value=(
                    ["harmful"],
                    ["harmless"],
                    {
                        "residue_examples": 0,
                        "residue_added_pairs": 0,
                        "total_pairs": 1,
                    },
                ),
            ),
            patch("obliteratus.abliterate.AbliterationPipeline") as pipeline_cls,
            patch("obliteratus.cli.console"),
        ):
            pipeline_cls.return_value.run.side_effect = commit_checkpoint
            main([
                "self-improve",
                "fake/model",
                "--audit", str(audit_path),
                "--output-dir", str(output_path),
            ])

        assert (output_path / "self_improve_plan.json").is_file()
        assert (output_path / "mined_residue.json").is_file()
        assert (output_path / "hard_negative_residue.json").is_file()
        assert not list(tmp_path.glob(".candidate.self-improve-*"))

    def test_self_improve_rejection_cleans_staging_without_publishing(self, tmp_path):
        """A rejected candidate leaves neither output artifacts nor staging debris."""
        from obliteratus.model_profile import ModelProfile

        audit_path = tmp_path / "audit.json"
        audit_path.write_text('{"examples": []}')
        output_path = tmp_path / "candidate"
        profile = ModelProfile(
            model="fake/model",
            source="test",
            total_params=None,
            total_params_b=None,
            active_params_b=None,
            num_layers=None,
            hidden_size=None,
            intermediate_size=None,
            vocab_size=None,
            model_type=None,
        )
        defaults = {
            "n_directions": 1,
            "regularization": 0.3,
            "refinement_passes": 1,
            "residue_weight": 1,
            "verify_sample_size": 30,
            "note": "test defaults",
        }

        with (
            patch("obliteratus.model_profile.profile_model", return_value=profile),
            patch(
                "obliteratus.model_profile.default_self_improve_params",
                return_value=defaults,
            ),
            patch(
                "obliteratus.hard_negative.build_weighted_prompt_pairs",
                return_value=(
                    ["harmful"],
                    ["harmless"],
                    {
                        "residue_examples": 0,
                        "residue_added_pairs": 0,
                        "total_pairs": 1,
                    },
                ),
            ),
            patch("obliteratus.abliterate.AbliterationPipeline") as pipeline_cls,
            patch("obliteratus.cli.console"),
            pytest.raises(RuntimeError, match="rejected"),
        ):
            pipeline_cls.return_value.run.side_effect = RuntimeError("rejected")
            main([
                "self-improve",
                "fake/model",
                "--audit", str(audit_path),
                "--output-dir", str(output_path),
            ])

        assert not output_path.exists()
        assert not list(tmp_path.glob(".candidate.self-improve-*"))

    def test_informed_self_improve_uses_analysis_pipeline_and_gate(self, tmp_path):
        """Hard-negative informed runs keep analysis and acceptance gating together."""
        from obliteratus.model_profile import ModelProfile

        audit_path = tmp_path / "audit.json"
        audit_path.write_text('{"examples": []}')
        output_path = tmp_path / "candidate"
        profile = ModelProfile(
            model="fake/model",
            source="test",
            total_params=None,
            total_params_b=None,
            active_params_b=None,
            num_layers=None,
            hidden_size=None,
            intermediate_size=None,
            vocab_size=None,
            model_type=None,
        )
        defaults = {
            "n_directions": 1,
            "regularization": 0.3,
            "refinement_passes": 1,
            "residue_weight": 1,
            "verify_sample_size": 30,
            "note": "test defaults",
        }
        args = SimpleNamespace(
            model="fake/model",
            output_dir=str(output_path),
            audit=[str(audit_path)],
            residue_out=None,
            dataset="builtin",
            residue_max=None,
            residue_weight=None,
            params_b=None,
            no_param_auto_scale=False,
            method="informed",
            direction_method=None,
            n_directions=None,
            regularization=None,
            refinement_passes=None,
            min_layer_fraction=None,
            max_layer_fraction=None,
            harmless_pc_count=None,
            shield_concept_count=None,
            shield_ridge=None,
            shield_residualize=None,
            shield_layer_penalty=None,
            projection_target="all",
            projection_row_fraction=None,
            device="cpu",
            dtype="float16",
            verify_sample_size=None,
            dry_run=False,
            damage_gate_enabled=True,
            damage_eval_size=64,
            max_ppl_ratio=1.02,
            max_sampled_token_kl=None,
            max_p95_sampled_token_kl=None,
            max_top1_flip_rate=None,
            max_coherence_drop=None,
            max_refusal_rate=0.10,
            project_lm_head=False,
            project_embeddings=False,
            overwrite_output=False,
        )

        def commit_informed():
            assert not output_path.exists()
            output_path.mkdir()
            return str(output_path), object()

        with (
            patch("obliteratus.model_profile.profile_model", return_value=profile),
            patch(
                "obliteratus.model_profile.default_self_improve_params",
                return_value=defaults,
            ),
            patch(
                "obliteratus.hard_negative.build_weighted_prompt_pairs",
                return_value=(
                    ["harmful"],
                    ["harmless"],
                    {
                        "residue_examples": 0,
                        "residue_added_pairs": 0,
                        "total_pairs": 1,
                    },
                ),
            ),
            patch("obliteratus.abliterate.AbliterationPipeline") as base_cls,
            patch(
                "obliteratus.informed_pipeline.InformedAbliterationPipeline"
            ) as informed_cls,
            patch("obliteratus.cli.console"),
        ):
            informed_cls.return_value.run_informed.side_effect = commit_informed
            _cmd_self_improve(args)

        base_cls.assert_not_called()
        informed_cls.return_value.run_informed.assert_called_once_with()
        forwarded = informed_cls.call_args.kwargs
        assert forwarded["damage_gate_enabled"] is True
        assert forwarded["project_lm_head"] is False
        assert forwarded["project_embeddings"] is False
        assert forwarded["projection_target"] == "all"
        assert forwarded[
            "damage_budget"
        ].damage.max_nll_increase_upper_ci == pytest.approx(math.log(1.02))
        assert forwarded[
            "damage_budget"
        ].efficacy.max_refusal_rate == pytest.approx(0.10)

    # 5. run requires config path
    def test_run_requires_config(self):
        """Calling main(['run']) without a config path should error."""
        stderr_text = _capture_exit(["run"], expect_code=2)
        assert "config" in stderr_text.lower() or "required" in stderr_text.lower()

    # 6. aggregate with nonexistent dir handles gracefully
    def test_aggregate_command_missing_dir(self):
        """Calling main(['aggregate']) with nonexistent dir should handle gracefully."""
        with patch("obliteratus.cli.console") as mock_console:
            main(["aggregate", "--dir", "/nonexistent/path/to/nowhere"])
        # The command prints a message about no contributions found and returns
        printed_text = " ".join(
            str(call) for call in mock_console.print.call_args_list
        )
        assert "no contributions found" in printed_text.lower() or mock_console.print.called

    # 7. --help flag prints help
    def test_help_flag(self):
        """Calling main(['--help']) should print help and exit 0."""
        buf = StringIO()
        with pytest.raises(SystemExit) as exc_info, patch("sys.stdout", buf):
            main(["--help"])
        assert exc_info.value.code == 0
        output = buf.getvalue()
        assert "obliteratus" in output.lower() or "usage" in output.lower()

    # 8. interactive subcommand is registered
    def test_interactive_command_exists(self):
        """Verify 'interactive' subcommand is registered and dispatches."""
        with patch("obliteratus.cli._cmd_interactive") as mock_cmd:
            main(["interactive"])
            mock_cmd.assert_called_once()

    # 9. --contribute and --contribute-notes are accepted on obliterate
    def test_contribute_flags_on_obliterate(self):
        """Verify --contribute and --contribute-notes are accepted args."""
        with patch("obliteratus.cli._cmd_abliterate") as mock_cmd:
            main([
                "obliterate", "fake/model",
                "--contribute",
                "--contribute-notes", "Testing contribution system",
            ])
            mock_cmd.assert_called_once()
            args_passed = mock_cmd.call_args[0][0]
            assert args_passed.contribute is True
            assert args_passed.contribute_notes == "Testing contribution system"
