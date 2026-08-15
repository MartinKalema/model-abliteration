"""Focused tests for safe remote CLI command construction."""

from __future__ import annotations

import shlex

import pytest

from obliteratus.remote import RemoteConfig, RemoteRunner


def test_remote_obliterate_forwards_damage_and_edit_policy():
    runner = RemoteRunner(RemoteConfig(host="gpu.example"))

    command = runner.build_obliterate_command(
        model="org/model name",
        output_dir="/tmp/output dir",
        damage_gate_enabled=False,
        damage_eval_size=96,
        max_ppl_ratio=1.03,
        max_sampled_token_kl=0.02,
        max_p95_sampled_token_kl=0.08,
        max_top1_flip_rate=0.01,
        max_coherence_drop=0.04,
        max_refusal_rate=0.15,
        project_lm_head=False,
        project_embeddings=True,
        min_layer_fraction=0.2,
        max_layer_fraction=0.75,
        harmless_pc_count=3,
        shield_concept_count=4,
        shield_ridge=0.07,
        shield_residualize=True,
        shield_layer_penalty=0.15,
        projection_target="auto",
        projection_row_fraction=0.5,
        overwrite_output=True,
    )

    argv = shlex.split(command)
    assert argv[:5] == ["python3", "-m", "obliteratus", "obliterate", "org/model name"]
    assert argv[argv.index("--output-dir") + 1] == "/tmp/output dir"
    assert "--unsafe-disable-damage-gate" in argv
    assert argv[argv.index("--damage-eval-size") + 1] == "96"
    assert argv[argv.index("--max-ppl-ratio") + 1] == "1.03"
    assert argv[argv.index("--max-sampled-token-kl") + 1] == "0.02"
    assert argv[argv.index("--max-p95-sampled-token-kl") + 1] == "0.08"
    assert argv[argv.index("--max-top1-flip-rate") + 1] == "0.01"
    assert argv[argv.index("--max-coherence-drop") + 1] == "0.04"
    assert argv[argv.index("--max-refusal-rate") + 1] == "0.15"
    assert "--no-project-lm-head" in argv
    assert "--project-embeddings" in argv
    assert argv[argv.index("--min-layer-fraction") + 1] == "0.2"
    assert argv[argv.index("--max-layer-fraction") + 1] == "0.75"
    assert argv[argv.index("--harmless-pc-count") + 1] == "3"
    assert argv[argv.index("--shield-concept-count") + 1] == "4"
    assert argv[argv.index("--shield-ridge") + 1] == "0.07"
    assert "--shield-residualize" in argv
    assert argv[argv.index("--shield-layer-penalty") + 1] == "0.15"
    assert argv[argv.index("--projection-target") + 1] == "auto"
    assert argv[argv.index("--projection-row-fraction") + 1] == "0.5"
    assert "--overwrite-output" in argv


def test_remote_obliterate_emits_fail_closed_defaults():
    runner = RemoteRunner(RemoteConfig(host="gpu.example"))

    argv = shlex.split(runner.build_obliterate_command("org/model"))

    assert "--damage-gate" in argv
    assert argv[argv.index("--damage-eval-size") + 1] == "64"
    assert "--unsafe-disable-damage-gate" not in argv
    assert "--project-lm-head" not in argv
    assert "--no-project-lm-head" not in argv
    assert "--project-embeddings" not in argv
    assert "--no-project-embeddings" not in argv
    assert "--overwrite-output" not in argv


@pytest.mark.parametrize(
    "kwargs",
    [
        {"damage_eval_size": 31},
        {"max_ppl_ratio": 0.99},
        {"max_sampled_token_kl": -0.1},
        {"max_p95_sampled_token_kl": float("nan")},
        {"max_top1_flip_rate": 1.1},
        {"max_coherence_drop": -0.1},
        {"max_refusal_rate": float("inf")},
        {"project_lm_head": "yes"},
        {"min_layer_fraction": -0.1},
        {"max_layer_fraction": 1.1},
        {"harmless_pc_count": True},
        {"shield_concept_count": -1},
        {"shield_ridge": float("nan")},
        {"shield_residualize": "yes"},
        {"shield_layer_penalty": -0.1},
        {"projection_target": "readers"},
        {"projection_row_fraction": 0.0},
        {"projection_target": "auto", "quantization": "4bit"},
        {"projection_target": "auto", "damage_eval_size": 63},
        {"overwrite_output": 1},
    ],
)
def test_remote_obliterate_rejects_invalid_safety_values(kwargs):
    runner = RemoteRunner(RemoteConfig(host="gpu.example"))

    with pytest.raises((TypeError, ValueError)):
        runner.build_obliterate_command("org/model", **kwargs)


def test_remote_gpu_prefix_rejects_shell_syntax():
    runner = RemoteRunner(RemoteConfig(host="gpu.example", gpus="0; touch /tmp/pwned"))

    with pytest.raises(ValueError, match="comma-separated"):
        runner.build_obliterate_command("org/model")


def test_remote_python_and_model_values_are_shell_quoted():
    runner = RemoteRunner(
        RemoteConfig(host="gpu.example", python="/opt/python env/bin/python", gpus="00,2")
    )

    command = runner.build_obliterate_command("org/model; echo injected")
    argv = shlex.split(command)

    assert argv[0] == "CUDA_VISIBLE_DEVICES=0,2"
    assert argv[1] == "/opt/python env/bin/python"
    assert argv[5] == "org/model; echo injected"
