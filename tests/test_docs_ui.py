"""Regression tests for the static documentation UI workflows."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from obliteratus.abliterate import UNAVAILABLE_METHODS, available_method_names
from obliteratus.study_presets import STUDY_PRESETS

ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")


def test_checkpoint_workflow_is_the_unambiguous_default() -> None:
    assert 'class="tab tab-obliterate active"' in DOCS_SOURCE
    assert '<div id="tab-abliterate" class="tab-content active"' in DOCS_SOURCE
    assert '<div id="tab-wizard" class="tab-content active"' not in DOCS_SOURCE
    assert "// STUDY BUILDER" in DOCS_SOURCE
    assert "Analysis studies do not create an obliterated checkpoint." in DOCS_SOURCE
    assert "These recipes modify model weights" in DOCS_SOURCE


def test_checkpoint_cards_match_cli_runnable_methods() -> None:
    displayed = set(
        re.findall(
            r'<label class="method-radio(?: selected)?" id="method-([^"]+)">',
            DOCS_SOURCE,
        )
    )

    assert displayed == set(available_method_names())
    assert displayed.isdisjoint(UNAVAILABLE_METHODS)
    assert "--method adaptive" not in DOCS_SOURCE


def test_study_cards_and_sample_options_match_backend_presets() -> None:
    match = re.search(r"const PRESETS = \[(.*?)\n\];", DOCS_SOURCE, re.DOTALL)
    assert match is not None
    preset_block = match.group(1)
    displayed_samples = {
        key: int(samples)
        for key, samples in re.findall(r'\{key:"([^"]+)".*?samples:(\d+)', preset_block)
    }

    assert set(displayed_samples) == {*STUDY_PRESETS, "custom"}
    assert {key: displayed_samples[key] for key in STUDY_PRESETS} == {
        key: preset.max_samples for key, preset in STUDY_PRESETS.items()
    }
    for samples in {preset.max_samples for preset in STUDY_PRESETS.values()}:
        assert f'<option value="{samples}"' in DOCS_SOURCE


def test_ui_state_and_custom_inputs_fail_safe() -> None:
    assert "event.target.classList.add('active')" not in DOCS_SOURCE
    assert "const btn = event.target" not in DOCS_SOURCE
    assert "logEl.innerHTML" not in DOCS_SOURCE
    assert "trust_remote_code: true" not in DOCS_SOURCE
    assert "trust_remote_code: false" in DOCS_SOURCE
    assert "shellQuote(ablSelectedModel)" in DOCS_SOURCE
    assert "state.model = null;" in DOCS_SOURCE
    assert "state.preset = null;" in DOCS_SOURCE
    assert "validateResultsData" in DOCS_SOURCE
    assert "updateStudyValidation" in DOCS_SOURCE
    assert "copyTextWithFallback" in DOCS_SOURCE
    assert "statsGrid.innerHTML" not in DOCS_SOURCE
    assert "chartEl.innerHTML" not in DOCS_SOURCE
    assert "tableEl.innerHTML" not in DOCS_SOURCE


def test_dense_search_commands_never_inherit_model_quantization_hints() -> None:
    assert (
        "const denseMethods = new Set(['gabliteration', 'rdo', 'som', "
        "'optimized', 'heretic']);"
    ) in DOCS_SOURCE
    assert "modelInfo?.quant && !denseMethods.has(ablMethod)" in DOCS_SOURCE


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_inline_javascript_parses() -> None:
    scripts = re.findall(r"<script>(.*?)</script>", DOCS_SOURCE, re.DOTALL)
    assert scripts
    result = subprocess.run(
        ["node", "--check", "-"],
        input="\n".join(scripts),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
