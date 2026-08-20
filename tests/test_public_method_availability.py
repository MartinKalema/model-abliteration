"""Regression checks for fail-closed public method selection surfaces."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from obliteratus.abliterate import UNAVAILABLE_METHODS, available_method_names

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _first_call_line(function: ast.FunctionDef, name: str) -> int:
    lines = [
        node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]
    assert lines, f"call to {name!r} not found in {function.name}"
    return min(lines)


def _load_app_method_validator():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    validator = _function(tree, "_validate_method_names")
    namespace = {
        "_available_method_names": available_method_names,
        "_UNAVAILABLE_METHODS": UNAVAILABLE_METHODS,
    }
    code = compile(ast.Module(body=[validator], type_ignores=[]), APP_PATH, "exec")
    exec(code, namespace)  # noqa: S102 - executes only the extracted local helper
    return namespace["_validate_method_names"]


def test_app_method_mapping_only_contains_runnable_presets():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "METHODS" for target in node.targets)
    )
    displayed_methods = set(ast.literal_eval(assignment.value).values())

    assert displayed_methods == {*available_method_names(), "adaptive"}
    assert displayed_methods.isdisjoint(UNAVAILABLE_METHODS)


def test_strength_sweep_excludes_non_scalar_and_specialized_workflows():
    source = APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    protocol_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_PROTOCOL_OWNED_METHODS"
            for target in node.targets
        )
    )
    protocol_set = next(
        node
        for node in ast.walk(protocol_assignment.value)
        if isinstance(node, ast.Set)
    )
    assert {
        elt.value for elt in protocol_set.elts if isinstance(elt, ast.Constant)
    } == {"gabliteration", "rdo", "som", "optimized", "heretic"}

    sweep_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_STRENGTH_SWEEP_METHODS"
            for target in node.targets
        )
    )
    sweep_excluded = next(
        node for node in ast.walk(sweep_assignment.value) if isinstance(node, ast.Set)
    )
    excluded_names = {
        elt.value for elt in sweep_excluded.elts if isinstance(elt, ast.Constant)
    }
    assert {"adaptive", "informed", "inverted", "nuclear"} <= excluded_names
    assert any(isinstance(elt, ast.Starred) for elt in sweep_excluded.elts)
    assert "METHODS.get(method_choice, \"advanced\")" not in source


def test_raw_ui_dispatch_defers_protocol_owned_edit_controls_to_presets():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "elif method in _PROTOCOL_OWNED_METHODS:" in source
    protocol_branch = source.split(
        "elif method in _PROTOCOL_OWNED_METHODS:", 1
    )[1].split("else:", 1)[0]
    assert '"cot_aware": bool(adv_cot_aware)' in protocol_branch
    assert '"kl_budget": float(adv_kl_budget)' in protocol_branch
    assert '"n_directions"' not in protocol_branch
    assert '"regularization"' not in protocol_branch
    assert '"use_kl_optimization"' not in protocol_branch
    assert "if _adaptive_requested and method in _PROTOCOL_OWNED_METHODS:" in source


def test_gradio_uses_method_aware_gpu_duration_for_named_protocols():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "@spaces.GPU(duration=_obliterate_gpu_duration)" in source
    assert 'if method == "som":\n        return 7_200' in source
    assert 'if method in {"optimized", "heretic"}:\n        return 3_600' in source
    assert 'if method in {"gabliteration", "rdo"}:\n        return 1_800' in source


def test_app_method_validator_rejects_unavailable_and_unknown_names():
    validate = _load_app_method_validator()

    assert validate(["advanced", "som_proxy"]) == ["advanced", "som_proxy"]
    for method in UNAVAILABLE_METHODS:
        with pytest.raises(ValueError, match="unavailable"):
            validate([method])
    with pytest.raises(ValueError, match="not a runnable preset"):
        validate(["not-a-method"])


def test_app_validates_raw_method_names_before_data_or_runner_setup():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))

    benchmark = _function(tree, "benchmark")
    assert _first_call_line(benchmark, "_validate_method_names") < _first_call_line(
        benchmark, "load_dataset_source"
    )

    multi_model = _function(tree, "benchmark_multi_model")
    assert _first_call_line(multi_model, "_validate_method_names") < _first_call_line(
        multi_model, "load_dataset_source"
    )

    tourney = _function(tree, "run_tourney")
    assert _first_call_line(tourney, "_validate_method_names") < _first_call_line(
        tourney, "TourneyRunner"
    )

    obliterate = _function(tree, "obliterate")
    assert _first_call_line(obliterate, "_validate_method_names") < _first_call_line(
        obliterate, "is_gated"
    )


def test_public_copy_does_not_advertise_unavailable_or_som_paper_methods():
    source = APP_PATH.read_text(encoding="utf-8")
    benchmark_plots = (ROOT / "obliteratus" / "evaluation" / "benchmark_plots.py").read_text(
        encoding="utf-8"
    )

    stale_claims = (
        "`surgical`, `optimized`, or `nuclear`",
        "Tests `surgical` + `optimized` + `nuclear`",
        "vs `optimized` (slow but smart)",
        "SOM-manifold",
    )
    assert all(claim not in source for claim in stale_claims)
    assert "surgical/optimized/nuclear" not in benchmark_plots
