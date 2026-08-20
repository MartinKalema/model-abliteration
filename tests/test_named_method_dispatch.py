"""Integration regressions for named model-forward method orchestration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from obliteratus.abliterate import AbliterationPipeline


def _stub_run_stages(pipeline: AbliterationPipeline, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_remove_activation_steering", lambda: 0)
    monkeypatch.setattr(pipeline, "_summon", lambda: None)
    monkeypatch.setattr(pipeline, "_probe", lambda: None)
    monkeypatch.setattr(pipeline, "_distill", lambda: None)
    monkeypatch.setattr(pipeline, "_capture_damage_baseline", lambda: None)
    monkeypatch.setattr(pipeline, "_free_gpu_memory", lambda: None)
    monkeypatch.setattr(pipeline, "_rebirth", lambda: Path("named-method-output"))


def test_gabliteration_run_bypasses_the_ordinary_excise_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AbliterationPipeline(model_name="offline", method="gabliteration")
    _stub_run_stages(pipeline, monkeypatch)
    calls: list[str] = []
    accepted = SimpleNamespace(accepted=True)
    monkeypatch.setattr(
        pipeline,
        "_run_gabliteration_checkpoint_search",
        lambda: calls.append("gabliteration") or accepted,
    )
    monkeypatch.setattr(
        pipeline,
        "_excise",
        lambda: pytest.fail("ordinary EXCISE must not run for Gabliteration"),
    )
    monkeypatch.setattr(
        pipeline,
        "_run_kl_preservation_search",
        lambda: pytest.fail("generic KL grid must not run for Gabliteration"),
    )

    assert pipeline.run() == Path("named-method-output")
    assert calls == ["gabliteration"]


@pytest.mark.parametrize("method", ["som", "optimized", "heretic"])
def test_exact_search_methods_route_through_candidate_orchestration(
    method: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AbliterationPipeline(model_name="offline", method=method)
    _stub_run_stages(pipeline, monkeypatch)
    calls: list[str] = []
    accepted = SimpleNamespace(accepted=True)
    monkeypatch.setattr(
        pipeline,
        "_run_kl_preservation_search",
        lambda: calls.append(method) or accepted,
    )
    monkeypatch.setattr(
        pipeline,
        "_excise",
        lambda: pytest.fail(f"ordinary EXCISE must not run for {method}"),
    )

    assert pipeline.run() == Path("named-method-output")
    assert calls == [method]


def test_rdo_distill_cannot_drop_model_forward_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AbliterationPipeline(model_name="offline", method="rdo")
    pipeline.handle = None
    pipeline._harmful_means = {}
    pipeline._harmless_means = {}
    pipeline._harmful_acts = {}
    pipeline._harmless_acts = {}
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "_run_rdo_direction_training",
        lambda: calls.append("rdo"),
    )
    monkeypatch.setattr(pipeline, "_emit", lambda *args, **kwargs: None)

    pipeline._distill()

    assert calls == ["rdo"]


def test_rdo_run_uses_transactional_checkpoint_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AbliterationPipeline(model_name="offline", method="rdo")
    _stub_run_stages(pipeline, monkeypatch)
    calls: list[str] = []
    accepted = SimpleNamespace(accepted=True)
    monkeypatch.setattr(
        pipeline,
        "_run_rdo_checkpoint_projection",
        lambda: calls.append("rdo-transaction") or accepted,
    )
    monkeypatch.setattr(
        pipeline,
        "_excise",
        lambda: pytest.fail("RDO must not use the unguarded ordinary run branch"),
    )

    assert pipeline.run() == Path("named-method-output")
    assert calls == ["rdo-transaction"]


def test_paper_som_distill_uses_the_dedicated_pool_trainer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AbliterationPipeline(model_name="offline", method="som")
    calls: list[float] = []
    monkeypatch.setattr(
        pipeline,
        "_distill_paper_som",
        lambda started_at: calls.append(started_at),
    )
    monkeypatch.setattr(pipeline, "_emit", lambda *args, **kwargs: None)

    pipeline._distill()

    assert len(calls) == 1


def test_gabliteration_verifier_exception_restores_the_immutable_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AbliterationPipeline(model_name="offline", method="gabliteration")
    pipeline.handle = SimpleNamespace(model=object(), tokenizer=object())
    pipeline._discovery_harmful = [f"harmful-{i}" for i in range(12)]
    pipeline._discovery_harmless = [f"harmless-{i}" for i in range(12)]
    manifest = SimpleNamespace(
        target="output",
        num_layers=2,
        entries_for_layer=lambda _layer: (object(),),
    )
    replay = SimpleNamespace(
        effective_layers=(0, 1),
        directions=torch.eye(2),
    )
    restored: list[str] = []
    monkeypatch.setattr(pipeline, "_assert_auto_projection_prerequisites", lambda *_: None)
    monkeypatch.setattr(pipeline, "_current_projection_manifest", lambda: manifest)
    monkeypatch.setattr(pipeline, "_gabliteration_hidden_batches", lambda _p: ())
    monkeypatch.setattr(pipeline, "_maybe_apply_chat_template", lambda prompts: prompts)
    monkeypatch.setattr(pipeline, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline,
        "_restore_auto_projection_baseline",
        lambda _weights, *, purpose: restored.append(purpose),
    )
    monkeypatch.setattr(
        "obliteratus.analysis.gabliteration.run_gabliteration_search",
        lambda **_kwargs: SimpleNamespace(replay_plan=replay),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify",
        lambda: (_ for _ in ()).throw(RuntimeError("verifier crashed")),
    )

    with pytest.raises(RuntimeError, match="verifier crashed"):
        pipeline._run_gabliteration_checkpoint_search()

    assert restored == ["method='gabliteration' behavioral search"]
    assert pipeline._gabliteration_search_result is None
    assert pipeline._excise_modified_count is None


def test_rdo_verifier_exception_restores_the_immutable_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AbliterationPipeline(model_name="offline", method="rdo")
    pipeline._rdo_result = object()
    restored: list[str] = []
    monkeypatch.setattr(pipeline, "_assert_auto_projection_prerequisites", lambda *_: None)
    monkeypatch.setattr(pipeline, "_excise", lambda: None)
    monkeypatch.setattr(pipeline, "_remove_activation_steering", lambda: 0)
    monkeypatch.setattr(pipeline, "_free_gpu_memory", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "_restore_auto_projection_baseline",
        lambda _weights, *, purpose: restored.append(purpose),
    )
    monkeypatch.setattr(
        pipeline,
        "_verify",
        lambda: (_ for _ in ()).throw(RuntimeError("verifier crashed")),
    )

    with pytest.raises(RuntimeError, match="verifier crashed"):
        pipeline._run_rdo_checkpoint_projection()

    assert restored == ["method='rdo' trained-direction checkpoint projection"]
    assert pipeline._rdo_result is None
    assert pipeline._excise_modified_count is None


def test_rdo_kl_search_cancellation_restores_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = AbliterationPipeline.__new__(AbliterationPipeline)
    pipeline.method = "rdo"
    pipeline._layer_excise_weights = {}
    pipeline._kl_search_results = []
    pipeline._kl_selected_regularization = None
    pipeline._requested_regularization = 0.0
    pipeline.regularization = 0.0
    pipeline.kl_budget = 0.05
    pipeline.log = lambda _message: None
    restored: list[float] = []
    monkeypatch.setattr(pipeline, "_assert_auto_projection_prerequisites", lambda *_: None)
    monkeypatch.setattr(pipeline, "_kl_regularization_candidates", lambda: (0.25,))
    monkeypatch.setattr(
        pipeline,
        "_restore_auto_projection_baseline",
        lambda _weights, *, purpose: restored.append(pipeline.regularization),
    )
    monkeypatch.setattr(
        pipeline,
        "_excise",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(pipeline, "_remove_activation_steering", lambda: 0)

    with pytest.raises(KeyboardInterrupt):
        pipeline._run_kl_preservation_search_inner([])

    assert len(restored) == 2
    assert pipeline.regularization == 0.0
    assert pipeline._kl_selected_regularization is None
