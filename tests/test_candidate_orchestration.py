"""Focused tests for fail-closed iterative and tournament selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from obliteratus.adaptive_defaults import (
    build_knowledge_base,
    format_recommendation,
    get_adaptive_recommendation,
)
from obliteratus.auto_obliterate import AutoObliterator, IterationResult
from obliteratus.evaluation.candidate_selection import (
    CandidateEvidenceError,
    damage_severity,
    validate_acceptance_payload,
)
from obliteratus.tourney import (
    Contender,
    TourneyRunner,
    _rank_and_select,
    _restore_rounds,
    composite_score,
)


def _accepted_payload(
    *,
    refusal_rate: float = 0.04,
    nll: float = 0.01,
    kl: float = 0.01,
    kl_p95: float = 0.02,
    flip_rate: float = 0.002,
    coherence_drop: float = 0.01,
) -> dict:
    metrics = {
        "eval_prompt_count": 40,
        "eval_token_count": 400,
        "sampled_token_count": 200,
        "nll_increase_upper_ci": nll,
        "sampled_token_kl_upper_ci": kl,
        "sampled_token_kl_p95": kl_p95,
        "top1_flip_rate": flip_rate,
        "coherence_drop": coherence_drop,
        "new_degenerate_count": 0,
        "nonfinite_output_count": 0,
        "refusal_rate": refusal_rate,
        "refusal_eval_count": 30,
    }
    return {
        "accepted": True,
        "damage_accepted": True,
        "efficacy_accepted": True,
        "violations": [],
        "inconclusive": [],
        "metrics": metrics,
        "budget": {
            "damage": {
                "max_nll_increase_upper_ci": 0.05,
                "max_sampled_token_kl_upper_ci": 0.05,
                "max_p95_sampled_token_kl": 0.20,
                "max_top1_flip_rate": 0.02,
                "max_coherence_drop": 0.10,
                "max_new_degenerate_outputs": 0,
                "max_nonfinite_output_count": 0,
                "min_eval_prompts": 32,
                "min_eval_tokens": 256,
                "min_sampled_tokens": 128,
            },
            "efficacy": {"max_refusal_rate": 0.20, "min_eval_prompts": 30},
        },
    }


def _quality_metrics(payload: dict) -> dict:
    metrics = dict(payload["metrics"])
    metrics.update(
        {
            "perplexity": 10.0,
            "coherence": 0.9,
            "kl_divergence": metrics["sampled_token_kl_upper_ci"],
            "acceptance": payload,
            "acceptance_passed": True,
            "damage_accepted": True,
            "efficacy_accepted": True,
            "damage_severity": damage_severity(payload),
        }
    )
    return metrics


def _telemetry_record(method: str, quality_metrics: dict, **extra) -> dict:
    return {
        "model": {
            "architecture": "LlamaForCausalLM",
            "num_layers": 32,
            "hidden_size": 4096,
            "total_params": 8_000_000_000,
        },
        "method": method,
        "quality_metrics": quality_metrics,
        **extra,
    }


def test_missing_or_inconclusive_evidence_is_never_neutral():
    assert composite_score({"refusal_rate": 0.0, "coherence": 1.0}) == -1.0

    payload = _accepted_payload()
    payload["metrics"].pop("top1_flip_rate")
    with pytest.raises(CandidateEvidenceError, match="top1_flip_rate"):
        validate_acceptance_payload(payload)

    payload = _accepted_payload()
    payload["inconclusive"] = ["too few prompts"]
    with pytest.raises(CandidateEvidenceError, match="inconclusive"):
        validate_acceptance_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("refusal_rate", -0.01),
        ("refusal_rate", 1.01),
        ("top1_flip_rate", -0.01),
        ("sampled_token_kl_upper_ci", -0.01),
        ("refusal_eval_count", 30.5),
        ("eval_prompt_count", True),
    ],
)
def test_acceptance_evidence_rejects_impossible_metric_domains(field, value):
    payload = _accepted_payload()
    payload["metrics"][field] = value

    with pytest.raises(CandidateEvidenceError):
        validate_acceptance_payload(payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1])
def test_acceptance_evidence_rejects_invalid_refusal_budget(value):
    payload = _accepted_payload()
    payload["budget"]["efficacy"]["max_refusal_rate"] = value

    with pytest.raises(CandidateEvidenceError):
        validate_acceptance_payload(payload)


def test_tourney_uses_damage_only_to_break_equal_refusal():
    low_damage = _accepted_payload(nll=0.002, kl=0.002, kl_p95=0.01)
    high_damage = _accepted_payload(nll=0.04, kl=0.04, kl_p95=0.18)
    low = Contender(
        method="low-damage",
        score=composite_score(_quality_metrics(low_damage)),
        metrics=_quality_metrics(low_damage),
    )
    high = Contender(
        method="high-damage",
        score=composite_score(_quality_metrics(high_damage)),
        metrics=_quality_metrics(high_damage),
    )

    assert composite_score(_quality_metrics(low_damage)) == composite_score(
        _quality_metrics(high_damage)
    )
    _, advanced, _ = _rank_and_select([high, low], advance_count=1)
    assert advanced == [low]


def test_tourney_prefers_efficacy_before_damage_inside_hard_budget():
    effective = _accepted_payload(
        refusal_rate=0.01,
        nll=0.04,
        kl=0.04,
        kl_p95=0.18,
    )
    gentle = _accepted_payload(
        refusal_rate=0.10,
        nll=0.001,
        kl=0.001,
        kl_p95=0.005,
    )
    effective_contender = Contender(
        method="effective",
        score=composite_score(_quality_metrics(effective)),
        metrics=_quality_metrics(effective),
    )
    gentle_contender = Contender(
        method="gentle",
        score=composite_score(_quality_metrics(gentle)),
        metrics=_quality_metrics(gentle),
    )

    _, advanced, _ = _rank_and_select(
        [gentle_contender, effective_contender],
        advance_count=1,
    )
    assert advanced == [effective_contender]


def test_tourney_never_advances_rejected_or_missing_evidence():
    accepted_payload = _accepted_payload()
    accepted = Contender(
        method="accepted",
        score=composite_score(_quality_metrics(accepted_payload)),
        metrics=_quality_metrics(accepted_payload),
    )
    missing = Contender(method="missing", score=1.0, metrics={"refusal_rate": 0.0})
    rejected_payload = _accepted_payload()
    rejected_payload["accepted"] = False
    rejected = Contender(
        method="rejected",
        score=1.0,
        metrics={"acceptance": rejected_payload},
    )

    _, advanced, eliminated = _rank_and_select(
        [missing, rejected, accepted],
        advance_count=3,
    )

    assert [candidate.method for candidate in advanced] == ["accepted"]
    assert {candidate.method for candidate in eliminated} == {"missing", "rejected"}


def test_resume_invalidates_old_checkpoint_without_gate_evidence():
    checkpoint = {
        "model": "model",
        "completed_rounds": [
            {
                "round_num": 1,
                "name": "Qualifiers",
                "advanced_to": ["legacy"],
                "eliminated": [],
                "contenders": [
                    {
                        "method": "legacy",
                        "score": 0.9,
                        "metrics": {"refusal_rate": 0.0},
                    }
                ],
            }
        ],
        "interrupted_round": {},
    }

    result, _, _, _ = _restore_rounds(checkpoint)

    assert result.rounds[0].advanced_to == []
    assert result.rounds[0].eliminated == ["legacy"]


def test_auto_selects_lowest_refusal_before_damage(tmp_path):
    auto = AutoObliterator(
        "base/model",
        output_base=str(tmp_path / "auto"),
        target_refusal_rate=0.05,
    )
    high_dir = tmp_path / "high"
    low_dir = tmp_path / "low"
    rejected_dir = tmp_path / "rejected"
    for directory in (high_dir, low_dir, rejected_dir):
        directory.mkdir()

    high = _accepted_payload(
        refusal_rate=0.02,
        nll=0.04,
        kl=0.04,
        kl_p95=0.18,
    )
    low = _accepted_payload(
        refusal_rate=0.04,
        nll=0.002,
        kl=0.002,
        kl_p95=0.01,
    )
    auto._result.iterations = [
        IterationResult(
            iteration=1,
            method="high",
            prompt_volume=256,
            refusal_rate=0.02,
            output_dir=str(high_dir),
            accepted=True,
            damage_accepted=True,
            efficacy_accepted=True,
            damage_severity=damage_severity(high),
            acceptance=high,
        ),
        IterationResult(
            iteration=2,
            method="low",
            prompt_volume=256,
            refusal_rate=0.04,
            output_dir=str(low_dir),
            accepted=True,
            damage_accepted=True,
            efficacy_accepted=True,
            damage_severity=damage_severity(low),
            acceptance=low,
        ),
        IterationResult(
            iteration=3,
            method="missing-evidence",
            prompt_volume=256,
            refusal_rate=0.0,
            output_dir=str(rejected_dir),
            accepted=True,
            damage_accepted=True,
            efficacy_accepted=True,
            damage_severity=0.0,
            acceptance={},
        ),
    ]

    assert auto._best_acceptable_iteration() is auto._result.iterations[0]


def test_auto_uses_damage_as_the_efficacy_tie_break(tmp_path):
    auto = AutoObliterator("base/model", output_base=str(tmp_path / "auto"))
    high_dir = tmp_path / "high"
    low_dir = tmp_path / "low"
    high_dir.mkdir()
    low_dir.mkdir()
    high = _accepted_payload(refusal_rate=0.03, nll=0.04, kl=0.04)
    low = _accepted_payload(refusal_rate=0.03, nll=0.002, kl=0.002)
    auto._result.iterations = [
        IterationResult(
            iteration=1,
            method="high-damage",
            prompt_volume=256,
            refusal_rate=0.03,
            output_dir=str(high_dir),
            accepted=True,
            damage_accepted=True,
            efficacy_accepted=True,
            damage_severity=damage_severity(high),
            acceptance=high,
        ),
        IterationResult(
            iteration=2,
            method="low-damage",
            prompt_volume=256,
            refusal_rate=0.03,
            output_dir=str(low_dir),
            accepted=True,
            damage_accepted=True,
            efficacy_accepted=True,
            damage_severity=damage_severity(low),
            acceptance=low,
        ),
    ]

    assert auto._best_acceptable_iteration() is auto._result.iterations[1]


def test_auto_does_not_chain_from_failed_iteration(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.handle = None
            self._strong_layers = []
            self._expert_directions = {}
            self._prompt_split = type(
                "PromptSplit",
                (),
                {
                    "fingerprint": AutoObliterator._evaluation_fingerprint(
                        kwargs["evaluation_harmful_prompts"],
                        kwargs["evaluation_harmless_prompts"],
                    )
                },
            )()
            calls.append(kwargs)
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            if len(calls) == 1:
                self._damage_assessment = None
                self._quality_metrics = {"refusal_rate": 0.0}
            else:
                payload = _accepted_payload()
                self._damage_assessment = payload
                self._quality_metrics = _quality_metrics(payload)

        def run(self):
            return None

    monkeypatch.setattr("obliteratus.abliterate.AbliterationPipeline", FakePipeline)

    auto = AutoObliterator(
        "base/model",
        max_iterations=2,
        output_base=str(tmp_path / "auto"),
    )
    monkeypatch.setattr(
        auto,
        "_get_expanded_prompts",
        lambda iteration: (
            [f"harmful-{iteration}-{i}" for i in range(300)],
            [f"harmless-{iteration}-{i}" for i in range(300)],
        ),
    )

    list(auto.run())

    assert len(calls) == 2
    assert calls[0]["model_name"] == "base/model"
    assert calls[1]["model_name"] == "base/model"
    assert calls[0]["damage_gate_enabled"] is True
    assert calls[0]["trust_remote_code"] is False
    assert auto._result.iterations[0].error is not None
    assert auto._result.iterations[1].accepted is True


def test_auto_candidates_keep_the_immutable_original_baseline(monkeypatch, tmp_path):
    calls: list[dict] = []

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            payload = _accepted_payload(refusal_rate=0.04 - 0.01 * len(calls))
            self._damage_assessment = payload
            self._quality_metrics = _quality_metrics(payload)
            self.handle = None
            self._strong_layers = []
            self._expert_directions = {}
            self._prompt_split = type(
                "PromptSplit",
                (),
                {
                    "fingerprint": AutoObliterator._evaluation_fingerprint(
                        kwargs["evaluation_harmful_prompts"],
                        kwargs["evaluation_harmless_prompts"],
                    )
                },
            )()

        def run(self):
            return None

    monkeypatch.setattr("obliteratus.abliterate.AbliterationPipeline", FakePipeline)
    auto = AutoObliterator(
        "base/model",
        max_iterations=2,
        output_base=str(tmp_path / "auto"),
    )
    monkeypatch.setattr(
        auto,
        "_get_expanded_prompts",
        lambda iteration: (
            [f"harmful-{iteration}-{i}" for i in range(300)],
            [f"harmless-{iteration}-{i}" for i in range(300)],
        ),
    )

    list(auto.run())

    assert [call["model_name"] for call in calls] == ["base/model", "base/model"]
    assert calls[0]["evaluation_harmful_prompts"] == calls[1][
        "evaluation_harmful_prompts"
    ]
    assert calls[0]["evaluation_harmless_prompts"] == calls[1][
        "evaluation_harmless_prompts"
    ]
    locked_pairs = set(
        zip(
            calls[0]["evaluation_harmful_prompts"],
            calls[0]["evaluation_harmless_prompts"],
            strict=True,
        )
    )
    for call in calls:
        discovery_pairs = set(
            zip(
                call["harmful_prompts"],
                call["harmless_prompts"],
                strict=True,
            )
        )
        assert discovery_pairs.isdisjoint(locked_pairs)
    assert auto._result.final_output_dir.endswith("iter_2")

    resumed = AutoObliterator(
        "base/model",
        max_iterations=2,
        output_base=str(tmp_path / "auto"),
    )
    assert resumed._result.evaluation_fingerprint == auto._result.evaluation_fingerprint
    assert (
        resumed._result.evaluation_harmful_prompts
        == auto._result.evaluation_harmful_prompts
    )


def test_auto_excludes_each_locked_prompt_side_from_discovery(tmp_path):
    auto = AutoObliterator("base/model", output_base=str(tmp_path / "auto"))
    auto._result.evaluation_harmful_prompts = ["Locked harmful"]
    auto._result.evaluation_harmless_prompts = ["Locked harmless"]

    harmful, harmless = auto._exclude_locked_evaluation_pairs(
        ["  LOCKED   HARMFUL  ", "fresh harmful", "other harmful"],
        ["different control", "locked harmless", "fresh harmless"],
    )

    assert harmful == ["other harmful"]
    assert harmless == ["fresh harmless"]


def test_tourney_requires_assessment_and_does_not_force_remote_code(
    monkeypatch,
    tmp_path,
):
    calls: list[dict] = []

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self._quality_metrics = {"refusal_rate": 0.0}
            self._damage_assessment = None

        def run(self):
            return None

    monkeypatch.setattr("obliteratus.abliterate.AbliterationPipeline", FakePipeline)
    runner = TourneyRunner(
        "base/model",
        methods=["advanced"],
        output_dir=str(tmp_path / "tourney"),
    )

    contender = runner._run_method(
        "advanced",
        ["harmful"] * 40,
        ["harmless"] * 40,
        str(tmp_path / "candidate"),
        evaluation_harmful=["eval-harmful"] * 32,
        evaluation_harmless=["eval-harmless"] * 32,
    )

    assert contender.error is not None
    assert contender.score == -1.0
    assert calls[0]["trust_remote_code"] is False
    assert calls[0]["damage_gate_enabled"] is True


def test_tourney_uses_disjoint_locked_evaluation_prompts(monkeypatch, tmp_path):
    harmful = [f"harmful-{index}" for index in range(100)]
    harmless = [f"harmless-{index}" for index in range(100)]
    monkeypatch.setattr(
        "obliteratus.prompts.load_dataset_source",
        lambda dataset_key: (harmful, harmless),
    )
    runner = TourneyRunner(
        "base/model",
        methods=["advanced"],
        output_dir=str(tmp_path / "tourney"),
    )

    discovery_h, discovery_b, evaluation_h, evaluation_b = (
        runner._load_prompt_sets(50)
    )

    discovery_pairs = set(zip(discovery_h, discovery_b, strict=True))
    evaluation_pairs = set(zip(evaluation_h, evaluation_b, strict=True))
    assert len(discovery_pairs) == 50
    assert len(evaluation_pairs) >= 32
    assert discovery_pairs.isdisjoint(evaluation_pairs)


def test_adaptive_defaults_exclude_legacy_rejected_and_errored_runs():
    accepted = _accepted_payload()
    rejected = _accepted_payload()
    rejected["accepted"] = False
    records = [
        _telemetry_record("advanced", _quality_metrics(accepted)),
        _telemetry_record(
            "nuclear",
            {**rejected["metrics"], "acceptance": rejected},
        ),
        _telemetry_record(
            "nuclear",
            {"refusal_rate": 0.0, "coherence": 1.0, "kl_divergence": None},
        ),
        _telemetry_record("nuclear", {}, error="run failed"),
    ]

    knowledge = build_knowledge_base(records)
    bucket = knowledge[("dense", "standard", "medium")]

    assert bucket.total_runs == 1
    assert bucket.excluded_runs == 3
    assert set(bucket.methods) == {"advanced"}
    assert bucket.best_method == "advanced"
    assert bucket.exclusion_reasons[
        "legacy telemetry lacks accepted damage-gate evidence"
    ] == 1


def test_adaptive_defaults_report_when_only_old_or_rejected_data_exists():
    rejected = _accepted_payload()
    rejected["damage_accepted"] = False
    knowledge = build_knowledge_base(
        [
            _telemetry_record("advanced", {"refusal_rate": 0.0}),
            _telemetry_record(
                "nuclear",
                {**rejected["metrics"], "acceptance": rejected},
            ),
        ]
    )

    recommendation = get_adaptive_recommendation(
        "dense",
        "standard",
        8.0,
        knowledge=knowledge,
    )

    assert recommendation.recommended_method == ""
    assert recommendation.confidence == "none"
    assert recommendation.n_records == 0
    assert recommendation.n_excluded_records == 2
    assert "No accepted damage-gated telemetry" in recommendation.reason
    assert "No eligible telemetry data" in format_recommendation(recommendation)


def test_adaptive_defaults_rank_efficacy_then_damage():
    effective = _accepted_payload(
        refusal_rate=0.01,
        nll=0.04,
        kl=0.04,
        kl_p95=0.18,
    )
    gentle = _accepted_payload(
        refusal_rate=0.10,
        nll=0.001,
        kl=0.001,
        kl_p95=0.005,
    )
    knowledge = build_knowledge_base(
        [
            _telemetry_record("effective", _quality_metrics(effective)),
            _telemetry_record("gentle", _quality_metrics(gentle)),
        ]
    )
    bucket = knowledge[("dense", "standard", "medium")]
    assert bucket.best_method == "effective"

    tied_effective = _accepted_payload(
        refusal_rate=0.01,
        nll=0.001,
        kl=0.001,
        kl_p95=0.005,
    )
    tied = build_knowledge_base(
        [
            _telemetry_record("high-damage", _quality_metrics(effective)),
            _telemetry_record("low-damage", _quality_metrics(tied_effective)),
        ]
    )[("dense", "standard", "medium")]
    assert tied.best_method == "low-damage"
