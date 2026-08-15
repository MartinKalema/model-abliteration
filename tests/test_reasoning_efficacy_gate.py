"""Protocol-aware harmful-efficacy gate regressions."""

from __future__ import annotations

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.reasoning_protocol import (
    ParsedResponse,
    ReasoningProtocol,
    ReasoningSetting,
)


def _pipeline() -> AbliterationPipeline:
    pipeline = AbliterationPipeline(model_name="offline-reasoning-gate", method="basic")
    direct = ReasoningSetting("direct", "direct", False, True)
    thinking = ReasoningSetting("thinking", "thinking", True)
    pipeline.reasoning_protocol = ReasoningProtocol(
        supports_direct=True,
        supports_thinking=True,
        default_mode="direct",
        control_kind="template_bool",
        control_argument="enable_thinking",
        trace_format="think_xml",
        confidence="confirmed",
        settings=(direct, thinking),
    )
    pipeline._free_gpu_memory = lambda: None
    return pipeline


def _complete(final_text: str, *, reasoning_text: str | None = None) -> ParsedResponse:
    return ParsedResponse(
        raw_text=(reasoning_text or "") + final_text,
        final_text=final_text,
        reasoning_text=reasoning_text,
        structured=None,
        status="complete",
        parser="fixture",
    )


def test_worst_required_reasoning_mode_controls_refusal_rate():
    pipeline = _pipeline()
    calls: list[tuple[str, str, int]] = []

    def generate(prompt, setting, *, max_new_tokens):
        calls.append((prompt, setting.name, max_new_tokens))
        if setting.name == "thinking":
            return _complete("I cannot help with that request."), 12
        return _complete("Here is the requested information."), 8

    pipeline._generate_parsed_response = generate
    metrics = pipeline._measure_refusal_efficacy(
        ["harmful one", "harmful two"],
        ["S1", "S2"],
    )

    assert metrics is not None
    assert metrics["refusal_rate"] == 1.0
    assert metrics["pattern_refusal_rate"] == 1.0
    assert metrics["refusal_eval_count"] == 2
    assert metrics["refusal_case_count"] == 4
    assert metrics["refusal_reasoning_settings"] == ["direct", "thinking"]
    assert {setting for _, setting, _ in calls} == {"direct", "thinking"}
    assert {budget for _, setting, budget in calls if setting == "thinking"} == {512}
    assert {budget for _, setting, budget in calls if setting == "direct"} == {256}


def test_only_parsed_final_answer_is_classified():
    pipeline = _pipeline()

    def generate(_prompt, _setting, *, max_new_tokens):
        assert max_new_tokens in {256, 512}
        return _complete(
            "The words analysis and assistant are ordinary nouns in this answer.",
            reasoning_text="I cannot comply with the unsafe idea in the thought trace. ",
        ), 20

    pipeline._generate_parsed_response = generate
    metrics = pipeline._measure_refusal_efficacy(["harmful"], ["S1"])

    assert metrics is not None
    assert metrics["refusal_rate"] == 0.0
    assert metrics["pattern_refusal_rate"] == 0.0


def test_inconclusive_response_in_any_required_mode_fails_closed():
    pipeline = _pipeline()

    def generate(_prompt, setting, *, max_new_tokens):
        assert max_new_tokens in {256, 512}
        if setting.name == "thinking":
            return ParsedResponse(
                raw_text="<think>unfinished",
                final_text=None,
                reasoning_text=None,
                structured=None,
                status="inconclusive",
                parser="fixture",
                error="generation reached max_new_tokens",
            ), max_new_tokens
        return _complete("Here is the requested information."), 8

    pipeline._generate_parsed_response = generate

    assert pipeline._measure_refusal_efficacy(["harmful"], ["S1"]) is None


def test_degenerate_final_answer_is_an_efficacy_failure():
    pipeline = _pipeline()
    pipeline.reasoning_protocol = ReasoningProtocol(
        supports_direct=True,
        supports_thinking=False,
        default_mode="direct",
        control_kind="fixed",
        control_argument=None,
        trace_format="none",
        confidence="confirmed",
        settings=(ReasoningSetting("direct", "direct", None, True),),
    )
    pipeline._generate_parsed_response = (
        lambda _prompt, _setting, *, max_new_tokens: (_complete("!!!!!!!"), 7)
    )

    metrics = pipeline._measure_refusal_efficacy(["harmful"], ["S1"])

    assert metrics is not None
    assert metrics["refusal_rate"] == 1.0
    assert metrics["harmful_degenerate_count"] == 1
