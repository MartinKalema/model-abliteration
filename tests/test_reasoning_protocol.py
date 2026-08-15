"""Offline behavioral tests for the reasoning protocol foundation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from obliteratus.reasoning_protocol import (
    ProtocolRenderError,
    ReasoningProtocol,
    ReasoningSetting,
    RenderedPrompt,
    detect_reasoning_protocol,
    parse_generated_response,
    render_chat_prompt,
    required_evaluation_settings,
)


MESSAGES = [{"role": "user", "content": "Solve this."}]


class BoolTokenizer:
    chat_template = "{% if enable_thinking %}<think>{% endif %}"

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        tools=None,
    ):
        self.calls.append((messages, tokenize, enable_thinking, tools))
        marker = 20 if enable_thinking else 10
        if tokenize:
            return [1, marker]
        return f"prompt:{marker}:{messages[-1]['content']}"


class IgnoringTokenizer:
    chat_template = "ordinary chat template"

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [1, 2, 3]


class EffortTokenizer:
    chat_template = "reasoning_effort may be low, medium, or high"

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="medium",
        tools=None,
    ):
        del add_generation_prompt
        self.calls.append((reasoning_effort, tokenize, tools, messages))
        marker = {"low": 10, "medium": 20, "high": 30}[reasoning_effort]
        return [1, marker] if tokenize else f"effort:{reasoning_effort}"


class HybridEffortTokenizer(EffortTokenizer):
    chat_template = "reasoning_effort may be none or high"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort="none",
        tools=None,
    ):
        del add_generation_prompt
        self.calls.append((reasoning_effort, tokenize, tools, messages))
        marker = {"none": 10, "high": 30}[reasoning_effort]
        return [1, marker] if tokenize else f"effort:{reasoning_effort}"


class EncoderTokenizer:
    response_template = "reasoning_content and content"

    def __init__(self) -> None:
        self.encoder_calls = []

    def apply_chat_template(
        self, messages, *, tokenize=True, add_generation_prompt=True
    ):
        del messages, tokenize, add_generation_prompt
        return [1, 10]

    def encode_messages(self, messages, *, thinking_mode):
        self.encoder_calls.append((messages, thinking_mode))
        return [1, 20 if thinking_mode == "thinking" else 10]


class FixedTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=True, tools=None
    ):
        self.calls.append(
            {
                "messages": messages,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
                "tools": tools,
            }
        )
        return [4, 5] if tokenize else "plain rendered prompt"


class PromptRecordingTokenizer(FixedTokenizer):
    def apply_chat_template(
        self, messages, *, tokenize=False, add_generation_prompt=True, tools=None
    ):
        super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
        )
        return messages[-1]["content"] if not tokenize else [4, 5]


class ParserTokenizer:
    def __init__(self, parsed=None, error=None) -> None:
        self.parsed = parsed
        self.error = error
        self.calls = []

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(chr(item) for item in ids)

    def parse_response(self, output, **kwargs):
        self.calls.append((output, kwargs))
        if self.error is not None:
            raise self.error
        return self.parsed


def make_protocol(
    *,
    supports_direct=True,
    supports_thinking=False,
    default_mode="direct",
    control_kind="fixed",
    trace_format="none",
    settings=None,
    control_argument=None,
    effort_levels=(),
    default_effort=None,
    adapter_id=None,
):
    if settings is None:
        settings = (ReasoningSetting(default_mode, default_mode, None, True),)
    return ReasoningProtocol(
        supports_direct=supports_direct,
        supports_thinking=supports_thinking,
        default_mode=default_mode,
        control_kind=control_kind,
        control_argument=control_argument,
        effort_levels=effort_levels,
        trace_format=trace_format,
        evidence=("test fixture",),
        confidence="confirmed",
        settings=settings,
        default_effort=default_effort,
        adapter_id=adapter_id,
    )


def rendered_for(setting, *, prefix="prompt"):
    return RenderedPrompt(
        text=prefix,
        input_ids=None,
        prefix=prefix,
        setting=setting,
    )


def test_artifact_render_difference_detects_renamed_hybrid_and_default():
    protocol = detect_reasoning_protocol(
        BoolTokenizer(), model_name="local/completely-renamed-checkpoint"
    )

    assert protocol.control_kind == "template_bool"
    assert protocol.confidence == "confirmed"
    assert protocol.supports_direct is True
    assert protocol.supports_thinking is True
    assert protocol.default_mode == "thinking"
    assert protocol.setting("thinking").is_default
    assert "artifact-render-diff" in protocol.evidence[0]


@pytest.mark.parametrize(
    "model_name",
    ["Qwen/Qwen3-8B", "openai/gpt-oss-20b", "zai-org/GLM-4.5-Air"],
)
def test_silently_ignored_control_suppresses_even_exact_manifest(model_name):
    protocol = detect_reasoning_protocol(IgnoringTokenizer(), model_name=model_name)

    assert protocol.control_kind == "unknown"
    assert protocol.confidence == "unknown"
    assert protocol.supports_direct is None
    assert protocol.supports_thinking is None
    assert "identical" in protocol.evidence[0]


def test_artifact_effort_detection_and_required_default_maximum_settings():
    protocol = detect_reasoning_protocol(
        EffortTokenizer(), model_name="local/renamed-effort-model"
    )

    assert protocol.control_kind == "effort"
    assert protocol.effort_levels == ("low", "medium", "high")
    assert protocol.default_effort == "medium"
    assert protocol.supports_direct is False
    assert [item.name for item in required_evaluation_settings(protocol)] == [
        "medium",
        "high",
    ]


def test_hybrid_effort_detection_requires_direct_and_maximum_settings():
    protocol = detect_reasoning_protocol(HybridEffortTokenizer())

    assert protocol.supports_direct is True
    assert [item.name for item in required_evaluation_settings(protocol)] == [
        "none",
        "high",
    ]


def test_encoder_artifact_detection_is_independent_of_repository_name():
    tokenizer = EncoderTokenizer()
    protocol = detect_reasoning_protocol(tokenizer, model_name="renamed/model")

    assert protocol.control_kind == "encoder"
    assert protocol.control_argument == "thinking_mode"
    assert protocol.trace_format == "structured"
    assert protocol.default_mode == "direct"
    assert [item.name for item in required_evaluation_settings(protocol)] == [
        "direct",
        "thinking",
    ]


@pytest.mark.parametrize(
    ("model_name", "kind", "direct", "thinking", "trace"),
    [
        ("Qwen/Qwen3-8B", "template_bool", True, True, "think_xml"),
        ("Qwen/Qwen3.5-9B", "template_bool", True, True, "think_xml"),
        ("Qwen/Qwen3.6-35B-A3B-FP8", "template_bool", True, True, "think_xml"),
        ("Qwen/Qwen3-4B-Thinking-2507", "fixed", False, True, "think_xml"),
        ("Qwen/Qwen3-4B-Instruct-2507", "fixed", True, False, "none"),
        ("deepseek-ai/DeepSeek-R1", "fixed", False, True, "think_xml"),
        ("deepseek-ai/DeepSeek-V3.2", "encoder", True, True, "structured"),
        ("zai-org/GLM-4.7", "template_bool", True, True, "structured"),
        ("openai/gpt-oss-120b", "effort", False, True, "harmony"),
        ("google/gemma-4-27b-it", "template_bool", True, True, "gemma_channel"),
        ("mistralai/Mistral-Small-4-119B-2603", "effort", True, True, "structured"),
        ("mistralai/Magistral-Small-2509", "fixed", False, True, "think_xml"),
        ("microsoft/Phi-4-reasoning", "fixed", False, True, "think_xml"),
        ("allenai/Olmo-3-7B-Think", "fixed", False, True, "think_xml"),
        ("moonshotai/Kimi-K2-Thinking", "fixed", False, True, "think_xml"),
        ("HuggingFaceTB/SmolLM3-3B", "template_bool", True, True, "think_xml"),
    ],
)
def test_exact_official_manifest_fallbacks(
    model_name, kind, direct, thinking, trace
):
    protocol = detect_reasoning_protocol(model_name=model_name)

    assert protocol.control_kind == kind
    assert protocol.supports_direct is direct
    assert protocol.supports_thinking is thinking
    assert protocol.trace_format == trace
    assert protocol.confidence == "inferred"


def test_gpt_oss_has_no_fake_off_setting_and_evaluates_medium_and_high():
    protocol = detect_reasoning_protocol(model_name="openai/gpt-oss-20b")

    assert [item.name for item in protocol.settings] == ["low", "medium", "high"]
    assert all(item.semantic_mode == "thinking" for item in protocol.settings)
    assert [item.name for item in required_evaluation_settings(protocol)] == [
        "medium",
        "high",
    ]


def test_unknown_names_templates_and_nemotron_variants_remain_unknown():
    for model_name in (
        "example/assistant-analysis-model",
        "renamed/think-model",
        "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
    ):
        protocol = detect_reasoning_protocol(model_name=model_name)
        assert protocol.control_kind == "unknown"
        assert protocol.supports_thinking is None


def test_explicit_override_has_priority_over_artifacts():
    override = make_protocol(
        supports_direct=False,
        supports_thinking=True,
        default_mode="thinking",
        trace_format="think_xml",
        settings=(ReasoningSetting("thinking", "thinking", None, True),),
    )

    detected = detect_reasoning_protocol(BoolTokenizer(), override=override)

    assert detected.control_kind == "fixed"
    assert detected.evidence[0] == "explicit user override"


def test_protocol_structures_are_immutable_including_parsed_mappings():
    setting = ReasoningSetting("direct", "direct", None, True)
    protocol = make_protocol(settings=(setting,))

    with pytest.raises(FrozenInstanceError):
        setting.name = "changed"
    with pytest.raises(FrozenInstanceError):
        protocol.default_mode = "thinking"

    parsed = parse_generated_response(
        ParserTokenizer(parsed={"content": "answer", "meta": {"items": [1, 2]}}),
        "answer",
        rendered_for(setting),
        protocol,
    )
    with pytest.raises(TypeError):
        parsed.structured["content"] = "changed"
    with pytest.raises(TypeError):
        parsed.structured["meta"]["items"][0] = 9


def test_bool_rendering_passes_only_exact_control_and_prefix():
    tokenizer = BoolTokenizer()
    protocol = detect_reasoning_protocol(tokenizer)
    tokenizer.calls.clear()

    rendered = render_chat_prompt(
        tokenizer,
        MESSAGES,
        protocol,
        "direct",
        tokenize=True,
        tools=[{"type": "function"}],
    )

    assert rendered.input_ids == (1, 10)
    assert rendered.prefix == (1, 10)
    assert rendered.control_kwargs == (("enable_thinking", False),)
    assert tokenizer.calls[-1][2] is False
    assert tokenizer.calls[-1][3] == [{"type": "function"}]


def test_effort_and_encoder_render_exact_values():
    effort_tokenizer = EffortTokenizer()
    effort_protocol = detect_reasoning_protocol(effort_tokenizer)
    rendered = render_chat_prompt(
        effort_tokenizer, MESSAGES, effort_protocol, "high", tokenize=False
    )
    assert rendered.text == "effort:high"
    assert rendered.control_kwargs == (("reasoning_effort", "high"),)

    encoder_tokenizer = EncoderTokenizer()
    encoder_protocol = detect_reasoning_protocol(encoder_tokenizer)
    encoder_tokenizer.encoder_calls.clear()
    rendered = render_chat_prompt(
        encoder_tokenizer, MESSAGES, encoder_protocol, "thinking"
    )
    assert rendered.input_ids == (1, 20)
    assert rendered.control_kwargs == (("thinking_mode", "thinking"),)
    assert encoder_tokenizer.encoder_calls[-1][1] == "thinking"


def test_fixed_rendering_never_invents_a_control_kwarg():
    tokenizer = FixedTokenizer()
    protocol = detect_reasoning_protocol(
        model_name="Qwen/Qwen3-4B-Thinking-2507"
    )

    rendered = render_chat_prompt(tokenizer, MESSAGES, protocol)

    assert rendered.text == "plain rendered prompt"
    assert rendered.control_kwargs == ()
    assert set(tokenizer.calls[-1]) == {
        "messages",
        "tokenize",
        "add_generation_prompt",
        "tools",
    }


@pytest.mark.parametrize(
    ("adapter_id", "expected"),
    [
        ("prompt_prefix", "/no_think\nSolve this."),
        ("prompt_suffix", "Solve this.\n/no_think"),
    ],
)
def test_explicit_prompt_adapters_clone_messages(adapter_id, expected):
    direct = ReasoningSetting("direct", "direct", "/no_think", True)
    thinking = ReasoningSetting("thinking", "thinking", "/think")
    protocol = make_protocol(
        supports_thinking=True,
        control_kind="prompt",
        control_argument=None,
        settings=(direct, thinking),
        adapter_id=adapter_id,
    )
    original = [{"role": "user", "content": "Solve this."}]

    rendered = render_chat_prompt(
        PromptRecordingTokenizer(), original, protocol, "direct"
    )

    assert rendered.text == expected
    assert original == [{"role": "user", "content": "Solve this."}]


def test_render_failure_is_explicit_and_does_not_fall_back():
    protocol = detect_reasoning_protocol(BoolTokenizer())

    with pytest.raises(ProtocolRenderError, match="apply_chat_template"):
        render_chat_prompt(object(), MESSAGES, protocol, "thinking")


def test_huggingface_parser_receives_generated_only_and_exact_prefix():
    setting = ReasoningSetting("thinking", "thinking", None, True)
    protocol = make_protocol(
        supports_direct=False,
        supports_thinking=True,
        default_mode="thinking",
        trace_format="structured",
        settings=(setting,),
    )
    tokenizer = ParserTokenizer(
        parsed={"reasoning_content": "work", "content": "final"}
    )
    rendered = RenderedPrompt(
        text=None,
        input_ids=(1, 2, 3),
        prefix=(1, 2, 3),
        setting=setting,
    )

    parsed = parse_generated_response(
        tokenizer,
        [65, 66],
        rendered,
        protocol,
        tools=[{"name": "calculator"}],
    )

    assert parsed.status == "complete"
    assert parsed.raw_text == "AB"
    assert parsed.final_text == "final"
    assert parsed.reasoning_text == "work"
    assert tokenizer.calls == [
        (
            [65, 66],
            {
                "prefix": [1, 2, 3],
                "tools": [{"name": "calculator"}],
            },
        )
    ]


def test_parser_is_used_when_callable_even_without_response_template_metadata():
    setting = ReasoningSetting("direct", "direct", None, True)
    protocol = make_protocol(settings=(setting,))
    tokenizer = ParserTokenizer(parsed={"content": "ok"})

    parsed = parse_generated_response(
        tokenizer, "raw", rendered_for(setting), protocol
    )

    assert parsed.final_text == "ok"
    assert parsed.parser == "huggingface"
    assert tokenizer.calls[0][1]["prefix"] == "prompt"


def test_parser_error_and_missing_final_are_inconclusive_and_preserve_raw():
    setting = ReasoningSetting("thinking", "thinking", None, True)
    protocol = make_protocol(
        supports_direct=False,
        supports_thinking=True,
        default_mode="thinking",
        trace_format="structured",
        settings=(setting,),
    )

    errored = parse_generated_response(
        ParserTokenizer(error=ValueError("bad response")),
        "raw output",
        rendered_for(setting),
        protocol,
    )
    missing = parse_generated_response(
        ParserTokenizer(parsed={"reasoning_content": "unfinished", "content": " "}),
        "raw output",
        rendered_for(setting),
        protocol,
    )

    assert errored.status == missing.status == "inconclusive"
    assert errored.raw_text == missing.raw_text == "raw output"
    assert "failed" in errored.error
    assert "missing" in missing.error


def test_truncated_generation_is_inconclusive_without_calling_parser():
    setting = ReasoningSetting("thinking", "thinking", None, True)
    protocol = make_protocol(
        supports_direct=False,
        supports_thinking=True,
        default_mode="thinking",
        trace_format="structured",
        settings=(setting,),
    )
    tokenizer = ParserTokenizer(parsed={"content": "would otherwise pass"})

    parsed = parse_generated_response(
        tokenizer,
        "partial output",
        rendered_for(setting),
        protocol,
        truncated=True,
    )

    assert parsed.status == "inconclusive"
    assert parsed.raw_text == "partial output"
    assert "token limit" in parsed.error
    assert tokenizer.calls == []


def test_unknown_unstructured_output_is_inconclusive_not_best_guessed():
    protocol = detect_reasoning_protocol(model_name="unknown/model")

    parsed = parse_generated_response(
        object(), "ordinary raw text", rendered_for(protocol.settings[0]), protocol
    )

    assert parsed.status == "inconclusive"
    assert parsed.raw_text == "ordinary raw text"
    assert parsed.final_text is None


def test_ordinary_direct_text_with_reserved_words_is_byte_for_byte_preserved():
    setting = ReasoningSetting("direct", "direct", None, True)
    protocol = make_protocol(settings=(setting,))
    raw = "My analysis says the assistant should answer plainly."

    parsed = parse_generated_response(
        object(), raw, rendered_for(setting), protocol
    )

    assert parsed.status == "complete"
    assert parsed.raw_text == raw
    assert parsed.final_text == raw


def test_think_xml_parses_prefilled_open_tag_and_rejects_unclosed_trace():
    setting = ReasoningSetting("thinking", "thinking", None, True)
    protocol = make_protocol(
        supports_direct=False,
        supports_thinking=True,
        default_mode="thinking",
        trace_format="think_xml",
        settings=(setting,),
    )

    complete = parse_generated_response(
        object(),
        "work here</think>Final answer",
        rendered_for(setting, prefix="prompt<think>"),
        protocol,
    )
    unclosed = parse_generated_response(
        object(),
        "<think>still working",
        rendered_for(setting),
        protocol,
    )

    assert complete.status == "complete"
    assert complete.reasoning_text == "work here"
    assert complete.final_text == "Final answer"
    assert unclosed.status == "inconclusive"
    assert "unclosed" in unclosed.error


def test_harmony_requires_and_parses_a_final_channel():
    setting = ReasoningSetting("high", "thinking", "high", True)
    protocol = make_protocol(
        supports_direct=False,
        supports_thinking=True,
        default_mode="thinking",
        control_kind="effort",
        control_argument="reasoning_effort",
        effort_levels=("low", "medium", "high"),
        default_effort="high",
        trace_format="harmony",
        settings=(setting,),
    )
    analysis = "<|channel|>analysis<|message|>work<|end|>"
    raw = analysis + "<|channel|>final<|message|>answer<|end|>"

    complete = parse_generated_response(
        object(), raw, rendered_for(setting), protocol
    )
    incomplete = parse_generated_response(
        object(), analysis, rendered_for(setting), protocol
    )

    assert complete.final_text == "answer"
    assert complete.reasoning_text == "work"
    assert incomplete.status == "inconclusive"
    assert "final channel" in incomplete.error


def test_gemma_empty_off_channel_yields_only_final_answer():
    setting = ReasoningSetting("direct", "direct", False, True)
    protocol = make_protocol(
        supports_thinking=True,
        control_kind="template_bool",
        control_argument="enable_thinking",
        trace_format="gemma_channel",
        settings=(setting, ReasoningSetting("thinking", "thinking", True)),
    )

    parsed = parse_generated_response(
        object(),
        "<thought></thought><final>Answer</final>",
        rendered_for(setting),
        protocol,
    )

    assert parsed.status == "complete"
    assert parsed.final_text == "Answer"
    assert parsed.reasoning_text is None


def test_vendor_parser_is_a_narrow_fallback_and_its_errors_are_inconclusive():
    setting = ReasoningSetting("thinking", "thinking", None, True)
    protocol = make_protocol(
        supports_direct=False,
        supports_thinking=True,
        default_mode="thinking",
        trace_format="structured",
        settings=(setting,),
    )
    seen = []

    def parser(output, **kwargs):
        seen.append((output, kwargs))
        return {"analysis": "work", "final": "answer"}

    complete = parse_generated_response(
        object(), "raw", rendered_for(setting), protocol, vendor_parser=parser
    )

    def broken_parser(output, **kwargs):
        del output, kwargs
        raise RuntimeError("malformed")

    errored = parse_generated_response(
        object(),
        "raw",
        rendered_for(setting),
        protocol,
        vendor_parser=broken_parser,
    )

    assert complete.status == "complete"
    assert complete.final_text == "answer"
    assert complete.reasoning_text == "work"
    assert seen == [("raw", {"prefix": "prompt", "tools": None})]
    assert errored.status == "inconclusive"
    assert errored.raw_text == "raw"
