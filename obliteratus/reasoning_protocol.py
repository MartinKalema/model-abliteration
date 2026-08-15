"""Reasoning/thinking protocol discovery, rendering, and response parsing.

This module deliberately keeps *inference protocol* separate from transformer
architecture.  A checkpoint may expose direct and deliberative modes without
changing a single layer, and two checkpoints with the same architecture may use
different response channels.

The implementation is intentionally dependency-light.  It works with tokenizer-
like objects by duck typing, performs no network access, and treats unsupported
or ambiguous behavior as unknown rather than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


_SEMANTIC_MODES = frozenset({"direct", "thinking", "unknown"})
_DEFAULT_MODES = frozenset({"direct", "thinking", "adaptive", "unknown"})
_CONTROL_KINDS = frozenset(
    {"template_bool", "effort", "prompt", "encoder", "fixed", "unknown"}
)
_CONFIDENCE_LEVELS = frozenset({"confirmed", "inferred", "unknown"})


class ReasoningProtocolError(RuntimeError):
    """Base error for invalid or unsupported protocol operations."""


class ProtocolRenderError(ReasoningProtocolError):
    """Raised when a requested protocol setting cannot be rendered safely."""


def _as_tuple(value: Any) -> tuple:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _freeze(value: Any) -> Any:
    """Recursively convert common containers into immutable equivalents."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class ReasoningSetting:
    """One concrete, renderable inference setting.

    ``semantic_mode`` describes behavior, while ``control_value`` is the exact
    tokenizer/encoder value.  For example, Mistral Small 4 maps ``none`` to the
    direct semantic mode and ``high`` to the thinking semantic mode.
    """

    name: str
    semantic_mode: str
    control_value: bool | str | None = None
    is_default: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError("ReasoningSetting.name must be a non-empty string")
        if self.semantic_mode not in _SEMANTIC_MODES:
            raise ValueError(
                "ReasoningSetting.semantic_mode must be direct, thinking, or unknown"
            )
        if self.control_value is not None and not isinstance(
            self.control_value, (bool, str)
        ):
            raise TypeError("ReasoningSetting.control_value must be bool, str, or None")


@dataclass(frozen=True)
class ReasoningProtocol:
    """Immutable description of a model's deliberation protocol.

    Unknown Boolean fields mean "not established", not ``False``.  ``evidence``
    records why the classification was made so callers can distinguish rendered
    artifact evidence from an exact-manifest fallback.
    """

    supports_direct: bool | None
    supports_thinking: bool | None
    default_mode: str
    control_kind: str
    control_argument: str | None
    effort_levels: tuple[str, ...] = ()
    trace_format: str = "unknown"
    history_policy: str = "strip"
    evidence: tuple[str, ...] = ()
    confidence: str = "unknown"
    settings: tuple[ReasoningSetting, ...] = ()
    default_effort: str | None = None
    adapter_id: str | None = None
    parser_kind: str = "auto"

    def __post_init__(self) -> None:
        if self.supports_direct not in (True, False, None):
            raise TypeError("supports_direct must be bool or None")
        if self.supports_thinking not in (True, False, None):
            raise TypeError("supports_thinking must be bool or None")
        if self.default_mode not in _DEFAULT_MODES:
            raise ValueError(
                "default_mode must be direct, thinking, adaptive, or unknown"
            )
        if self.control_kind not in _CONTROL_KINDS:
            raise ValueError(f"Unsupported control_kind: {self.control_kind!r}")
        if self.confidence not in _CONFIDENCE_LEVELS:
            raise ValueError(f"Unsupported confidence: {self.confidence!r}")
        object.__setattr__(self, "effort_levels", _as_tuple(self.effort_levels))
        object.__setattr__(self, "evidence", _as_tuple(self.evidence))
        object.__setattr__(self, "settings", _as_tuple(self.settings))
        if any(not isinstance(setting, ReasoningSetting) for setting in self.settings):
            raise TypeError("settings must contain ReasoningSetting instances")
        default_count = sum(1 for setting in self.settings if setting.is_default)
        if default_count > 1:
            raise ValueError("A protocol may have at most one default setting")

    def setting(self, name: str) -> ReasoningSetting:
        """Return a named setting or raise a clear error."""

        for setting in self.settings:
            if setting.name == name:
                return setting
        available = ", ".join(setting.name for setting in self.settings) or "default"
        raise ValueError(f"Unknown reasoning setting {name!r}; available: {available}")


@dataclass(frozen=True)
class RenderedPrompt:
    """A rendered conversation and the exact prefix needed for response parsing."""

    text: str | None
    input_ids: tuple[int, ...] | None
    prefix: str | tuple[int, ...]
    setting: ReasoningSetting
    control_kwargs: tuple[tuple[str, bool | str], ...] = ()
    adapter_id: str | None = None

    def __post_init__(self) -> None:
        if (self.text is None) == (self.input_ids is None):
            raise ValueError("RenderedPrompt requires exactly one of text or input_ids")
        if self.input_ids is not None:
            object.__setattr__(self, "input_ids", tuple(int(item) for item in self.input_ids))
        if isinstance(self.prefix, list):
            object.__setattr__(self, "prefix", tuple(int(item) for item in self.prefix))
        normalized_kwargs = []
        for key, value in self.control_kwargs:
            if not isinstance(key, str) or not isinstance(value, (bool, str)):
                raise TypeError("control_kwargs entries must be (str, bool | str)")
            normalized_kwargs.append((key, value))
        object.__setattr__(self, "control_kwargs", tuple(normalized_kwargs))

    @property
    def model_input(self) -> str | tuple[int, ...]:
        """Return the value to tokenize or pass directly to a model."""

        return self.text if self.text is not None else self.input_ids  # type: ignore[return-value]


@dataclass(frozen=True)
class ParsedResponse:
    """A parsed assistant response with an explicit conclusive/inconclusive state."""

    raw_text: str
    final_text: str | None
    reasoning_text: str | None
    structured: Mapping[str, Any] | None
    status: str
    parser: str
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"complete", "inconclusive"}:
            raise ValueError("ParsedResponse.status must be complete or inconclusive")
        if self.status == "complete" and (
            self.final_text is None or not self.final_text.strip()
        ):
            raise ValueError("A complete ParsedResponse requires non-blank final_text")
        if self.structured is not None:
            object.__setattr__(self, "structured", _freeze(self.structured))

    @property
    def is_conclusive(self) -> bool:
        return self.status == "complete"


@dataclass(frozen=True)
class _RenderAttempt:
    accepted: bool
    signature: Any = None
    value: Any = None
    error: str | None = None


def _normalise_ids(value: Any) -> tuple[int, ...] | None:
    """Normalize a tokenizer/encoder return value without importing torch/numpy."""

    if isinstance(value, Mapping):
        value = value.get("input_ids")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    if isinstance(value, list) and all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        return tuple(value)
    return None


def _render_signature(value: Any) -> Any:
    ids = _normalise_ids(value)
    if ids is not None:
        return ("ids", ids)
    if isinstance(value, str):
        return ("text", value)
    return ("unsupported", type(value).__name__, repr(value))


def _call_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    control_kwargs: Mapping[str, bool | str] | None = None,
) -> _RenderAttempt:
    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        return _RenderAttempt(False, error="apply_chat_template unavailable")
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    kwargs.update(control_kwargs or {})
    try:
        value = apply(list(messages), **kwargs)
    except Exception as exc:  # tokenizer implementations use heterogeneous errors
        return _RenderAttempt(False, error=f"{type(exc).__name__}: {exc}")
    return _RenderAttempt(True, _render_signature(value), value)


def _call_encoder(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    control_kwargs: Mapping[str, bool | str],
) -> _RenderAttempt:
    encoder = getattr(tokenizer, "encode_messages", None)
    if not callable(encoder):
        return _RenderAttempt(False, error="encode_messages unavailable")
    try:
        value = encoder(list(messages), **dict(control_kwargs))
    except Exception as exc:
        return _RenderAttempt(False, error=f"{type(exc).__name__}: {exc}")
    return _RenderAttempt(True, _render_signature(value), value)


def _artifact_text(tokenizer: Any) -> str:
    parts = []
    for name in ("chat_template", "response_template"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            parts.append(str(value))
    return "\n".join(parts).lower()


def _trace_format_from_artifacts(tokenizer: Any) -> tuple[str, str, str]:
    text = _artifact_text(tokenizer)
    has_parser = callable(getattr(tokenizer, "parse_response", None))
    parser_kind = "huggingface" if has_parser else "auto"
    if "<|channel|>" in text and "analysis" in text and "final" in text:
        return "harmony", "strip", parser_kind
    if "<thought>" in text or "thought" in text and "start_of_turn" in text:
        return "gemma_channel", "strip", parser_kind
    if "reasoning_content" in text:
        return "structured", "strip", parser_kind
    if "<think>" in text or "</think>" in text:
        return "think_xml", "strip", parser_kind
    if has_parser:
        return "structured", "strip", parser_kind
    return "unknown", "strip", parser_kind


def _default_setting(
    settings: tuple[ReasoningSetting, ...],
    default_signature: Any,
    signatures: Mapping[str, Any],
) -> tuple[tuple[ReasoningSetting, ...], str, str | None]:
    matches = [name for name, signature in signatures.items() if signature == default_signature]
    if len(matches) != 1:
        return settings, "unknown", None
    default_name = matches[0]
    updated = tuple(
        replace(setting, is_default=(setting.name == default_name)) for setting in settings
    )
    semantic = next(setting.semantic_mode for setting in updated if setting.is_default)
    return updated, semantic, default_name


def _unknown_protocol(
    *,
    trace_format: str = "unknown",
    history_policy: str = "strip",
    parser_kind: str = "auto",
    evidence: Sequence[str] = (),
) -> ReasoningProtocol:
    return ReasoningProtocol(
        supports_direct=None,
        supports_thinking=None,
        default_mode="unknown",
        control_kind="unknown",
        control_argument=None,
        trace_format=trace_format,
        history_policy=history_policy,
        evidence=tuple(evidence) or ("no protocol behavior established",),
        confidence="unknown",
        settings=(ReasoningSetting("default", "unknown", None, True),),
        parser_kind=parser_kind,
    )


def _detect_rendered_control(tokenizer: Any) -> tuple[ReasoningProtocol | None, set[str]]:
    """Return a confirmed protocol and the kwargs accepted but shown ineffective."""

    messages = ({"role": "user", "content": "protocol probe"},)
    trace_format, history_policy, parser_kind = _trace_format_from_artifacts(tokenizer)
    default = _call_chat_template(tokenizer, messages)
    accepted_but_unchanged: set[str] = set()

    # Boolean controls used by Qwen, Gemma, GLM, and SmolLM families.
    false_render = _call_chat_template(
        tokenizer, messages, control_kwargs={"enable_thinking": False}
    )
    true_render = _call_chat_template(
        tokenizer, messages, control_kwargs={"enable_thinking": True}
    )
    if false_render.accepted and true_render.accepted:
        if false_render.signature != true_render.signature:
            settings = (
                ReasoningSetting("direct", "direct", False),
                ReasoningSetting("thinking", "thinking", True),
            )
            default_mode = "unknown"
            if default.accepted:
                settings, default_mode, _ = _default_setting(
                    settings,
                    default.signature,
                    {"direct": false_render.signature, "thinking": true_render.signature},
                )
            return (
                ReasoningProtocol(
                    supports_direct=True,
                    supports_thinking=True,
                    default_mode=default_mode,
                    control_kind="template_bool",
                    control_argument="enable_thinking",
                    trace_format=trace_format,
                    history_policy=(
                        "configurable"
                        if "preserve_thinking" in _artifact_text(tokenizer)
                        else history_policy
                    ),
                    evidence=(
                        "artifact-render-diff: enable_thinking=False != True",
                    ),
                    confidence="confirmed",
                    settings=settings,
                    parser_kind=parser_kind,
                ),
                accepted_but_unchanged,
            )
        accepted_but_unchanged.add("enable_thinking")

    # DeepSeek's local encoder uses thinking_mode=chat|thinking.  Prefer the
    # encoder when present, otherwise probe a template exposing the same values.
    chat_render = _call_encoder(
        tokenizer, messages, control_kwargs={"thinking_mode": "chat"}
    )
    thinking_render = _call_encoder(
        tokenizer, messages, control_kwargs={"thinking_mode": "thinking"}
    )
    control_kind = "encoder"
    if not (chat_render.accepted and thinking_render.accepted):
        chat_render = _call_chat_template(
            tokenizer, messages, control_kwargs={"thinking_mode": "chat"}
        )
        thinking_render = _call_chat_template(
            tokenizer, messages, control_kwargs={"thinking_mode": "thinking"}
        )
        control_kind = "encoder"
    if chat_render.accepted and thinking_render.accepted:
        if chat_render.signature != thinking_render.signature:
            settings = (
                ReasoningSetting("direct", "direct", "chat"),
                ReasoningSetting("thinking", "thinking", "thinking"),
            )
            default_mode = "unknown"
            if default.accepted:
                settings, default_mode, _ = _default_setting(
                    settings,
                    default.signature,
                    {"direct": chat_render.signature, "thinking": thinking_render.signature},
                )
            return (
                ReasoningProtocol(
                    supports_direct=True,
                    supports_thinking=True,
                    default_mode=default_mode,
                    control_kind=control_kind,
                    control_argument="thinking_mode",
                    trace_format=(
                        trace_format if trace_format != "unknown" else "structured"
                    ),
                    history_policy=history_policy,
                    evidence=(
                        "artifact-render-diff: thinking_mode=chat != thinking",
                    ),
                    confidence="confirmed",
                    settings=settings,
                    adapter_id="deepseek_encoder",
                    parser_kind=parser_kind,
                ),
                accepted_but_unchanged,
            )
        accepted_but_unchanged.add("thinking_mode")

    # Effort probing is restricted to values actually named by the template.
    # This avoids treating a template that blindly interpolates arbitrary input
    # as proof that an undocumented effort level is valid.
    artifact_text = _artifact_text(tokenizer)
    if "reasoning_effort" in artifact_text:
        candidate_levels = tuple(
            level for level in ("none", "low", "medium", "high") if level in artifact_text
        )
        successful: dict[str, _RenderAttempt] = {}
        for level in candidate_levels:
            attempt = _call_chat_template(
                tokenizer, messages, control_kwargs={"reasoning_effort": level}
            )
            if attempt.accepted:
                successful[level] = attempt
        unique_signatures = {attempt.signature for attempt in successful.values()}
        if len(successful) >= 2 and len(unique_signatures) >= 2:
            levels = tuple(successful)
            settings = tuple(
                ReasoningSetting(
                    level,
                    "direct" if level == "none" else "thinking",
                    level,
                )
                for level in levels
            )
            default_mode = "unknown"
            default_effort = None
            if default.accepted:
                settings, default_mode, default_effort = _default_setting(
                    settings,
                    default.signature,
                    {level: successful[level].signature for level in levels},
                )
            return (
                ReasoningProtocol(
                    supports_direct="none" in levels,
                    supports_thinking=any(level != "none" for level in levels),
                    default_mode=default_mode,
                    control_kind="effort",
                    control_argument="reasoning_effort",
                    effort_levels=levels,
                    default_effort=default_effort,
                    trace_format=trace_format,
                    history_policy=history_policy,
                    evidence=(
                        "artifact-render-diff: reasoning_effort levels alter token sequence",
                    ),
                    confidence="confirmed",
                    settings=settings,
                    parser_kind=parser_kind,
                ),
                accepted_but_unchanged,
            )
        if len(successful) >= 2:
            accepted_but_unchanged.add("reasoning_effort")

    return None, accepted_but_unchanged


def _manifest_control_is_silently_ignored(
    tokenizer: Any,
    protocol: ReasoningProtocol,
) -> bool:
    """Check the exact fallback control rather than trusting an accepted kwarg."""

    messages = ({"role": "user", "content": "protocol probe"},)
    argument = protocol.control_argument
    if argument is None:
        return False
    if protocol.control_kind == "template_bool":
        first_value, second_value = False, True
    elif protocol.control_kind == "encoder":
        first_value, second_value = "chat", "thinking"
    elif protocol.control_kind == "effort" and len(protocol.effort_levels) >= 2:
        first_value, second_value = protocol.effort_levels[0], protocol.effort_levels[-1]
    else:
        return False

    if protocol.control_kind == "encoder" and callable(
        getattr(tokenizer, "encode_messages", None)
    ):
        first = _call_encoder(
            tokenizer, messages, control_kwargs={argument: first_value}
        )
        second = _call_encoder(
            tokenizer, messages, control_kwargs={argument: second_value}
        )
    else:
        first = _call_chat_template(
            tokenizer, messages, control_kwargs={argument: first_value}
        )
        second = _call_chat_template(
            tokenizer, messages, control_kwargs={argument: second_value}
        )
    return (
        first.accepted
        and second.accepted
        and first.signature == second.signature
    )


def _manifest_protocol(model_name: str) -> ReasoningProtocol | None:
    """Return a conservative exact-family fallback.

    This registry runs only after behavioral artifact probing.  It exists for
    fixed checkpoints and older tokenizer versions that cannot expose modern
    response metadata.  Patterns are anchored to official repository namespaces;
    arbitrary occurrences of words such as "think" never classify a model.
    """

    name = model_name.strip().lower()
    if not name:
        return None

    def protocol(
        *,
        supports_direct: bool,
        supports_thinking: bool,
        default_mode: str,
        control_kind: str,
        control_argument: str | None,
        settings: tuple[ReasoningSetting, ...],
        trace_format: str,
        effort_levels: tuple[str, ...] = (),
        default_effort: str | None = None,
        adapter_id: str | None = None,
        history_policy: str = "strip",
    ) -> ReasoningProtocol:
        return ReasoningProtocol(
            supports_direct=supports_direct,
            supports_thinking=supports_thinking,
            default_mode=default_mode,
            control_kind=control_kind,
            control_argument=control_argument,
            effort_levels=effort_levels,
            default_effort=default_effort,
            trace_format=trace_format,
            history_policy=history_policy,
            evidence=(f"exact official repository manifest: {model_name}",),
            confidence="inferred",
            settings=settings,
            adapter_id=adapter_id,
        )

    # Fixed Qwen 2507 variants must precede the general Qwen3 matcher.
    if re.match(r"^qwen/qwen3-[^/]*-thinking-2507(?:$|[-_])", name):
        return protocol(
            supports_direct=False,
            supports_thinking=True,
            default_mode="thinking",
            control_kind="fixed",
            control_argument=None,
            settings=(ReasoningSetting("thinking", "thinking", None, True),),
            trace_format="think_xml",
        )
    if re.match(r"^qwen/qwen3-[^/]*-instruct-2507(?:$|[-_])", name):
        return protocol(
            supports_direct=True,
            supports_thinking=False,
            default_mode="direct",
            control_kind="fixed",
            control_argument=None,
            settings=(ReasoningSetting("direct", "direct", None, True),),
            trace_format="none",
        )
    if re.match(r"^qwen/qwen3\.(?:5|6)-", name):
        return protocol(
            supports_direct=True,
            supports_thinking=True,
            default_mode="thinking",
            control_kind="template_bool",
            control_argument="enable_thinking",
            settings=(
                ReasoningSetting("direct", "direct", False),
                ReasoningSetting("thinking", "thinking", True, True),
            ),
            trace_format="think_xml",
            history_policy="configurable" if name.startswith("qwen/qwen3.6-") else "strip",
        )
    if re.match(r"^qwen/qwen3-(?!.*(?:thinking|instruct)-2507)[^/]+$", name):
        return protocol(
            supports_direct=True,
            supports_thinking=True,
            default_mode="thinking",
            control_kind="template_bool",
            control_argument="enable_thinking",
            settings=(
                ReasoningSetting("direct", "direct", False),
                ReasoningSetting("thinking", "thinking", True, True),
            ),
            trace_format="think_xml",
        )

    if re.match(r"^deepseek-ai/deepseek-r1(?:$|-)", name):
        return protocol(
            supports_direct=False,
            supports_thinking=True,
            default_mode="thinking",
            control_kind="fixed",
            control_argument=None,
            settings=(ReasoningSetting("thinking", "thinking", None, True),),
            trace_format="think_xml",
        )
    if re.match(r"^deepseek-ai/deepseek-v3\.(?:1|2)(?:$|-)", name):
        return protocol(
            supports_direct=True,
            supports_thinking=True,
            default_mode="unknown",
            control_kind="encoder",
            control_argument="thinking_mode",
            settings=(
                ReasoningSetting("direct", "direct", "chat"),
                ReasoningSetting("thinking", "thinking", "thinking"),
            ),
            trace_format="structured",
            adapter_id="deepseek_encoder",
        )

    if re.match(r"^(?:zai-org|thudm)/glm-(?:4\.[5-9]|5)(?:$|-)", name):
        return protocol(
            supports_direct=True,
            supports_thinking=True,
            default_mode="unknown",
            control_kind="template_bool",
            control_argument="enable_thinking",
            settings=(
                ReasoningSetting("direct", "direct", False),
                ReasoningSetting("thinking", "thinking", True),
            ),
            trace_format="structured",
            history_policy="configurable",
            adapter_id="glm_thinking",
        )

    if re.match(r"^openai/gpt-oss-(?:20b|120b)(?:$|-)", name):
        return protocol(
            supports_direct=False,
            supports_thinking=True,
            default_mode="thinking",
            control_kind="effort",
            control_argument="reasoning_effort",
            effort_levels=("low", "medium", "high"),
            default_effort="medium",
            settings=(
                ReasoningSetting("low", "thinking", "low"),
                ReasoningSetting("medium", "thinking", "medium", True),
                ReasoningSetting("high", "thinking", "high"),
            ),
            trace_format="harmony",
            adapter_id="harmony",
        )

    if re.match(r"^google/gemma-4(?:$|-)", name):
        return protocol(
            supports_direct=True,
            supports_thinking=True,
            default_mode="thinking",
            control_kind="template_bool",
            control_argument="enable_thinking",
            settings=(
                ReasoningSetting("direct", "direct", False),
                ReasoningSetting("thinking", "thinking", True, True),
            ),
            trace_format="gemma_channel",
            adapter_id="gemma_thought",
        )

    if re.match(r"^mistralai/mistral-small-4(?:$|-)", name):
        return protocol(
            supports_direct=True,
            supports_thinking=True,
            default_mode="unknown",
            control_kind="effort",
            control_argument="reasoning_effort",
            effort_levels=("none", "high"),
            settings=(
                ReasoningSetting("none", "direct", "none"),
                ReasoningSetting("high", "thinking", "high"),
            ),
            trace_format="structured",
        )

    fixed_thinking_patterns = (
        r"^mistralai/magistral-(?:small|medium)(?:$|-)",
        r"^microsoft/phi-4-reasoning(?:$|-)",
        r"^allenai/olmo-(?:3|3\.1)-[^/]*-think(?:$|-)",
        r"^moonshotai/kimi-k2-thinking(?:$|-)",
    )
    if any(re.match(pattern, name) for pattern in fixed_thinking_patterns):
        return protocol(
            supports_direct=False,
            supports_thinking=True,
            default_mode="thinking",
            control_kind="fixed",
            control_argument=None,
            settings=(ReasoningSetting("thinking", "thinking", None, True),),
            trace_format="think_xml",
        )

    if re.match(r"^huggingfacetb/smollm3-(?:3b|[^/]+)(?:$|-)", name):
        return protocol(
            supports_direct=True,
            supports_thinking=True,
            default_mode="thinking",
            control_kind="template_bool",
            control_argument="enable_thinking",
            settings=(
                ReasoningSetting("direct", "direct", False),
                ReasoningSetting("thinking", "thinking", True, True),
            ),
            trace_format="think_xml",
        )

    # Nemotron releases use multiple incompatible prompt/template controls.
    # Merely recognizing the family is not enough to select one safely.
    if re.match(r"^nvidia/(?:nvidia-)?nemotron(?:$|-)", name):
        return _unknown_protocol(
            evidence=(
                "exact Nemotron family manifest requires artifact evidence or explicit override",
            )
        )

    return None


def detect_reasoning_protocol(
    tokenizer: Any = None,
    config: Any = None,
    model_name: str = "",
    override: ReasoningProtocol | None = None,
) -> ReasoningProtocol:
    """Detect a deliberation protocol without network access.

    Detection is artifact-first.  A control is confirmed only when rendering
    the same conversation with two values changes the resulting token sequence.
    An accepted-but-ignored keyword explicitly suppresses a hybrid manifest
    fallback, preventing a known-looking repository name from turning a no-op
    argument into a purported control.
    """

    del config  # Reserved for future explicit metadata; never substring-guessed.
    if override is not None:
        if not isinstance(override, ReasoningProtocol):
            raise TypeError("override must be a ReasoningProtocol")
        return replace(
            override,
            evidence=("explicit user override",) + tuple(override.evidence),
        )

    trace_format = "unknown"
    history_policy = "strip"
    parser_kind = "auto"
    ignored: set[str] = set()
    if tokenizer is not None:
        trace_format, history_policy, parser_kind = _trace_format_from_artifacts(tokenizer)
        detected, ignored = _detect_rendered_control(tokenizer)
        if detected is not None:
            return detected

    fallback = _manifest_protocol(model_name)
    if fallback is not None:
        expected = fallback.control_argument
        silently_ignored = (
            tokenizer is not None
            and fallback.control_kind in {"template_bool", "effort", "encoder"}
            and _manifest_control_is_silently_ignored(tokenizer, fallback)
        )
        if (
            fallback.control_kind in {"template_bool", "effort", "encoder"}
            and expected is not None
            and (expected in ignored or silently_ignored)
        ):
            return _unknown_protocol(
                trace_format=trace_format,
                history_policy=history_policy,
                parser_kind=parser_kind,
                evidence=(
                    f"{expected} was accepted but rendered identical token sequences",
                    "manifest fallback suppressed by contradictory artifact behavior",
                ),
            )
        # Preserve stronger parser/trace metadata without changing manifest mode
        # semantics.  Unknown artifact format must not erase a known exact format.
        return replace(
            fallback,
            trace_format=(
                trace_format if trace_format != "unknown" else fallback.trace_format
            ),
            history_policy=(
                history_policy
                if history_policy != "strip" or fallback.history_policy == "strip"
                else fallback.history_policy
            ),
            parser_kind=(
                parser_kind if parser_kind != "auto" else fallback.parser_kind
            ),
        )

    return _unknown_protocol(
        trace_format=trace_format,
        history_policy=history_policy,
        parser_kind=parser_kind,
        evidence=(
            "tokenizer artifacts did not prove a supported reasoning control",
        ),
    )


def required_evaluation_settings(
    protocol: ReasoningProtocol,
) -> tuple[ReasoningSetting, ...]:
    """Return the minimal settings that must pass an evaluation gate.

    Hybrid controls test direct and thinking.  Effort-only models without a
    direct setting test their default and maximum effort.  Unknown protocols
    return an explicit ``unknown`` default so integration code can fail closed.
    """

    settings = protocol.settings or (
        ReasoningSetting("default", "unknown", None, True),
    )
    if protocol.control_kind == "effort":
        direct = [setting for setting in settings if setting.semantic_mode == "direct"]
        thinking = [setting for setting in settings if setting.semantic_mode == "thinking"]
        if direct:
            # Hybrid effort controls (e.g. none/high): direct plus maximum effort.
            maximum = None
            for level in reversed(protocol.effort_levels):
                maximum = next((item for item in thinking if item.name == level), None)
                if maximum is not None:
                    break
            return tuple(direct[:1] + ([maximum] if maximum is not None else thinking[-1:]))

        default = next((setting for setting in thinking if setting.is_default), None)
        if default is None and protocol.default_effort:
            default = next(
                (setting for setting in thinking if setting.name == protocol.default_effort),
                None,
            )
        maximum = None
        for level in reversed(protocol.effort_levels):
            maximum = next((item for item in thinking if item.name == level), None)
            if maximum is not None:
                break
        selected = []
        for setting in (default, maximum):
            if setting is not None and setting not in selected:
                selected.append(setting)
        return tuple(selected or thinking[:1] or settings[:1])

    if protocol.control_kind in {"template_bool", "encoder", "prompt"}:
        direct = next(
            (setting for setting in settings if setting.semantic_mode == "direct"),
            None,
        )
        thinking = next(
            (setting for setting in settings if setting.semantic_mode == "thinking"),
            None,
        )
        selected = [setting for setting in (direct, thinking) if setting is not None]
        return tuple(selected or settings[:1])

    return tuple(settings[:1])


def _resolve_setting(
    protocol: ReasoningProtocol,
    setting: ReasoningSetting | str | None,
) -> ReasoningSetting:
    if isinstance(setting, ReasoningSetting):
        if setting not in protocol.settings:
            raise ValueError("The requested setting does not belong to this protocol")
        return setting
    if isinstance(setting, str):
        return protocol.setting(setting)
    default = next((item for item in protocol.settings if item.is_default), None)
    if default is not None:
        return default
    if len(protocol.settings) == 1:
        return protocol.settings[0]
    raise ValueError("Protocol has no known default; choose a setting explicitly")


def _with_prompt_directive(
    messages: Sequence[Mapping[str, Any]],
    directive: str,
    *,
    position: str,
) -> list[dict[str, Any]]:
    cloned = [dict(message) for message in messages]
    for index in range(len(cloned) - 1, -1, -1):
        if cloned[index].get("role") != "user":
            continue
        content = cloned[index].get("content")
        if not isinstance(content, str):
            raise ProtocolRenderError("Prompt controls require string user content")
        if position == "prefix":
            cloned[index]["content"] = f"{directive}\n{content}"
        else:
            cloned[index]["content"] = f"{content}\n{directive}"
        return cloned
    raise ProtocolRenderError("Prompt controls require at least one user message")


def _rendered_prompt_from_value(
    value: Any,
    *,
    setting: ReasoningSetting,
    control_kwargs: Mapping[str, bool | str],
    adapter_id: str | None,
) -> RenderedPrompt:
    if isinstance(value, str):
        return RenderedPrompt(
            text=value,
            input_ids=None,
            prefix=value,
            setting=setting,
            control_kwargs=tuple(control_kwargs.items()),
            adapter_id=adapter_id,
        )
    ids = _normalise_ids(value)
    if ids is None:
        raise ProtocolRenderError(
            f"Tokenizer returned unsupported rendered value {type(value).__name__}"
        )
    return RenderedPrompt(
        text=None,
        input_ids=ids,
        prefix=ids,
        setting=setting,
        control_kwargs=tuple(control_kwargs.items()),
        adapter_id=adapter_id,
    )


def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    protocol: ReasoningProtocol,
    setting: ReasoningSetting | str | None = None,
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = True,
    tools: Sequence[Mapping[str, Any]] | None = None,
) -> RenderedPrompt:
    """Render one setting without silently falling back to another mode."""

    selected = _resolve_setting(protocol, setting)
    rendered_messages: Sequence[Mapping[str, Any]] = [dict(item) for item in messages]
    control_kwargs: dict[str, bool | str] = {}

    if protocol.control_kind in {"template_bool", "effort", "encoder"}:
        if protocol.control_argument is None or selected.control_value is None:
            raise ProtocolRenderError("Controlled protocol lacks an argument/value")
        control_kwargs[protocol.control_argument] = selected.control_value
    elif protocol.control_kind == "prompt":
        if not isinstance(selected.control_value, str):
            raise ProtocolRenderError("Prompt control requires a string directive")
        if protocol.adapter_id == "prompt_suffix":
            rendered_messages = _with_prompt_directive(
                rendered_messages, selected.control_value, position="suffix"
            )
        elif protocol.adapter_id in {"prompt_prefix", "qwen_slash"}:
            rendered_messages = _with_prompt_directive(
                rendered_messages, selected.control_value, position="prefix"
            )
        else:
            raise ProtocolRenderError(
                "Prompt-controlled protocols require an exact prompt adapter"
            )
    elif protocol.control_kind not in {"fixed", "unknown"}:
        raise ProtocolRenderError(f"Unsupported control kind: {protocol.control_kind}")

    if protocol.control_kind == "encoder" and callable(
        getattr(tokenizer, "encode_messages", None)
    ):
        try:
            value = tokenizer.encode_messages(rendered_messages, **control_kwargs)
        except Exception as exc:
            raise ProtocolRenderError(
                f"Encoder could not render {selected.name!r}: {type(exc).__name__}: {exc}"
            ) from exc
        return _rendered_prompt_from_value(
            value,
            setting=selected,
            control_kwargs=control_kwargs,
            adapter_id=protocol.adapter_id,
        )

    apply = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply):
        raise ProtocolRenderError("Tokenizer has no apply_chat_template")
    kwargs: dict[str, Any] = {
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = list(tools)
    kwargs.update(control_kwargs)
    try:
        value = apply(list(rendered_messages), **kwargs)
    except Exception as exc:
        raise ProtocolRenderError(
            f"Chat template could not render {selected.name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    return _rendered_prompt_from_value(
        value,
        setting=selected,
        control_kwargs=control_kwargs,
        adapter_id=protocol.adapter_id,
    )


def _decode_raw(tokenizer: Any, generated: Any) -> str:
    if isinstance(generated, str):
        return generated
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return ""
    ids = _normalise_ids(generated)
    value: Any = list(ids) if ids is not None else generated
    try:
        return str(decode(value, skip_special_tokens=False))
    except TypeError:
        return str(decode(value))


def _prefix_for_parser(rendered: RenderedPrompt) -> str | list[int]:
    if isinstance(rendered.prefix, tuple):
        return list(rendered.prefix)
    return rendered.prefix


def _text_content(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    return None


def _normalise_parsed_message(parsed: Mapping[str, Any]) -> tuple[str | None, str | None]:
    final = None
    for field in ("content", "final", "final_answer"):
        candidate = _text_content(parsed.get(field))
        if candidate is not None:
            final = candidate
            break
    reasoning_parts = []
    for field in ("reasoning_content", "analysis", "thought", "thinking", "reasoning"):
        candidate = _text_content(parsed.get(field))
        if candidate:
            reasoning_parts.append(candidate)
    reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
    return final, reasoning


def _prefix_text(tokenizer: Any, rendered: RenderedPrompt) -> str:
    if isinstance(rendered.prefix, str):
        return rendered.prefix
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        return ""
    try:
        return str(decode(list(rendered.prefix), skip_special_tokens=False))
    except TypeError:
        return str(decode(list(rendered.prefix)))


def _structured_trace_error(
    protocol: ReasoningProtocol,
    raw_text: str,
    prefix_text: str,
) -> str | None:
    combined = prefix_text + raw_text
    if protocol.trace_format == "think_xml":
        if combined.count("<think>") > combined.count("</think>"):
            return "unclosed <think> trace"
    elif protocol.trace_format == "gemma_channel":
        if combined.count("<thought>") > combined.count("</thought>"):
            return "unclosed <thought> trace"
    elif protocol.trace_format == "harmony":
        if "<|channel|>analysis" in raw_text and "<|channel|>final" not in raw_text:
            return "missing Harmony final channel"
    return None


def _strict_think_xml_parse(
    raw_text: str,
    prefix_text: str,
    setting: ReasoningSetting,
) -> tuple[str | None, str | None]:
    reasoning = None
    full_match = list(re.finditer(r"<think>(.*?)</think>", raw_text, re.DOTALL))
    if full_match:
        match = full_match[-1]
        reasoning = match.group(1)
        return raw_text[match.end() :], reasoning
    close = raw_text.rfind("</think>")
    if close >= 0 and prefix_text.rfind("<think>") > prefix_text.rfind("</think>"):
        return raw_text[close + len("</think>") :], raw_text[:close]
    if setting.semantic_mode == "direct":
        return raw_text, None
    return None, None


def _strict_harmony_parse(raw_text: str) -> tuple[str | None, str | None]:
    final_matches = list(
        re.finditer(
            r"<\|channel\|>final(?:<\|constrain\|>[^<]*)?<\|message\|>"
            r"(.*?)(?=<\|end\|>|<\|start\|>|$)",
            raw_text,
            re.DOTALL,
        )
    )
    if not final_matches:
        return None, None
    analysis_matches = list(
        re.finditer(
            r"<\|channel\|>analysis(?:<\|constrain\|>[^<]*)?<\|message\|>"
            r"(.*?)(?=<\|end\|>|<\|start\|>|$)",
            raw_text,
            re.DOTALL,
        )
    )
    reasoning = analysis_matches[-1].group(1) if analysis_matches else None
    return final_matches[-1].group(1), reasoning


def _strict_gemma_parse(
    raw_text: str,
    setting: ReasoningSetting,
) -> tuple[str | None, str | None]:
    matches = list(re.finditer(r"<thought>(.*?)</thought>", raw_text, re.DOTALL))
    if not matches:
        return (raw_text, None) if setting.semantic_mode == "direct" else (None, None)
    match = matches[-1]
    final = raw_text[match.end() :]
    final_match = re.fullmatch(r"\s*<final>(.*?)</final>\s*", final, re.DOTALL)
    if final_match:
        final = final_match.group(1)
    return final, match.group(1) or None


def _inconclusive(raw_text: str, parser: str, error: str) -> ParsedResponse:
    return ParsedResponse(
        raw_text=raw_text,
        final_text=None,
        reasoning_text=None,
        structured=None,
        status="inconclusive",
        parser=parser,
        error=error,
    )


def parse_generated_response(
    tokenizer: Any,
    generated: Any,
    rendered: RenderedPrompt,
    protocol: ReasoningProtocol,
    *,
    truncated: bool = False,
    tools: Sequence[Mapping[str, Any]] | None = None,
    vendor_parser: Callable[..., Mapping[str, Any]] | None = None,
) -> ParsedResponse:
    """Parse generated output while preserving raw text on every failure.

    ``generated`` must contain only newly generated output.  ``rendered.prefix``
    is passed to Hugging Face's parser exactly because templates can prefill an
    opening trace marker.  The function never searches ordinary prose for words
    such as "analysis" or "assistant".
    """

    raw_text = _decode_raw(tokenizer, generated)
    if truncated:
        return _inconclusive(
            raw_text,
            "none",
            "generation reached its token limit before a confirmed stop",
        )

    prefix_text = _prefix_text(tokenizer, rendered)
    trace_error = _structured_trace_error(protocol, raw_text, prefix_text)
    if trace_error is not None:
        return _inconclusive(raw_text, "none", trace_error)

    parse_response = getattr(tokenizer, "parse_response", None)
    if callable(parse_response):
        kwargs: dict[str, Any] = {"prefix": _prefix_for_parser(rendered)}
        if tools is not None:
            kwargs["tools"] = list(tools)
        parse_input: Any = generated
        generated_ids = _normalise_ids(generated)
        if generated_ids is not None:
            parse_input = list(generated_ids)
        try:
            parsed = parse_response(parse_input, **kwargs)
        except Exception as exc:
            return _inconclusive(
                raw_text,
                "huggingface",
                f"response parser failed: {type(exc).__name__}: {exc}",
            )
        if not isinstance(parsed, Mapping):
            return _inconclusive(
                raw_text,
                "huggingface",
                "response parser did not return a message mapping",
            )
        final, reasoning = _normalise_parsed_message(parsed)
        if final is None or not final.strip():
            return ParsedResponse(
                raw_text=raw_text,
                final_text=None,
                reasoning_text=reasoning,
                structured=parsed,
                status="inconclusive",
                parser="huggingface",
                error="parsed response is missing non-blank final content",
            )
        return ParsedResponse(
            raw_text=raw_text,
            final_text=final,
            reasoning_text=reasoning,
            structured=parsed,
            status="complete",
            parser="huggingface",
        )

    if vendor_parser is not None:
        try:
            parsed = vendor_parser(
                generated,
                prefix=_prefix_for_parser(rendered),
                tools=list(tools) if tools is not None else None,
            )
        except Exception as exc:
            return _inconclusive(
                raw_text,
                "vendor",
                f"vendor parser failed: {type(exc).__name__}: {exc}",
            )
        if not isinstance(parsed, Mapping):
            return _inconclusive(raw_text, "vendor", "vendor parser returned no message mapping")
        final, reasoning = _normalise_parsed_message(parsed)
        if final is None or not final.strip():
            return ParsedResponse(
                raw_text=raw_text,
                final_text=None,
                reasoning_text=reasoning,
                structured=parsed,
                status="inconclusive",
                parser="vendor",
                error="vendor response is missing non-blank final content",
            )
        return ParsedResponse(
            raw_text=raw_text,
            final_text=final,
            reasoning_text=reasoning,
            structured=parsed,
            status="complete",
            parser="vendor",
        )

    final = reasoning = None
    parser = "none"
    if protocol.trace_format == "think_xml":
        final, reasoning = _strict_think_xml_parse(
            raw_text, prefix_text, rendered.setting
        )
        parser = "think_xml"
    elif protocol.trace_format == "harmony":
        final, reasoning = _strict_harmony_parse(raw_text)
        parser = "harmony"
    elif protocol.trace_format == "gemma_channel":
        final, reasoning = _strict_gemma_parse(raw_text, rendered.setting)
        parser = "gemma_channel"
    elif (
        protocol.supports_direct is True
        and protocol.supports_thinking is False
        and protocol.trace_format in {"none", "plain", "unknown"}
    ):
        final = raw_text
        parser = "direct"

    if final is None or not final.strip():
        return ParsedResponse(
            raw_text=raw_text,
            final_text=None,
            reasoning_text=reasoning,
            structured=None,
            status="inconclusive",
            parser=parser,
            error="no trustworthy non-blank final answer could be parsed",
        )
    return ParsedResponse(
        raw_text=raw_text,
        final_text=final,
        reasoning_text=reasoning,
        structured=None,
        status="complete",
        parser=parser,
    )


__all__ = [
    "ParsedResponse",
    "ProtocolRenderError",
    "ReasoningProtocol",
    "ReasoningProtocolError",
    "ReasoningSetting",
    "RenderedPrompt",
    "detect_reasoning_protocol",
    "parse_generated_response",
    "render_chat_prompt",
    "required_evaluation_settings",
]
