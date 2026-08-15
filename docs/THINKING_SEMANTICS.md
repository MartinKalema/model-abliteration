Research result: “thinking” and “reasoning” should not be one binary architecture flag.

“Reasoning” is the general ability to solve multi-step problems. “Thinking mode” usually means the model generates an explicit deliberation trace before its final answer. A non-thinking model can still reason, and the same checkpoint can often switch between direct and thinking modes.

When an interface says “Thinking…”, it usually means the serving layer is parsing extra generated tokens into a separate field. It does not imply a separate reasoning module inside the transformer.

## Primary-source model matrix

| Family | Actual behavior |
|---|---|
| Qwen3 original | Hybrid checkpoint. `enable_thinking=True/False`; also stateful `/think` and `/no_think`. Raw trace uses `<think>…</think>`. [Official Qwen docs](https://qwen.readthedocs.io/en/stable/inference/transformers.html#thinking-non-thinking-mode) |
| Qwen3 2507 | Separate fixed variants. `Thinking-2507` is thinking-only; `Instruct-2507` is non-thinking-only, despite still performing well on “reasoning” benchmarks. The Thinking template prefills `<think>`, so generated tokens may contain only the closing `</think>`. [Thinking model card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507), [Instruct model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Qwen3.5 | Hybrid hard switch through `enable_thinking`; `/think` and `/nothink` are explicitly unsupported. Thinking is default. [Official model card](https://huggingface.co/Qwen/Qwen3.5-9B), [official template](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja) |
| Qwen3.6 | Hybrid, plus optional preservation of historical thinking with `preserve_thinking`. [Official model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8) |
| DeepSeek-R1 | Reasoning-oriented checkpoint with a `<think>` pattern; original R1 does not expose a dependable generic Boolean off switch. [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| DeepSeek-V3.1/V3.2 | Hybrid modes selected by chat encoding. V3.2 officially uses `thinking_mode="chat"|"thinking"` and a structured `reasoning_content` field. [V3.1 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1), [V3.2 encoder/parser](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/encoding/encoding_dsv32.py) |
| GLM 4.5 onward | Hybrid family with several distinct controls: ordinary thinking, interleaved tool thinking, preserved thinking, and turn-level thinking. APIs use a `thinking` object; some local templates use `enable_thinking`. [Official Z.ai documentation](https://docs.z.ai/guides/capabilities/thinking-mode) |
| GPT-OSS | Always a reasoning model; there is no advertised off mode. `reasoning_effort` is `low`, `medium`, or `high`, defaulting to medium. Harmony separates `analysis` and `final` channels. [Official Harmony guide](https://github.com/openai/openai-cookbook/blob/main/articles/openai-harmony.md), [official template](https://huggingface.co/openai/gpt-oss-20b/blob/main/chat_template.jinja) |
| Gemma 4 | Hybrid Boolean thinking mode. It uses a `thought` channel rather than ordinary `<think>` tags; some sizes emit an empty thought channel even when thinking is off. [Official thinking guide](https://ai.google.dev/gemma/docs/capabilities/thinking), [prompt format](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4) |
| Mistral Small 4 | Hybrid effort control: `reasoning_effort="none"` or `"high"`. [Official model card](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603) |
| Magistral, Phi-4-reasoning, OLMo-Think, Kimi-K2-Thinking | Separate reasoning-oriented checkpoints rather than one hybrid checkpoint. [Magistral](https://huggingface.co/mistralai/Magistral-Small-2509), [Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning), [OLMo 3 Think](https://huggingface.co/allenai/Olmo-3-7B-Think), [Kimi K2 Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking) |
| SmolLM3 and Nemotron | Hybrid models not identifiable from their names. SmolLM3 supports both `/think` and `enable_thinking`; Nemotron generations use several different prompt/template controls. [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3-3B), [Nemotron Nano v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-12B-v2), [Nemotron 3](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) |

These examples also prove that reasoning behavior is usually post-training and prompt protocol, not structural architecture:

- Phi-4 and Phi-4-reasoning have the same dense architecture.
- Qwen3 Thinking and Instruct variants have the same layer structure.
- DeepSeek-V3.1 changes mode by changing the chat prefix.

Therefore, projection names and layer discovery must remain separate from reasoning-mode discovery.

## Current code audit

The current implementation is materially incomplete:

1. [architecture_profiles.py](/Users/martin/Desktop/OBLITERATUS/obliteratus/architecture_profiles.py:120) detects reasoning only from `r1`, `think`, `qwq`, `o1`, and `o3` in the repository name. Config metadata is never consulted for reasoning.

   In a local check, all of these were incorrectly classified as `standard`:

   - Qwen3, Qwen3.5, Qwen3.6
   - GPT-OSS
   - GLM 4.5, 4.7, and 5
   - Gemma 4
   - DeepSeek-V3.1 and V3.2
   - Magistral and Mistral Small 4
   - Nemotron Nano
   - Phi-4-reasoning
   - SmolLM3

2. [abliterate.py](/Users/martin/Desktop/OBLITERATUS/obliteratus/abliterate.py:1928) blindly passes `enable_thinking=False` to every tokenizer. Hugging Face explicitly documents that unsupported templates may silently ignore this option. GPT-OSS needs `reasoning_effort`; DeepSeek-V3.2 needs `thinking_mode`; older Nemotron variants use prompt text.

3. That method always disables thinking for Qwen-style models, while the architecture profile simultaneously claims to preserve reasoning. The two paths are internally contradictory.

4. Fixed-thinking models can spend the entire 128-token refusal-evaluation budget generating their thought trace. The current evaluator then classifies the unparsed trace as though it were the final answer.

5. [app.py](/Users/martin/Desktop/OBLITERATUS/app.py:2373) uses a generic regular expression to remove reasoning. It does not recognize the official `<think>` tag, Harmony channels, or Gemma’s thought channel. It can also damage an ordinary answer merely because it contains words such as “analysis” or “assistant.”

6. The “CoT-aware” activation code does not collect generated chain-of-thought tokens. It runs a forward pass over the user prompt and averages prompt positions, despite comments calling them “reasoning tokens.” Its top harmless principal component is consequently not demonstrated to be a reasoning direction.

7. Telemetry repeats a separate substring heuristic and can classify a model as reasoning merely because the user enabled `cot_aware`.

## Recommended representation

Keep structural architecture and deliberation protocol independent:

```python
@dataclass(frozen=True)
class ReasoningProtocol:
    supports_direct: bool | None
    supports_thinking: bool | None
    default_mode: str              # direct, thinking, adaptive, unknown
    control_kind: str              # template_bool, effort, prompt, encoder, fixed
    control_argument: str | None   # enable_thinking, reasoning_effort, thinking_mode
    effort_levels: tuple[str, ...]
    trace_format: str              # think_xml, harmony, gemma_channel, structured
    history_policy: str            # strip, preserve_tools, configurable
    evidence: tuple[str, ...]
    confidence: str                # confirmed, inferred, unknown
```

Detection priority should be:

1. Explicit user override.
2. Tokenizer/processor response metadata.
3. Prove that rendering the same conversation with alternative kwargs changes the token sequence.
4. Known protocol adapters for Harmony, DeepSeek encoders, GLM, and prompt-controlled Nemotron.
5. Exact official repository manifest as fallback.
6. Otherwise return `unknown`, never silently call it `standard`.

Model names can provide a weak hint but must not authorize reasoning-specific weight-edit settings.

For output parsing, use Hugging Face’s `tokenizer.parse_response(output, prefix=input_ids)` when available. Its official documentation explains why the original prompt prefix is required: some templates open the thinking field inside the prompt. Use official vendor parsers as fallback. If parsing is unsupported, preserve raw text and mark the result inconclusive instead of applying a destructive regex. [Hugging Face response parsing](https://huggingface.co/docs/transformers/en/chat_response_parsing)

## Required behavioral changes

For the project’s “highest removal, lowest damage” goal:

- Hybrid models must be evaluated in both direct and thinking modes.
- Effort-controlled models should be tested at default and maximum effort.
- Refusal classification must use only the parsed final answer.
- Missing final answers, truncated thoughts, and parser errors must reject the candidate as inconclusive.
- Reasoning preservation must be measured with actual reasoning benchmarks in thinking mode.
- The speculative “reasoning direction” derived from generic harmless prompts should not affect weights until validated against genuine generated traces.
- Do not automatically increase directions or refinement passes merely because a model is labelled reasoning; let the damage-gated candidate search decide.

## Concrete tests

Add offline official-template fixtures covering:

- Qwen3 hybrid, Qwen3.5 hard switch, and Qwen3 2507 fixed variants.
- DeepSeek-R1 fixed thinking and V3.1/V3.2 hybrid encoders.
- GLM interleaved and preserved-thinking history.
- GPT-OSS low/medium/high Harmony parsing, with no fake off mode.
- Gemma 4 thought-channel parsing, including the empty off-mode channel.
- Mistral Small 4 `none/high`.
- SmolLM3 and Nemotron controls.
- Phi-4-reasoning, OLMo-Think, Kimi Thinking, and their ordinary counterparts.
- Renamed repositories: behavior must come from artifacts, not the name.
- Unknown templates: must remain `unknown`.
- Ordinary answers containing “analysis” or “assistant” must remain byte-for-byte intact.
- Prefilled opening tags and malformed/unclosed traces.
- A candidate that passes direct mode but damages or still refuses in thinking mode must fail the overall gate.
- A test proving that “CoT-aware” activation collection actually observes generated trace tokens.

No files were changed.