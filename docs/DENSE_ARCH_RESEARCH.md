Research complete, pinned to official Transformers commit [`96fe6dc`](https://github.com/huggingface/transformers/commit/96fe6dce36cc929a5ffd3e34296554c4cb6b669e).

The current lists leave several real gaps.

| Family / `model_type` | Text decoder layout | Projection names | Current support |
|---|---|---|---|
| Llama 3 / `llama` | `model.layers`, `self_attn`, `mlp` | q/k/v/o; gate/up/down | Full |
| Llama 3.2 Vision / `mllama` | `model.layers`; self or cross-attention layers | q/k/v/o; gate/up/down | Partial; `cross_attn` missing |
| Llama 4 / `llama4`, `llama4_text` | `model.layers`; FFN is `feed_forward` | q/k/v/o; dense gate/up/down; fused expert gate-up/down | FFN missing |
| Gemma 2 / `gemma2` | Standard Llama-like | Standard | Full |
| Gemma 3 / `gemma3` | `model.language_model.layers` | Standard | Broken: current path says `model.layers` |
| Gemma 3 text / `gemma3_text` | `model.layers` | Standard | Heuristic only |
| Gemma 3n / `gemma3n` | `model.language_model.layers` | Standard | Heuristic and potentially ambiguous |
| Qwen 2/2.5 text / `qwen2` | Standard | Standard | Full |
| Qwen 2/2.5 VL | `model.language_model.layers` | Standard | Loader cannot normally load full VL checkpoint |
| Qwen 3 / `qwen3` | Standard | Standard | Full |
| Qwen 3 VL / `qwen3_vl` | `model.language_model.layers` | Standard | Loader/path missing |
| Qwen 3.5 / `qwen3_5` | Alternating `self_attn` and `linear_attn` | MHA q/k/v/o; DeltaNet input/output projections | Partial |
| Mistral / `mistral` | Standard | Standard | Full |
| Mistral 3 wrapper / `mistral3` | `model.language_model.layers` | Underlying Mistral/Ministral names | Loader/path missing |
| Ministral / `ministral`, `ministral3` | Standard | Standard | Heuristic only |
| Phi-3 and Phi-4 reasoning / `phi3` | `model.layers` | `qkv_proj`, `o_proj`; `gate_up_proj`, `down_proj` | FFN writer works; fused FFN reader missing |
| Cohere 2 / `cohere2` | Standard | Standard | Full |
| OLMo 2 / `olmo2` | Standard | Standard | Full |
| OLMo 3 / `olmo3` | Standard | Standard | Heuristic only |
| SmolLM3 / `smollm3` | Standard | Standard | Full |
| InternLM3 / `internlm3` | Standard | Standard | Full |
| Granite / `granite` | Standard | Standard | Full |
| Granite SWA / `granite_swa` | Standard | Standard | Heuristic only |
| LFM2 / `lfm2` | Alternating attention and short-conv; FFN is `feed_forward` | q/k/v/out or in/out; FFN w1/w3/w2 | Actual gap |

Important sources:

- [Llama 4](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/llama4/modeling_llama4.py)
- [Gemma 3](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/gemma3/modeling_gemma3.py)
- [Gemma 3n](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/gemma3n/modeling_gemma3n.py)
- [Qwen 3.5](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- [Mistral 3](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/mistral3/modeling_mistral3.py)
- [Phi-3/Phi-4 text backbone](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/phi3/modeling_phi3.py)
- [LFM2](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/lfm2/modeling_lfm2.py)
- [InternLM3 official custom source](https://huggingface.co/internlm/internlm3-8b-instruct/blob/main/modeling_internlm3.py)
- [Official AutoModel mappings](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/auto/modeling_auto.py#L1067-L1144)

Other important dense model types that should be explicitly registered instead of relying on the heuristic are:

`apertus`, `exaone4`, `seed_oss`, `hunyuan_v1_dense`, `arcee`, `nemotron`, `jais2`, `ernie4_5`, `helium`, `hyperclovax`, `vaultgemma`, `granite_swa`, and `olmo3`.

Recommended changes:

1. Replace the single path dictionaries with architecture specifications containing alternative text paths and per-layer variants.
2. Fix `gemma3` immediately.
3. Add `gate_up_proj`, Qwen DeltaNet’s `in_proj_z/in_proj_b/in_proj_a`, and LFM2’s `in_proj`.
4. Map Llama 4 and LFM2 FFNs to `feed_forward`.
5. Support `cross_attn`, `linear_attn`, and LFM2 `conv` as explicit variants.
6. Preserve complete multimodal wrappers when loading and saving. Text-only loading of a multimodal checkpoint can discard its vision tower.
7. Never select the first or largest matching module silently. Validate the exact text prefix, layer count, hidden size, and complete parameter manifest before editing.
8. Resolve embeddings through `get_input_embeddings()` and validate against text vocabulary/hidden size; otherwise the first `nn.Embedding` could be a vision positional embedding.
9. Resolve output heads through `get_output_embeddings()`.
10. Add tests with competing vision/text `ModuleList`s, mixed layer types, expected fully qualified tensor names, and save/reload class parity.

“Thinking” is related to reasoning, but they are not identical:

- Reasoning is the model’s ability to solve multi-step problems.
- Thinking is usually an inference/output mode that exposes a reasoning trace.
- It normally does not change projection names or architecture.

For example, Qwen3.5, SmolLM3, and EXAONE4 switch thinking through templates or flags while retaining the same weights. Phi-4 Reasoning is differently post-trained but still uses `model_type="phi3"`. OLMo 3 Think remains `olmo3`; Ministral Reasoning remains `mistral3` with a `ministral3` text backbone.

Evaluation should use [`tokenizer.parse_response`](https://huggingface.co/docs/transformers/main/en/chat_response_parsing) when available, measure refusal on final `content`, and separately verify that the thinking/reasoning field remains coherent.