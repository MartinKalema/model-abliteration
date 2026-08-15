Research complete. The current compatibility lists do leave out major architectures.

Highest-priority findings:

- DeepSeek V4 must be rejected for now. Its residual state is `[batch, sequence, hyper-streams, hidden]`, so current activation collection derives the wrong-sized direction. Its factorized attention writer is also `o_a_proj → o_b_proj`, not `o_proj`. [Official Transformers source](https://github.com/huggingface/transformers/blob/96fe6dce36cc929a5ffd3e34296554c4cb6b669e/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py)
- DBRX and Nemotron-H are effectively unsupported by current lookup.
- Llama 4, Arctic, Granite MoE, Jamba, Falcon-H1, and LFM2 only receive partial attention edits; their FFN/MoE or state-space branches are missed.
- Qwen3-Next and Qwen3.5 MoE skip most Gated DeltaNet layers.
- Current fused MoE support edits `down_proj` but misses the widespread fused `gate_up_proj`.
- Native GPT-OSS MXFP4, MiniMax FP8, and Kimi K2 Thinking INT4 should be rejected on every editing path until exact quantization round-trips are tested.
- A nonzero edit count is not enough: partially supported candidates can appear artificially “safe” because most of the model was never edited.

Required architectural changes:

1. Replace the one-attention/one-FFN string mapping with per-family adapters returning all layer branches.
2. Build and validate a pre-edit parameter manifest with exact names, roles, orientations, shapes, storage identity, and expected layer coverage.
3. Add scoped support for:
   - MLA: `q_a_proj`, `kv_a_proj_with_mqa`, `o_proj`
   - Qwen DeltaNet: `in_proj_qkvz`, `in_proj_ba`, or `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, plus `out_proj`
   - Mamba/short-conv: `in_proj` and `out_proj`; leave `A_log`, `D`, state projections, and convolution kernels untouched
   - Fused experts: `gate_up_proj` and `down_proj`, including transposed GPT-OSS/Llama-4 layouts
   - Nested routers such as DBRX `router.layer` and Hunyuan `gate.wg`
   - Multiple FFN branches such as Arctic and Granite shared-MoE models
4. Require full expected coverage before a projection candidate can compete in automatic selection.
5. Add tiny genuine Transformers-model tests per architecture, plus legacy/current Mixtral layout fixtures.

“Thinking” is not another architecture category. Reasoning is the capability; thinking normally means visible intermediate reasoning tokens or a special response channel.

- Qwen3 switches thinking through the chat template but retains the same weights. [Official model card](https://huggingface.co/Qwen/Qwen3-235B-A22B)
- DeepSeek-R1 is reasoning post-training on the DeepSeek-V3 architecture. [Official model card](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- GPT-OSS reasoning effort changes generated analysis behavior, not its module layout. [Official model card](https://huggingface.co/openai/gpt-oss-20b)
- Kimi K2 Thinking retains the same DeepSeek-V3-style MoE/MLA graph as Kimi K2. [Official model card](https://huggingface.co/moonshotai/Kimi-K2-Thinking)

The current `cot_aware` implementation does not actually observe generated thinking. It disables Qwen thinking in the template and only processes prompt tokens. Its “reasoning direction” is PC1 of harmless-prompt activations, so it should be renamed as a heuristic rather than presented as validated reasoning preservation.