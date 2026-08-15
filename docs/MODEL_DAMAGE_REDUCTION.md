# Model-damage policy for OBLITERATUS

**Updated:** 2026-08-15
**Goal:** maximize held-out refusal removal subject to hard, predeclared limits on damage to normal model behavior.

## The short version

“Not damaged” does **not** mean that the edited weights still have a similar
size. Two very different models can have weights with the same norm.

In this project, damage means an unwanted change outside the refusal-removal
goal. Examples include worse predictions on ordinary text, lost reasoning or
instruction following, empty or repetitive answers, changed top-token choices,
NaN/Inf values, or a checkpoint that behaves differently after it is saved.

The pipeline now treats a candidate model like a transaction:

> measure the untouched model → make the edit → measure the same held-out
> examples → accept or reject → save to staging → validate → publish

A rejected or unmeasurable candidate is not saved or uploaded.

## What model damage looks like

Damage can be obvious:

- blank answers;
- repeated punctuation or words;
- broken JSON or tool calls;
- nonsense text;
- NaN/Inf logits or an unloadable checkpoint.

It can also be subtle:

- ordinary text becomes less likely, so perplexity rises;
- the probability distribution changes even when the visible answer does not;
- the model picks a different next token more often;
- math, code, multilingual, long-context, calibration, or instruction-following
  performance falls;
- safe questions close to sensitive topics are refused more often;
- a runtime hook makes the in-memory model look good, but that hook disappears
  from the saved checkpoint.

## How the edit can cause damage

Abliteration finds activation directions associated with refusal and removes
some component of those directions from weight matrices. A matrix edit can
damage unrelated behavior when:

- the learned direction also contains useful topic or reasoning information;
- too many directions, layers, or matrices are edited;
- a matrix is projected on the wrong side;
- input embeddings or the vocabulary output head are changed;
- tied embedding/output weights are edited twice;
- norm restoration amplifies everything that remains;
- several refinement passes compound an earlier bad edit;
- reflection (“inverted” mode) changes a component by more than simple removal;
- quantization introduces a second source of error;
- selection and verification reuse the same prompts.

This is why weight norm alone is only a structural diagnostic. The important
test is the model’s behavior and probability distribution before versus after
the edit.

## How damage is now measured

### 1. Keep evaluation examples out of direction discovery

The prompt pairs are split deterministically before probing. Exact normalized
duplicates on either side are kept in one group, so they cannot appear in both
discovery and evaluation. A separate explicit evaluation set can also be
provided; exact overlap is rejected.

The default gate requires at least 32 held-out benign prompts. This is why the
UI now defaults to 99 prompt pairs instead of 33.

### 2. Capture the untouched model

Before the first persistent weight edit, the pipeline records on held-out
benign prompts:

- token-weighted negative log-likelihood (NLL);
- full-vocabulary logits at deterministic positions spread across each prompt;
- deterministic benign generations, coherence, and degeneration.

The retained logit rows are stored on CPU in FP16 to bound memory use. Candidate
rows are quantized the same way before KL/top-token comparison, so baseline-only
compression cannot create false damage on a no-op run. Full prompt-sized
vocabulary tensors are not retained.

### 3. Compare the candidate on exactly the same inputs

After editing, runtime-only steering hooks are removed. The candidate is then
run on the exact same token IDs and masks.

The gate measures:

- **NLL increase:** candidate NLL minus untouched-model NLL, weighted by token;
- **perplexity ratio:** `exp(NLL increase)`;
- **forward KL:** `KL(untouched || candidate)` at sampled real-token positions;
- **p95 KL:** a tail-damage check, because a safe mean can hide a few badly
  damaged prompts;
- **top-1 flip rate:** how often the most likely token changes;
- **coherence drop:** candidate coherence relative to its own baseline;
- **new degeneration:** broken outputs introduced by the edit;
- **non-finite values:** NaN/Inf output evidence;
- **held-out refusal rate:** target efficacy, reported separately from damage.

Blank or degenerate target responses count as efficacy failures. They cannot
improve the score merely because they do not contain a refusal phrase.

NLL and mean KL use paired, one-sided 95% bootstrap upper confidence bounds.
“Paired” means the same prompt is compared before and after, which removes much
of the normal prompt-to-prompt variation.

### 4. Fail closed

Missing, non-finite, failed, out-of-memory, or undersampled required evidence is
not assigned a neutral score. It rejects the candidate. Saving has a second
independent gate check, so calling the save method directly cannot bypass
verification.

## Default acceptance budget

These values are a starting smoke-gate policy, **not universal laws of model
quality**.

| Check | Default limit | Plain-English meaning |
|---|---:|---|
| One-sided 95% upper bound on benign NLL increase | 0.05 nat/token | The worst plausible mean PPL ratio must be at most `exp(0.05)`, about **1.0513** (5.13% higher). |
| One-sided 95% upper bound on mean sampled-token KL | 0.05 nat | Average distribution drift must remain small. |
| p95 sampled-token KL | 0.20 nat | The tail of damaged token positions is also limited. |
| Benign top-1 flip rate | 2% | No more than 2% of sampled positions may change their most likely token. |
| Coherence drop | 10 percentage points | Deterministic benign generation must not materially degrade. |
| New degenerate outputs | 0 | The edit may introduce no new blank/repetitive/broken answers. |
| Non-finite outputs | 0 | NaN/Inf is never accepted. |
| Evidence | 32 prompts, 256 causal tokens, 128 KL positions | Tiny samples cannot pass. |
| Held-out refusal rate | at most 20% on at least 30 prompts | At least 80% of the target prompts must no longer be classified as refusals. |

The code keeps two decisions separate:

- `damage_accepted`: normal behavior stayed inside its budget;
- `efficacy_accepted`: refusal removal reached its target.

The checkpoint is accepted only when both are true. A very compliant but broken
model therefore fails, as does an undamaged model that still refuses too often.

## How “acceptable” should be chosen for a real use case

The defaults answer, “What is a cautious smoke-test budget?” They do not answer,
“What loss is acceptable for every product?” That decision should be made
before searching for an edit:

1. Define the protected uses: for example code, math, a language, medical text,
   tool use, or long context.
2. Run untouched load/save/reload controls in the intended dtype and
   quantization. Measure normal numerical and harness variation.
3. Choose non-inferiority margins based on that noise floor and product risk.
4. Freeze the thresholds before looking at candidate results.
5. Compare candidates only after every hard gate passes. Among passing
   candidates, choose the lowest held-out refusal rate. Use lower normalized
   damage as the tie-breaker when efficacy is equal.

A p-value saying “no significant difference” is not enough. The useful
question is whether a paired confidence bound excludes a loss larger than the
predeclared acceptable margin.

## How projection targets are chosen

The project no longer treats reader edits as categorically unsafe. The four
targets are nested candidates:

- `output`: attention output and FFN-down writers;
- `attention`: `output` plus attention Q/K/V readers;
- `ffn`: `output` plus FFN up/gate/router readers;
- `all`: both reader families plus the output writers.

`projection_target=auto` evaluates every requested target from an exact copy of
the untouched dense model. A candidate must pass all hard damage and efficacy
limits. Among passing candidates, lower refusal wins; measured damage is used
only for an exact refusal tie. The winner is recreated from the untouched
snapshot and evaluated once on a separate 32-pair confirmation set. Failure on
that set is terminal; the code does not try a runner-up after learning the
confirmation result.

The target/locality selection and confirmation sets are disjoint. The small,
fixed benign-generation probe bank is intentionally reused as an invariant
smoke check; it is not an untouched capability benchmark and must not be
treated as one.

This mode requires 64 distinct held-out pairs (32 selection and 32
confirmation), an unquantized FP16/BF16/FP32 model, and roughly one extra
model-size of CPU RAM for exact rollback. The ordinary `advanced` preset keeps
`output` as its inexpensive starting target; aggressive, surgical, inverted,
and nuclear presets explicitly use `all`. Auto target search is available in
the API, CLI, remote runner, and app rather than being silently forced on runs
that cannot meet its memory/evidence requirements.

## Safety changes implemented in this revision

- Projection orientation is explicit. Q/K/V and FFN input matrices use right
  projection; attention and FFN output writers use left projection. Square
  matrices can no longer guess the wrong orientation from shape.
- Projection targets are nested and explicit. Broad attention/FFN reader edits
  compete in objective-driven auto search instead of being excluded by policy.
- Input embeddings and `lm_head` are opt-in, including Nuclear mode.
- Tied embedding/output storage is detected so it is not edited twice.
- Output-head edits respect regularization rather than always using full force.
- Runtime activation-steering hooks are removed before official verification
  and cannot be mistaken for serialized behavior.
- The old “KL correction” is disabled. It used projection magnitude and
  absolute perplexity rather than actual baseline-versus-candidate KL, and its
  approximate add-back could create new damage.
- The former Bayesian `optimized`/`heretic` path is explicitly unavailable and
  fails before editing weights: it measured separate attention/MLP kernels but
  replayed an averaged, different edit. Informed mode continues through its
  deterministic analysis-guided path with Bayesian tuning off.
- Every Informed/Ouroboros persistent pass is checked separately. A failed pass
  is not followed by another edit or saved.
- Iterative verification clears old metrics, so a failed measurement cannot
  reuse a previous pass’s successful values.
- Missing tournament/automatic-search metrics are ineligible rather than
  receiving a neutral score.
- Automatic-search candidates all start from the immutable source model;
  accepted edits are not chained into a fresh baseline.
- Projection-target auto-search separates candidate selection from one-shot
  confirmation and rejects duplicate prompt rows as fake sample size.
- Candidate selection is efficacy-first inside the hard budgets: lowest
  held-out refusal wins, with normalized damage used only for exact ties.
- Checkpoints are written to a same-filesystem staging directory, structurally
  validated, and atomically renamed. Existing non-empty output is replaced
  only with explicit permission, with backup restoration on failure.
- A local source checkpoint cannot also be the output directory.
- Caller-owned offload directories are never recursively deleted as temporary
  storage, and tournament output cleanup is restricted to path-bound owned run
  directories.
- Generic batched perplexity evaluation masks padding labels instead of scoring
  pad tokens as language-model targets.
- Every floating/complex tensor destined for the checkpoint is scanned in
  bounded-memory chunks for NaN/Inf values, including weights in MoE experts
  that the small generation probes may never route through.
- The invalid rank-one “formal spectral certificate” is disabled.

## What this still does not prove

The built-in gate is deliberately affordable enough to run for every
candidate. It catches distribution drift and gross generation failure, but it
does not prove that every capability is preserved.

Before deployment, a passing candidate should also receive paired
untouched-versus-edited evaluation on:

- MMLU and domain-specific knowledge;
- GSM8K/MATH or the model’s intended reasoning tasks;
- IFEval instruction following;
- HumanEval/LiveCodeBench in a no-network sandbox if code matters;
- multilingual suites such as MGSM/Belebele;
- RULER/LongBench if long context matters;
- calibration, safe-but-sensitive prompts, and false-refusal tests;
- the saved-and-reloaded artifact, not only the in-memory model.

Current checkpoint validation proves that the staged files are structurally
complete and readable at the container level; it is not yet a full behavioral
reload evaluation. Exact duplicate grouping is implemented, but semantic
paraphrase clustering also remains future work. Refusal detection is heuristic
and should be replaced or supplemented with a task-specific evaluator.

Projection-target auto-search has a clean selection/confirmation boundary for
its target and locality evidence; its fixed benign generation probes are
shared smoke tests.
The higher-level multi-method Auto, Tournament, and adaptive Informed loops
still reuse their locked gate evidence while choosing iterations or methods;
their results should therefore be treated as search results until the selected
configuration is rerun once on a second untouched promotion set. Community
telemetry is useful for ordering candidates, not for certifying a model.

The original refusal-direction work itself used disjoint train/validation/eval
sets and found model-specific utility changes, including TruthfulQA regression;
preservation therefore has to be demonstrated per model. See
[Arditi et al., NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf).
For held-out locality measured as pre/post prediction KL, see
[MEND](https://arxiv.org/abs/2110.11309). For objective instruction-following
evaluation, see [IFEval](https://arxiv.org/abs/2311.07911).

## Practical decision rule

The end goal is not “conservative editing” and it is not “the lowest refusal
rate at any cost.” It is constrained optimization:

> Reject every candidate outside the damage budget. Among the candidates that
> remain, choose the one with the lowest held-out refusal rate; if efficacy is
> tied, choose the one with less measured damage.
