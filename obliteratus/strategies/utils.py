"""Utilities for navigating different HF model architectures."""

from __future__ import annotations

from torch import nn

from obliteratus.models.loader import ModelHandle

# Mapping from model_type -> attribute path to the list of transformer layers.
#
# Keep this explicit even though get_layer_modules() has a structural fallback.
# Modern multimodal models frequently contain several similarly-sized
# ModuleLists (language, vision, audio, and experts), so guessing by length is
# unsafe.  Composite language-model paths are handled separately below.
_LAYER_ATTR_PATHS: dict[str, list[str]] = {
    "arcee": ["model", "layers"],
    "arctic": ["model", "layers"],
    "apertus": ["model", "layers"],
    "gpt2": ["transformer", "h"],
    "gpt_neo": ["transformer", "h"],
    "gpt_neox": ["gpt_neox", "layers"],
    "llama": ["model", "layers"],
    "mllama": ["language_model", "model", "layers"],
    "mistral": ["model", "layers"],
    "gemma": ["model", "layers"],
    "gemma2": ["model", "layers"],
    "phi": ["model", "layers"],
    "phi3": ["model", "layers"],
    "qwen2": ["model", "layers"],
    "qwen2_moe": ["model", "layers"],
    "qwen2_vl": ["model", "layers"],
    "qwen2_5_vl": ["model", "layers"],
    "qwen3": ["model", "layers"],
    "qwen3_moe": ["model", "layers"],
    "qwen3_vl": ["model", "layers"],
    "qwen3_vl_moe": ["model", "layers"],
    "qwen3_next": ["model", "layers"],
    "qwen3_5": ["model", "layers"],
    "qwen3_5_moe": ["model", "layers"],
    "qwen3_5_moe_text": ["model", "layers"],
    "qwen3_5_text": ["model", "layers"],
    "llama4": ["model", "layers"],
    "llama4_text": ["model", "layers"],
    "mistral3": ["model", "layers"],
    "mistral4": ["model", "layers"],
    "ministral": ["model", "layers"],
    "ministral3": ["model", "layers"],
    "gemma3_text": ["model", "layers"],
    "gemma3n": ["model", "layers"],
    "gemma3n_text": ["model", "layers"],
    "gemma4": ["model", "layers"],
    "gemma4_text": ["model", "layers"],
    "gemma4_unified": ["model", "layers"],
    "gemma4_unified_text": ["model", "layers"],
    "minimax_m2": ["model", "layers"],
    "glm_moe_dsa": ["model", "layers"],
    "deepseek_v2": ["model", "layers"],
    "deepseek_v3": ["model", "layers"],
    "deepseek_v32": ["model", "layers"],
    "glm4": ["model", "layers"],
    "glm4_moe": ["model", "layers"],
    "glm4_moe_lite": ["model", "layers"],
    "minicpm3": ["model", "layers"],
    "internlm3": ["model", "layers"],
    "lfm2": ["model", "layers"],
    "lfm2_moe": ["model", "layers"],
    "lfm2_vl": ["model", "layers"],
    "jamba": ["model", "layers"],
    "falcon_h1": ["model", "layers"],
    "nemotron_h": ["model", "layers"],
    "granitemoehybrid": ["model", "layers"],
    "falcon": ["transformer", "h"],
    "opt": ["model", "decoder", "layers"],
    "bloom": ["transformer", "h"],
    "mpt": ["transformer", "blocks"],
    "stablelm": ["model", "layers"],
    "chatglm": ["transformer", "encoder", "layers"],
    "glm": ["model", "layers"],
    "gpt_oss": ["model", "layers"],
    "smollm3": ["model", "layers"],
    "cohere": ["model", "layers"],
    "cohere2": ["model", "layers"],
    "cohere2_moe": ["model", "layers"],
    "olmo": ["model", "layers"],
    "olmo2": ["model", "layers"],
    "olmo3": ["model", "layers"],
    "olmoe": ["model", "layers"],
    "flex_olmo": ["model", "layers"],
    "internlm2": ["model", "layers"],
    "granite": ["model", "layers"],
    "granite_swa": ["model", "layers"],
    "granitemoe": ["model", "layers"],
    "granitemoe_swa": ["model", "layers"],
    "granitemoeshared": ["model", "layers"],
    "gemma3": ["model", "layers"],
    "ernie": ["model", "layers"],
    "ernie4_5": ["model", "layers"],
    "ernie4_5_moe": ["model", "layers"],
    "exaone4": ["model", "layers"],
    "exaone_moe": ["model", "layers"],
    "hunyuan_v1_dense": ["model", "layers"],
    "hunyuan_v1_moe": ["model", "layers"],
    "nemotron": ["model", "layers"],
    "jais2": ["model", "layers"],
    "helium": ["model", "layers"],
    "hyperclovax": ["model", "layers"],
    "phi4_multimodal": ["model", "layers"],
    "seed_oss": ["model", "layers"],
    "solar_open": ["model", "layers"],
    "vaultgemma": ["model", "layers"],
    "dbrx": ["transformer", "blocks"],
}

# Language-first paths for multimodal/composite wrappers.  These are checked
# before architecture-specific direct paths so a vision ``layers`` list can
# never win merely because it has the same length as the text stack.
_NESTED_LANGUAGE_LAYER_PATHS: tuple[str, ...] = (
    "model.language_model.model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "language_model.layers",
    "model.text_model.model.layers",
    "model.text_model.layers",
    "text_model.model.layers",
    "text_model.layers",
)

# Safe semantic paths for otherwise unknown text-only wrappers.  Structural
# discovery is used only after these paths and must be unambiguous.
_COMMON_DIRECT_LAYER_PATHS: tuple[str, ...] = (
    "model.layers",
    "transformer.blocks",
    "transformer.h",
    "gpt_neox.layers",
    "model.decoder.layers",
)

# These model types may look Llama-like at the module-name level, but their
# hidden-state shape violates the pipeline's [batch, sequence, hidden]
# assumption.  Failing here prevents a plausible-looking edit along the wrong
# tensor axis.
_UNSUPPORTED_ARCHITECTURES: dict[str, str] = {
    "deepseek_v4": (
        "DeepSeek V4 uses mHC hyper-connections whose layer activations include "
        "an extra hyper-connection axis. OBLITERATUS currently expects "
        "[batch, sequence, hidden] activations and cannot edit it safely."
    ),
}

_ATTENTION_ATTR: dict[str, str] = {
    "arcee": "self_attn",
    "arctic": "self_attn",
    "apertus": "self_attn",
    "gpt2": "attn",
    "gpt_neo": "attn.attention",
    "gpt_neox": "attention",
    "llama": "self_attn",
    "mllama": "self_attn",
    "mistral": "self_attn",
    "gemma": "self_attn",
    "gemma2": "self_attn",
    "phi": "self_attn",
    "phi3": "self_attn",
    "qwen2": "self_attn",
    "qwen2_moe": "self_attn",
    "qwen2_vl": "self_attn",
    "qwen2_5_vl": "self_attn",
    "qwen3": "self_attn",
    "qwen3_moe": "self_attn",
    "qwen3_vl": "self_attn",
    "qwen3_vl_moe": "self_attn",
    "qwen3_next": "self_attn",
    "qwen3_5": "self_attn",
    "qwen3_5_moe": "self_attn",
    "qwen3_5_moe_text": "self_attn",
    "qwen3_5_text": "self_attn",
    "llama4": "self_attn",
    "llama4_text": "self_attn",
    "mistral3": "self_attn",
    "mistral4": "self_attn",
    "ministral": "self_attn",
    "ministral3": "self_attn",
    "gemma3_text": "self_attn",
    "gemma3n": "self_attn",
    "gemma3n_text": "self_attn",
    "gemma4": "self_attn",
    "gemma4_text": "self_attn",
    "gemma4_unified": "self_attn",
    "gemma4_unified_text": "self_attn",
    "minimax_m2": "self_attn",
    "glm_moe_dsa": "self_attn",
    "deepseek_v3": "self_attn",
    "deepseek_v2": "self_attn",
    "deepseek_v32": "self_attn",
    "glm4": "self_attn",
    "glm4_moe": "self_attn",
    "glm4_moe_lite": "self_attn",
    "minicpm3": "self_attn",
    "internlm3": "self_attn",
    "lfm2": "self_attn",
    "lfm2_moe": "self_attn",
    "lfm2_vl": "self_attn",
    "jamba": "self_attn",
    "falcon_h1": "self_attn",
    "nemotron_h": "mixer",
    "granitemoehybrid": "self_attn",
    "falcon": "self_attention",
    "opt": "self_attn",
    "bloom": "self_attention",
    "mpt": "attn",
    "stablelm": "self_attn",
    "chatglm": "self_attention",
    "glm": "self_attn",
    "gpt_oss": "self_attn",
    "smollm3": "self_attn",
    "cohere": "self_attn",
    "cohere2": "self_attn",
    "cohere2_moe": "self_attn",
    "olmo": "self_attn",
    "olmo2": "self_attn",
    "olmo3": "self_attn",
    "olmoe": "self_attn",
    "flex_olmo": "self_attn",
    "internlm2": "attention",
    "granite": "self_attn",
    "granite_swa": "self_attn",
    "granitemoe": "self_attn",
    "granitemoe_swa": "self_attn",
    "granitemoeshared": "self_attn",
    "gemma3": "self_attn",
    "ernie": "self_attn",
    "ernie4_5": "self_attn",
    "ernie4_5_moe": "self_attn",
    "exaone4": "self_attn",
    "exaone_moe": "self_attn",
    "hunyuan_v1_dense": "self_attn",
    "hunyuan_v1_moe": "self_attn",
    "nemotron": "self_attn",
    "jais2": "self_attn",
    "helium": "self_attn",
    "hyperclovax": "self_attn",
    "phi4_multimodal": "self_attn",
    "seed_oss": "self_attn",
    "solar_open": "self_attn",
    "vaultgemma": "self_attn",
    "dbrx": "norm_attn_norm.attn",
}

_FFN_ATTR: dict[str, str] = {
    "arcee": "mlp",
    "arctic": "block_sparse_moe",
    "apertus": "mlp",
    "gpt2": "mlp",
    "gpt_neo": "mlp",
    "gpt_neox": "mlp",
    "llama": "mlp",
    "mllama": "mlp",
    "mistral": "mlp",
    "gemma": "mlp",
    "gemma2": "mlp",
    "phi": "mlp",
    "phi3": "mlp",
    "qwen2": "mlp",
    "qwen2_moe": "mlp",
    "qwen2_vl": "mlp",
    "qwen2_5_vl": "mlp",
    "qwen3": "mlp",
    "qwen3_moe": "mlp",
    "qwen3_vl": "mlp",
    "qwen3_vl_moe": "mlp",
    "qwen3_next": "mlp",
    "qwen3_5": "mlp",
    "qwen3_5_moe": "mlp",
    "qwen3_5_moe_text": "mlp",
    "qwen3_5_text": "mlp",
    "llama4": "feed_forward",
    "llama4_text": "feed_forward",
    "mistral3": "mlp",
    "mistral4": "mlp",
    "ministral": "mlp",
    "ministral3": "mlp",
    "gemma3_text": "mlp",
    "gemma3n": "mlp",
    "gemma3n_text": "mlp",
    "gemma4": "mlp",
    "gemma4_text": "mlp",
    "gemma4_unified": "mlp",
    "gemma4_unified_text": "mlp",
    "minimax_m2": "mlp",
    "glm_moe_dsa": "mlp",
    "deepseek_v3": "mlp",
    "deepseek_v2": "mlp",
    "deepseek_v32": "mlp",
    "glm4": "mlp",
    "glm4_moe": "mlp",
    "glm4_moe_lite": "mlp",
    "minicpm3": "mlp",
    "internlm3": "mlp",
    "lfm2": "feed_forward",
    "lfm2_moe": "feed_forward",
    "lfm2_vl": "feed_forward",
    "jamba": "feed_forward",
    "falcon_h1": "feed_forward",
    "nemotron_h": "mixer",
    "granitemoehybrid": "block_sparse_moe",
    "falcon": "mlp",
    # OPT: fc1/fc2 live directly on the layer — handled by _FLAT_FFN_ARCHS
    "bloom": "mlp",
    "mpt": "ffn",
    "stablelm": "mlp",
    "chatglm": "mlp",
    "glm": "mlp",
    "gpt_oss": "mlp",
    "smollm3": "mlp",
    "cohere": "mlp",
    "cohere2": "mlp",
    "cohere2_moe": "mlp",
    "olmo": "mlp",
    "olmo2": "mlp",
    "olmo3": "mlp",
    "olmoe": "mlp",
    "flex_olmo": "mlp",
    "internlm2": "feed_forward",
    "granite": "mlp",
    "granite_swa": "mlp",
    "granitemoe": "block_sparse_moe",
    "granitemoe_swa": "block_sparse_moe",
    "granitemoeshared": "block_sparse_moe",
    "gemma3": "mlp",
    "ernie": "mlp",
    "ernie4_5": "mlp",
    "ernie4_5_moe": "mlp",
    "exaone4": "mlp",
    "exaone_moe": "mlp",
    "hunyuan_v1_dense": "mlp",
    "hunyuan_v1_moe": "mlp",
    "nemotron": "mlp",
    "jais2": "mlp",
    "helium": "mlp",
    "hyperclovax": "mlp",
    "phi4_multimodal": "mlp",
    "seed_oss": "mlp",
    "solar_open": "mlp",
    "vaultgemma": "mlp",
    "dbrx": "ffn",
}


# Architectures with hybrid attention (e.g. Qwen3.5 mixes standard multi-head
# attention with GatedDeltaNet).  On layers that lack the primary attribute,
# try the fallbacks in order.
_ATTENTION_ATTR_FALLBACKS: dict[str, list[str]] = {
    "qwen3_5": ["linear_attn"],
    "qwen3_5_moe": ["linear_attn"],
    "qwen3_5_moe_text": ["linear_attn"],
    "qwen3_5_text": ["linear_attn"],
    "qwen3_next": ["linear_attn"],
    "mllama": ["cross_attn"],
    "lfm2": ["conv"],
    "lfm2_moe": ["conv"],
    "lfm2_vl": ["conv"],
    "jamba": ["mamba"],
    "falcon_h1": ["mamba"],
    "granitemoehybrid": ["mamba"],
}

# Families with multiple simultaneously active FFN branches.  Most entries in
# ``_FFN_ATTR`` are alternatives across layer variants; these are true sibling
# residual writers and must all be returned to the edit manifest.
_FFN_ATTR_EXTRA_BRANCHES: dict[str, tuple[str, ...]] = {
    "arctic": ("residual_layer",),
    "granitemoehybrid": ("shared_mlp",),
    "granitemoeshared": ("shared_mlp",),
    # Gemma 4 keeps the dense MLP plus optional router/experts directly on the
    # decoder layer. ``@self`` is included only when those attributes exist.
    "gemma4": ("@self",),
    "gemma4_text": ("@self",),
}

_FFN_ATTR_FALLBACKS: tuple[str, ...] = ("feed_forward", "block_sparse_moe", "ffn")

# Architectures where FFN weights (fc1/fc2) live directly on the layer module
# instead of inside a dedicated MLP submodule.  For these, get_ffn_module
# returns the layer itself so _project_out_advanced can find fc1/fc2.
_FLAT_FFN_ARCHS: set[str] = {"opt"}


def _resolve_attr(obj, dotted_path: str):
    """Resolve a dotted attribute path like 'model.layers'."""
    for attr in dotted_path.split("."):
        obj = getattr(obj, attr)
    return obj


def _choose_explicit_layer_stack(
    model: nn.Module,
    paths: tuple[str, ...],
    *,
    architecture: str,
    expected_layers: int,
    source: str,
) -> nn.ModuleList | None:
    """Resolve semantic layer paths, rejecting conflicting or wrong-size stacks."""
    found: list[tuple[str, nn.ModuleList]] = []
    seen_ids: set[int] = set()
    for path in paths:
        try:
            candidate = _resolve_attr(model, path)
        except AttributeError:
            continue
        if not isinstance(candidate, nn.ModuleList) or id(candidate) in seen_ids:
            continue
        seen_ids.add(id(candidate))
        found.append((path, candidate))

    if not found:
        return None

    eligible = (
        [(path, stack) for path, stack in found if len(stack) == expected_layers]
        if expected_layers
        else found
    )
    if len(eligible) == 1:
        return eligible[0][1]

    details = ", ".join(f"{path} ({len(stack)} layers)" for path, stack in found)
    if not eligible:
        raise RuntimeError(
            f"Found {source} layer path(s) for architecture {architecture!r}, but none "
            f"matched the configured {expected_layers} layers: {details}."
        )
    raise RuntimeError(
        f"Ambiguous {source} layer paths for architecture {architecture!r}: {details}. "
        "Refusing to guess which stack is the language model."
    )


def get_layer_modules(handle: ModelHandle) -> nn.ModuleList:
    """Return the nn.ModuleList of transformer layers for this model."""
    arch = handle.architecture
    if arch in _UNSUPPORTED_ARCHITECTURES:
        raise RuntimeError(_UNSUPPORTED_ARCHITECTURES[arch])

    # Composite models often contain vision/audio layer lists as well as the
    # language stack.  Prefer explicit language paths before every direct or
    # structural lookup.
    layers = _choose_explicit_layer_stack(
        handle.model,
        _NESTED_LANGUAGE_LAYER_PATHS,
        architecture=arch,
        expected_layers=handle.num_layers,
        source="nested language-model",
    )
    if layers is not None:
        return layers

    if arch in _LAYER_ATTR_PATHS:
        architecture_path = ".".join(_LAYER_ATTR_PATHS[arch])
        layers = _choose_explicit_layer_stack(
            handle.model,
            (architecture_path,),
            architecture=arch,
            expected_layers=handle.num_layers,
            source="architecture-specific",
        )
        if layers is not None:
            return layers

    layers = _choose_explicit_layer_stack(
        handle.model,
        _COMMON_DIRECT_LAYER_PATHS,
        architecture=arch,
        expected_layers=handle.num_layers,
        source="common text-model",
    )
    if layers is not None:
        return layers

    # Last resort: structural discovery.  Never select the first or largest
    # list when several candidates exist; that can silently choose a vision or
    # expert stack and edit unrelated weights.
    candidates = [
        (name or "<root>", module)
        for name, module in handle.model.named_modules()
        if isinstance(module, nn.ModuleList) and len(module) > 1
    ]
    wrong_sized_candidates = False
    if handle.num_layers:
        candidates_with_expected_length = [
            (name, module) for name, module in candidates if len(module) == handle.num_layers
        ]
        if len(candidates_with_expected_length) == 1:
            return candidates_with_expected_length[0][1]
        wrong_sized_candidates = not candidates_with_expected_length and bool(candidates)
        candidates_to_report = candidates_with_expected_length or candidates
    else:
        if len(candidates) == 1:
            return candidates[0][1]
        candidates_to_report = candidates

    if candidates_to_report:
        details = ", ".join(
            f"{name} ({len(module)} layers)" for name, module in candidates_to_report
        )
        if wrong_sized_candidates:
            raise RuntimeError(
                f"Cannot locate a transformer stack with the configured {handle.num_layers} "
                f"layers for architecture {arch!r}. Found: {details}."
            )
        raise RuntimeError(
            f"Ambiguous transformer layer stacks for architecture {arch!r}: {details}. "
            "Refusing to guess which stack is the language model."
        )
    raise RuntimeError(
        f"Cannot locate transformer layers for architecture {arch!r}. "
        f"Supported: {sorted(_LAYER_ATTR_PATHS)}"
    )


def is_registered_architecture(architecture: str) -> bool:
    """Return whether layer and branch discovery has an explicit family adapter."""
    return (
        architecture in _LAYER_ATTR_PATHS
        and architecture in _ATTENTION_ATTR
        and architecture in _FFN_ATTR
    )


def get_layer_module_path(handle: ModelHandle, layers: nn.ModuleList | None = None) -> str:
    """Return the exact qualified path of the resolved text decoder stack."""
    resolved = layers if layers is not None else get_layer_modules(handle)
    matches = [
        name
        for name, module in handle.model.named_modules()
        if module is resolved
    ]
    if len(matches) != 1:
        details = matches or ["<unregistered>"]
        raise RuntimeError(
            "Resolved text layer stack does not have one unique qualified path: "
            + ", ".join(details)
        )
    return matches[0]


def _resolve_layer_branches(
    layer_module: nn.Module,
    attr_paths: list[str],
) -> list[tuple[str, nn.Module]]:
    """Resolve every distinct, non-null branch named by an architecture adapter."""
    branches: list[tuple[str, nn.Module]] = []
    seen: set[int] = set()
    for attr_path in dict.fromkeys(attr_paths):
        if attr_path == "@self":
            candidate = layer_module
            # ``@self`` is reserved for layouts such as Gemma 4 where the
            # router and fused experts are siblings of the ordinary MLP.
            if not (
                getattr(layer_module, "experts", None) is not None
                and getattr(layer_module, "router", None) is not None
            ):
                continue
        else:
            try:
                candidate = _resolve_attr(layer_module, attr_path)
            except AttributeError:
                continue
        if candidate is None or not isinstance(candidate, nn.Module):
            continue
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        branches.append((attr_path, candidate))
    return branches


def get_attention_modules(
    layer_module: nn.Module,
    architecture: str,
) -> list[tuple[str, nn.Module]]:
    """Return every residual mixer branch in a decoder layer.

    The path is returned with the module so callers can build exact qualified
    parameter names. Hybrid families may have one branch selected per layer
    (Qwen DeltaNet, LFM2, Jamba) or multiple coexisting branches (Falcon-H1).
    """
    if architecture == "nemotron_h":
        block_type = getattr(layer_module, "block_type", None)
        if block_type not in {"full_attention", "linear_attention"}:
            return []
    attr_paths = [
        _ATTENTION_ATTR.get(architecture, "self_attn"),
        *_ATTENTION_ATTR_FALLBACKS.get(architecture, []),
    ]
    return _resolve_layer_branches(layer_module, attr_paths)


def get_attention_module(layer_module: nn.Module, architecture: str) -> nn.Module:
    """Return the first mixer branch for backward-compatible single-branch callers."""
    branches = get_attention_modules(layer_module, architecture)
    if branches:
        return branches[0][1]
    attr_paths = [
        _ATTENTION_ATTR.get(architecture, "self_attn"),
        *_ATTENTION_ATTR_FALLBACKS.get(architecture, []),
    ]
    raise AttributeError(
        f"Cannot locate attention module for architecture {architecture!r}; "
        f"tried {attr_paths}."
    )


def get_ffn_modules(
    layer_module: nn.Module,
    architecture: str,
) -> list[tuple[str, nn.Module]]:
    """Return every dense, sparse, and shared FFN branch in a layer."""
    if architecture in _FLAT_FFN_ARCHS:
        return [("@self", layer_module)]
    if architecture == "nemotron_h":
        block_type = getattr(layer_module, "block_type", None)
        if block_type not in {"mlp", "moe"}:
            return []
    attr_paths = [
        _FFN_ATTR.get(architecture, "mlp"),
        *_FFN_ATTR_EXTRA_BRANCHES.get(architecture, ()),
        *_FFN_ATTR_FALLBACKS,
    ]
    return _resolve_layer_branches(layer_module, attr_paths)


def get_ffn_module(layer_module: nn.Module, architecture: str) -> nn.Module:
    """Return the first FFN branch for backward-compatible single-branch callers."""
    branches = get_ffn_modules(layer_module, architecture)
    if branches:
        return branches[0][1]
    attr_paths = [
        _FFN_ATTR.get(architecture, "mlp"),
        *_FFN_ATTR_EXTRA_BRANCHES.get(architecture, ()),
        *_FFN_ATTR_FALLBACKS,
    ]
    raise AttributeError(
        f"Cannot locate FFN module for architecture {architecture!r}; tried {attr_paths}."
    )


def get_embedding_module(handle: ModelHandle) -> nn.Embedding:
    """Return the token embedding module."""
    model = handle.model
    # Hugging Face wrappers know which embedding belongs to the text model.
    # This must precede path or structural discovery: multimodal checkpoints
    # often register a vision positional embedding first.
    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embedding = get_embeddings()
        if isinstance(embedding, nn.Embedding):
            hidden_size = int(getattr(handle, "hidden_size", 0) or 0)
            if hidden_size and embedding.embedding_dim != hidden_size:
                raise RuntimeError(
                    "get_input_embeddings() returned an embedding whose width "
                    f"({embedding.embedding_dim}) does not match the text hidden "
                    f"size ({hidden_size})"
                )
            return embedding
    # Try common paths
    for path in [
        "transformer.wte",
        "model.embed_tokens",
        "gpt_neox.embed_in",
        "model.decoder.embed_tokens",
        "transformer.word_embeddings",
    ]:
        try:
            emb = _resolve_attr(model, path)
            if isinstance(emb, nn.Embedding):
                return emb
        except AttributeError:
            continue

    # Never pick the first embedding structurally. A wrong choice can mutate a
    # vision/audio positional table while leaving the text model unchanged.
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Embedding)
        and (not handle.hidden_size or module.embedding_dim == handle.hidden_size)
    ]
    if len(candidates) == 1:
        return candidates[0][1]
    if candidates:
        names = ", ".join(name or "<root>" for name, _ in candidates)
        raise RuntimeError(
            f"Ambiguous text embedding candidates: {names}. The model wrapper "
            "must implement get_input_embeddings()."
        )
    raise RuntimeError("Cannot locate a validated text embedding module.")
