"""Lightweight model parameter profiling for OBLITERATUS planning/eval.

Avoids loading full weights.  For local safetensors artifacts it can count
parameters exactly from tensor metadata, and for local GGUF files it reads the
memory-mapped header and tensor shapes without materializing weights in Torch.
For Hub/config-only targets it falls back to architecture-aware config
estimates, including composite Qwen configs whose text dimensions live under
``text_config``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    model: str
    source: str
    total_params: int | None
    total_params_b: float | None
    active_params_b: float | None
    num_layers: int | None
    hidden_size: int | None
    intermediate_size: int | None
    vocab_size: int | None
    model_type: str | None
    dtype: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _text_cfg(cfg: Any) -> Any:
    return _cfg_get(cfg, "text_config") or {}


def _dim(cfg: Any, name: str, default: Any = 0) -> Any:
    val = _cfg_get(cfg, name, None)
    if val not in (None, 0):
        return val
    return _cfg_get(_text_cfg(cfg), name, default)


def _local_config(model: str | Path) -> dict[str, Any] | None:
    path = Path(model).expanduser()
    cfg_path = path / "config.json" if path.is_dir() else path
    if cfg_path.exists() and cfg_path.name == "config.json":
        return json.loads(cfg_path.read_text())
    return None


def _count_safetensors_params(model_dir: Path) -> int | None:
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    if not model_dir.is_dir():
        return None
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None
    seen: set[str] = set()
    total = 0
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as sf:
            # ``safe_open`` exposes tensor names through ``keys()``; the
            # context object itself is not an iterable in current
            # safetensors releases.
            for key in tuple(sf.keys()):
                if key in seen:
                    continue
                seen.add(key)
                try:
                    shape = sf.get_slice(key).get_shape()
                except Exception:  # noqa: BLE001 - fall back across safetensors backends
                    shape = sf.get_tensor(key).shape
                total += math.prod(int(x) for x in shape)
    return total


def _gguf_field_value(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    if field is None:
        return None
    return field.contents()


def _gguf_int(reader: Any, key: str) -> int | None:
    value = _gguf_field_value(reader, key)
    if isinstance(value, (list, tuple)):
        values = {int(item) for item in value if int(item) > 0}
        return values.pop() if len(values) == 1 else None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _gguf_vocab_size(reader: Any, architecture: str) -> int | None:
    explicit = _gguf_int(reader, f"{architecture}.vocab_size")
    if explicit is not None:
        return explicit

    # Most GGUFs expose the vocabulary as an array. ReaderField.data contains
    # one index per array element, so its length can be inspected without
    # decoding every token string into Python objects.
    tokens = reader.get_field("tokenizer.ggml.tokens")
    if tokens is None:
        return None
    size = len(tokens.data)
    return size or None


def _gguf_dtype(reader: Any) -> str | None:
    file_type = _gguf_int(reader, "general.file_type")
    if file_type is None:
        return None
    try:
        from gguf import LlamaFileType

        return LlamaFileType(file_type).name
    except (ImportError, ValueError):
        return f"GGUF_FILE_TYPE_{file_type}"


def _profile_local_gguf(path: Path, *, dtype: str | None = None) -> ModelProfile:
    if not path.is_file():
        raise FileNotFoundError(f"Local GGUF file does not exist or is not a file: {path}")

    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError(
            "Profiling a local GGUF requires gguf==0.19.0; install it with "
            "`pip install gguf==0.19.0`."
        ) from exc

    reader = None
    try:
        # GGUFReader memory-maps tensor storage; reading fields and shapes does
        # not dequantize or allocate the model's weights in Torch.
        reader = GGUFReader(str(path), "r")
        architecture = _gguf_field_value(reader, "general.architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("missing general.architecture metadata")

        total_params = sum(
            math.prod(int(dimension) for dimension in tensor.shape) for tensor in reader.tensors
        )
        if total_params <= 0:
            raise ValueError("GGUF contains no tensor parameters")

        config = {
            "model_type": architecture,
            "num_hidden_layers": _gguf_int(reader, f"{architecture}.block_count"),
            "hidden_size": _gguf_int(reader, f"{architecture}.embedding_length"),
            "intermediate_size": _gguf_int(reader, f"{architecture}.feed_forward_length")
            or _gguf_int(reader, f"{architecture}.expert_feed_forward_length"),
            "moe_intermediate_size": _gguf_int(
                reader, f"{architecture}.expert_feed_forward_length"
            ),
            "vocab_size": _gguf_vocab_size(reader, architecture),
            "num_local_experts": _gguf_int(reader, f"{architecture}.expert_count"),
            "num_experts_per_tok": _gguf_int(reader, f"{architecture}.expert_used_count"),
        }
        total_b = total_params / 1e9
        active_b = estimate_active_params_b(config, total_b)

        return ModelProfile(
            model=str(path),
            source="local_gguf",
            total_params=total_params,
            total_params_b=round(total_b, 6),
            active_params_b=round(active_b, 6),
            num_layers=config["num_hidden_layers"],
            hidden_size=config["hidden_size"],
            intermediate_size=config["intermediate_size"],
            vocab_size=config["vocab_size"],
            model_type=architecture,
            dtype=dtype or _gguf_dtype(reader),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Unable to profile local GGUF {path}: invalid or unsupported GGUF file ({exc})"
        ) from exc
    finally:
        if reader is not None:
            close = getattr(reader, "close", None)
            if callable(close):
                close()


def estimate_total_params(config: Any) -> int | None:
    """Estimate total params from config, falling through to text_config.

    Prefer exact-ish explicit attributes when present; otherwise use an
    architecture-aware transformer estimate. Local artifacts should use
    safetensors metadata for exact counts instead.
    """

    for attr in ("num_parameters", "n_params", "total_params"):
        val = _dim(config, attr, None)
        if val and float(val) > 1000:
            return int(val)

    h = int(_dim(config, "hidden_size", 0) or 0)
    layers = int(_dim(config, "num_hidden_layers", 0) or 0)
    vocab = int(_dim(config, "vocab_size", 0) or 0)
    inter = int(_dim(config, "intermediate_size", h * 4) or h * 4)
    if h <= 0 or layers <= 0:
        return None

    n_heads = int(_dim(config, "num_attention_heads", 0) or max(1, h // 128))
    head_dim = int(_dim(config, "head_dim", 0) or max(1, h // n_heads))
    kv_heads = int(_dim(config, "num_key_value_heads", 0) or n_heads)
    n_experts = int(_dim(config, "num_local_experts", 0) or _dim(config, "num_experts", 1) or 1)
    moe_inter = int(_dim(config, "moe_intermediate_size", 0) or inter)

    # Attention projections: q, k, v, o. Handles GQA/MQA by using kv_heads.
    attn = h * (n_heads * head_dim) + 2 * h * (kv_heads * head_dim) + (n_heads * head_dim) * h
    ffn = 3 * h * moe_inter * n_experts
    if n_experts > 1 and _dim(config, "moe_intermediate_size", None) is not None:
        ffn += 3 * h * inter
        ffn += h * n_experts  # router-ish
    norms = 4 * h
    embeddings = 2 * vocab * h
    return int(layers * (attn + ffn + norms) + embeddings)


def estimate_active_params_b(config: Any, total_params_b: float) -> float:
    n_experts = int(_dim(config, "num_local_experts", 0) or _dim(config, "num_experts", 1) or 1)
    if n_experts <= 1:
        return total_params_b
    top_k = int(_dim(config, "num_experts_per_tok", 0) or _dim(config, "top_k", 2) or 2)
    h = int(_dim(config, "hidden_size", 0) or 0)
    layers = int(_dim(config, "num_hidden_layers", 0) or 0)
    inter = int(_dim(config, "intermediate_size", h * 4) or h * 4)
    moe_inter = int(_dim(config, "moe_intermediate_size", inter) or inter)
    if h <= 0 or layers <= 0:
        return total_params_b
    ffn_per_expert = 3 * h * moe_inter
    ffn_all = ffn_per_expert * n_experts * layers
    ffn_active = ffn_per_expert * min(top_k, n_experts) * layers
    non_ffn = total_params_b * 1e9 - ffn_all
    return max((non_ffn + ffn_active) / 1e9, 0.1)


def profile_model(
    model: str, *, dtype: str | None = None, trust_remote_code: bool = True
) -> ModelProfile:
    path = Path(model).expanduser()
    if path.suffix.lower() == ".gguf":
        return _profile_local_gguf(path, dtype=dtype)

    config: Any | None = _local_config(path)
    source = "local_config" if config is not None else "hf_config"
    exact_params = _count_safetensors_params(path) if path.is_dir() else None

    if config is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model, trust_remote_code=trust_remote_code)

    total_params = exact_params or estimate_total_params(config)
    total_b = (total_params / 1e9) if total_params else None
    active_b = estimate_active_params_b(config, total_b) if total_b else None
    if exact_params is not None:
        source = "local_safetensors"

    return ModelProfile(
        model=str(path) if path.exists() else model,
        source=source,
        total_params=total_params,
        total_params_b=round(total_b, 6) if total_b is not None else None,
        active_params_b=round(active_b, 6) if active_b is not None else None,
        num_layers=int(_dim(config, "num_hidden_layers", 0) or 0) or None,
        hidden_size=int(_dim(config, "hidden_size", 0) or 0) or None,
        intermediate_size=int(_dim(config, "intermediate_size", 0) or 0) or None,
        vocab_size=int(_dim(config, "vocab_size", 0) or 0) or None,
        model_type=str(_dim(config, "model_type", "") or "") or None,
        dtype=dtype
        or str(_dim(config, "dtype", "") or _dim(config, "torch_dtype", "") or "")
        or None,
    )


def default_self_improve_params(profile: ModelProfile) -> dict[str, Any]:
    """Conservative size-aware defaults for recursive residue runs."""

    p = profile.total_params_b or 0.0
    if p >= 20:
        return {
            "n_directions": 3,
            "regularization": 0.30,
            "refinement_passes": 1,
            "residue_weight": 3,
            "verify_sample_size": 30,
            "note": "20B+ model: conservative residue weight and one refinement pass to reduce collapse risk.",
        }
    if p >= 7:
        return {
            "n_directions": 3,
            "regularization": 0.28,
            "refinement_passes": 1,
            "residue_weight": 5,
            "verify_sample_size": 40,
            "note": "7B–20B model: moderate residue pressure.",
        }
    return {
        "n_directions": 2,
        "regularization": 0.25,
        "refinement_passes": 1,
        "residue_weight": 5,
        "verify_sample_size": 50,
        "note": "Small model: lower direction count, larger verify sample.",
    }
