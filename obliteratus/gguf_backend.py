"""Resolve GGUF inputs without conflating model and tokenizer provenance.

GGUF is an import format for OBLITERATUS.  Transformers dequantizes the
checkpoint into ordinary floating-point PyTorch parameters before any model
editing takes place; the original GGUF file is never modified in place.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

logger = logging.getLogger(__name__)

SourceFormat = Literal["hf", "gguf"]


@dataclass(frozen=True)
class ModelSource:
    """Resolved model/checkpoint and tokenizer locations.

    ``model_root`` and ``gguf_file`` are passed separately to Transformers.
    This is required both for Hub repositories (repo id + filename) and local
    GGUF files (parent directory + basename).  ``tokenizer_source`` is kept
    independent so chat templates and special-token ids come from the
    canonical model rather than being reconstructed from quantized metadata.
    """

    source_format: SourceFormat
    requested_model: str
    model_root: str
    gguf_file: str | None
    canonical_model_id: str | None
    tokenizer_source: str
    is_local: bool

    @property
    def format(self) -> SourceFormat:
        """Compatibility alias for callers that use the shorter field name."""
        return self.source_format

    @property
    def is_gguf(self) -> bool:
        return self.source_format == "gguf"

    @property
    def source_file(self) -> str | None:
        """Return a stable artifact identifier for provenance."""
        if self.gguf_file is None:
            return None
        if self.is_local:
            return str(Path(self.model_root, self.gguf_file))
        return self.gguf_file

    def summary(self) -> dict[str, object]:
        return {
            "format": self.source_format,
            "requested_model": self.requested_model,
            "model_root": self.model_root,
            "source_file": self.source_file,
            "canonical_model_id": self.canonical_model_id,
            "tokenizer_source": self.tokenizer_source,
            "is_local": self.is_local,
            "in_memory_format": "dense" if self.is_gguf else "native",
        }


@dataclass(frozen=True)
class _GGUFCompatibilityMetadata:
    """Small metadata view used to repair Transformers' GGUF config import.

    ``GGUFReader`` memory-maps tensor storage.  This view reads scalar metadata,
    tensor names, and shapes only; it never accesses or dequantizes tensor data.
    """

    architecture: str | None
    fields: dict[str, Any]
    tensor_names: frozenset[str]


_TRANSFORMERS_GGUF_COMPAT_LOCK = threading.RLock()
_TRANSFORMERS_GGUF_COMPAT_DEPTH = 0
_GGUF_LOAD_DTYPE: ContextVar[Any | None] = ContextVar("obliteratus_gguf_load_dtype", default=None)
_GGUF_FUSED_PARTS: ContextVar[dict[tuple[str, str], set[str]] | None] = ContextVar(
    "obliteratus_gguf_fused_parts", default=None
)


def _field_contents(reader: Any, key: str) -> Any:
    field = reader.get_field(key)
    return None if field is None else field.contents()


def _read_compatibility_metadata(
    gguf_path: str | os.PathLike[str],
    reader_class: type,
) -> _GGUFCompatibilityMetadata:
    """Read only config-relevant metadata from a GGUF file."""

    reader = reader_class(os.fspath(gguf_path), "r")
    try:
        architecture = _field_contents(reader, "general.architecture")
        keys: tuple[str, ...]
        if architecture == "gemma4":
            keys = (
                "gemma4.attention.head_count_kv",
                "gemma4.attention.sliding_window_pattern",
                "gemma4.attention.key_length",
                "gemma4.attention.key_length_swa",
                "gemma4.expert_count",
                "gemma4.expert_used_count",
                "gemma4.expert_feed_forward_length",
                "gemma4.embedding_length_per_layer_input",
                "gemma4.final_logit_softcapping",
                "gemma4.attention.shared_kv_layers",
            )
        else:
            keys = ()
        return _GGUFCompatibilityMetadata(
            architecture=architecture if isinstance(architecture, str) else None,
            fields={key: _field_contents(reader, key) for key in keys},
            tensor_names=frozenset(tensor.name for tensor in reader.tensors),
        )
    finally:
        # gguf 0.19 readers own NumPy memmaps rather than an explicit public
        # close() method.  Newer releases may expose close(), so use it when
        # present and otherwise let reference counting release the maps.
        close = getattr(reader, "close", None)
        if callable(close):
            close()


def read_gguf_file_type(gguf_path: str | os.PathLike[str]) -> str | None:
    """Return the GGUF ``general.file_type`` enum name without loading weights.

    For example, both challenge Q4_K_M files return ``"MOSTLY_Q4_K_M"``.
    Unknown numeric values are retained as ``"GGUF_FILE_TYPE_<n>"`` so source
    quantization provenance is never silently discarded.
    """

    path = Path(gguf_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Local GGUF file does not exist: {path}")
    try:
        from gguf import GGUFReader, LlamaFileType
    except ImportError as exc:
        raise RuntimeError(
            "Reading GGUF metadata requires the pinned gguf==0.19.0 package"
        ) from exc

    reader = GGUFReader(str(path), "r")
    try:
        value = _field_contents(reader, "general.file_type")
        if value is None:
            return None
        numeric_value = int(value)
        try:
            return LlamaFileType(numeric_value).name
        except ValueError:
            return f"GGUF_FILE_TYPE_{numeric_value}"
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()


def _repair_gpt_oss_reader_fields(reader: Any) -> None:
    """Correct the scalar view used by Transformers 5.14.1 for GPT-OSS RoPE.

    Transformers' generic GGUF parser correctly uses ``field.data``.  Its
    GPT-OSS-specific RoPE block instead reads ``field.parts[0]``, which is the
    GGUF key-length array (for example ``array([27])`` for ``factor``), not the
    field value.  Change only that erroneous compatibility view; the actual
    data part and the mapped tensor storage remain untouched.
    """

    if _field_contents(reader, "general.architecture") != "gpt-oss":
        return
    prefix = "gpt-oss.rope.scaling."
    for key, field in reader.fields.items():
        if not key.startswith(prefix):
            continue
        value = field.contents()
        if isinstance(value, (str, int, float, bool)):
            field.parts[0] = value


def _compatible_reader_class(reader_class: type) -> type:
    class TransformersCompatibleGGUFReader(reader_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _repair_gpt_oss_reader_fields(self)

    TransformersCompatibleGGUFReader.__name__ = reader_class.__name__
    TransformersCompatibleGGUFReader.__qualname__ = reader_class.__qualname__
    return TransformersCompatibleGGUFReader


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"GGUF metadata {field} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"GGUF metadata {field} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"GGUF metadata {field} must be positive, got {parsed}")
    return parsed


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"GGUF metadata {field} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"GGUF metadata {field} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"GGUF metadata {field} must be non-negative, got {parsed}")
    return parsed


def _repair_gemma4_config(
    config: dict[str, Any],
    metadata: _GGUFCompatibilityMetadata,
) -> None:
    """Translate Gemma 4's per-layer llama.cpp metadata to HF config fields."""

    if metadata.architecture != "gemma4" or config.get("model_type") != "gemma4_text":
        return

    fields = metadata.fields
    kv_heads = fields.get("gemma4.attention.head_count_kv")
    sliding_pattern = fields.get("gemma4.attention.sliding_window_pattern")
    if not isinstance(kv_heads, list) or not isinstance(sliding_pattern, list):
        raise TypeError("Gemma 4 GGUF is missing per-layer KV-head or sliding-attention metadata")
    if len(kv_heads) != len(sliding_pattern) or len(kv_heads) != config.get("num_hidden_layers"):
        raise RuntimeError(
            "Gemma 4 GGUF per-layer attention metadata does not match num_hidden_layers"
        )

    normalized_pattern = [bool(value) for value in sliding_pattern]
    sliding_heads = {
        _positive_int(value, field="gemma4.attention.head_count_kv")
        for value, is_sliding in zip(kv_heads, normalized_pattern, strict=True)
        if is_sliding
    }
    global_heads = {
        _positive_int(value, field="gemma4.attention.head_count_kv")
        for value, is_sliding in zip(kv_heads, normalized_pattern, strict=True)
        if not is_sliding
    }
    if len(sliding_heads) != 1 or len(global_heads) != 1:
        raise RuntimeError(
            "Gemma 4 GGUF must use one KV-head count for sliding layers and one for global layers"
        )

    config["num_key_value_heads"] = sliding_heads.pop()
    config["num_global_key_value_heads"] = global_heads.pop()
    config["layer_types"] = [
        "sliding_attention" if is_sliding else "full_attention" for is_sliding in normalized_pattern
    ]
    config["head_dim"] = _positive_int(
        fields.get("gemma4.attention.key_length_swa"),
        field="gemma4.attention.key_length_swa",
    )
    config["global_head_dim"] = _positive_int(
        fields.get("gemma4.attention.key_length"),
        field="gemma4.attention.key_length",
    )

    full_layer_indices = [
        index for index, is_sliding in enumerate(normalized_pattern) if not is_sliding
    ]
    if not full_layer_indices:
        raise RuntimeError("Gemma 4 GGUF contains no global-attention layers")
    has_shared_global_values = all(
        f"blk.{index}.attn_k.weight" in metadata.tensor_names
        and f"blk.{index}.attn_v.weight" not in metadata.tensor_names
        for index in full_layer_indices
    )
    if not has_shared_global_values:
        raise RuntimeError(
            "Gemma 4 global attention tensors do not match the expected shared K=V layout"
        )
    config["attention_k_eq_v"] = True

    config["num_experts"] = _positive_int(
        fields.get("gemma4.expert_count"), field="gemma4.expert_count"
    )
    config["top_k_experts"] = _positive_int(
        fields.get("gemma4.expert_used_count"), field="gemma4.expert_used_count"
    )
    config["moe_intermediate_size"] = _positive_int(
        fields.get("gemma4.expert_feed_forward_length"),
        field="gemma4.expert_feed_forward_length",
    )
    config["enable_moe_block"] = True
    config["hidden_size_per_layer_input"] = _nonnegative_int(
        fields.get("gemma4.embedding_length_per_layer_input"),
        field="gemma4.embedding_length_per_layer_input",
    )
    config["num_kv_shared_layers"] = _nonnegative_int(
        fields.get("gemma4.attention.shared_kv_layers"),
        field="gemma4.attention.shared_kv_layers",
    )
    softcap = fields.get("gemma4.final_logit_softcapping")
    try:
        config["final_logit_softcapping"] = float(softcap)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("GGUF metadata gemma4.final_logit_softcapping must be numeric") from exc


def _repair_gpt_oss_config(
    config: dict[str, Any],
    metadata: _GGUFCompatibilityMetadata,
) -> None:
    if metadata.architecture != "gpt-oss" or config.get("model_type") != "gpt_oss":
        return
    rope_scaling = config.get("rope_scaling")
    if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "yarn":
        # These are GptOssConfig's architecture defaults.  llama.cpp stores
        # factor and original context length but omits the standard YaRN beta
        # values and truncation flag from GGUF metadata.
        rope_scaling.setdefault("beta_fast", 32.0)
        rope_scaling.setdefault("beta_slow", 1.0)
        rope_scaling.setdefault("truncate", False)


def _copy_tensor_for_gguf_load(weights):
    """Copy a dequantized NumPy tensor directly into the requested load dtype."""

    import torch

    tensor = torch.from_numpy(weights.copy())
    target_dtype = _GGUF_LOAD_DTYPE.get()
    return tensor.to(dtype=target_dtype) if target_dtype is not None else tensor


def _build_project_tensor_processors(gguf_utils):
    """Build processors against the pinned Transformers private GGUF API.

    Transformers 5.14.1 has no Gemma 4 processor. Its GPT-OSS processor also
    targets old ``*_projs``/per-expert names, while current llama.cpp emits
    fused ``*_exps`` tensors for ``GptOssForCausalLM``. Keeping these classes
    local to the compatibility context prevents permanent global registration.
    """

    TensorProcessor = gguf_utils.TensorProcessor
    GGUFTensor = gguf_utils.GGUFTensor

    class GptOssTensorProcessor(TensorProcessor):
        _DOWN_BIAS = re.compile(r"(?:model\.)?layers\.(?P<bid>\d+)\.mlp\.experts\.down_proj_bias")
        _GATE_UP = re.compile(r"(?:model\.)?layers\.(?P<bid>\d+)\.mlp\.experts\.gate_up_proj")
        _GATE_UP_BIAS = re.compile(
            r"(?:model\.)?layers\.(?P<bid>\d+)\.mlp\.experts\.gate_up_proj_bias"
        )
        _GGUF_GATE_UP = re.compile(
            r"blk\.(?P<bid>\d+)\.ffn_(?P<part>gate|up)_exps\."
            r"(?P<kind>weight|bias)"
        )

        def perform_fallback_tensor_mapping(
            self,
            gguf_to_hf_name_map: dict[str, str],
            suffix: str,
            qual_name: str,
            hf_name: str,
        ):
            del suffix
            if match := self._DOWN_BIAS.fullmatch(hf_name):
                gguf_to_hf_name_map[f"blk.{match['bid']}.ffn_down_exps.bias"] = qual_name + hf_name
            elif match := self._GATE_UP.fullmatch(hf_name):
                target = qual_name + hf_name
                gguf_to_hf_name_map[f"blk.{match['bid']}.ffn_gate_exps.weight"] = target
                gguf_to_hf_name_map[f"blk.{match['bid']}.ffn_up_exps.weight"] = target
            elif match := self._GATE_UP_BIAS.fullmatch(hf_name):
                target = qual_name + hf_name
                gguf_to_hf_name_map[f"blk.{match['bid']}.ffn_gate_exps.bias"] = target
                gguf_to_hf_name_map[f"blk.{match['bid']}.ffn_up_exps.bias"] = target

        @staticmethod
        def normalize_source_name(name: str) -> str:
            return name.replace(".attn_sinks.weight", ".attn_sinks").replace(
                ".ffn_down_exps.weight", ".ffn_down_exps"
            )

        def process(self, weights, name: str, **kwargs):
            # llama.cpp transposes GPT-OSS expert matrices during conversion.
            if name.endswith(".ffn_down_exps.weight"):
                return GGUFTensor(weights.swapaxes(-1, -2), self.normalize_source_name(name), {})

            match = self._GGUF_GATE_UP.fullmatch(name)
            if match is None:
                return GGUFTensor(weights, self.normalize_source_name(name), {})

            tensor_key_mapping = kwargs.get("tensor_key_mapping") or {}
            parsed_parameters = kwargs.get("parsed_parameters")
            target = tensor_key_mapping.get(name)
            if target is None or not isinstance(parsed_parameters, dict):
                raise RuntimeError(f"GPT-OSS GGUF tensor has no verified target: {name}")

            part = match["part"]
            kind = match["kind"]
            if kind == "weight":
                weights = weights.swapaxes(-1, -2)
            shard = _copy_tensor_for_gguf_load(weights)
            shape = list(shard.shape)
            shape[-1] *= 2
            destination = parsed_parameters["tensors"].get(target)
            if destination is None:
                import torch

                destination = torch.zeros(shape, dtype=shard.dtype)
                parsed_parameters["tensors"][target] = destination
            if tuple(destination.shape) != tuple(shape):
                raise RuntimeError(
                    f"GPT-OSS fused tensor shape mismatch for {target}: "
                    f"expected {tuple(shape)}, got {tuple(destination.shape)}"
                )

            # GptOssExperts consumes gate/up values interleaved at even/odd
            # positions, matching llama.cpp's split during HF-to-GGUF export.
            offset = 0 if part == "gate" else 1
            destination[..., offset::2].copy_(shard)
            tracker = _GGUF_FUSED_PARTS.get()
            if tracker is not None:
                tracker.setdefault((target, kind), set()).add(part)
            return GGUFTensor(weights, None, {})

    class Gemma4TensorProcessor(TensorProcessor):
        _ROUTER_SCALE = re.compile(r"(?:model\.)?layers\.(?P<bid>\d+)\.router\.scale")
        _ROUTER_PER_EXPERT_SCALE = re.compile(
            r"(?:model\.)?layers\.(?P<bid>\d+)\.router\.per_expert_scale"
        )

        def perform_fallback_tensor_mapping(
            self,
            gguf_to_hf_name_map: dict[str, str],
            suffix: str,
            qual_name: str,
            hf_name: str,
        ):
            del suffix
            if match := self._ROUTER_SCALE.fullmatch(hf_name):
                gguf_to_hf_name_map[f"blk.{match['bid']}.ffn_gate_inp.scale"] = qual_name + hf_name
            elif match := self._ROUTER_PER_EXPERT_SCALE.fullmatch(hf_name):
                gguf_to_hf_name_map[f"blk.{match['bid']}.ffn_down_exps.scale"] = qual_name + hf_name

        @staticmethod
        def normalize_source_name(name: str) -> str:
            return (
                name.replace(".layer_output_scale.weight", ".layer_output_scale")
                .replace(".ffn_gate_up_exps.weight", ".ffn_gate_up_exps")
                .replace(".ffn_down_exps.weight", ".ffn_down_exps")
            )

        def process(self, weights, name: str, **kwargs):
            del kwargs
            return GGUFTensor(weights, self.normalize_source_name(name), {})

    return {
        "gpt-oss": GptOssTensorProcessor,
        "gpt_oss": GptOssTensorProcessor,
        "gemma4": Gemma4TensorProcessor,
    }


def _validate_gguf_tensor_mapping(
    metadata: _GGUFCompatibilityMetadata,
    model_to_load: Any,
    gguf_utils: Any,
    processor_class: type,
) -> None:
    """Fail before dequantization if a source or core target cannot be mapped."""

    config = getattr(model_to_load, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else {}
    processor = processor_class(config=config_dict)
    mapping = gguf_utils.get_gguf_hf_weights_map(model_to_load, processor)
    normalize = getattr(processor, "normalize_source_name", lambda name: name)

    allowed_source = {"rope_freqs.weight"} if metadata.architecture == "gemma4" else set()
    unmapped_source = sorted(
        name
        for name in metadata.tensor_names
        if name not in allowed_source and normalize(name) not in mapping
    )

    expected_targets = set(model_to_load.state_dict())
    mapped_targets = {
        mapping[normalize(name)] for name in metadata.tensor_names if normalize(name) in mapping
    }
    declared_targets = set(mapping.values())
    allowed_targets: set[str] = set()
    if getattr(config, "tie_word_embeddings", False):
        allowed_targets.update(
            name
            for name in expected_targets
            if name.endswith(("lm_head.weight", "embed_out.weight", "output.weight"))
        )
    missing_targets = sorted(expected_targets - mapped_targets - allowed_targets)
    invalid_targets = sorted(declared_targets - expected_targets)

    if unmapped_source or missing_targets or invalid_targets:
        details = []
        if unmapped_source:
            details.append(f"unmapped source tensors={unmapped_source[:20]}")
        if missing_targets:
            details.append(f"unmapped model tensors={missing_targets[:20]}")
        if invalid_targets:
            details.append(f"invalid model targets={invalid_targets[:20]}")
        raise RuntimeError(
            "GGUF tensor map is incomplete; refusing to dequantize: " + "; ".join(details)
        )


@contextlib.contextmanager
def transformers_gguf_compatibility() -> Iterator[None]:
    """Apply scoped Transformers 5.14.1 GGUF metadata compatibility repairs.

    The patch is installed only while OBLITERATUS calls a Transformers GGUF
    loader, is serialized with a re-entrant lock, and is restored afterward.
    Other architectures are delegated byte-for-byte to Transformers' original
    loader.  This avoids a permanent process-wide monkeypatch while retaining
    Transformers' own tensor mapping and dequantization implementation.
    """

    global _TRANSFORMERS_GGUF_COMPAT_DEPTH

    try:
        import gguf
        import transformers
        import transformers.modeling_gguf_pytorch_utils as gguf_utils
        import transformers.tokenization_utils_tokenizers as tokenizer_utils
        from transformers import configuration_utils
    except ImportError:
        # Let Transformers emit its normal dependency error at the call site.
        yield
        return

    # These repairs target two confirmed defects in the pinned runtime.  Do
    # not silently carry private compatibility behavior into newer releases.
    if transformers.__version__ != "5.14.1":
        raise RuntimeError(
            "Verified GGUF import requires transformers==5.14.1; the project "
            f"compatibility layer does not support {transformers.__version__}"
        )

    with _TRANSFORMERS_GGUF_COMPAT_LOCK:
        if _TRANSFORMERS_GGUF_COMPAT_DEPTH:
            _TRANSFORMERS_GGUF_COMPAT_DEPTH += 1
            try:
                yield
            finally:
                _TRANSFORMERS_GGUF_COMPAT_DEPTH -= 1
            return

        _TRANSFORMERS_GGUF_COMPAT_DEPTH = 1
        original_reader = gguf.GGUFReader
        original_loader = gguf_utils.load_gguf_checkpoint
        compatible_reader = _compatible_reader_class(original_reader)
        project_processors = _build_project_tensor_processors(gguf_utils)
        original_processors = {
            name: gguf_utils.TENSOR_PROCESSORS.get(name) for name in project_processors
        }

        def compatible_loader(gguf_checkpoint_path, *args, **kwargs):
            metadata = _read_compatibility_metadata(gguf_checkpoint_path, original_reader)
            model_to_load = kwargs.get("model_to_load")
            if model_to_load is None and len(args) >= 2:
                model_to_load = args[1]
            return_tensors = kwargs.get("return_tensors", args[0] if args else False)
            torch_dtype = kwargs.get("torch_dtype", args[2] if len(args) >= 3 else None)
            processor_class = project_processors.get(metadata.architecture or "")
            if return_tensors and model_to_load is not None and processor_class is not None:
                _validate_gguf_tensor_mapping(metadata, model_to_load, gguf_utils, processor_class)

            dtype_token = _GGUF_LOAD_DTYPE.set(torch_dtype)
            fused_token = _GGUF_FUSED_PARTS.set({})
            try:
                result = original_loader(gguf_checkpoint_path, *args, **kwargs)
                fused_parts = _GGUF_FUSED_PARTS.get() or {}
                incomplete = {
                    target: sorted(parts)
                    for target, parts in fused_parts.items()
                    if parts != {"gate", "up"}
                }
                if incomplete:
                    raise RuntimeError(
                        f"GPT-OSS GGUF has incomplete fused expert tensors: {incomplete}"
                    )
            finally:
                _GGUF_FUSED_PARTS.reset(fused_token)
                _GGUF_LOAD_DTYPE.reset(dtype_token)
            if not isinstance(result, dict) or not isinstance(result.get("config"), dict):
                return result
            _repair_gpt_oss_config(result["config"], metadata)
            _repair_gemma4_config(result["config"], metadata)
            return result

        bindings = (
            (gguf, "GGUFReader", original_reader),
            (gguf_utils, "load_gguf_checkpoint", original_loader),
            (
                configuration_utils,
                "load_gguf_checkpoint",
                configuration_utils.load_gguf_checkpoint,
            ),
            (
                tokenizer_utils,
                "load_gguf_checkpoint",
                tokenizer_utils.load_gguf_checkpoint,
            ),
        )
        try:
            gguf.GGUFReader = compatible_reader
            gguf_utils.load_gguf_checkpoint = compatible_loader
            configuration_utils.load_gguf_checkpoint = compatible_loader
            tokenizer_utils.load_gguf_checkpoint = compatible_loader
            gguf_utils.TENSOR_PROCESSORS.update(project_processors)
            yield
        finally:
            for name, original in original_processors.items():
                if original is None:
                    gguf_utils.TENSOR_PROCESSORS.pop(name, None)
                else:
                    gguf_utils.TENSOR_PROCESSORS[name] = original
            for module, name, original in reversed(bindings):
                setattr(module, name, original)
            _TRANSFORMERS_GGUF_COMPAT_DEPTH = 0


def _nonblank(value: str | os.PathLike[str] | None, *, name: str) -> str | None:
    if value is None:
        return None
    normalized = os.fspath(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be blank")
    return normalized


def _validate_gguf_filename(filename: str) -> None:
    path = PurePosixPath(filename.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "gguf_file must be a repository-relative filename without '..' path segments"
        )
    if path.suffix.lower() != ".gguf":
        raise ValueError(f"gguf_file must end in .gguf, got {filename!r}")


def resolve_model_source(
    model_name: str | os.PathLike[str],
    *,
    gguf_file: str | None = None,
    canonical_model_id: str | None = None,
    tokenizer_source: str | os.PathLike[str] | None = None,
) -> ModelSource:
    """Resolve a Hugging Face checkpoint or a GGUF import source.

    A local path ending in ``.gguf`` is detected automatically.  A GGUF file
    stored on the Hub is expressed as ``model_name=<repo id>`` plus the
    repository-relative ``gguf_file``.  GGUF imports intentionally require an
    explicit canonical tokenizer source (either ``canonical_model_id`` or
    ``tokenizer_source``) so tool/chat tokens are never guessed from incomplete
    GGUF metadata.
    """

    requested = _nonblank(model_name, name="model_name")
    assert requested is not None
    explicit_gguf = _nonblank(gguf_file, name="gguf_file")
    canonical = _nonblank(canonical_model_id, name="canonical_model_id")
    explicit_tokenizer = _nonblank(tokenizer_source, name="tokenizer_source")

    requested_path = Path(requested).expanduser()
    looks_like_local_gguf = requested_path.suffix.lower() == ".gguf"

    if looks_like_local_gguf:
        if explicit_gguf is not None:
            explicit_path = Path(explicit_gguf).expanduser()
            same_local_file = explicit_path.name == requested_path.name and (
                explicit_path.parent == Path(".")
                or explicit_path.resolve() == requested_path.resolve()
            )
            if not same_local_file:
                raise ValueError(
                    "model_name is a local GGUF file but gguf_file identifies a different artifact"
                )
        if not requested_path.is_file():
            raise FileNotFoundError(f"Local GGUF file does not exist: {requested_path}")
        resolved_file = requested_path.resolve()
        model_root = str(resolved_file.parent)
        resolved_gguf_file = resolved_file.name
        is_local = True
        source_format: SourceFormat = "gguf"
    elif explicit_gguf is not None:
        _validate_gguf_filename(explicit_gguf)
        model_root = requested
        resolved_gguf_file = explicit_gguf
        is_local = requested_path.is_dir()
        if is_local:
            local_candidate = (requested_path / explicit_gguf).resolve()
            if not local_candidate.is_file():
                raise FileNotFoundError(f"Local GGUF file does not exist: {local_candidate}")
            model_root = str(requested_path.resolve())
        source_format = "gguf"
    else:
        model_root = requested
        resolved_gguf_file = None
        is_local = requested_path.exists()
        source_format = "hf"

    if source_format == "gguf" and explicit_tokenizer is None and canonical is None:
        raise ValueError(
            "GGUF import requires canonical_model_id or tokenizer_source so chat templates "
            "and special-token ids come from the canonical tokenizer"
        )

    resolved_tokenizer = explicit_tokenizer or canonical or model_root
    if explicit_tokenizer is not None:
        tokenizer_path = Path(explicit_tokenizer).expanduser()
        if tokenizer_path.exists():
            resolved_tokenizer = str(tokenizer_path.resolve())

    return ModelSource(
        source_format=source_format,
        requested_model=requested,
        model_root=model_root,
        gguf_file=resolved_gguf_file,
        canonical_model_id=canonical,
        tokenizer_source=resolved_tokenizer,
        is_local=is_local,
    )
