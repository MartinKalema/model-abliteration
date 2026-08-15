"""Fail-closed architecture adapters and projection manifests.

The editor must know every residual reader/writer it intends to change before
the first mutation.  A non-zero edit count is not evidence of architecture
support: hybrid and MoE layers can otherwise look successful while entire
branches remain untouched.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from obliteratus.models.loader import ModelHandle
from obliteratus.strategies.utils import (
    get_attention_modules,
    get_ffn_modules,
    get_layer_module_path,
    get_layer_modules,
    is_registered_architecture,
)

ATTENTION_OUTPUT_NAMES: tuple[str, ...] = (
    "o_proj", "o_b_proj", "out_proj", "dense", "c_proj", "wo",
)
ATTENTION_INPUT_NAMES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj",
    "q_a_proj", "kv_a_proj_with_mqa", "kv_proj",
    "qkv_proj", "Wqkv", "wqkv",
    "in_proj_qkv", "in_proj_qkvz", "in_proj_ba",
    "in_proj_z", "in_proj_b", "in_proj_a", "in_proj",
    "c_attn", "query_key_value", "W_pack",
)
FFN_OUTPUT_NAMES: tuple[str, ...] = (
    "down_proj", "output_linear", "c_proj", "dense_4h_to_h",
    "fc_out", "fc2", "w2",
)
FFN_INPUT_NAMES: tuple[str, ...] = (
    "up_proj", "gate_proj", "gate_up_proj", "input_linear",
    "w1", "v1", "w3", "fc1", "c_fc", "dense_h_to_4h",
)
ROUTER_PATHS: tuple[str, ...] = (
    "gate", "router", "wg", "router.layer", "router.proj", "gate.wg",
)
SHARED_EXPERT_PATHS: tuple[str, ...] = (
    "shared_expert", "shared_experts", "shared_mlp",
)


class ArchitectureCoverageError(RuntimeError):
    """Raised before editing when an architecture cannot be covered exactly."""


@dataclass(frozen=True)
class ProjectionManifestEntry:
    """One unique residual-axis tensor scheduled for projection."""

    qualified_name: str
    aliases: tuple[str, ...]
    layer_indices: tuple[int, ...]
    branch_kind: str
    branch_paths: tuple[str, ...]
    component: str
    role: str
    orientation: str
    shape: tuple[int, ...]
    dtype: str
    storage_identity: str
    residual_axis: int
    expert_axis: int | None
    projection_kind: str
    owner: nn.Module = field(repr=False, compare=False)
    attribute_path: str = field(repr=False, compare=False)
    parameter: torch.Tensor = field(repr=False, compare=False)
    expert_index: int | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.qualified_name,
            "aliases": list(self.aliases),
            "layers": list(self.layer_indices),
            "branch_kind": self.branch_kind,
            "branch_paths": list(self.branch_paths),
            "component": self.component,
            "role": self.role,
            "orientation": self.orientation,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "storage_identity": self.storage_identity,
            "residual_axis": self.residual_axis,
            "expert_axis": self.expert_axis,
            "projection_kind": self.projection_kind,
            "expert_index": self.expert_index,
        }


@dataclass(frozen=True)
class ProjectionManifest:
    architecture: str
    target: str
    layer_path: str
    hidden_size: int
    num_layers: int
    entries: tuple[ProjectionManifestEntry, ...]
    branch_coverage: tuple[dict[str, Any], ...]

    def entries_for_layer(self, layer_index: int) -> tuple[ProjectionManifestEntry, ...]:
        return tuple(entry for entry in self.entries if layer_index in entry.layer_indices)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "target": self.target,
            "layer_path": self.layer_path,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "entry_count": len(self.entries),
            "entries": [entry.to_metadata() for entry in self.entries],
            "branch_coverage": list(self.branch_coverage),
        }


@dataclass
class _RawEntry:
    qualified_name: str
    layer_index: int
    branch_kind: str
    branch_path: str
    component: str
    role: str
    orientation: str
    residual_axis: int
    expert_axis: int | None
    projection_kind: str
    owner: nn.Module
    attribute_path: str
    parameter: torch.Tensor
    expert_index: int | None = None


def _resolve_parent(obj: Any, dotted_path: str) -> tuple[Any, str, Any]:
    parts = dotted_path.split(".")
    parent = obj
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    return parent, leaf, getattr(parent, leaf)


def _maybe_resolve(obj: Any, dotted_path: str) -> Any | None:
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
        return obj
    except AttributeError:
        return None


def _storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    if tensor.device.type == "meta":
        return ("meta", id(tensor), tuple(tensor.shape), str(tensor.dtype))
    try:
        storage = tensor.untyped_storage()
        return (
            str(tensor.device), storage.data_ptr(), tensor.storage_offset(), storage.nbytes(),
        )
    except (AttributeError, RuntimeError):
        return (str(tensor.device), tensor.data_ptr(), tensor.storage_offset(), tensor.numel())


def _storage_identity(tensor: torch.Tensor) -> str:
    return ":".join(str(part) for part in _storage_key(tensor))


def _direct_parameter_axis(
    parameter: torch.Tensor,
    *,
    hidden_size: int,
    orientation: str,
    architecture: str,
    attribute_path: str,
) -> int:
    shape = tuple(parameter.shape)
    if len(shape) < 2:
        raise ArchitectureCoverageError(
            f"{attribute_path} has shape {shape}; a residual projection needs a matrix"
        )

    matching = [idx for idx, size in enumerate(shape) if size == hidden_size]
    if not matching:
        raise ArchitectureCoverageError(
            f"{attribute_path} shape {shape} has no text hidden axis {hidden_size}"
        )
    if len(matching) == 1:
        return matching[0]

    # Fused experts use both current and legacy/transposed layouts. Role, not
    # coincidental square test dimensions, determines the intended axis.
    leaf = attribute_path.rsplit(".", 1)[-1]
    if architecture in {"llama4", "llama4_text"}:
        if leaf == "gate_up_proj":
            return len(shape) - 2
        if leaf == "down_proj":
            return len(shape) - 1
    if architecture == "gpt_oss":
        # GPT-OSS stores fused experts as
        #   gate_up_proj: [experts, hidden, 2 * intermediate]
        #   down_proj:    [experts, intermediate, hidden]
        # The released models use hidden == intermediate, so matching tensor
        # dimensions cannot distinguish the two semantic axes.  Keep this
        # explicit: the generic orientation preference selects axis 1 for the
        # square down projection, which is the intermediate rather than the
        # residual output axis.
        if leaf == "gate_up_proj":
            return len(shape) - 2
        if leaf == "down_proj":
            return len(shape) - 1
    if architecture == "dbrx" and leaf in {"w1", "v1", "w2"}:
        return len(shape) - 1

    preferred = (len(shape) - 1) if orientation == "input" else (len(shape) - 2)
    if preferred in matching:
        return preferred
    raise ArchitectureCoverageError(
        f"Ambiguous hidden axis for {attribute_path} shape {shape}; add an explicit "
        f"{architecture!r} layout rule before editing"
    )


def _add_projection(
    raw: list[_RawEntry],
    *,
    model_prefix: str,
    layer_index: int,
    branch_kind: str,
    branch_path: str,
    branch: nn.Module,
    relative_path: str,
    component: str,
    orientation: str,
    hidden_size: int,
    architecture: str,
    expert_index: int | None = None,
) -> bool:
    obj = _maybe_resolve(branch, relative_path)
    if obj is None:
        return False

    role = "reader" if orientation == "input" else "writer"
    qualified_base = model_prefix
    if branch_path != "@self":
        qualified_base += f".{branch_path}"
    qualified_base += f".{relative_path}"

    if isinstance(obj, nn.Module) and hasattr(obj, "weight"):
        parameter = obj.weight
        if not isinstance(parameter, torch.Tensor):
            return False
        # Hugging Face's GPT-style Conv1D stores weights as [in, out], the
        # transpose of nn.Linear's [out, in].  Treating it as a Linear would
        # either reject GPT-2/Neo or project the intermediate rather than the
        # residual axis on square fixtures.
        transposed_linear = obj.__class__.__name__ == "Conv1D"
        if transposed_linear:
            residual_axis = 0 if orientation == "input" else 1
        else:
            residual_axis = 1 if orientation == "input" else 0
        if parameter.ndim != 2 or parameter.shape[residual_axis] != hidden_size:
            raise ArchitectureCoverageError(
                f"{qualified_base}.weight shape {tuple(parameter.shape)} does not "
                f"match {orientation} residual axis {hidden_size}"
            )
        raw.append(
            _RawEntry(
                qualified_name=f"{qualified_base}.weight",
                layer_index=layer_index,
                branch_kind=branch_kind,
                branch_path=branch_path,
                component=component,
                role=role,
                orientation=orientation,
                residual_axis=residual_axis,
                expert_axis=None,
                projection_kind="module_weight",
                owner=branch,
                attribute_path=relative_path,
                parameter=parameter,
                expert_index=expert_index,
            )
        )
        return True

    if isinstance(obj, (nn.Parameter, torch.Tensor)):
        residual_axis = _direct_parameter_axis(
            obj,
            hidden_size=hidden_size,
            orientation=orientation,
            architecture=architecture,
            attribute_path=relative_path,
        )
        expert_axis = 0 if obj.ndim >= 3 and residual_axis != 0 else None
        raw.append(
            _RawEntry(
                qualified_name=qualified_base,
                layer_index=layer_index,
                branch_kind=branch_kind,
                branch_path=branch_path,
                component=component,
                role=role,
                orientation=orientation,
                residual_axis=residual_axis,
                expert_axis=expert_axis,
                projection_kind="parameter_axis",
                owner=branch,
                attribute_path=relative_path,
                parameter=obj,
                expert_index=expert_index,
            )
        )
        return True
    return False


def _add_dense_branch_entries(
    raw: list[_RawEntry],
    *,
    model_prefix: str,
    layer_index: int,
    branch_kind: str,
    branch_path: str,
    branch: nn.Module,
    output_names: Iterable[str],
    input_names: Iterable[str],
    include_inputs: bool,
    hidden_size: int,
    architecture: str,
    component_prefix: str,
    expert_index: int | None = None,
) -> tuple[int, int]:
    outputs = 0
    inputs = 0
    for name in output_names:
        outputs += int(
            _add_projection(
                raw,
                model_prefix=model_prefix,
                layer_index=layer_index,
                branch_kind=branch_kind,
                branch_path=branch_path,
                branch=branch,
                relative_path=name,
                component=f"{component_prefix}_output",
                orientation="output",
                hidden_size=hidden_size,
                architecture=architecture,
                expert_index=expert_index,
            )
        )
    if include_inputs:
        for name in input_names:
            inputs += int(
                _add_projection(
                    raw,
                    model_prefix=model_prefix,
                    layer_index=layer_index,
                    branch_kind=branch_kind,
                    branch_path=branch_path,
                    branch=branch,
                    relative_path=name,
                    component=f"{component_prefix}_input",
                    orientation="input",
                    hidden_size=hidden_size,
                    architecture=architecture,
                    expert_index=expert_index,
                )
            )
    return outputs, inputs


def _build_attention_branch(
    raw: list[_RawEntry],
    *,
    model_prefix: str,
    layer_index: int,
    branch_path: str,
    branch: nn.Module,
    include_inputs: bool,
    hidden_size: int,
    architecture: str,
) -> dict[str, Any]:
    outputs, inputs = _add_dense_branch_entries(
        raw,
        model_prefix=model_prefix,
        layer_index=layer_index,
        branch_kind="attention",
        branch_path=branch_path,
        branch=branch,
        output_names=ATTENTION_OUTPUT_NAMES,
        input_names=ATTENTION_INPUT_NAMES,
        include_inputs=include_inputs,
        hidden_size=hidden_size,
        architecture=architecture,
        component_prefix="attention",
    )
    if outputs == 0 or (include_inputs and inputs == 0):
        raise ArchitectureCoverageError(
            f"Layer {layer_index} mixer branch {branch_path!r} is incomplete: "
            f"writers={outputs}, readers={inputs}"
        )
    return {
        "layer": layer_index,
        "kind": "attention",
        "path": branch_path,
        "writers": outputs,
        "readers": inputs,
    }


def _build_ffn_branch(
    raw: list[_RawEntry],
    *,
    model_prefix: str,
    layer_index: int,
    branch_path: str,
    branch: nn.Module,
    include_inputs: bool,
    hidden_size: int,
    architecture: str,
) -> dict[str, Any]:
    if architecture == "dbrx":
        _validate_dbrx_ffn_layout(branch, hidden_size=hidden_size)

    direct_outputs, direct_inputs = _add_dense_branch_entries(
        raw,
        model_prefix=model_prefix,
        layer_index=layer_index,
        branch_kind="ffn",
        branch_path=branch_path,
        branch=branch,
        output_names=FFN_OUTPUT_NAMES,
        input_names=FFN_INPUT_NAMES,
        include_inputs=include_inputs,
        hidden_size=hidden_size,
        architecture=architecture,
        component_prefix="ffn",
    )

    expert_outputs = expert_inputs = router_inputs = 0
    shared_outputs = shared_inputs = 0
    experts = getattr(branch, "experts", None)
    if experts is not None:
        if include_inputs:
            for router_path in ROUTER_PATHS:
                router_inputs += int(
                    _add_projection(
                        raw,
                        model_prefix=model_prefix,
                        layer_index=layer_index,
                        branch_kind="ffn",
                        branch_path=branch_path,
                        branch=branch,
                        relative_path=router_path,
                        component="router_input",
                        orientation="input",
                        hidden_size=hidden_size,
                        architecture=architecture,
                    )
                )
            if router_inputs == 0:
                raise ArchitectureCoverageError(
                    f"Layer {layer_index} MoE branch {branch_path!r} has experts but "
                    "no explicitly supported router projection"
                )

        if isinstance(experts, nn.ModuleList):
            for expert_index, expert in enumerate(experts):
                sub_path = f"experts.{expert_index}"
                eo, ei = _add_dense_branch_entries(
                    raw,
                    model_prefix=model_prefix,
                    layer_index=layer_index,
                    branch_kind="ffn",
                    branch_path=branch_path,
                    branch=branch,
                    output_names=(f"{sub_path}.{name}" for name in FFN_OUTPUT_NAMES),
                    input_names=(f"{sub_path}.{name}" for name in FFN_INPUT_NAMES),
                    include_inputs=include_inputs,
                    hidden_size=hidden_size,
                    architecture=architecture,
                    component_prefix="expert",
                    expert_index=expert_index,
                )
                expert_outputs += eo
                expert_inputs += ei
        else:
            expert_paths = ("experts", "experts.mlp")
            for expert_path in expert_paths:
                if _maybe_resolve(branch, expert_path) is None:
                    continue
                eo, ei = _add_dense_branch_entries(
                    raw,
                    model_prefix=model_prefix,
                    layer_index=layer_index,
                    branch_kind="ffn",
                    branch_path=branch_path,
                    branch=branch,
                    output_names=(f"{expert_path}.{name}" for name in FFN_OUTPUT_NAMES),
                    input_names=(f"{expert_path}.{name}" for name in FFN_INPUT_NAMES),
                    include_inputs=include_inputs,
                    hidden_size=hidden_size,
                    architecture=architecture,
                    component_prefix="expert",
                )
                expert_outputs += eo
                expert_inputs += ei

        if expert_outputs == 0 or (include_inputs and expert_inputs == 0):
            raise ArchitectureCoverageError(
                f"Layer {layer_index} MoE branch {branch_path!r} expert coverage is "
                f"incomplete: writers={expert_outputs}, readers={expert_inputs}"
            )

        for shared_path in SHARED_EXPERT_PATHS:
            shared = _maybe_resolve(branch, shared_path)
            if shared is None or not isinstance(shared, nn.Module):
                continue
            so, si = _add_dense_branch_entries(
                raw,
                model_prefix=model_prefix,
                layer_index=layer_index,
                branch_kind="ffn",
                branch_path=branch_path,
                branch=branch,
                output_names=(f"{shared_path}.{name}" for name in FFN_OUTPUT_NAMES),
                input_names=(f"{shared_path}.{name}" for name in FFN_INPUT_NAMES),
                include_inputs=include_inputs,
                hidden_size=hidden_size,
                architecture=architecture,
                component_prefix="shared_expert",
            )
            shared_outputs += so
            shared_inputs += si
            if so == 0 or (include_inputs and si == 0):
                raise ArchitectureCoverageError(
                    f"Layer {layer_index} shared expert {shared_path!r} is incomplete"
                )

        if (
            include_inputs
            and getattr(branch, "shared_expert_gate", None) is not None
            and not _add_projection(
                raw,
                model_prefix=model_prefix,
                layer_index=layer_index,
                branch_kind="ffn",
                branch_path=branch_path,
                branch=branch,
                relative_path="shared_expert_gate",
                component="shared_expert_gate_input",
                orientation="input",
                hidden_size=hidden_size,
                architecture=architecture,
            )
        ):
            raise ArchitectureCoverageError(
                f"Layer {layer_index} shared_expert_gate is not editable"
            )

    writers = direct_outputs + expert_outputs + shared_outputs
    readers = direct_inputs + expert_inputs + shared_inputs + router_inputs
    if writers == 0 or (include_inputs and readers == 0):
        raise ArchitectureCoverageError(
            f"Layer {layer_index} FFN branch {branch_path!r} is incomplete: "
            f"writers={writers}, readers={readers}"
        )
    return {
        "layer": layer_index,
        "kind": "ffn",
        "path": branch_path,
        "writers": writers,
        "readers": readers,
        "router_readers": router_inputs,
        "expert_writers": expert_outputs,
        "expert_readers": expert_inputs,
        "shared_writers": shared_outputs,
        "shared_readers": shared_inputs,
    }


def _validate_dbrx_ffn_layout(branch: nn.Module, *, hidden_size: int) -> None:
    """Accept only DBRX packed experts whose physical residual axis is explicit.

    The legacy Hugging Face layout stores each packed expert tensor as
    ``[experts * intermediate, hidden]`` and routes directly from ``hidden``.
    Some Transformers releases expose a refactored
    ``[experts * hidden, intermediate]`` layout instead.  Its expert and
    residual axes are fused together, which this manifest cannot project
    independently.  Shape coincidences (notably square tiny fixtures) must not
    turn that unsupported layout into a plausible-looking edit.
    """
    router = _maybe_resolve(branch, "router.layer")
    experts = _maybe_resolve(branch, "experts")
    mlp = _maybe_resolve(branch, "experts.mlp")
    if not isinstance(router, nn.Module) or not isinstance(experts, nn.Module):
        raise ArchitectureCoverageError(
            "DBRX FFN is missing its explicitly supported router or packed experts"
        )
    if not isinstance(mlp, nn.Module):
        raise ArchitectureCoverageError(
            "DBRX FFN does not expose the supported experts.mlp packed layout"
        )

    router_weight = getattr(router, "weight", None)
    if (
        not isinstance(router_weight, torch.Tensor)
        or router_weight.ndim != 2
        or router_weight.shape[1] != hidden_size
    ):
        shape = tuple(router_weight.shape) if isinstance(router_weight, torch.Tensor) else None
        raise ArchitectureCoverageError(
            "DBRX router is not residual-aligned: router.layer.weight shape "
            f"{shape} must have input width {hidden_size}"
        )

    expert_hidden = int(getattr(mlp, "hidden_size", 0) or 0)
    expert_intermediate = int(getattr(mlp, "ffn_hidden_size", 0) or 0)
    num_experts = int(
        getattr(experts, "num_experts", 0)
        or getattr(mlp, "moe_num_experts", 0)
        or router_weight.shape[0]
    )
    if (
        expert_hidden != hidden_size
        or expert_intermediate <= 0
        or expert_intermediate == hidden_size
        or num_experts <= 0
    ):
        raise ArchitectureCoverageError(
            "DBRX packed-expert layout is unsupported or ambiguous: expected "
            "experts.mlp tensors shaped [experts * intermediate, hidden] with "
            f"hidden={hidden_size} and a distinct intermediate width"
        )

    expected_shape = (num_experts * expert_intermediate, hidden_size)
    for name in ("w1", "v1", "w2"):
        parameter = getattr(mlp, name, None)
        if not isinstance(parameter, torch.Tensor) or tuple(parameter.shape) != expected_shape:
            shape = tuple(parameter.shape) if isinstance(parameter, torch.Tensor) else None
            raise ArchitectureCoverageError(
                f"DBRX experts.mlp.{name} shape {shape} does not match the supported "
                f"packed layout {expected_shape}"
            )


def _merge_aliases(raw: list[_RawEntry]) -> tuple[ProjectionManifestEntry, ...]:
    grouped: dict[tuple[Any, ...], list[_RawEntry]] = {}
    for entry in raw:
        grouped.setdefault(_storage_key(entry.parameter), []).append(entry)

    merged: list[ProjectionManifestEntry] = []
    for entries in grouped.values():
        first = entries[0]
        axes = {entry.residual_axis for entry in entries}
        if len(axes) != 1:
            names = ", ".join(entry.qualified_name for entry in entries)
            raise ArchitectureCoverageError(
                f"Aliased storage has conflicting residual axes: {names}"
            )
        merged.append(
            ProjectionManifestEntry(
                qualified_name=first.qualified_name,
                aliases=tuple(dict.fromkeys(entry.qualified_name for entry in entries)),
                layer_indices=tuple(sorted({entry.layer_index for entry in entries})),
                branch_kind=first.branch_kind,
                branch_paths=tuple(dict.fromkeys(entry.branch_path for entry in entries)),
                component=first.component,
                role=first.role,
                orientation=first.orientation,
                shape=tuple(first.parameter.shape),
                dtype=str(first.parameter.dtype),
                storage_identity=_storage_identity(first.parameter),
                residual_axis=first.residual_axis,
                expert_axis=first.expert_axis,
                projection_kind=first.projection_kind,
                owner=first.owner,
                attribute_path=first.attribute_path,
                parameter=first.parameter,
                expert_index=first.expert_index,
            )
        )
    return tuple(sorted(merged, key=lambda entry: entry.qualified_name))


def build_projection_manifest(handle: ModelHandle, target: str) -> ProjectionManifest:
    """Resolve and validate every parameter required by a projection target."""
    if target not in {"output", "attention", "ffn", "all"}:
        raise ValueError("target must be output, attention, ffn, or all")
    architecture = str(handle.architecture)
    if not is_registered_architecture(architecture):
        raise ArchitectureCoverageError(
            f"Architecture {architecture!r} has no explicit complete-edit adapter"
        )
    if architecture == "deepseek_v4":
        # Kept here as a second line of defense if callers bypass layer lookup.
        raise ArchitectureCoverageError("DeepSeek V4 mHC activations are unsupported")

    layers = get_layer_modules(handle)
    if len(layers) != int(handle.num_layers):
        raise ArchitectureCoverageError(
            f"Resolved {len(layers)} layers but config declares {handle.num_layers}"
        )
    hidden_size = int(handle.hidden_size)
    if hidden_size <= 0:
        raise ArchitectureCoverageError("Text hidden size is unknown; refusing shape guesses")
    layer_path = get_layer_module_path(handle, layers)
    named_parameters = dict(handle.model.named_parameters(remove_duplicate=False))

    raw: list[_RawEntry] = []
    coverage: list[dict[str, Any]] = []
    include_attention_inputs = target in {"attention", "all"}
    include_ffn_inputs = target in {"ffn", "all"}

    for layer_index, layer in enumerate(layers):
        prefix = f"{layer_path}.{layer_index}" if layer_path else str(layer_index)
        attention_branches = get_attention_modules(layer, architecture)
        ffn_branches = get_ffn_modules(layer, architecture)
        if not attention_branches and not ffn_branches:
            raise ArchitectureCoverageError(
                f"Layer {layer_index} exposes no registered residual mixer or FFN branch"
            )
        for branch_path, branch in attention_branches:
            coverage.append(
                _build_attention_branch(
                    raw,
                    model_prefix=prefix,
                    layer_index=layer_index,
                    branch_path=branch_path,
                    branch=branch,
                    include_inputs=include_attention_inputs,
                    hidden_size=hidden_size,
                    architecture=architecture,
                )
            )
        for branch_path, branch in ffn_branches:
            coverage.append(
                _build_ffn_branch(
                    raw,
                    model_prefix=prefix,
                    layer_index=layer_index,
                    branch_path=branch_path,
                    branch=branch,
                    include_inputs=include_ffn_inputs,
                    hidden_size=hidden_size,
                    architecture=architecture,
                )
            )

    for entry in raw:
        actual = named_parameters.get(entry.qualified_name)
        if actual is None:
            raise ArchitectureCoverageError(
                f"Manifest target {entry.qualified_name!r} is not a named model parameter"
            )
        # Parameter wrappers may recreate Python objects but must retain the
        # exact same storage identity.
        if (
            actual is not entry.parameter
            and _storage_key(actual) != _storage_key(entry.parameter)
        ):
            raise ArchitectureCoverageError(
                f"Manifest target {entry.qualified_name!r} resolves to different storage"
            )

    entries = _merge_aliases(raw)
    if not entries:
        raise ArchitectureCoverageError("Projection manifest contains no editable parameters")
    return ProjectionManifest(
        architecture=architecture,
        target=target,
        layer_path=layer_path,
        hidden_size=hidden_size,
        num_layers=len(layers),
        entries=entries,
        branch_coverage=tuple(coverage),
    )
