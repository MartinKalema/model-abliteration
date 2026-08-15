"""Transactional saving and lightweight validation for model artifacts.

Serializers write into a temporary directory next to the requested output
directory.  The completed artifact is validated there before a same-filesystem
rename makes it visible at the final path.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch


class CheckpointTransactionError(RuntimeError):
    """Base error for transactional checkpoint saves."""


class SourceDestinationCollisionError(CheckpointTransactionError):
    """Raised when a local source checkpoint and output resolve to one path."""


class CheckpointValidationError(CheckpointTransactionError):
    """Raised when a staged directory is not a minimally valid checkpoint."""


class CheckpointDestinationExistsError(CheckpointTransactionError):
    """Raised rather than replacing an existing non-empty destination."""


class CheckpointCommitError(CheckpointTransactionError):
    """Raised when the validated checkpoint cannot be committed."""


Serializer = Callable[[Path], None]
Validator = Callable[[Path], None]

def _tensor_is_finite_in_chunks(
    tensor: torch.Tensor,
    *,
    chunk_elements: int,
) -> bool:
    """Check one dense tensor without allocating a tensor-sized boolean mask."""

    if tensor.numel() == 0 or not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    if tensor.device.type == "meta":
        return False
    if tensor.layout != torch.strided:
        raise CheckpointValidationError(
            f"Unsupported non-strided tensor layout during finite scan: {tensor.layout}"
        )
    detached = tensor.detach()
    if detached.ndim <= 1 or detached.is_contiguous():
        flat = detached if detached.ndim == 1 else detached.reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            if not bool(torch.isfinite(flat[start : start + chunk_elements]).all().item()):
                return False
        return True

    # Avoid reshape() copying an entire non-contiguous tensor. Each recursive
    # slice is smaller and eventually reaches a one-dimensional strided view.
    return all(
        _tensor_is_finite_in_chunks(part, chunk_elements=chunk_elements)
        for part in detached.unbind(0)
    )


def validate_finite_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    chunk_elements: int = 1_048_576,
) -> None:
    """Reject NaN/Inf/meta tensors before checkpoint serialization.

    Generation probes exercise only a small part of a model (and may not route
    through every MoE expert). This scan covers every floating/complex tensor
    that is about to be saved while bounding temporary memory per check.
    """

    if not isinstance(chunk_elements, int) or isinstance(chunk_elements, bool):
        raise TypeError("chunk_elements must be an integer")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise CheckpointValidationError("state_dict must map string names to tensors")
        try:
            finite = _tensor_is_finite_in_chunks(
                tensor,
                chunk_elements=chunk_elements,
            )
        except CheckpointValidationError:
            raise
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CheckpointValidationError(
                f"Could not scan state tensor {name!r} for finite values: {exc}"
            ) from exc
        if not finite:
            raise CheckpointValidationError(
                f"State tensor {name!r} contains NaN/Inf data or is still on meta device"
            )


def _path_exists(path: Path) -> bool:
    """Like ``lexists`` but with a ``Path`` interface."""

    return path.exists() or path.is_symlink()


def _is_nonempty_destination(path: Path) -> bool:
    # Files and symlinks are user-owned objects even when they point at an empty
    # directory, so replacing either requires explicit overwrite permission.
    if path.is_symlink() or not path.is_dir():
        return True
    return next(path.iterdir(), None) is not None


def _resolved_inside(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise CheckpointValidationError(
            f"Checkpoint file escapes the staging directory: {candidate}"
        ) from exc
    return resolved_candidate


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"Could not read {description} at {path.name}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise CheckpointValidationError(f"{description} must contain a JSON object")
    return value


def _validate_safetensors(paths: list[Path]) -> None:
    """Open safetensors files one at a time without materializing their tensors."""

    try:
        from safetensors import safe_open
    except ImportError:  # pragma: no cover - safetensors is a project dependency
        return

    for path in sorted(paths):
        try:
            with safe_open(str(path), framework="pt", device="cpu") as reader:
                tensor_names = list(reader.keys())
                if not tensor_names:
                    raise ValueError("file has no tensors")
                # Accessing each slice validates tensor metadata and byte ranges
                # while avoiding a second model-sized allocation.
                for name in tensor_names:
                    reader.get_slice(name).get_shape()
        except Exception as exc:
            raise CheckpointValidationError(
                f"Safetensors file is not readable: {path.name}: {exc}"
            ) from exc


def validate_hf_checkpoint(checkpoint_dir: str | os.PathLike[str]) -> None:
    """Validate the minimum structure needed to reload an HF checkpoint.

    Validation intentionally stays lightweight: it checks ``config.json``, a
    recognized model-weight file or index, every index reference, and the
    headers/ranges of safetensors files.  PyTorch pickle weights are only checked
    for existence and a non-zero size; loading them here would be both expensive
    and unsafe for an otherwise untrusted artifact.
    """

    root = Path(checkpoint_dir)
    if not root.is_dir() or root.is_symlink():
        raise CheckpointValidationError("Staged checkpoint must be a real directory")

    config_path = root / "config.json"
    if not config_path.is_file():
        raise CheckpointValidationError("Staged checkpoint is missing config.json")
    _resolved_inside(root, config_path)
    _read_json_object(config_path, "config.json")

    index_paths = [
        path
        for path in (
            root / "model.safetensors.index.json",
            root / "pytorch_model.bin.index.json",
        )
        if path.is_file()
    ]
    referenced_weights: set[Path] = set()

    for index_path in index_paths:
        _resolved_inside(root, index_path)
        index = _read_json_object(index_path, index_path.name)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointValidationError(
                f"{index_path.name} must contain a non-empty weight_map object"
            )
        for tensor_name, relative_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(relative_name, str):
                raise CheckpointValidationError(
                    f"{index_path.name} weight_map entries must map strings to strings"
                )
            relative_path = Path(relative_name)
            if relative_path.is_absolute():
                raise CheckpointValidationError(
                    f"{index_path.name} contains an absolute weight path"
                )
            weight_path = _resolved_inside(root, root / relative_path)
            if not weight_path.is_file():
                raise CheckpointValidationError(
                    f"{index_path.name} references missing weight file {relative_name}"
                )
            referenced_weights.add(weight_path)

    if index_paths:
        weight_paths = sorted(referenced_weights)
    else:
        weight_paths = sorted(
            {
                *root.glob("model.safetensors"),
                *root.glob("model-*-of-*.safetensors"),
                *root.glob("pytorch_model.bin"),
                *root.glob("pytorch_model-*-of-*.bin"),
            }
        )

    if not weight_paths:
        raise CheckpointValidationError(
            "Staged checkpoint has no model weights or recognized weight index"
        )

    for weight_path in weight_paths:
        _resolved_inside(root, weight_path)
        if not weight_path.is_file():
            raise CheckpointValidationError(f"Missing weight file: {weight_path.name}")
        try:
            size = weight_path.stat().st_size
        except OSError as exc:
            raise CheckpointValidationError(
                f"Could not inspect weight file {weight_path.name}: {exc}"
            ) from exc
        if size == 0:
            raise CheckpointValidationError(f"Weight file is empty: {weight_path.name}")

    _validate_safetensors([path for path in weight_paths if path.suffix == ".safetensors"])


def _reserve_backup_path(parent: Path, destination_name: str) -> Path:
    reserved = Path(tempfile.mkdtemp(prefix=f".{destination_name}.backup-", dir=str(parent)))
    reserved.rmdir()
    return reserved


def _remove_owned_tree(path: Path) -> None:
    if not _path_exists(path):
        return
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)


def _assert_distinct_local_source(source: str | os.PathLike[str] | None, destination: Path) -> None:
    if source is None:
        return
    source_path = Path(source).expanduser()
    if not _path_exists(source_path):
        # A non-existent string may be a Hugging Face repository identifier.
        return

    try:
        same = os.path.samefile(source_path, destination) if _path_exists(destination) else False
    except OSError:
        same = False
    resolved_source = source_path.resolve()
    resolved_destination = destination.resolve()
    if same or resolved_source == resolved_destination:
        raise SourceDestinationCollisionError(
            "Local source checkpoint and output directory resolve to the same path"
        )

    # A local source may be a file inside an existing output directory.
    # Replacing that directory would delete the source before serialization.
    try:
        resolved_source.relative_to(resolved_destination)
    except ValueError:
        pass
    else:
        raise SourceDestinationCollisionError(
            "Local source artifact is inside the output directory and would be replaced"
        )


def _save_directory_transactionally(
    output_dir: str | os.PathLike[str],
    serializer: Serializer,
    *,
    source: str | os.PathLike[str] | None,
    overwrite: bool,
    validator: Validator,
    artifact_name: str,
) -> Path:
    """Shared same-parent staging and rollback implementation."""

    destination = Path(output_dir).expanduser()
    _assert_distinct_local_source(source, destination)

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(destination) and _is_nonempty_destination(destination) and not overwrite:
        raise CheckpointDestinationExistsError(
            f"Refusing to replace non-empty {artifact_name} directory: {destination}"
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=str(parent)))
    backup: Path | None = None
    try:
        serializer(staging)
        validator(staging)

        # Re-check immediately before the commit in case another process created
        # the destination while serialization was running.
        if _path_exists(destination):
            if _is_nonempty_destination(destination) and not overwrite:
                raise CheckpointDestinationExistsError(
                    f"Refusing to replace non-empty {artifact_name} directory: {destination}"
                )
            backup = _reserve_backup_path(parent, destination.name)
            os.replace(destination, backup)

        try:
            os.replace(staging, destination)
        except OSError as commit_exc:
            if backup is not None and _path_exists(backup):
                try:
                    if _path_exists(destination):
                        raise CheckpointCommitError(
                            "Commit failed and another object appeared at the output path; "
                            f"the original {artifact_name} remains at {backup}"
                        ) from commit_exc
                    os.replace(backup, destination)
                    backup = None
                except CheckpointCommitError:
                    raise
                except OSError as restore_exc:
                    raise CheckpointCommitError(
                        f"Commit failed and the original {artifact_name} could not be restored; "
                        f"its backup remains at {backup}: {restore_exc}"
                    ) from commit_exc
            raise CheckpointCommitError(
                f"Could not atomically publish {artifact_name} at {destination}: {commit_exc}"
            ) from commit_exc

        # The rename succeeded, so staging no longer exists.
        if backup is not None:
            try:
                _remove_owned_tree(backup)
            except OSError as exc:  # Publishing succeeded; retain a recoverable backup.
                warnings.warn(
                    f"{artifact_name.capitalize()} was saved, but its previous backup remains "
                    f"at {backup}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return destination
    finally:
        if _path_exists(staging):
            _remove_owned_tree(staging)


def save_hf_checkpoint_transactionally(
    output_dir: str | os.PathLike[str],
    serializer: Serializer,
    *,
    source: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
    validator: Validator = validate_hf_checkpoint,
) -> Path:
    """Serialize, validate, and atomically publish a Hugging Face checkpoint.

    ``serializer`` receives an empty temporary directory located in the same
    parent as ``output_dir``.  Existing non-empty outputs are never replaced
    unless ``overwrite=True``.  An overwritten output is first renamed to a
    private backup and is restored if publishing the staged checkpoint fails.
    """

    return _save_directory_transactionally(
        output_dir,
        serializer,
        source=source,
        overwrite=overwrite,
        validator=validator,
        artifact_name="checkpoint",
    )
