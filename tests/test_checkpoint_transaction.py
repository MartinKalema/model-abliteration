import json
import os
from pathlib import Path

import pytest
import torch

from obliteratus.checkpoint_transaction import (
    CheckpointCommitError,
    CheckpointDestinationExistsError,
    CheckpointValidationError,
    SourceDestinationCollisionError,
    save_hf_checkpoint_transactionally,
    validate_finite_state_dict,
    validate_hf_checkpoint,
)


def _write_minimal_checkpoint(directory: Path, marker: str = "new") -> None:
    (directory / "config.json").write_text(
        json.dumps({"model_type": "test", "marker": marker}), encoding="utf-8"
    )
    # The structural validator intentionally does not deserialize pickle files.
    (directory / "pytorch_model.bin").write_bytes(b"model weights")


def _transaction_debris(parent: Path, output_name: str) -> list[Path]:
    return [
        *parent.glob(f".{output_name}.staging-*"),
        *parent.glob(f".{output_name}.backup-*"),
    ]


def test_rejects_equal_local_source_and_output(tmp_path):
    source = tmp_path / "model"
    source.mkdir()

    called = False

    def serializer(_staging: Path) -> None:
        nonlocal called
        called = True

    with pytest.raises(SourceDestinationCollisionError, match="same path"):
        save_hf_checkpoint_transactionally(source, serializer, source=source)

    assert called is False


def test_rejects_source_file_inside_destination_before_serializing(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    source = output / "source-checkpoint.bin"
    source.write_bytes(b"source")
    called = False

    def serializer(_staging: Path) -> None:
        nonlocal called
        called = True

    with pytest.raises(SourceDestinationCollisionError, match="inside the output"):
        save_hf_checkpoint_transactionally(
            output,
            serializer,
            source=source,
            overwrite=True,
            validator=lambda _path: None,
        )

    assert called is False
    assert source.read_bytes() == b"source"


def test_rejects_symlink_equivalent_source_and_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "source-alias"
    alias.symlink_to(source, target_is_directory=True)

    with pytest.raises(SourceDestinationCollisionError, match="same path"):
        save_hf_checkpoint_transactionally(alias, lambda _path: None, source=source)


def test_serializer_failure_preserves_existing_checkpoint_and_cleans_stage(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    def serializer(staging: Path) -> None:
        (staging / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("serializer failed")

    with pytest.raises(RuntimeError, match="serializer failed"):
        save_hf_checkpoint_transactionally(output, serializer, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert _transaction_debris(tmp_path, output.name) == []


def test_validator_failure_preserves_existing_checkpoint_and_cleans_stage(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    def validator(_staging: Path) -> None:
        raise CheckpointValidationError("validator failed")

    with pytest.raises(CheckpointValidationError, match="validator failed"):
        save_hf_checkpoint_transactionally(
            output,
            _write_minimal_checkpoint,
            overwrite=True,
            validator=validator,
        )

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert _transaction_debris(tmp_path, output.name) == []


def test_existing_nonempty_destination_is_refused_by_default(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    called = False

    def serializer(_staging: Path) -> None:
        nonlocal called
        called = True

    with pytest.raises(CheckpointDestinationExistsError, match="non-empty"):
        save_hf_checkpoint_transactionally(output, serializer)

    assert called is False
    assert sentinel.read_text(encoding="utf-8") == "original"


def test_successful_commit_is_same_parent_and_leaves_no_debris(tmp_path):
    output = tmp_path / "output"
    observed_staging: list[Path] = []

    def serializer(staging: Path) -> None:
        observed_staging.append(staging)
        assert staging.parent == output.parent
        _write_minimal_checkpoint(staging)

    result = save_hf_checkpoint_transactionally(output, serializer)

    assert result == output
    assert json.loads((output / "config.json").read_text(encoding="utf-8"))["marker"] == "new"
    assert observed_staging and not observed_staging[0].exists()
    assert _transaction_debris(tmp_path, output.name) == []


def test_overwrite_restores_backup_when_commit_rename_fails(tmp_path, monkeypatch):
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    import obliteratus.checkpoint_transaction as transaction

    real_replace = os.replace
    replace_calls = 0

    def fail_publish_once(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(transaction.os, "replace", fail_publish_once)

    with pytest.raises(CheckpointCommitError, match="atomically publish"):
        save_hf_checkpoint_transactionally(output, _write_minimal_checkpoint, overwrite=True)

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert _transaction_debris(tmp_path, output.name) == []


def test_validator_checks_index_references_and_safetensors_readability(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(CheckpointValidationError, match="missing weight file"):
        validate_hf_checkpoint(checkpoint)

    (checkpoint / "missing.safetensors").write_bytes(b"not safetensors")
    with pytest.raises(CheckpointValidationError, match="not readable"):
        validate_hf_checkpoint(checkpoint)


def test_state_dict_finite_scan_checks_all_chunks_and_noncontiguous_views():
    finite = torch.arange(24, dtype=torch.float32).reshape(4, 6).T
    validate_finite_state_dict(
        {"finite.weight": finite, "integer.buffer": torch.tensor([1, 2])},
        chunk_elements=3,
    )

    broken = finite.clone()
    broken[5, 3] = float("nan")
    with pytest.raises(CheckpointValidationError, match="broken.weight.*NaN/Inf"):
        validate_finite_state_dict({"broken.weight": broken}, chunk_elements=3)


def test_state_dict_finite_scan_rejects_invalid_chunk_size():
    with pytest.raises(ValueError, match="positive"):
        validate_finite_state_dict({"weight": torch.ones(1)}, chunk_elements=0)
