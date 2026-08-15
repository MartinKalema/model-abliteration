import json
import os
import struct
import sys
import types
from pathlib import Path

import pytest

from obliteratus.checkpoint_transaction import (
    CheckpointCommitError,
    CheckpointValidationError,
    SourceDestinationCollisionError,
    save_gguf_bundle_transactionally,
    validate_gguf_bundle,
)


def _write_minimal_bundle(directory: Path, marker: str = "new") -> None:
    # A zero-tensor GGUF v3 is enough for lightweight parser/header tests. Pad
    # the header to the default GGUF alignment so real GGUFReader builds can
    # memory-map it without reading beyond EOF.
    header = struct.pack("<4sIQQ", b"GGUF", 3, 0, 0)
    (directory / "model-Q4_K_M.gguf").write_bytes(header.ljust(32, b"\0"))
    (directory / "abliteration_metadata.json").write_text(
        json.dumps({"marker": marker}),
        encoding="utf-8",
    )


def _transaction_debris(parent: Path, output_name: str) -> list[Path]:
    return [
        *parent.glob(f".{output_name}.staging-*"),
        *parent.glob(f".{output_name}.backup-*"),
    ]


def test_successful_bundle_commit_uses_same_parent_and_injected_validator(tmp_path):
    output = tmp_path / "published"
    observed: list[Path] = []

    def serializer(staging: Path) -> None:
        observed.append(staging)
        assert staging.parent == output.parent
        _write_minimal_bundle(staging)

    def validator(staging: Path) -> None:
        assert (staging / "model-Q4_K_M.gguf").read_bytes()[:4] == b"GGUF"
        assert json.loads(
            (staging / "abliteration_metadata.json").read_text(encoding="utf-8")
        ) == {"marker": "new"}

    result = save_gguf_bundle_transactionally(output, serializer, validator=validator)

    assert result == output
    assert (output / "model-Q4_K_M.gguf").is_file()
    assert observed and not observed[0].exists()
    assert _transaction_debris(tmp_path, output.name) == []


def test_serializer_failure_preserves_output_and_cleans_staging(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    def serializer(staging: Path) -> None:
        (staging / "partial.gguf").write_bytes(b"partial")
        raise RuntimeError("conversion failed")

    with pytest.raises(RuntimeError, match="conversion failed"):
        save_gguf_bundle_transactionally(
            output,
            serializer,
            overwrite=True,
            validator=lambda _path: None,
        )

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert _transaction_debris(tmp_path, output.name) == []


def test_validator_failure_preserves_output_and_cleans_staging(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("original", encoding="utf-8")

    def validator(_staging: Path) -> None:
        raise CheckpointValidationError("verification failed")

    with pytest.raises(CheckpointValidationError, match="verification failed"):
        save_gguf_bundle_transactionally(
            output,
            _write_minimal_bundle,
            overwrite=True,
            validator=validator,
        )

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert _transaction_debris(tmp_path, output.name) == []


def test_overwrite_restores_original_when_publish_rename_fails(tmp_path, monkeypatch):
    output = tmp_path / "published"
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
            raise OSError("simulated GGUF publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(transaction.os, "replace", fail_publish_once)

    with pytest.raises(CheckpointCommitError, match="atomically publish GGUF bundle"):
        save_gguf_bundle_transactionally(
            output,
            _write_minimal_bundle,
            overwrite=True,
            validator=lambda _path: None,
        )

    assert sentinel.read_text(encoding="utf-8") == "original"
    assert _transaction_debris(tmp_path, output.name) == []


def test_rejects_source_file_inside_destination_before_serializing(tmp_path):
    output = tmp_path / "published"
    output.mkdir()
    source = output / "source.gguf"
    source.write_bytes(b"source")
    called = False

    def serializer(_staging: Path) -> None:
        nonlocal called
        called = True

    with pytest.raises(SourceDestinationCollisionError, match="inside the output"):
        save_gguf_bundle_transactionally(
            output,
            serializer,
            source=source,
            overwrite=True,
            validator=lambda _path: None,
        )

    assert called is False
    assert source.read_bytes() == b"source"


def test_validator_rejects_missing_and_multiple_gguf_files(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CheckpointValidationError, match="missing a .gguf"):
        validate_gguf_bundle(bundle)

    header = struct.pack("<4sI", b"GGUF", 3)
    (bundle / "one.gguf").write_bytes(header)
    (bundle / "two.GGUF").write_bytes(header)
    with pytest.raises(CheckpointValidationError, match="exactly one.*found 2"):
        validate_gguf_bundle(bundle)


@pytest.mark.parametrize(
    ("header", "message"),
    [
        (b"NOPE" + struct.pack("<I", 3), "invalid magic"),
        (b"GGUF" + struct.pack("<I", 99), "unsupported version 99"),
    ],
)
def test_validator_rejects_invalid_magic_and_version(tmp_path, header, message):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.gguf").write_bytes(header)
    (bundle / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(CheckpointValidationError, match=message):
        validate_gguf_bundle(bundle)


def test_validator_requires_json_metadata(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.gguf").write_bytes(struct.pack("<4sI", b"GGUF", 3))

    with pytest.raises(CheckpointValidationError, match="JSON metadata"):
        validate_gguf_bundle(bundle)


def test_reader_is_used_when_available(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    calls: list[tuple[str, str]] = []

    class FakeReader:
        def __init__(self, path: str, mode: str):
            calls.append((path, mode))

    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFReader = FakeReader
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)

    validate_gguf_bundle(bundle)

    assert calls == [(str(bundle / "model-Q4_K_M.gguf"), "r")]


def test_strict_validation_fails_clearly_without_gguf_dependency(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    _write_minimal_bundle(bundle)
    monkeypatch.setitem(sys.modules, "gguf", None)

    # Header-only validation remains useful in minimal installations.
    validate_gguf_bundle(bundle)

    with pytest.raises(CheckpointValidationError, match="Strict.*optional 'gguf' package"):
        validate_gguf_bundle(bundle, strict=True)
