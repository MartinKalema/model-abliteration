"""Offload cleanup must never delete a caller-owned directory."""

from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from obliteratus.abliterate import AbliterationPipeline
from obliteratus.models.loader import ModelHandle


def _handle(path, *, owned: bool) -> ModelHandle:
    return ModelHandle(
        model=nn.Linear(2, 2),
        tokenizer=object(),
        config=SimpleNamespace(model_type="test"),
        model_name="test",
        task="causal_lm",
        _offload_dir=str(path),
        _owns_offload_dir=owned,
    )


def test_model_handle_preserves_caller_owned_offload_directory(tmp_path):
    offload = tmp_path / "caller-offload"
    offload.mkdir()
    sentinel = offload / "keep.txt"
    sentinel.write_text("owned by caller", encoding="utf-8")

    handle = _handle(offload, owned=False)
    handle.cleanup()

    assert sentinel.read_text(encoding="utf-8") == "owned by caller"
    assert handle._offload_dir is None


def test_model_handle_removes_only_auto_created_offload_directory(tmp_path):
    offload = tmp_path / "automatic-offload"
    offload.mkdir()
    (offload / "weights.dat").write_bytes(b"weights")

    handle = _handle(offload, owned=True)
    handle.cleanup()

    assert not offload.exists()
    assert handle._offload_dir is None


def test_pipeline_cleanup_preserves_caller_owned_offload_directory(tmp_path):
    offload = tmp_path / "caller-offload"
    offload.mkdir()
    sentinel = offload / "keep.txt"
    sentinel.write_text("owned by caller", encoding="utf-8")
    pipeline = object.__new__(AbliterationPipeline)
    pipeline.handle = _handle(offload, owned=False)
    pipeline.log = lambda _message: None

    pipeline._cleanup_offload_dir()

    assert sentinel.exists()
