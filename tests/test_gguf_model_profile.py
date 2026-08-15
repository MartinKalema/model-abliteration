from __future__ import annotations

import builtins

import numpy as np
import pytest

from obliteratus.model_profile import profile_model

gguf = pytest.importorskip("gguf")


def _write_tiny_gguf(path):
    writer = gguf.GGUFWriter(path, "llama")
    writer.add_name("tiny-profile-fixture")
    writer.add_block_count(2)
    writer.add_embedding_length(8)
    writer.add_feed_forward_length(16)
    writer.add_vocab_size(32)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_tensor("token_embd.weight", np.zeros((3, 4), dtype=np.float32))
    writer.add_tensor("blk.0.attn.weight", np.zeros((2, 5), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_profile_local_gguf_counts_shapes_and_reads_architecture_metadata(tmp_path):
    model = tmp_path / "tiny.GGUF"
    _write_tiny_gguf(model)

    profile = profile_model(str(model))

    assert profile.source == "local_gguf"
    assert profile.total_params == 22
    assert profile.model_type == "llama"
    assert profile.num_layers == 2
    assert profile.hidden_size == 8
    assert profile.intermediate_size == 16
    assert profile.vocab_size == 32
    assert profile.dtype == "MOSTLY_F16"


def test_profile_local_gguf_preserves_explicit_dtype(tmp_path):
    model = tmp_path / "tiny.gguf"
    _write_tiny_gguf(model)

    assert profile_model(str(model), dtype="float16").dtype == "float16"


def test_profile_local_gguf_fails_clearly_when_dependency_is_missing(tmp_path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    real_import = builtins.__import__

    def import_without_gguf(name, *args, **kwargs):
        if name == "gguf":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gguf)

    with pytest.raises(RuntimeError, match=r"requires gguf==0\.19\.0"):
        profile_model(str(model))


def test_profile_local_gguf_fails_clearly_for_invalid_file(tmp_path):
    model = tmp_path / "broken.gguf"
    model.write_bytes(b"not a GGUF")

    with pytest.raises(RuntimeError, match="invalid or unsupported GGUF"):
        profile_model(str(model))


def test_profile_local_gguf_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        profile_model(str(tmp_path / "missing.gguf"))
