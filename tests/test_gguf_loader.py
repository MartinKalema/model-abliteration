"""Offline tests for GGUF source resolution and dense Transformers import."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

from obliteratus.gguf_backend import resolve_model_source
from obliteratus.models import loader


def _write_gguf_fixture(path: Path) -> Path:
    # Loading is mocked; only source resolution needs a real, local file.
    path.write_bytes(b"GGUF-offline-fixture")
    return path


def test_resolve_local_gguf_splits_parent_and_filename(tmp_path):
    gguf_path = _write_gguf_fixture(tmp_path / "model-Q4_K_M.GGUF")

    source = resolve_model_source(
        gguf_path,
        canonical_model_id="openai/gpt-oss-20b",
    )

    assert source.is_gguf
    assert source.model_root == str(tmp_path.resolve())
    assert source.gguf_file == gguf_path.name
    assert source.source_file == str(gguf_path.resolve())
    assert source.tokenizer_source == "openai/gpt-oss-20b"
    assert source.summary()["in_memory_format"] == "dense"


def test_resolve_local_gguf_accepts_identical_cli_forwarded_filename(tmp_path):
    gguf_path = _write_gguf_fixture(tmp_path / "model.gguf")

    source = resolve_model_source(
        gguf_path,
        gguf_file=str(gguf_path),
        canonical_model_id="openai/gpt-oss-20b",
    )

    assert source.source_file == str(gguf_path.resolve())


def test_resolve_local_gguf_rejects_conflicting_filename(tmp_path):
    gguf_path = _write_gguf_fixture(tmp_path / "model.gguf")

    with pytest.raises(ValueError, match="different artifact"):
        resolve_model_source(
            gguf_path,
            gguf_file="other.gguf",
            canonical_model_id="openai/gpt-oss-20b",
        )


def test_resolve_hub_gguf_keeps_repo_filename_and_tokenizer_separate():
    source = resolve_model_source(
        "quantizer/gpt-oss-20b-GGUF",
        gguf_file="Q4/model-Q4_K_M.gguf",
        canonical_model_id="openai/gpt-oss-20b",
        tokenizer_source="/offline/canonical-tokenizer",
    )

    assert source.is_gguf
    assert not source.is_local
    assert source.model_root == "quantizer/gpt-oss-20b-GGUF"
    assert source.gguf_file == "Q4/model-Q4_K_M.gguf"
    assert source.canonical_model_id == "openai/gpt-oss-20b"
    assert source.tokenizer_source == "/offline/canonical-tokenizer"


def test_gguf_requires_explicit_canonical_tokenizer(tmp_path):
    gguf_path = _write_gguf_fixture(tmp_path / "model.gguf")

    with pytest.raises(ValueError, match="canonical_model_id or tokenizer_source"):
        resolve_model_source(gguf_path)


def test_canonical_gguf_config_selects_gemma_text_and_strips_storage_quantization():
    text_config = SimpleNamespace(model_type="gemma4_text", quantization_config={"x": 1})
    wrapper = SimpleNamespace(model_type="gemma4", text_config=text_config)

    result = loader._prepare_gguf_canonical_config(wrapper)

    assert result is text_config
    assert not hasattr(result, "quantization_config")


def test_missing_local_gguf_fails_before_transformers_loading(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_model_source(
            tmp_path / "missing.gguf",
            canonical_model_id="openai/gpt-oss-20b",
        )


@pytest.mark.parametrize("filename", ["../model.gguf", "/tmp/model.gguf", "model.bin"])
def test_hub_gguf_filename_is_repository_relative(filename):
    with pytest.raises(ValueError, match="gguf_file"):
        resolve_model_source(
            "quantizer/repo",
            gguf_file=filename,
            canonical_model_id="openai/gpt-oss-20b",
        )


class _DenseModel(nn.Module):
    def __init__(self, dtype: torch.dtype):
        super().__init__()
        self.core = nn.Linear(2, 2, bias=False, dtype=dtype)


class _IntegerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.core = nn.Parameter(
            torch.ones(2, 2, dtype=torch.int8),
            requires_grad=False,
        )


class _EmbeddingModel(nn.Module):
    def __init__(self, *, tied: bool):
        super().__init__()
        self.input_embeddings = nn.Embedding(4, 2)
        self.output_embeddings = nn.Linear(2, 4, bias=False)
        if tied:
            self.output_embeddings.weight = self.input_embeddings.weight

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings


class _FakeAutoConfig:
    calls: ClassVar[list[tuple[str, dict]]] = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls.append((model_name, kwargs))
        return SimpleNamespace(
            model_type="gpt_oss",
            architectures=["GptOssForCausalLM"],
            num_hidden_layers=1,
            num_attention_heads=1,
            hidden_size=2,
            intermediate_size=4,
            vocab_size=8,
            tie_word_embeddings=False,
        )


class _FakeAutoModel:
    calls: ClassVar[list[dict]] = []
    loading_info: ClassVar[dict] = {}
    integer_parameters = False

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.calls.append(kwargs)
        if cls.integer_parameters:
            model = _IntegerModel()
        else:
            model = _DenseModel(kwargs["dtype"])
        return model, dict(cls.loading_info)


class _FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"


class _FakeAutoTokenizer:
    calls: ClassVar[list[tuple[str, dict]]] = []

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls.append((model_name, kwargs))
        return _FakeTokenizer()


@pytest.fixture(autouse=True)
def _mock_offline_transformers(monkeypatch):
    _FakeAutoConfig.calls = []
    _FakeAutoModel.calls = []
    _FakeAutoModel.loading_info = {}
    _FakeAutoModel.integer_parameters = False
    _FakeAutoTokenizer.calls = []

    monkeypatch.setattr(loader, "AutoConfig", _FakeAutoConfig)
    monkeypatch.setattr(loader, "AutoTokenizer", _FakeAutoTokenizer)
    monkeypatch.setitem(loader.TASK_MODEL_MAP, "causal_lm", _FakeAutoModel)
    monkeypatch.setattr(loader, "_apply_deferred_shims", lambda: None)
    monkeypatch.setattr(loader, "_available_gpu_memory_gb", lambda: 0.0)
    monkeypatch.setattr(loader, "read_gguf_file_type", lambda _path: "MOSTLY_Q4_K_M")
    monkeypatch.setattr(loader.dev, "empty_cache", lambda: None)


def _load_local_gguf(tmp_path, **kwargs):
    gguf_path = _write_gguf_fixture(tmp_path / "model-Q4_K_M.gguf")
    return loader.load_model(
        str(gguf_path),
        device="cpu",
        dtype="bfloat16",
        canonical_model_id="openai/gpt-oss-20b",
        **kwargs,
    )


def test_load_local_gguf_dequantizes_dense_and_uses_canonical_tokenizer(tmp_path):
    handle = _load_local_gguf(tmp_path)

    config_root, config_kwargs = _FakeAutoConfig.calls[0]
    assert config_root == "openai/gpt-oss-20b"
    assert "gguf_file" not in config_kwargs

    model_kwargs = _FakeAutoModel.calls[0]
    assert model_kwargs["pretrained_model_name_or_path"] == str(tmp_path.resolve())
    assert model_kwargs["gguf_file"] == "model-Q4_K_M.gguf"
    assert model_kwargs["dtype"] is torch.bfloat16
    assert "torch_dtype" not in model_kwargs
    assert model_kwargs["low_cpu_mem_usage"] is True
    assert model_kwargs["output_loading_info"] is True
    assert all(parameter.dtype is torch.bfloat16 for parameter in handle.model.parameters())
    assert handle.source_quantization == "MOSTLY_Q4_K_M"

    tokenizer_name, tokenizer_kwargs = _FakeAutoTokenizer.calls[0]
    assert tokenizer_name == "openai/gpt-oss-20b"
    assert "gguf_file" not in tokenizer_kwargs
    assert handle.tokenizer.pad_token == "<eos>"

    assert handle.source_format == "gguf"
    assert handle.canonical_model_id == "openai/gpt-oss-20b"
    assert handle.in_memory_format == "dense"
    assert handle.in_memory_dtype == "bfloat16"
    assert handle._original_state is None
    assert handle.summary()["source"]["file"] == str((tmp_path / "model-Q4_K_M.gguf").resolve())


def test_load_hub_gguf_uses_optional_repository_filename():
    handle = loader.load_model(
        "quantizer/gemma-4-GGUF",
        gguf_file="Q4/gemma-4-Q4_K_M.gguf",
        canonical_model_id="google/gemma-4-26B-A4B-it",
        device="cpu",
        dtype="float16",
    )

    assert _FakeAutoConfig.calls[0][0] == "google/gemma-4-26B-A4B-it"
    assert "gguf_file" not in _FakeAutoConfig.calls[0][1]
    assert _FakeAutoModel.calls[0]["gguf_file"] == "Q4/gemma-4-Q4_K_M.gguf"
    assert _FakeAutoTokenizer.calls[0][0] == "google/gemma-4-26B-A4B-it"
    assert handle.source_file == "Q4/gemma-4-Q4_K_M.gguf"
    assert handle.in_memory_dtype == "float16"


def test_gguf_snapshot_is_created_only_when_explicitly_forced(tmp_path):
    handle = _load_local_gguf(tmp_path, skip_snapshot=False)

    assert handle._original_state is not None
    assert "core.weight" in handle._original_state


@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
def test_gguf_rejects_bitsandbytes_before_loading(tmp_path, quantization):
    gguf_path = _write_gguf_fixture(tmp_path / "model.gguf")

    with pytest.raises(ValueError, match="BitsAndBytes"):
        loader.load_model(
            str(gguf_path),
            device="cpu",
            dtype="float16",
            quantization=quantization,
            canonical_model_id="openai/gpt-oss-20b",
        )

    assert not _FakeAutoConfig.calls
    assert not _FakeAutoModel.calls


def test_gguf_requires_direct_half_precision_dequantization(tmp_path):
    gguf_path = _write_gguf_fixture(tmp_path / "model.gguf")

    with pytest.raises(ValueError, match=r"float16.*bfloat16"):
        loader.load_model(
            str(gguf_path),
            device="cpu",
            dtype="float32",
            canonical_model_id="openai/gpt-oss-20b",
        )


@pytest.mark.parametrize(
    ("loading_info", "message"),
    [
        ({"missing_keys": ["core.weight"]}, "missing core keys"),
        ({"unexpected_keys": ["model.layers.0.mlp.weight"]}, "unexpected core keys"),
        ({"mismatched_keys": [("core.weight", (2, 2), (3, 3))]}, "mismatched keys"),
        ({"error_msgs": ["synthetic conversion failure"]}, "loader errors"),
    ],
)
def test_gguf_loading_report_fails_closed_on_core_mapping_errors(tmp_path, loading_info, message):
    _FakeAutoModel.loading_info = loading_info

    with pytest.raises(RuntimeError, match=message):
        _load_local_gguf(tmp_path)


def test_gguf_loading_report_allows_only_known_deterministic_buffers(tmp_path):
    _FakeAutoModel.loading_info = {
        "missing_keys": ["model.rotary_emb.inv_freq"],
        "unexpected_keys": ["transformer.position_ids"],
    }

    handle = _load_local_gguf(tmp_path)

    assert handle.model.core.weight.dtype is torch.bfloat16


def test_missing_output_embedding_is_allowed_only_when_storage_is_really_tied():
    config = SimpleNamespace(tie_word_embeddings=True)

    loader._validate_gguf_loading_info(
        _EmbeddingModel(tied=True),
        config,
        {"missing_keys": ["lm_head.weight"]},
    )
    with pytest.raises(RuntimeError, match="missing core keys"):
        loader._validate_gguf_loading_info(
            _EmbeddingModel(tied=False),
            config,
            {"missing_keys": ["lm_head.weight"]},
        )


def test_gguf_rejects_non_dense_integer_parameters(tmp_path):
    _FakeAutoModel.integer_parameters = True

    with pytest.raises(RuntimeError, match="non-editable or non-dense"):
        _load_local_gguf(tmp_path)
