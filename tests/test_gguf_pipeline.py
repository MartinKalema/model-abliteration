"""Focused pipeline integration tests for GGUF import and publication."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from torch import nn

import obliteratus.abliterate as abliterate_module
import obliteratus.checkpoint_transaction as transaction_module
import obliteratus.gguf_export as export_module
from obliteratus.abliterate import AbliterationPipeline
from obliteratus.gguf_export import (
    GGUFExportResult,
    LlamaCppToolchain,
)
from obliteratus.informed_pipeline import InformedAbliterationPipeline


def _prompts(prefix: str) -> list[str]:
    return [f"{prefix} {index}" for index in range(4)]


def _pipeline(tmp_path: Path, **kwargs) -> AbliterationPipeline:
    return AbliterationPipeline(
        model_name="/models/source/model-Q4_K_M.gguf",
        output_dir=str(tmp_path / "published"),
        harmful_prompts=_prompts("harmful"),
        harmless_prompts=_prompts("harmless"),
        damage_gate_enabled=False,
        **kwargs,
    )


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 2.0]]))
        self.save_calls: list[dict[str, object]] = []

    def save_pretrained(self, directory: Path, **kwargs) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.save_calls.append(dict(kwargs))
        (directory / "config.json").write_text(
            json.dumps({"model_type": "test"}), encoding="utf-8"
        )
        (directory / "pytorch_model.bin").write_bytes(b"dense-weights")


class _FakeTokenizer:
    chat_template = "{{ messages[0]['content'] }}"
    special_tokens_map: ClassVar[dict[str, str]] = {"eos_token": "<eos>"}

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tokenizer_config.json").write_text(
            json.dumps({"chat_template": self.chat_template}), encoding="utf-8"
        )

    def apply_chat_template(self, _messages, **_kwargs) -> list[int]:
        return [7, 8, 9]


def _handle(*, source_format: str = "hf", model: nn.Module | None = None):
    return SimpleNamespace(
        model=model or _FakeModel(),
        tokenizer=_FakeTokenizer(),
        config=SimpleNamespace(model_type="test", quantization_config=None),
        source_format=source_format,
        source_model="quantizer/repository",
        source_file="/models/source/model-Q4_K_M.gguf",
        canonical_model_id="openai/gpt-oss-20b",
        tokenizer_source="openai/gpt-oss-20b",
        in_memory_dtype="float16",
        _offload_dir=None,
        _owns_offload_dir=False,
    )


def test_constructor_normalizes_and_validates_gguf_controls(tmp_path):
    pipeline = _pipeline(
        tmp_path,
        output_format="GGUF",
        gguf_quant=" q5_k_m ",
        gguf_file="weights.gguf",
        base_model_id="openai/gpt-oss-20b",
        tokenizer_path="/models/tokenizer",
        llama_cpp_dir="/tools/llama.cpp",
        llama_cpp_python="/python",
        gguf_imatrix="importance.dat",
        keep_dense_intermediate=True,
        post_quant_verify=False,
    )

    assert pipeline.output_format == "gguf"
    assert pipeline.gguf_quant == "Q5_K_M"
    assert pipeline.gguf_file == "weights.gguf"
    assert pipeline.base_model_id == "openai/gpt-oss-20b"
    assert pipeline.tokenizer_path == "/models/tokenizer"
    assert pipeline.llama_cpp_dir == "/tools/llama.cpp"
    assert pipeline.llama_cpp_python == "/python"
    assert pipeline.gguf_imatrix == "importance.dat"
    assert pipeline.keep_dense_intermediate is True
    assert pipeline.post_quant_verify is False

    with pytest.raises(ValueError, match="output_format"):
        _pipeline(tmp_path, output_format="safetensors")
    with pytest.raises(ValueError, match="gguf_quant"):
        _pipeline(tmp_path, output_format="gguf", gguf_quant="F16")
    with pytest.raises(ValueError, match="gguf_quant"):
        _pipeline(tmp_path, output_format="both", gguf_quant="Q4-K-M")


def test_informed_pipeline_forwards_gguf_controls_to_base(tmp_path):
    pipeline = InformedAbliterationPipeline(
        model_name="repository/model",
        output_dir=str(tmp_path / "informed"),
        harmful_prompts=_prompts("harmful"),
        harmless_prompts=_prompts("harmless"),
        damage_gate_enabled=False,
        gguf_file="model.gguf",
        base_model_id="google/gemma-4-26B-A4B-it",
        tokenizer_path="/models/gemma-tokenizer",
        output_format="both",
        gguf_quant="q4_k_m",
        llama_cpp_dir="/tools/llama.cpp",
        llama_cpp_python="python3",
        gguf_imatrix="importance.dat",
        keep_dense_intermediate=True,
        post_quant_verify=False,
    )

    assert pipeline.gguf_file == "model.gguf"
    assert pipeline.base_model_id == "google/gemma-4-26B-A4B-it"
    assert pipeline.tokenizer_path == "/models/gemma-tokenizer"
    assert pipeline.output_format == "both"
    assert pipeline.gguf_quant == "Q4_K_M"
    assert pipeline.llama_cpp_dir == "/tools/llama.cpp"
    assert pipeline.llama_cpp_python == "python3"
    assert pipeline.gguf_imatrix == "importance.dat"
    assert pipeline.keep_dense_intermediate is True
    assert pipeline.post_quant_verify is False


def test_summon_forwards_gguf_source_and_records_provenance(tmp_path, monkeypatch):
    pipeline = _pipeline(
        tmp_path,
        gguf_file="Q4/model.gguf",
        base_model_id="openai/gpt-oss-20b",
        tokenizer_path="/tokenizers/gpt-oss",
    )
    loaded_handle = _handle(source_format="gguf")
    loaded_handle.summary = lambda: {
        "model_name": "test",
        "architecture": "gpt_oss",
        "task": "causal_lm",
        "num_layers": 2,
        "num_heads": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "total_params": 2,
        "source_format": "gguf",
        "in_memory_dtype": "float16",
    }
    captured: dict[str, object] = {}

    def fake_load_model(**kwargs):
        captured.update(kwargs)
        return loaded_handle

    monkeypatch.setattr(abliterate_module, "load_model", fake_load_model)
    monkeypatch.setattr(abliterate_module, "required_evaluation_settings", lambda _p: ())
    pipeline._get_reasoning_protocol = lambda: SimpleNamespace(
        control_kind="none", trace_format="plain", confidence="high"
    )
    pipeline._prepare_projection_manifests = lambda: None

    pipeline._summon()

    assert captured == {
        "model_name": "/models/source/model-Q4_K_M.gguf",
        "task": "causal_lm",
        "device": "auto",
        "dtype": "float16",
        "trust_remote_code": False,
        "quantization": None,
        "gguf_file": "Q4/model.gguf",
        "canonical_model_id": "openai/gpt-oss-20b",
        "tokenizer_source": "/tokenizers/gpt-oss",
        "skip_snapshot": None,
    }
    assert pipeline._input_source_metadata == {
        "format": "gguf",
        "model": "quantizer/repository",
        "file": "/models/source/model-Q4_K_M.gguf",
        "canonical_model_id": "openai/gpt-oss-20b",
        "tokenizer_source": "openai/gpt-oss-20b",
        "in_memory_dtype": "float16",
    }


def test_storage_gate_accepts_dequantized_gguf_and_rejects_packed_parameters(tmp_path):
    pipeline = _pipeline(tmp_path)
    pipeline.handle = _handle(source_format="gguf")
    pipeline._assert_supported_storage_format()

    class PackedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.packed = nn.Parameter(
                torch.tensor([1, 2], dtype=torch.int8), requires_grad=False
            )

    pipeline.handle = _handle(source_format="gguf", model=PackedModel())
    with pytest.raises(RuntimeError, match="did not dequantize every editable parameter"):
        pipeline._assert_supported_storage_format()


@pytest.mark.parametrize("output_format", ["gguf", "both"])
def test_rebirth_converts_dense_hf_then_transactionally_publishes_gguf_bundle(
    tmp_path,
    monkeypatch,
    output_format,
):
    pipeline = _pipeline(
        tmp_path,
        output_format=output_format,
        gguf_quant="Q4_K_M",
        llama_cpp_dir="/tools/llama.cpp",
        post_quant_verify=False,
    )
    model = _FakeModel()
    pipeline.handle = _handle(model=model)
    pipeline._input_source_metadata = {
        "format": "gguf",
        "model": "quantizer/repository",
        "file": "/models/source/model-Q4_K_M.gguf",
        "canonical_model_id": "openai/gpt-oss-20b",
        "tokenizer_source": "openai/gpt-oss-20b",
        "in_memory_dtype": "float16",
    }
    pipeline._free_gpu_memory = lambda: None
    export_calls: list[dict[str, object]] = []
    validation_calls: list[tuple[str, Path, object]] = []

    def fake_export(hf_checkpoint, bundle_dir, **kwargs):
        hf_dir = Path(hf_checkpoint)
        bundle = Path(bundle_dir)
        assert (hf_dir / "config.json").is_file()
        assert (hf_dir / "pytorch_model.bin").is_file()
        export_calls.append({"hf_dir": hf_dir, "bundle": bundle, **kwargs})
        final_path = bundle / str(kwargs["final_name"])
        final_path.write_bytes(b"GGUF\x03\x00\x00\x00mock")
        toolchain = LlamaCppToolchain(
            root=Path("/tools/llama.cpp"),
            converter=Path("/tools/llama.cpp/convert_hf_to_gguf.py"),
            quantizer=Path("/tools/llama.cpp/llama-quantize"),
            cli=None,
            python="python3",
            revision="test-revision",
        )
        return GGUFExportResult(
            final_path=final_path,
            dense_path=None,
            quantization=str(kwargs["quantization"]),
            dense_outtype=str(kwargs["dense_outtype"]),
            commands=(("convert",), ("quantize",)),
            toolchain=toolchain,
        )

    def fake_validate_gguf(path, *, strict=False):
        validation_calls.append(("gguf", Path(path), strict))

    def fake_validate_hf(path):
        validation_calls.append(("hf", Path(path), None))

    def fake_validate_quantization(path, expected):
        validation_calls.append(("quant", Path(path), expected))
        return f"MOSTLY_{expected}"

    monkeypatch.setattr(export_module, "export_hf_checkpoint_to_gguf", fake_export)
    monkeypatch.setattr(
        export_module,
        "validate_gguf_quantization",
        fake_validate_quantization,
    )
    monkeypatch.setattr(transaction_module, "validate_gguf_bundle", fake_validate_gguf)
    monkeypatch.setattr(transaction_module, "validate_hf_checkpoint", fake_validate_hf)

    result = pipeline._rebirth()

    assert result == pipeline.output_dir
    assert result.is_dir()
    gguf_files = list(result.glob("*.gguf"))
    assert [path.name for path in gguf_files] == [
        "model-OBLITERATED-Q4_K_M.gguf"
    ]
    assert (result / "tokenizer_config.json").is_file()
    metadata = json.loads(
        (result / "abliteration_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["model_source"] == pipeline._input_source_metadata
    assert metadata["output"]["format"] == output_format
    assert metadata["output"]["gguf_quant"] == "Q4_K_M"
    assert metadata["output"]["gguf_export"]["final_file"] == gguf_files[0].name
    assert export_calls[0]["quantization"] == "Q4_K_M"
    assert export_calls[0]["dense_outtype"] == "f16"
    assert export_calls[0]["keep_dense_intermediate"] is False
    assert model.save_calls[0]["save_original_format"] is False
    assert validation_calls[0][0] == "gguf"
    assert validation_calls[0][2] is True
    assert validation_calls[1][0] == "quant"
    assert validation_calls[1][2] == "Q4_K_M"

    if output_format == "both":
        assert (result / "hf" / "config.json").is_file()
        assert any(kind == "hf" for kind, _path, _strict in validation_calls)
    else:
        assert not (result / ".hf-staging").exists()
        assert all(kind != "hf" for kind, _path, _strict in validation_calls)

    assert pipeline._gguf_export_result is not None
    assert pipeline._gguf_export_result.final_path == gguf_files[0]
