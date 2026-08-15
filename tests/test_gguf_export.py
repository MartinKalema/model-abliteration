from __future__ import annotations

import subprocess
from pathlib import Path
from typing import ClassVar

import pytest

from obliteratus.gguf_export import (
    GGUFExportError,
    export_hf_checkpoint_to_gguf,
    normalize_gguf_quantization,
    resolve_llama_cpp_toolchain,
    run_llama_cpp_smoke_test,
    validate_gguf_quantization,
)


def _toolchain(tmp_path: Path) -> Path:
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# converter\n", encoding="utf-8")
    for name in ("llama-quantize", "llama-cli"):
        executable = root / "build" / "bin" / name
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
    return root


def _hf_checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    return checkpoint


class FakeRunner:
    def __init__(self, *, fail_quantize: bool = False):
        self.commands: list[tuple[str, ...]] = []
        self.fail_quantize = fail_quantize

    def __call__(self, command, **_kwargs):
        argv = tuple(str(item) for item in command)
        self.commands.append(argv)
        if argv[:4] == ("git", "-C", argv[2], "rev-parse"):
            return subprocess.CompletedProcess(argv, 0, "abc123\n")
        if "convert_hf_to_gguf.py" in argv[1]:
            output = Path(argv[argv.index("--outfile") + 1])
            output.write_bytes(b"GGUF-dense")
            return subprocess.CompletedProcess(argv, 0, "converted\n")
        if argv[0].endswith("llama-quantize"):
            if self.fail_quantize:
                return subprocess.CompletedProcess(argv, 7, "bad quantization\n")
            final_path = Path(argv[-2])
            final_path.write_bytes(b"GGUF-quantized")
            return subprocess.CompletedProcess(argv, 0, "quantized\n")
        if argv[0].endswith("llama-cli"):
            return subprocess.CompletedProcess(argv, 0, "OK\n")
        raise AssertionError(f"unexpected command: {argv}")


def test_export_converts_dense_then_quantizes_once(tmp_path: Path):
    root = _toolchain(tmp_path)
    hf = _hf_checkpoint(tmp_path)
    imatrix = tmp_path / "importance.dat"
    imatrix.write_bytes(b"importance")
    runner = FakeRunner()

    result = export_hf_checkpoint_to_gguf(
        hf,
        tmp_path / "bundle",
        llama_cpp_dir=root,
        python_executable="/bin/sh",
        quantization="q4_k_m",
        imatrix=imatrix,
        runner=runner,
    )

    assert result.final_path.read_bytes() == b"GGUF-quantized"
    assert result.dense_path is None
    assert result.quantization == "Q4_K_M"
    assert not (tmp_path / "bundle" / ".gguf-work").exists()
    flattened = [part for command in runner.commands for part in command]
    assert "--allow-requantize" not in flattened
    quantize = next(command for command in runner.commands if command[0].endswith("llama-quantize"))
    assert quantize[1:3] == ("--imatrix", str(imatrix.resolve()))


def test_export_can_retain_dense_intermediate_below_top_level(tmp_path: Path):
    root = _toolchain(tmp_path)
    runner = FakeRunner()
    result = export_hf_checkpoint_to_gguf(
        _hf_checkpoint(tmp_path),
        tmp_path / "bundle",
        llama_cpp_dir=root,
        python_executable="/bin/sh",
        keep_dense_intermediate=True,
        runner=runner,
    )

    assert result.dense_path == tmp_path / "bundle" / "intermediate" / "model-BF16.gguf"
    assert result.dense_path.is_file()
    assert list((tmp_path / "bundle").glob("*.gguf")) == [result.final_path]


def test_export_rejects_direct_dense_or_copy_final_type(tmp_path: Path):
    root = _toolchain(tmp_path)
    hf = _hf_checkpoint(tmp_path)
    for quant in ("copy", "bf16", "Q4-K-M", "", "NOT_A_REAL_QUANT"):
        with pytest.raises(ValueError, match="gguf_quant"):
            export_hf_checkpoint_to_gguf(
                hf,
                tmp_path / f"bundle-{quant or 'empty'}",
                llama_cpp_dir=root,
                python_executable="/bin/sh",
                quantization=quant,
                runner=FakeRunner(),
            )


def test_quantization_aliases_are_canonicalized():
    assert normalize_gguf_quantization("q4_k") == "Q4_K_M"


def test_export_surfaces_quantizer_failure_and_output_tail(tmp_path: Path):
    with pytest.raises(GGUFExportError, match="bad quantization"):
        export_hf_checkpoint_to_gguf(
            _hf_checkpoint(tmp_path),
            tmp_path / "bundle",
            llama_cpp_dir=_toolchain(tmp_path),
            python_executable="/bin/sh",
            runner=FakeRunner(fail_quantize=True),
        )


def test_toolchain_requires_checkout(tmp_path: Path):
    with pytest.raises(GGUFExportError, match="llama.cpp directory"):
        resolve_llama_cpp_toolchain(tmp_path / "missing")


def test_smoke_test_generates_one_token_on_cpu(tmp_path: Path):
    runner = FakeRunner()
    toolchain = resolve_llama_cpp_toolchain(
        _toolchain(tmp_path),
        python_executable="/bin/sh",
        runner=runner,
    )
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")

    run_llama_cpp_smoke_test(model, toolchain, runner=runner)

    command = runner.commands[-1]
    assert command[0].endswith("llama-cli")
    assert command[command.index("--n-predict") + 1] == "1"
    assert command[command.index("--n-gpu-layers") + 1] == "0"


def test_quantization_validator_checks_general_file_type(tmp_path: Path, monkeypatch):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")

    class FakeField:
        data: ClassVar[list[int]] = [0]
        parts: ClassVar[list[list[int]]] = [[15]]

    class FakeReader:
        def __init__(self, path, mode):
            assert Path(path) == model
            assert mode == "r"

        def get_field(self, name):
            assert name == "general.file_type"
            return FakeField()

    import gguf

    monkeypatch.setattr(gguf, "GGUFReader", FakeReader)
    assert validate_gguf_quantization(model, "q4_k_m") == "MOSTLY_Q4_K_M"

    with pytest.raises(GGUFExportError, match="quantization mismatch"):
        validate_gguf_quantization(model, "Q5_K_M")
