"""Safe HF-to-GGUF export through a local llama.cpp checkout.

GGUF is an inference container, not a mutable PyTorch storage format.  The
abliteration pipeline therefore saves a dense Hugging Face checkpoint first,
converts that checkpoint to a 16-bit GGUF, and only then quantizes the converted
file.  In particular, this module never passes ``--allow-requantize`` to
``llama-quantize``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class GGUFExportError(RuntimeError):
    """Raised when the llama.cpp export toolchain cannot produce an artifact."""


LogCallback = Callable[[str], None]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]

_QUANT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{1,31}$")
_DENSE_OUTTYPES = {"f16", "bf16"}
_QUANT_ALIASES = {"Q3_K": "Q3_K_M", "Q4_K": "Q4_K_M", "Q5_K": "Q5_K_M"}
_SUPPORTED_QUANTIZATIONS = {
    "Q1_0",
    "Q2_0",
    "Q4_0",
    "Q4_1",
    "MXFP4_MOE",
    "Q5_0",
    "Q5_1",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ1_S",
    "IQ1_M",
    "TQ1_0",
    "TQ2_0",
    "Q2_K",
    "Q2_K_S",
    "IQ3_XXS",
    "IQ3_S",
    "IQ3_M",
    "IQ3_XS",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "IQ4_NL",
    "IQ4_XS",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
}


@dataclass(frozen=True)
class LlamaCppToolchain:
    """Resolved executables from one llama.cpp checkout."""

    root: Path
    converter: Path
    quantizer: Path
    cli: Path | None
    python: str
    revision: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "converter": str(self.converter),
            "quantizer": str(self.quantizer),
            "cli": str(self.cli) if self.cli is not None else None,
            "python": self.python,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class GGUFExportResult:
    """Paths and provenance for a completed dense-to-quantized conversion."""

    final_path: Path
    dense_path: Path | None
    quantization: str
    dense_outtype: str
    commands: tuple[tuple[str, ...], ...]
    toolchain: LlamaCppToolchain

    def to_metadata(self, *, bundle_dir: Path | None = None) -> dict[str, Any]:
        def display(path: Path | None) -> str | None:
            if path is None:
                return None
            if bundle_dir is not None:
                try:
                    return str(path.relative_to(bundle_dir))
                except ValueError:
                    pass
            return str(path)

        return {
            "final_file": display(self.final_path),
            "dense_intermediate": display(self.dense_path),
            "quantization": self.quantization,
            "dense_outtype": self.dense_outtype,
            "commands": [list(command) for command in self.commands],
            "toolchain": self.toolchain.to_metadata(),
            "requantized_packed_source_directly": False,
        }


def _real_file(path: Path, *, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise GGUFExportError(f"{label} was not found: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise GGUFExportError(f"{label} is not executable: {resolved}")
    return resolved


def _first_executable(candidates: Sequence[Path], name: str) -> Path:
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    from_path = shutil.which(name)
    if from_path:
        return Path(from_path).resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise GGUFExportError(
        f"Could not find {name}. Checked {rendered}. Build llama.cpp first or "
        "pass --llama-cpp-dir."
    )


def _git_revision(root: Path, runner: CommandRunner) -> str | None:
    try:
        result = runner(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    revision = (result.stdout or "").strip().splitlines()
    return revision[-1] if revision else None


def resolve_llama_cpp_toolchain(
    llama_cpp_dir: str | os.PathLike[str] | None,
    *,
    python_executable: str | os.PathLike[str] | None = None,
    runner: CommandRunner = subprocess.run,
) -> LlamaCppToolchain:
    """Resolve a converter, quantizer, and optional CLI without shell execution."""

    requested = llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR")
    if requested is None:
        candidates = (
            Path.cwd() / "vendor" / "llama.cpp",
            Path.cwd() / "llama.cpp",
        )
        root = next((path.resolve() for path in candidates if path.is_dir()), None)
        if root is None:
            raise GGUFExportError(
                "A llama.cpp checkout is required for GGUF output. Pass "
                "--llama-cpp-dir or set LLAMA_CPP_DIR."
            )
    else:
        root = Path(requested).expanduser().resolve()
        if not root.is_dir():
            raise GGUFExportError(f"llama.cpp directory was not found: {root}")

    converter = _real_file(root / "convert_hf_to_gguf.py", label="llama.cpp converter")
    quantizer = _first_executable(
        (
            root / "build" / "bin" / "llama-quantize",
            root / "llama-quantize",
            root / "quantize",
        ),
        "llama-quantize",
    )
    cli: Path | None = None
    for candidate in (
        root / "build" / "bin" / "llama-cli",
        root / "llama-cli",
        Path(shutil.which("llama-cli") or ""),
    ):
        if str(candidate) and candidate.expanduser().is_file() and os.access(candidate, os.X_OK):
            cli = candidate.expanduser().resolve()
            break

    python = str(Path(python_executable).expanduser()) if python_executable else sys.executable
    if os.path.sep in python:
        _real_file(Path(python), label="llama.cpp Python interpreter", executable=True)
    elif shutil.which(python) is None:
        raise GGUFExportError(f"Python interpreter was not found: {python}")

    return LlamaCppToolchain(
        root=root,
        converter=converter,
        quantizer=quantizer,
        cli=cli,
        python=python,
        revision=_git_revision(root, runner),
    )


def normalize_gguf_quantization(value: str) -> str:
    """Validate and canonicalize quantizers supported by the pinned toolchain."""

    quant = str(value).strip().upper()
    if quant in {"COPY", "F16", "BF16", "F32"} or not _QUANT_RE.fullmatch(quant):
        raise ValueError(
            "gguf_quant must be a llama.cpp quantization name such as Q4_K_M; "
            "dense/copy output is not accepted as the final quantized artifact"
        )
    quant = _QUANT_ALIASES.get(quant, quant)
    if quant not in _SUPPORTED_QUANTIZATIONS:
        raise ValueError(
            f"Unsupported gguf_quant {quant!r} for the tested llama.cpp toolchain"
        )
    return quant


_validate_quantization = normalize_gguf_quantization


def _run_checked(
    command: Sequence[str],
    *,
    label: str,
    runner: CommandRunner,
    log: LogCallback,
    cwd: Path | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    argv = [str(item) for item in command]
    log(f"{label}: {' '.join(argv)}")
    try:
        result = runner(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GGUFExportError(f"{label} timed out after {timeout:.0f}s") from exc
    except OSError as exc:
        raise GGUFExportError(f"Could not start {label}: {exc}") from exc

    output = result.stdout or ""
    if result.returncode != 0:
        tail = "\n".join(output.splitlines()[-40:])
        raise GGUFExportError(
            f"{label} failed with exit code {result.returncode}"
            + (f":\n{tail}" if tail else "")
        )
    if output.strip():
        log("\n".join(output.splitlines()[-10:]))
    return result


def export_hf_checkpoint_to_gguf(
    hf_checkpoint: str | os.PathLike[str],
    bundle_dir: str | os.PathLike[str],
    *,
    llama_cpp_dir: str | os.PathLike[str] | None,
    python_executable: str | os.PathLike[str] | None = None,
    quantization: str = "Q4_K_M",
    imatrix: str | os.PathLike[str] | None = None,
    dense_outtype: str = "bf16",
    final_name: str = "model-abliterated-Q4_K_M.gguf",
    keep_dense_intermediate: bool = False,
    log: LogCallback | None = None,
    runner: CommandRunner = subprocess.run,
) -> GGUFExportResult:
    """Convert a dense HF checkpoint and quantize it exactly once."""

    logger = log or (lambda _message: None)
    hf_dir = Path(hf_checkpoint).expanduser().resolve()
    if not hf_dir.is_dir() or not (hf_dir / "config.json").is_file():
        raise GGUFExportError(f"Expected a dense Hugging Face checkpoint directory: {hf_dir}")

    out_dir = Path(bundle_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    quant = _validate_quantization(quantization)
    outtype = str(dense_outtype).lower()
    if outtype not in _DENSE_OUTTYPES:
        raise ValueError(f"dense_outtype must be one of {sorted(_DENSE_OUTTYPES)}")

    final_basename = Path(final_name).name
    if final_basename != final_name or not final_basename.lower().endswith(".gguf"):
        raise ValueError("final_name must be a plain .gguf filename")
    final_path = out_dir / final_basename
    dense_dir = out_dir / ".gguf-work"
    dense_dir.mkdir(parents=True, exist_ok=True)
    dense_path = dense_dir / f"model-{outtype.upper()}.gguf"
    if final_path.exists() or dense_path.exists():
        raise GGUFExportError("GGUF staging paths must not already exist")

    imatrix_path: Path | None = None
    if imatrix is not None:
        imatrix_path = _real_file(Path(imatrix), label="importance matrix")

    toolchain = resolve_llama_cpp_toolchain(
        llama_cpp_dir,
        python_executable=python_executable,
        runner=runner,
    )
    convert_command = (
        toolchain.python,
        str(toolchain.converter),
        str(hf_dir),
        "--outfile",
        str(dense_path),
        "--outtype",
        outtype,
        "--use-temp-file",
    )
    _run_checked(
        convert_command,
        label="llama.cpp HF-to-GGUF conversion",
        runner=runner,
        log=logger,
        cwd=toolchain.root,
    )
    if not dense_path.is_file() or dense_path.stat().st_size <= 0:
        raise GGUFExportError("llama.cpp converter reported success but produced no dense GGUF")

    quantize_parts = [str(toolchain.quantizer)]
    if imatrix_path is not None:
        quantize_parts.extend(("--imatrix", str(imatrix_path)))
    quantize_parts.extend((str(dense_path), str(final_path), quant))
    quantize_command = tuple(quantize_parts)
    _run_checked(
        quantize_command,
        label=f"llama.cpp {quant} quantization",
        runner=runner,
        log=logger,
        cwd=toolchain.root,
    )
    if not final_path.is_file() or final_path.stat().st_size <= 0:
        raise GGUFExportError("llama-quantize reported success but produced no final GGUF")

    retained_dense: Path | None = None
    if keep_dense_intermediate:
        intermediate_dir = out_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        retained_dense = intermediate_dir / dense_path.name
        dense_path.replace(retained_dense)
    shutil.rmtree(dense_dir, ignore_errors=True)

    return GGUFExportResult(
        final_path=final_path,
        dense_path=retained_dense,
        quantization=quant,
        dense_outtype=outtype,
        commands=(tuple(convert_command), quantize_command),
        toolchain=toolchain,
    )


def run_llama_cpp_smoke_test(
    gguf_path: str | os.PathLike[str],
    toolchain: LlamaCppToolchain,
    *,
    log: LogCallback | None = None,
    runner: CommandRunner = subprocess.run,
    timeout: float = 900.0,
) -> None:
    """Load and generate one token with llama.cpp after dense weights are freed."""

    if toolchain.cli is None:
        raise GGUFExportError(
            "Post-quantization verification requires llama-cli; build it in the "
            "selected llama.cpp checkout or disable --post-quant-verify explicitly."
        )
    model = _real_file(Path(gguf_path), label="quantized GGUF")
    command = (
        str(toolchain.cli),
        "--model",
        str(model),
        "--prompt",
        "Reply with OK.",
        "--n-predict",
        "1",
        "--ctx-size",
        "256",
        "--n-gpu-layers",
        "0",
        "--no-warmup",
    )
    _run_checked(
        command,
        label="llama.cpp post-quantization load/generation smoke test",
        runner=runner,
        log=log or (lambda _message: None),
        cwd=toolchain.root,
        timeout=timeout,
    )


def validate_gguf_quantization(
    gguf_path: str | os.PathLike[str],
    expected: str,
) -> str:
    """Verify that GGUF metadata records the requested llama.cpp file type."""

    model = _real_file(Path(gguf_path), label="quantized GGUF")
    quant = _validate_quantization(expected)
    try:
        from gguf import GGUFReader, LlamaFileType
    except ImportError as exc:  # pragma: no cover - dependency is pinned at runtime
        raise GGUFExportError(
            "GGUF quantization validation requires the pinned 'gguf' package"
        ) from exc

    try:
        reader = GGUFReader(model, "r")
        field = reader.get_field("general.file_type")
        if field is None or not field.data:
            raise GGUFExportError("GGUF is missing required general.file_type metadata")
        raw = field.parts[field.data[0]]
        if isinstance(raw, (list, tuple)) or getattr(raw, "ndim", 0):
            raw = raw[0]
        value = int(raw)
        recorded = LlamaFileType(value).name
    except GGUFExportError:
        raise
    except (IndexError, TypeError, ValueError, OSError) as exc:
        raise GGUFExportError(
            f"Could not read GGUF quantization metadata from {model}: {exc}"
        ) from exc

    expected_name = f"MOSTLY_{quant}"
    if recorded != expected_name:
        raise GGUFExportError(
            f"GGUF quantization mismatch: requested {quant}, metadata records {recorded}"
        )
    return recorded


__all__ = [
    "GGUFExportError",
    "GGUFExportResult",
    "LlamaCppToolchain",
    "export_hf_checkpoint_to_gguf",
    "normalize_gguf_quantization",
    "resolve_llama_cpp_toolchain",
    "run_llama_cpp_smoke_test",
    "validate_gguf_quantization",
]
