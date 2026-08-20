"""Remote execution support for Obliteratus.

Run abliteration pipelines on remote GPU nodes via SSH. The remote machine
must have CUDA-capable GPUs and a Python environment. Obliteratus will be
auto-installed if not present.

Usage (CLI):
    obliteratus obliterate meta-llama/Llama-3.1-8B-Instruct \
        --remote user@gpu-node \
        --ssh-key ~/.ssh/id_rsa

Usage (YAML config):
    remote:
      host: gpu-node
      user: root
      ssh_key: ~/.ssh/id_rsa
      remote_dir: /tmp/obliteratus_run
"""

from __future__ import annotations

import math
import os
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

console = Console()


@dataclass
class RemoteConfig:
    """SSH connection and remote execution settings."""

    host: str
    user: str = "root"
    port: int = 22
    ssh_key: str | None = None
    remote_dir: str = "/tmp/obliteratus_run"
    install_timeout: int = 600  # seconds
    python: str = "python3"  # remote python binary
    sync_results: bool = True
    gpus: str | None = None  # comma-separated GPU IDs or "all"

    @property
    def ssh_target(self) -> str:
        return f"{self.user}@{self.host}"

    @classmethod
    def from_cli_args(cls, remote_str: str, **kwargs) -> RemoteConfig:
        """Parse 'user@host' or just 'host' from CLI --remote flag."""
        if "@" in remote_str:
            user, host = remote_str.rsplit("@", 1)
        else:
            user = "root"
            host = remote_str
        return cls(host=host, user=user, **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> RemoteConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RemoteRunner:
    """Execute Obliteratus commands on a remote machine via SSH."""

    def __init__(
        self,
        config: RemoteConfig,
        on_log: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.on_log = on_log or (lambda msg: console.print(f"[dim][remote][/] {msg}"))

    def _ssh_base_cmd(self) -> list[str]:
        """Build base SSH command with common options."""
        cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=30",
            "-p", str(self.config.port),
        ]
        if self.config.ssh_key:
            key_path = os.path.expanduser(self.config.ssh_key)
            cmd.extend(["-i", key_path])
        cmd.append(self.config.ssh_target)
        return cmd

    def _scp_base_cmd(self) -> list[str]:
        """Build base SCP command."""
        cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-P", str(self.config.port),
            "-r",
        ]
        if self.config.ssh_key:
            key_path = os.path.expanduser(self.config.ssh_key)
            cmd.extend(["-i", key_path])
        return cmd

    def run_ssh(self, remote_cmd: str, stream: bool = False, timeout: int | None = None) -> subprocess.CompletedProcess | int:
        """Run a command on the remote host.

        If stream=True, streams stdout/stderr in real-time and returns the
        exit code. Otherwise returns CompletedProcess.
        """
        cmd = self._ssh_base_cmd() + [remote_cmd]

        if stream:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            try:
                for line in proc.stdout:
                    line = line.rstrip("\n")
                    self.on_log(line)
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                self.on_log("[red]Remote command timed out[/]")
                return 1
            return proc.returncode
        else:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

    def check_connection(self) -> bool:
        """Verify SSH connectivity."""
        self.on_log(f"Testing SSH connection to {self.config.ssh_target}...")
        result = self.run_ssh("echo ok", timeout=30)
        if isinstance(result, subprocess.CompletedProcess) and result.returncode == 0:
            self.on_log("SSH connection OK")
            return True
        self.on_log("[red]SSH connection failed[/]")
        return False

    def check_gpu(self) -> str | None:
        """Check for CUDA GPUs on remote. Returns nvidia-smi output or None."""
        result = self.run_ssh(
            "nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader",
            timeout=30,
        )
        if isinstance(result, subprocess.CompletedProcess) and result.returncode == 0:
            gpu_info = result.stdout.strip()
            lines = gpu_info.split("\n")
            self.on_log(f"Remote GPUs ({len(lines)} detected):")
            for line in lines:
                self.on_log(f"  {line.strip()}")
            if self.config.gpus and self.config.gpus.lower() != "all":
                self.on_log(f"  Selected GPUs: {self.config.gpus}")
            else:
                self.on_log(f"  Using: all {len(lines)} GPUs")
            return gpu_info
        self.on_log("[yellow]No GPUs detected on remote (nvidia-smi failed)[/]")
        return None

    def _env_prefix(self) -> str:
        """Build environment variable prefix for remote commands (e.g. CUDA_VISIBLE_DEVICES)."""
        parts = []
        if self.config.gpus and self.config.gpus.lower() != "all":
            raw_ids = [item.strip() for item in self.config.gpus.split(",")]
            if not raw_ids or any(not item.isdigit() for item in raw_ids):
                raise ValueError("gpus must be 'all' or a comma-separated list of integer IDs")
            gpu_ids = ",".join(str(int(item)) for item in raw_ids)
            parts.append(f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_ids)}")
        return " ".join(parts) + " " if parts else ""

    def ensure_obliteratus(self) -> bool:
        """Install or update obliteratus on the remote if needed."""
        # Check if already installed
        check = self.run_ssh(
            f"{shlex.quote(self.config.python)} "
            '-c "import obliteratus; print(obliteratus.__version__)"',
            timeout=30,
        )
        if isinstance(check, subprocess.CompletedProcess) and check.returncode == 0:
            version = check.stdout.strip()
            self.on_log(f"Obliteratus {version} already installed on remote")
            return True

        # Install from PyPI or git
        self.on_log("Installing obliteratus on remote...")
        install_cmd = (
            f"{shlex.quote(self.config.python)} -m pip install --quiet "
            f"git+https://github.com/StellaAthena/OBLITERATUS.git"
        )
        rc = self.run_ssh(install_cmd, stream=True, timeout=self.config.install_timeout)
        if rc != 0:
            self.on_log("[red]Failed to install obliteratus on remote[/]")
            return False

        self.on_log("Obliteratus installed successfully")
        return True

    def sync_results_back(self, remote_output_dir: str, local_output_dir: str) -> bool:
        """Copy results from remote back to local machine via scp."""
        local_path = Path(local_output_dir)
        local_path.mkdir(parents=True, exist_ok=True)

        self.on_log(f"Syncing results: {self.config.ssh_target}:{remote_output_dir} -> {local_output_dir}")

        cmd = self._scp_base_cmd() + [
            f"{self.config.ssh_target}:{remote_output_dir}/",
            str(local_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            self.on_log(f"Results synced to {local_output_dir}")
            return True
        else:
            self.on_log(f"[red]SCP failed: {result.stderr}[/]")
            return False

    def build_obliterate_command(
        self,
        model: str,
        output_dir: str | None = None,
        method: str = "advanced",
        device: str = "auto",
        dtype: str = "float16",
        quantization: str | None = None,
        n_directions: int | None = None,
        direction_method: str | None = None,
        regularization: float | None = None,
        kl_preservation: bool = False,
        kl_budget: float = 0.05,
        kl_search_steps: int = 5,
        cot_preservation: bool = False,
        cot_reasoning_ce_budget: float = 0.25,
        cot_answer_ce_budget: float = 0.15,
        refinement_passes: int | None = None,
        min_layer_fraction: float | None = None,
        max_layer_fraction: float | None = None,
        harmless_pc_count: int | None = None,
        shield_concept_count: int | None = None,
        shield_ridge: float | None = None,
        shield_residualize: bool | None = None,
        shield_layer_penalty: float | None = None,
        projection_target: str | None = None,
        projection_row_fraction: float | None = None,
        large_model: bool = False,
        verify_sample_size: int | None = None,
        damage_gate_enabled: bool = True,
        damage_eval_size: int = 64,
        max_ppl_ratio: float | None = None,
        max_sampled_token_kl: float | None = None,
        max_p95_sampled_token_kl: float | None = None,
        max_top1_flip_rate: float | None = None,
        max_coherence_drop: float | None = None,
        max_refusal_rate: float | None = None,
        project_lm_head: bool | None = None,
        project_embeddings: bool | None = None,
        overwrite_output: bool = False,
    ) -> str:
        """Build the remote obliteratus CLI command."""
        if not isinstance(damage_gate_enabled, bool):
            raise TypeError("damage_gate_enabled must be a boolean")
        if not isinstance(kl_preservation, bool):
            raise TypeError("kl_preservation must be a boolean")
        if not isinstance(cot_preservation, bool):
            raise TypeError("cot_preservation must be a boolean")
        if (
            not isinstance(kl_search_steps, int)
            or isinstance(kl_search_steps, bool)
            or kl_search_steps < 2
        ):
            raise ValueError("kl_search_steps must be an integer of at least 2")
        if not isinstance(damage_eval_size, int) or isinstance(damage_eval_size, bool):
            raise TypeError("damage_eval_size must be an integer")
        if damage_eval_size < 32:
            raise ValueError("damage_eval_size must be an integer greater than or equal to 32")

        def _validate_finite(name: str, value: float | None, minimum: float) -> None:
            if value is None:
                return
            if (
                isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < minimum
            ):
                raise ValueError(f"{name} must be finite and greater than or equal to {minimum}")

        def _validate_rate(name: str, value: float | None) -> None:
            if value is None:
                return
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")

        _validate_finite("max_ppl_ratio", max_ppl_ratio, 1.0)
        _validate_finite("kl_budget", kl_budget, 0.0)
        _validate_finite("cot_reasoning_ce_budget", cot_reasoning_ce_budget, 0.0)
        _validate_finite("cot_answer_ce_budget", cot_answer_ce_budget, 0.0)
        _validate_finite("max_sampled_token_kl", max_sampled_token_kl, 0.0)
        _validate_finite("max_p95_sampled_token_kl", max_p95_sampled_token_kl, 0.0)
        _validate_rate("max_top1_flip_rate", max_top1_flip_rate)
        _validate_rate("max_coherence_drop", max_coherence_drop)
        _validate_rate("max_refusal_rate", max_refusal_rate)
        for name, value in (
            ("min_layer_fraction", min_layer_fraction),
            ("max_layer_fraction", max_layer_fraction),
        ):
            _validate_rate(name, value)
        for name, value in (
            ("harmless_pc_count", harmless_pc_count),
            ("shield_concept_count", shield_concept_count),
        ):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        _validate_finite("shield_ridge", shield_ridge, 0.0)
        _validate_finite("shield_layer_penalty", shield_layer_penalty, 0.0)
        if shield_residualize is not None and not isinstance(shield_residualize, bool):
            raise TypeError("shield_residualize must be a boolean or None")
        if projection_target is not None and projection_target not in {
            "auto", "all", "attention", "ffn", "output",
        }:
            raise ValueError(
                "projection_target must be one of: auto, all, attention, ffn, output"
            )
        if projection_row_fraction is not None and (
            not math.isfinite(float(projection_row_fraction))
            or not 0.0 < float(projection_row_fraction) <= 1.0
        ):
            raise ValueError("projection_row_fraction must be finite and in (0, 1]")
        if projection_target == "auto" and quantization is not None:
            raise ValueError("projection_target='auto' is not compatible with quantization")
        if projection_target == "auto" and damage_eval_size < 64:
            raise ValueError("projection_target='auto' requires damage_eval_size >= 64")
        if kl_preservation and projection_target == "auto":
            raise ValueError(
                "kl_preservation cannot be combined with projection_target='auto'"
            )
        if (kl_preservation or cot_preservation) and quantization is not None:
            raise ValueError("KL/CoT preservation is not compatible with quantization")
        if kl_preservation and damage_eval_size < 64:
            raise ValueError("kl_preservation requires damage_eval_size >= 64")
        for name, value in (
            ("project_lm_head", project_lm_head),
            ("project_embeddings", project_embeddings),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean or None")
        if not isinstance(overwrite_output, bool):
            raise TypeError("overwrite_output must be a boolean")

        remote_output = output_dir or f"{self.config.remote_dir}/output/{model.replace('/', '_')}"

        parts = [
            self._env_prefix() + shlex.quote(self.config.python), "-m", "obliteratus",
            "obliterate", shlex.quote(model),
            "--output-dir", shlex.quote(remote_output),
            "--method", shlex.quote(method),
            "--device", shlex.quote(device),
            "--dtype", shlex.quote(dtype),
        ]
        if quantization:
            parts.extend(["--quantization", shlex.quote(quantization)])
        if n_directions is not None:
            parts.extend(["--n-directions", shlex.quote(str(n_directions))])
        if direction_method:
            parts.extend(["--direction-method", shlex.quote(direction_method)])
        if regularization is not None:
            parts.extend(["--regularization", shlex.quote(str(regularization))])
        if kl_preservation:
            parts.extend(
                [
                    "--kl-preservation",
                    "--kl-budget",
                    shlex.quote(str(kl_budget)),
                    "--kl-search-steps",
                    shlex.quote(str(kl_search_steps)),
                ]
            )
        if cot_preservation:
            parts.extend(
                [
                    "--cot-preservation",
                    "--cot-reasoning-ce-budget",
                    shlex.quote(str(cot_reasoning_ce_budget)),
                    "--cot-answer-ce-budget",
                    shlex.quote(str(cot_answer_ce_budget)),
                ]
            )
        if refinement_passes is not None:
            parts.extend(["--refinement-passes", shlex.quote(str(refinement_passes))])
        for flag, value in (
            ("--min-layer-fraction", min_layer_fraction),
            ("--max-layer-fraction", max_layer_fraction),
            ("--harmless-pc-count", harmless_pc_count),
            ("--shield-concept-count", shield_concept_count),
            ("--shield-ridge", shield_ridge),
            ("--shield-layer-penalty", shield_layer_penalty),
            ("--projection-target", projection_target),
            ("--projection-row-fraction", projection_row_fraction),
        ):
            if value is not None:
                parts.extend([flag, shlex.quote(str(value))])
        if shield_residualize:
            parts.append("--shield-residualize")
        if large_model:
            parts.append("--large-model")
        if verify_sample_size is not None:
            parts.extend(["--verify-sample-size", shlex.quote(str(verify_sample_size))])

        parts.append("--damage-gate" if damage_gate_enabled else "--unsafe-disable-damage-gate")
        parts.extend(["--damage-eval-size", str(damage_eval_size)])
        for flag, value in (
            ("--max-ppl-ratio", max_ppl_ratio),
            ("--max-sampled-token-kl", max_sampled_token_kl),
            ("--max-p95-sampled-token-kl", max_p95_sampled_token_kl),
            ("--max-top1-flip-rate", max_top1_flip_rate),
            ("--max-coherence-drop", max_coherence_drop),
            ("--max-refusal-rate", max_refusal_rate),
        ):
            if value is not None:
                parts.extend([flag, shlex.quote(str(value))])
        if project_lm_head is not None:
            parts.append("--project-lm-head" if project_lm_head else "--no-project-lm-head")
        if project_embeddings is not None:
            parts.append(
                "--project-embeddings" if project_embeddings else "--no-project-embeddings"
            )
        if overwrite_output:
            parts.append("--overwrite-output")

        return " ".join(parts)

    def build_run_command(self, remote_config_path: str, output_dir: str | None = None, preset: str | None = None) -> str:
        """Build remote 'obliteratus run' command."""
        parts = [
            self._env_prefix() + shlex.quote(self.config.python), "-m", "obliteratus",
            "run", shlex.quote(remote_config_path),
        ]
        if output_dir:
            parts.extend(["--output-dir", shlex.quote(output_dir)])
        if preset:
            parts.extend(["--preset", preset])
        return " ".join(parts)

    def build_tourney_command(
        self,
        model: str,
        output_dir: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        quantization: str | None = None,
        methods: list[str] | None = None,
        hub_org: str | None = None,
        hub_repo: str | None = None,
        dataset: str = "builtin",
    ) -> str:
        """Build remote 'obliteratus tourney' command."""
        remote_output = output_dir or f"{self.config.remote_dir}/tourney/{model.replace('/', '_')}"

        parts = [
            self._env_prefix() + shlex.quote(self.config.python), "-m", "obliteratus",
            "tourney", shlex.quote(model),
            "--output-dir", shlex.quote(remote_output),
            "--device", device,
            "--dtype", dtype,
            "--dataset", dataset,
        ]
        if quantization:
            parts.extend(["--quantization", quantization])
        if hub_org:
            parts.extend(["--hub-org", hub_org])
        if hub_repo:
            parts.extend(["--hub-repo", hub_repo])
        if methods:
            parts.extend(["--methods"] + methods)
        return " ".join(parts)

    def upload_config(self, local_config_path: str) -> str:
        """Upload a YAML config file to the remote."""
        remote_path = f"{self.config.remote_dir}/config.yaml"
        self.run_ssh(f"mkdir -p {shlex.quote(self.config.remote_dir)}")

        cmd = self._scp_base_cmd()
        # scp uses -P not -p, already handled in _scp_base_cmd
        cmd += [local_config_path, f"{self.config.ssh_target}:{remote_path}"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to upload config: {result.stderr}")
        self.on_log(f"Config uploaded to {remote_path}")
        return remote_path

    def run_obliterate(
        self,
        model: str,
        local_output_dir: str | None = None,
        **kwargs,
    ) -> str | None:
        """Full remote obliteration: setup, run, sync results.

        Returns local path to results, or None on failure.
        """
        # 1. Verify connection
        if not self.check_connection():
            return None

        # 2. Check GPUs
        self.check_gpu()

        # 3. Ensure obliteratus is installed
        if not self.ensure_obliteratus():
            return None

        # 4. Create remote working directory
        self.run_ssh(f"mkdir -p {shlex.quote(self.config.remote_dir)}")

        # 5. Build and run the command
        remote_output = f"{self.config.remote_dir}/output/{model.replace('/', '_')}"
        cmd = self.build_obliterate_command(model, output_dir=remote_output, **kwargs)
        self.on_log(f"Running: {cmd}")

        rc = self.run_ssh(cmd, stream=True)
        if rc != 0:
            self.on_log(f"[red]Remote obliteration failed (exit code {rc})[/]")
            return None

        # 6. Sync results back
        if self.config.sync_results:
            local_output = local_output_dir or f"abliterated/{model.replace('/', '_')}"
            if self.sync_results_back(remote_output, local_output):
                return local_output
            return None

        self.on_log(f"Results on remote: {remote_output}")
        return remote_output

    def run_config(
        self,
        local_config_path: str,
        local_output_dir: str | None = None,
        preset: str | None = None,
    ) -> str | None:
        """Upload config, run study remotely, sync results."""
        if not self.check_connection():
            return None
        self.check_gpu()
        if not self.ensure_obliteratus():
            return None

        # Upload config
        remote_config = self.upload_config(local_config_path)

        # Determine remote output dir
        remote_output = f"{self.config.remote_dir}/results"
        cmd = self.build_run_command(remote_config, output_dir=remote_output, preset=preset)
        self.on_log(f"Running: {cmd}")

        rc = self.run_ssh(cmd, stream=True)
        if rc != 0:
            self.on_log(f"[red]Remote run failed (exit code {rc})[/]")
            return None

        if self.config.sync_results:
            local_output = local_output_dir or "results"
            if self.sync_results_back(remote_output, local_output):
                return local_output
            return None

        return remote_output

    def run_tourney(
        self,
        model: str,
        local_output_dir: str | None = None,
        **kwargs,
    ) -> str | None:
        """Run tournament remotely, sync results."""
        if not self.check_connection():
            return None
        self.check_gpu()
        if not self.ensure_obliteratus():
            return None

        remote_output = f"{self.config.remote_dir}/tourney/{model.replace('/', '_')}"
        cmd = self.build_tourney_command(model, output_dir=remote_output, **kwargs)
        self.on_log(f"Running: {cmd}")

        rc = self.run_ssh(cmd, stream=True)
        if rc != 0:
            self.on_log(f"[red]Remote tourney failed (exit code {rc})[/]")
            return None

        if self.config.sync_results:
            local_output = local_output_dir or f"/tmp/obliteratus_tourney/{model.replace('/', '_')}"
            if self.sync_results_back(remote_output, local_output):
                return local_output
            return None

        return remote_output
