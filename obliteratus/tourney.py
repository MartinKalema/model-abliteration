"""OBLITERATUS Tourney — March Madness-style tournament to find the best abliteration method.

Run all methods head-to-head in elimination rounds.  The winner gets auto-pushed
to HuggingFace Hub so the community can use the best possible abliteration.

Usage (CLI):
    obliteratus tourney meta-llama/Llama-3.1-8B-Instruct --hub-org my-org

Usage (Python):
    from obliteratus.tourney import TourneyRunner
    runner = TourneyRunner("meta-llama/Llama-3.1-8B-Instruct", hub_org="my-org")
    winner = runner.run()
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from obliteratus.abliterate import UNAVAILABLE_METHODS, available_method_names
from obliteratus.evaluation.candidate_selection import (
    CandidateEvidenceError,
    add_acceptance_evidence,
    damage_severity,
    validate_acceptance_payload,
)

# ---------------------------------------------------------------------------
# All tournament-eligible methods.
#
# Excluded:
#   - compute-heavy paper/model-forward search presets — runnable explicitly,
#     but omitted from the default tournament because each entrant performs
#     its own multi-trial search and full-snapshot replay
#   - 'nuclear'   — collapsed in essentially every telemetry run (n=7,545)
#   - 'basic'     — fast but quality is unusable across architectures
# ---------------------------------------------------------------------------

TOURNEY_METHODS = [
    "advanced",
    "aggressive",
    "spectral_cascade",
    "informed",
    "surgical",
    "inverted",
    "failspy",
]


def validate_tourney_methods(methods: list[str] | None) -> list[str]:
    """Return runnable tournament methods or reject stale/raw preset names."""
    if isinstance(methods, (str, bytes)):
        raise TypeError("Tournament methods must be provided as a list of preset names")
    selected = list(TOURNEY_METHODS) if not methods else list(methods)
    available = frozenset(available_method_names())

    problems: list[str] = []
    for method in selected:
        if not isinstance(method, str):
            problems.append(f"method names must be strings, got {type(method).__name__}")
        elif method in UNAVAILABLE_METHODS:
            problems.append(f"`{method}` is unavailable: {UNAVAILABLE_METHODS[method]}")
        elif method not in available:
            problems.append(f"`{method}` is not a runnable preset")

    if problems:
        raise ValueError("Invalid tournament method selection: " + "; ".join(problems))
    return selected


# Tournament output directories are recursively cleaned between fresh runs.
# A path must carry this exact, path-bound marker before it is eligible for
# that cleanup.  If a caller points at an unrelated non-empty directory, the
# runner works in a dedicated child instead of claiming or deleting the
# caller's files.
TOURNEY_OWNER_FILENAME = ".obliteratus-tourney-owner.json"
TOURNEY_OWNER_SCHEMA = 1
TOURNEY_OWNER_TOOL = "obliteratus.tourney"
TOURNEY_RUN_SUBDIR = ".obliteratus-tourney-run"


class TourneyOutputSafetyError(RuntimeError):
    """Raised when a tournament output path cannot be used safely."""


def _path_exists(path: Path) -> bool:
    """Return true for ordinary paths and dangling symlinks."""

    return path.exists() or path.is_symlink()


def _directory_is_empty(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is None


def _owner_marker(path: Path) -> dict[str, Any] | None:
    """Return a validated ownership marker, or ``None``.

    Binding the marker to the resolved directory prevents a copied marker from
    making some other directory eligible for recursive cleanup.
    """

    if path.is_symlink() or not path.is_dir():
        return None
    marker = path / TOURNEY_OWNER_FILENAME
    if marker.is_symlink() or not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("schema") != TOURNEY_OWNER_SCHEMA
        or payload.get("tool") != TOURNEY_OWNER_TOOL
        or payload.get("kind") not in {
            "exclusive-run-directory",
            "resume-run-directory",
        }
        or payload.get("resolved_path") != str(path.resolve())
        or not isinstance(payload.get("owner_id"), str)
        or not payload.get("owner_id")
        or not isinstance(payload.get("owned_children"), list)
        or not all(isinstance(name, str) for name in payload["owned_children"])
    ):
        return None
    return payload


def _exclusive_owner_marker(path: Path) -> dict[str, Any] | None:
    payload = _owner_marker(path)
    if payload is None or payload.get("kind") != "exclusive-run-directory":
        return None
    return payload


def _write_owner_marker(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist the run-directory ownership manifest."""

    marker = path / TOURNEY_OWNER_FILENAME
    temporary = path / f".{TOURNEY_OWNER_FILENAME}.{uuid.uuid4().hex}.tmp"
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, marker)


def _initialize_owned_run_dir(path: Path) -> None:
    if _path_exists(path) and (path.is_symlink() or not path.is_dir()):
        raise TourneyOutputSafetyError(
            f"Tournament output must be a real directory, not a file or symlink: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)
    if not _directory_is_empty(path):
        raise TourneyOutputSafetyError(
            f"Refusing to claim a non-empty directory as tournament-owned: {path}"
        )
    _write_owner_marker(
        path,
        {
            "schema": TOURNEY_OWNER_SCHEMA,
            "tool": TOURNEY_OWNER_TOOL,
            "kind": "exclusive-run-directory",
            "resolved_path": str(path.resolve()),
            "owner_id": uuid.uuid4().hex,
            "created_at": datetime.now().isoformat(),
            "owned_children": [],
        },
    )


def _initialize_resume_run_dir(path: Path) -> None:
    """Add a non-exclusive marker to a legacy checkpoint directory.

    This supports checkpoints written before ownership markers existed without
    ever making their containing directory eligible for recursive reset.
    Existing candidate directories are intentionally not claimed.
    """

    if path.is_symlink() or not path.is_dir():
        raise TourneyOutputSafetyError(
            f"Legacy tournament output must be a real directory: {path}"
        )
    marker = path / TOURNEY_OWNER_FILENAME
    if _path_exists(marker):
        raise TourneyOutputSafetyError(
            f"Refusing to replace an invalid or unowned marker file: {marker}"
        )
    _write_owner_marker(
        path,
        {
            "schema": TOURNEY_OWNER_SCHEMA,
            "tool": TOURNEY_OWNER_TOOL,
            "kind": "resume-run-directory",
            "resolved_path": str(path.resolve()),
            "owner_id": uuid.uuid4().hex,
            "created_at": datetime.now().isoformat(),
            "owned_children": [],
        },
    )


def _dangerous_cleanup_target(path: Path) -> bool:
    """Reject broad roots even if somebody has planted a marker in them."""

    resolved = path.resolve()
    home = Path.home().resolve()
    return resolved == Path(resolved.anchor) or resolved == home or len(resolved.parts) <= 2


def _reset_owned_run_dir(path: Path) -> None:
    """Clear and recreate a directory only after validating its marker."""

    if _exclusive_owner_marker(path) is None:
        raise TourneyOutputSafetyError(
            f"Refusing to clean tournament output without a valid ownership marker: {path}"
        )
    if _dangerous_cleanup_target(path):
        raise TourneyOutputSafetyError(
            f"Refusing recursive cleanup of a broad filesystem location: {path}"
        )
    shutil.rmtree(path)
    _initialize_owned_run_dir(path)


def _has_legacy_resume_checkpoint(path: Path) -> bool:
    checkpoint = path / "tourney_checkpoint.json"
    if checkpoint.is_symlink() or not checkpoint.is_file():
        return False
    try:
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and payload.get("version") == 1


def resolve_tourney_output_dir(
    output_dir: str | os.PathLike[str],
    *,
    resume: bool = False,
) -> Path:
    """Resolve a safe run directory without deleting or creating anything.

    New/empty paths and valid tool-owned paths are used directly.  A caller's
    unrelated non-empty directory is treated as a container, and tournament
    files go into a stable dedicated child.  If that child name is also
    occupied by unrelated data, a numbered sibling is selected.
    """

    requested = Path(output_dir).expanduser()
    if _path_exists(requested):
        if requested.is_symlink() or not requested.is_dir():
            raise TourneyOutputSafetyError(
                f"Tournament output must be a real directory, not a file or symlink: {requested}"
            )
        owner = _owner_marker(requested)
        if (
            _directory_is_empty(requested)
            or _exclusive_owner_marker(requested) is not None
            or (resume and owner is not None)
            or (resume and _has_legacy_resume_checkpoint(requested))
        ):
            return requested
    else:
        return requested

    index = 1
    while True:
        suffix = "" if index == 1 else f"-{index}"
        candidate = requested / f"{TOURNEY_RUN_SUBDIR}{suffix}"
        if not _path_exists(candidate):
            return candidate
        if candidate.is_symlink() or not candidate.is_dir():
            index += 1
            continue
        owner = _owner_marker(candidate)
        if (
            _directory_is_empty(candidate)
            or _exclusive_owner_marker(candidate) is not None
            or (resume and owner is not None)
            or (resume and _has_legacy_resume_checkpoint(candidate))
        ):
            return candidate
        index += 1

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def composite_score(metrics: dict[str, Any]) -> float:
    """Return accepted refusal-removal efficacy on [0, 1].

    Missing gate evidence is ineligible (``-1``), never a neutral score.
    Tournament ranking is lexicographic: this efficacy is maximized first and
    paired collateral damage is consulted only when refusal rates tie.
    """

    try:
        payload = validate_acceptance_payload(metrics.get("acceptance", {}))
    except (CandidateEvidenceError, TypeError, ValueError):
        return -1.0

    refusal_rate = float(payload["metrics"]["refusal_rate"])
    return 1.0 - min(1.0, max(0.0, refusal_rate))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Contender:
    """A single method's result in the tournament."""

    method: str
    score: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)
    output_dir: str = ""
    time_s: float = 0.0
    error: str | None = None
    round_eliminated: int = 0  # 0 = still alive / winner
    direction_method: str = ""  # which direction extraction was used
    spectral_cert: str = ""  # GREEN/YELLOW/RED/""


@dataclass
class TourneyRound:
    """One round of the tournament."""

    round_num: int
    name: str
    contenders: list[Contender] = field(default_factory=list)
    prompt_volume: int = 0
    advanced_to: list[str] = field(default_factory=list)
    eliminated: list[str] = field(default_factory=list)


@dataclass
class TourneyResult:
    """Full tournament results."""

    model: str
    winner: Contender | None = None
    rounds: list[TourneyRound] = field(default_factory=list)
    total_time_s: float = 0.0
    hub_repo: str | None = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "winner": {
                "method": self.winner.method,
                "score": self.winner.score,
                "metrics": self.winner.metrics,
                "time_s": self.winner.time_s,
            } if self.winner else None,
            "rounds": [
                {
                    "round": r.round_num,
                    "name": r.name,
                    "prompt_volume": r.prompt_volume,
                    "contenders": [
                        {
                            "method": c.method,
                            "score": c.score,
                            "metrics": c.metrics,
                            "time_s": c.time_s,
                            "error": c.error,
                            "direction_method": c.direction_method,
                            "spectral_cert": c.spectral_cert,
                        }
                        for c in sorted(r.contenders, key=_contender_rank_key)
                    ],
                    "advanced": r.advanced_to,
                    "eliminated": r.eliminated,
                }
                for r in self.rounds
            ],
            "total_time_s": self.total_time_s,
            "hub_repo": self.hub_repo,
            "timestamp": self.timestamp,
        }


def _contender_is_eligible(contender: Contender) -> bool:
    """Return whether a contender may advance or be crowned."""

    if contender.error is not None or contender.score < 0.0:
        return False
    try:
        validate_acceptance_payload(contender.metrics.get("acceptance", {}))
    except (CandidateEvidenceError, TypeError, ValueError):
        return False
    return True


def _contender_rank_key(contender: Contender) -> tuple[float, float, float, str]:
    """Put accepted candidates first, ordered by efficacy then damage."""

    if not _contender_is_eligible(contender):
        return (1.0, 1.0, float("inf"), contender.method)
    payload = validate_acceptance_payload(contender.metrics["acceptance"])
    return (
        0.0,
        float(payload["metrics"]["refusal_rate"]),
        damage_severity(payload),
        contender.method,
    )


def _rank_and_select(
    contenders: list[Contender],
    advance_count: int,
) -> tuple[list[Contender], list[Contender], list[Contender]]:
    """Rank all results while allowing only gate-accepted ones to advance."""

    ranked = sorted(contenders, key=_contender_rank_key)
    eligible = [contender for contender in ranked if _contender_is_eligible(contender)]
    advanced = eligible[: max(0, advance_count)]
    advanced_ids = {id(contender) for contender in advanced}
    eliminated = [contender for contender in ranked if id(contender) not in advanced_ids]
    return ranked, advanced, eliminated


CHECKPOINT_FILENAME = "tourney_checkpoint.json"


def _save_checkpoint(
    output_dir: Path,
    result: TourneyResult,
    current_round_num: int,
    current_round_name: str,
    current_round_volume: int,
    current_round_advance: int,
    current_round_verify: int,
    completed_methods: list[Contender],
    remaining_methods: list[str],
    alive: list[str],
    model_name: str,
    dataset_key: str,
    quantization: str | None,
    methods: list[str],
) -> Path:
    """Save tournament progress so it can be resumed after quota exhaustion."""
    checkpoint = {
        "version": 1,
        "model": model_name,
        "dataset_key": dataset_key,
        "quantization": quantization,
        "methods": methods,
        "alive": alive,
        "completed_rounds": [
            {
                "round_num": r.round_num,
                "name": r.name,
                "prompt_volume": r.prompt_volume,
                "advanced_to": r.advanced_to,
                "eliminated": r.eliminated,
                "contenders": [
                    {
                        "method": c.method,
                        "score": c.score,
                        "metrics": c.metrics,
                        "output_dir": c.output_dir,
                        "time_s": c.time_s,
                        "error": c.error,
                        "round_eliminated": c.round_eliminated,
                        "direction_method": c.direction_method,
                        "spectral_cert": c.spectral_cert,
                    }
                    for c in r.contenders
                ],
            }
            for r in result.rounds
        ],
        "interrupted_round": {
            "round_num": current_round_num,
            "name": current_round_name,
            "prompt_volume": current_round_volume,
            "advance_count": current_round_advance,
            "verify_sample_size": current_round_verify,
            "completed_methods": [
                {
                    "method": c.method,
                    "score": c.score,
                    "metrics": c.metrics,
                    "output_dir": c.output_dir,
                    "time_s": c.time_s,
                    "error": c.error,
                    "round_eliminated": c.round_eliminated,
                    "direction_method": c.direction_method,
                    "spectral_cert": c.spectral_cert,
                }
                for c in completed_methods
            ],
            "remaining_methods": remaining_methods,
        },
        "timestamp": datetime.now().isoformat(),
    }
    path = output_dir / CHECKPOINT_FILENAME
    path.write_text(json.dumps(checkpoint, indent=2))
    return path


def _load_checkpoint(output_dir: Path) -> dict | None:
    """Load a tournament checkpoint if one exists. Returns None if absent or corrupt."""
    path = output_dir / CHECKPOINT_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("version") != 1:
            return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


def _checkpoint_matches(
    checkpoint: dict,
    model_name: str,
    dataset_key: str,
    quantization: str | None,
) -> bool:
    """Check if a checkpoint is for the same model/dataset/quantization config."""
    return (
        checkpoint.get("model") == model_name
        and checkpoint.get("dataset_key") == dataset_key
        and checkpoint.get("quantization") == quantization
    )


def _restore_rounds(checkpoint: dict) -> tuple[TourneyResult, list[Contender], list[str], dict]:
    """Restore completed rounds and interrupted round state from checkpoint.

    Returns:
        (result_with_completed_rounds, partial_contenders, remaining_methods, interrupted_round_spec)
    """
    result = TourneyResult(
        model=checkpoint["model"],
        timestamp=checkpoint.get("timestamp", ""),
    )

    for rnd_data in checkpoint.get("completed_rounds", []):
        rnd = TourneyRound(
            round_num=rnd_data["round_num"],
            name=rnd_data["name"],
            prompt_volume=rnd_data.get("prompt_volume", 0),
            advanced_to=rnd_data.get("advanced_to", []),
            eliminated=rnd_data.get("eliminated", []),
        )
        for c_data in rnd_data.get("contenders", []):
            restored_metrics = c_data.get("metrics", {})
            rnd.contenders.append(Contender(
                method=c_data["method"],
                score=composite_score(restored_metrics),
                metrics=restored_metrics,
                output_dir=c_data.get("output_dir", ""),
                time_s=c_data.get("time_s", 0.0),
                error=c_data.get("error"),
                round_eliminated=c_data.get("round_eliminated", 0),
                direction_method=c_data.get("direction_method", ""),
                spectral_cert=c_data.get("spectral_cert", ""),
            ))
        eligible_methods = {
            contender.method
            for contender in rnd.contenders
            if _contender_is_eligible(contender)
        }
        rnd.advanced_to = [
            method for method in rnd.advanced_to if method in eligible_methods
        ]
        rnd.eliminated = [
            contender.method
            for contender in rnd.contenders
            if contender.method not in rnd.advanced_to
        ]
        result.rounds.append(rnd)

    ir = checkpoint.get("interrupted_round", {})
    partial_contenders = []
    for c_data in ir.get("completed_methods", []):
        restored_metrics = c_data.get("metrics", {})
        partial_contenders.append(Contender(
            method=c_data["method"],
            score=composite_score(restored_metrics),
            metrics=restored_metrics,
            output_dir=c_data.get("output_dir", ""),
            time_s=c_data.get("time_s", 0.0),
            error=c_data.get("error"),
            round_eliminated=c_data.get("round_eliminated", 0),
        ))

    remaining = ir.get("remaining_methods", [])

    return result, partial_contenders, remaining, ir


# ---------------------------------------------------------------------------
# Bracket renderer
# ---------------------------------------------------------------------------


def render_bracket(result: TourneyResult) -> str:
    """Render the tournament bracket as a markdown string."""
    lines = []
    lines.append(f"# OBLITERATUS TOURNEY — {result.model}")
    lines.append("")
    lines.append(f"**Winner: `{result.winner.method}`** "
                 f"(score: {result.winner.score:.4f})" if result.winner else "**No winner**")
    lines.append(f"Total time: {result.total_time_s / 60:.1f} minutes")
    if result.hub_repo:
        lines.append(f"Pushed to: [{result.hub_repo}](https://huggingface.co/{result.hub_repo})")
    lines.append("")

    for rnd in result.rounds:
        lines.append(f"## Round {rnd.round_num}: {rnd.name}")
        lines.append(f"*{len(rnd.contenders)} contenders, {rnd.prompt_volume} prompt pairs*")
        lines.append("")
        lines.append("| Rank | Method | Dir | Score | Refusal | Coherence | KL Div | PPL | Cert | Time |")
        lines.append("|------|--------|-----|-------|---------|-----------|--------|-----|------|------|")

        sorted_contenders = sorted(rnd.contenders, key=_contender_rank_key)
        for i, c in enumerate(sorted_contenders, 1):
            if c.error:
                lines.append(
                    f"| {i} | {c.method} | — | ERROR | — | — | — | — | — | {c.time_s:.0f}s |"
                )
                continue
            m = c.metrics
            # Only annotate elimination for non-final rounds
            if c.method in rnd.advanced_to:
                marker = ""
            elif rnd.round_num < len(result.rounds):
                marker = " *out*"
            else:
                marker = ""
            rr = f"{m.get('refusal_rate', 0):.1%}" if m.get('refusal_rate') is not None else "—"
            co = f"{m.get('coherence', 0):.3f}" if m.get('coherence') is not None else "—"
            kl_val = m.get('kl_divergence')
            kl_str = f"{kl_val:.4f}" if kl_val is not None else "—"
            pp = f"{m.get('perplexity', 0):.1f}" if m.get('perplexity') is not None else "—"
            dir_m = c.direction_method or m.get("direction_method", "—")
            cert = c.spectral_cert or "—"
            lines.append(
                f"| {i} | **{c.method}**{marker} | {dir_m} | {c.score:.4f} "
                f"| {rr} | {co} | {kl_str} | {pp} | {cert} | {c.time_s:.0f}s |"
            )
        lines.append("")

    return "\n".join(lines)


def render_bracket_html(result: TourneyResult) -> str:
    """Render the tournament bracket as a styled HTML bracket visualization."""
    import html as html_mod

    model_short = result.model.split("/")[-1] if "/" in result.model else result.model

    # ── CSS ──────────────────────────────────────────────────────────────
    css = """
    <style>
    .tourney-wrap {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: #e0e0e0;
        max-width: 100%;
        overflow-x: auto;
    }
    .tourney-header {
        text-align: center;
        padding: 18px 20px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        border: 1px solid #333;
    }
    .tourney-header h2 {
        margin: 0 0 4px 0;
        font-size: 1.4em;
        color: #fff;
        letter-spacing: 1px;
    }
    .tourney-header .model-name {
        font-size: 0.85em;
        color: #8892b0;
        font-family: 'Courier New', monospace;
    }
    .tourney-header .champion-box {
        margin-top: 14px;
        padding: 12px 18px;
        background: linear-gradient(135deg, #2d1f00 0%, #3d2a00 100%);
        border: 1px solid #f0c040;
        border-radius: 8px;
        display: inline-block;
    }
    .tourney-header .champion-box .trophy { font-size: 1.4em; }
    .tourney-header .champion-box .champ-name {
        font-size: 1.15em;
        font-weight: 700;
        color: #f0c040;
        font-family: 'Courier New', monospace;
    }
    .tourney-header .champion-box .champ-score {
        font-size: 0.85em;
        color: #cca030;
        margin-top: 2px;
    }
    .tourney-header .no-winner {
        margin-top: 14px;
        padding: 10px 16px;
        background: #2a1a1a;
        border: 1px solid #cc4444;
        border-radius: 8px;
        display: inline-block;
        color: #ff6b6b;
        font-weight: 600;
    }
    .tourney-header .time-info {
        font-size: 0.78em;
        color: #666;
        margin-top: 8px;
    }

    /* ── Bracket flow ── */
    .bracket-flow {
        display: flex;
        gap: 12px;
        align-items: stretch;
        padding: 4px 0;
    }
    .round-col {
        flex: 1;
        min-width: 200px;
        max-width: 340px;
    }
    .round-title {
        text-align: center;
        font-size: 0.82em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #8892b0;
        padding: 6px 0 8px 0;
        border-bottom: 2px solid #333;
        margin-bottom: 8px;
    }
    .round-subtitle {
        text-align: center;
        font-size: 0.7em;
        color: #555;
        margin-top: 2px;
    }

    /* ── Method cards ── */
    .method-card {
        padding: 8px 10px;
        margin: 4px 0;
        border-radius: 6px;
        border-left: 3px solid #444;
        background: #1c1c2e;
        transition: all 0.2s;
    }
    .method-card.advanced {
        border-left-color: #4ecca3;
        background: #1a2e28;
    }
    .method-card.champion {
        border-left-color: #f0c040;
        background: #2d2a1a;
        box-shadow: 0 0 8px rgba(240, 192, 64, 0.15);
    }
    .method-card.eliminated {
        border-left-color: #cc4444;
        background: #1e1a1a;
        opacity: 0.7;
    }
    .method-card.errored {
        border-left-color: #ff4444;
        background: #2a1a1a;
        opacity: 0.6;
    }
    .card-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .card-rank {
        font-size: 0.7em;
        color: #666;
        font-weight: 700;
        min-width: 18px;
    }
    .card-name {
        font-weight: 600;
        font-size: 0.88em;
        font-family: 'Courier New', monospace;
        flex: 1;
        margin: 0 6px;
    }
    .card-score {
        font-weight: 700;
        font-size: 0.88em;
        font-family: 'Courier New', monospace;
    }
    .card-score.good { color: #4ecca3; }
    .card-score.mid { color: #f0c040; }
    .card-score.bad { color: #cc4444; }
    .card-metrics {
        display: flex;
        gap: 8px;
        margin-top: 4px;
        flex-wrap: wrap;
    }
    .metric {
        font-size: 0.68em;
        color: #777;
    }
    .metric .val {
        color: #aaa;
        font-family: 'Courier New', monospace;
    }
    .card-badge {
        font-size: 0.65em;
        font-weight: 700;
        padding: 1px 5px;
        border-radius: 3px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-adv { background: #1a3a2e; color: #4ecca3; }
    .badge-out { background: #2a1a1a; color: #cc6666; }
    .badge-champ { background: #3d2a00; color: #f0c040; }
    .badge-err { background: #2a1a1a; color: #ff6666; }

    /* ── Arrow column ── */
    .arrow-col {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        width: 30px;
        min-width: 30px;
        color: #444;
        font-size: 1.2em;
    }
    </style>
    """

    # ── Header ───────────────────────────────────────────────────────────
    header_parts = [
        '<div class="tourney-header">',
        '<h2>OBLITERATUS TOURNEY</h2>',
        f'<div class="model-name">{html_mod.escape(model_short)}</div>',
    ]

    if result.winner and not result.winner.error:
        w = result.winner
        m = w.metrics or {}
        rr = f"{m.get('refusal_rate', 0):.1%}" if m.get("refusal_rate") is not None else "—"
        co = f"{m.get('coherence', 0):.3f}" if m.get("coherence") is not None else "—"
        header_parts.append('<div class="champion-box">')
        header_parts.append(
            f'<span class="trophy">&#x1F3C6;</span> '
            f'<span class="champ-name">{html_mod.escape(w.method)}</span>'
        )
        dir_m = w.direction_method or "—"
        cert = w.spectral_cert or "—"
        header_parts.append(
            f'<div class="champ-score">'
            f'Score: {w.score:.4f} &nbsp;|&nbsp; Refusal: {rr} &nbsp;|&nbsp; '
            f'Coherence: {co} &nbsp;|&nbsp; Dir: {html_mod.escape(dir_m)} &nbsp;|&nbsp; Cert: {html_mod.escape(cert)}'
            f'</div>'
        )
        header_parts.append("</div>")
    else:
        header_parts.append('<div class="no-winner">No winner determined</div>')

    if result.total_time_s:
        header_parts.append(
            f'<div class="time-info">{result.total_time_s / 60:.1f} min total</div>'
        )
    header_parts.append("</div>")

    # ── Bracket columns ──────────────────────────────────────────────────
    bracket_parts = ['<div class="bracket-flow">']
    n_rounds = len(result.rounds)

    for ri, rnd in enumerate(result.rounds):
        if ri > 0:
            bracket_parts.append('<div class="arrow-col">&#x25B6;</div>')

        bracket_parts.append('<div class="round-col">')
        bracket_parts.append(
            f'<div class="round-title">{html_mod.escape(rnd.name)}'
            f'<div class="round-subtitle">{rnd.prompt_volume} pairs</div></div>'
        )

        sorted_c = sorted(rnd.contenders, key=_contender_rank_key)
        is_final = ri == n_rounds - 1

        for rank, c in enumerate(sorted_c, 1):
            if c.error:
                css_cls = "errored"
                badge = '<span class="card-badge badge-err">ERR</span>'
            elif is_final and rank == 1 and result.winner and not result.winner.error:
                css_cls = "champion"
                badge = '<span class="card-badge badge-champ">&#x2605; CHAMP</span>'
            elif c.method in (rnd.advanced_to or []):
                css_cls = "advanced"
                badge = '<span class="card-badge badge-adv">ADV</span>'
            else:
                css_cls = "eliminated"
                badge = '<span class="card-badge badge-out">OUT</span>'

            # Score color
            if c.error:
                score_html = '<span class="card-score bad">ERR</span>'
            elif c.score >= 0.7:
                score_html = f'<span class="card-score good">{c.score:.4f}</span>'
            elif c.score >= 0.4:
                score_html = f'<span class="card-score mid">{c.score:.4f}</span>'
            else:
                score_html = f'<span class="card-score bad">{c.score:.4f}</span>'

            # Compact metrics
            m = c.metrics or {}
            metric_spans = []
            if not c.error:
                dm = c.direction_method or m.get("direction_method", "")
                if dm:
                    metric_spans.append(
                        f'<span class="metric">dir <span class="val">{html_mod.escape(dm)}</span></span>'
                    )
                rr = m.get("refusal_rate")
                if rr is not None:
                    metric_spans.append(
                        f'<span class="metric">ref <span class="val">{rr:.0%}</span></span>'
                    )
                co = m.get("coherence")
                if co is not None:
                    metric_spans.append(
                        f'<span class="metric">coh <span class="val">{co:.3f}</span></span>'
                    )
                sc = c.spectral_cert or m.get("spectral_certification", "")
                if sc:
                    cert_color = {"GREEN": "#4ecca3", "YELLOW": "#f0c040", "RED": "#cc4444"}.get(sc, "#777")
                    metric_spans.append(
                        f'<span class="metric">cert <span class="val" style="color:{cert_color}">{html_mod.escape(sc)}</span></span>'
                    )
                kl = m.get("kl_divergence")
                if kl is not None:
                    metric_spans.append(
                        f'<span class="metric">kl <span class="val">{kl:.4f}</span></span>'
                    )
                pp = m.get("perplexity")
                if pp is not None:
                    metric_spans.append(
                        f'<span class="metric">ppl <span class="val">{pp:.1f}</span></span>'
                    )
            metrics_html = "".join(metric_spans)

            bracket_parts.append(f'<div class="method-card {css_cls}">')
            bracket_parts.append(
                f'<div class="card-top">'
                f'<span class="card-rank">#{rank}</span>'
                f'<span class="card-name">{html_mod.escape(c.method)}</span>'
                f'{score_html}'
                f'{badge}'
                f'</div>'
            )
            if metrics_html:
                bracket_parts.append(f'<div class="card-metrics">{metrics_html}</div>')
            bracket_parts.append("</div>")

        bracket_parts.append("</div>")

    bracket_parts.append("</div>")

    return css + '<div class="tourney-wrap">' + "\n".join(header_parts + bracket_parts) + "</div>"


def generate_model_card(result: TourneyResult) -> str:
    """Generate a HuggingFace model card for the tournament winner."""
    w = result.winner
    if not w:
        return ""

    short_model = result.model.split("/")[-1] if "/" in result.model else result.model
    bracket = render_bracket(result)

    return f"""---
language: en
tags:
  - obliteratus
  - abliteration
  - uncensored
  - tourney
base_model: {result.model}
---

# {short_model} — Obliterated (Tourney Winner)

This model was abliterated using the **`{w.method}`** method, selected by an
automated [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) tournament
that pitted **{len(TOURNEY_METHODS)} abliteration techniques** against each other
in elimination rounds.

## Winning Method: `{w.method}`

| Metric | Value |
|--------|-------|
| Refusal-removal efficacy | **{w.score:.4f}** |
| Direction Method | {w.direction_method or 'N/A'} |
| Refusal Rate | {f'{w.metrics["refusal_rate"]:.1%}' if w.metrics.get('refusal_rate') is not None else 'N/A'} |
| Coherence | {f'{w.metrics["coherence"]:.3f}' if w.metrics.get('coherence') is not None else 'N/A'} |
| KL Divergence | {f'{w.metrics["kl_divergence"]:.4f}' if w.metrics.get('kl_divergence') is not None else 'N/A'} |
| Perplexity | {f'{w.metrics["perplexity"]:.1f}' if w.metrics.get('perplexity') is not None else 'N/A'} |
| Spectral Cert | {w.spectral_cert or 'N/A'} |

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{result.hub_repo or 'this-repo'}")
tokenizer = AutoTokenizer.from_pretrained("{result.hub_repo or 'this-repo'}")
```

## Full Tournament Bracket

{bracket}

---

*Generated by [OBLITERATUS](https://github.com/elder-plinius/OBLITERATUS) tourney on {result.timestamp}*
"""


def _noop_log(msg: str) -> None:
    """Picklable no-op log callback (lambdas can't be pickled by ZeroGPU)."""
    pass


def _noop_round(r: TourneyRound) -> None:
    """Picklable no-op round callback."""
    pass


class _MethodLogger:
    """Picklable per-method log adapter that prefixes messages.

    ZeroGPU pickles bound methods (and their ``self``) when shipping work to
    the GPU worker process.  Plain lambdas like
    ``lambda msg: self.log(f"  [{method}] {msg}")`` can't survive that, so
    this small class replaces them.
    """

    def __init__(self, parent_log: Callable[[str], None], method: str):
        self._parent = parent_log
        self._method = method

    def __call__(self, msg: str):
        self._parent(f"  [{self._method}] {msg}")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


class TourneyRunner:
    """Run a March Madness-style tournament across all abliteration methods.

    Round 1 — Qualifiers:  All methods, reduced prompts.  Bottom half eliminated.
    Round 2 — Semifinals:  Survivors, full prompts.  Bottom half eliminated.
    Round 3 — Finals:      Top contenders, maximum prompts + extended verify.
    Winner  — Auto-pushed to HuggingFace Hub (if hub_org is set).
    """

    def __init__(
        self,
        model_name: str,
        hub_org: str | None = None,
        hub_repo: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        dataset_key: str = "builtin",
        quantization: str | None = None,
        methods: list[str] | None = None,
        output_dir: str = "/tmp/obliteratus_tourney",
        on_log: Callable[[str], None] | None = None,
        on_round: Callable[[TourneyRound], None] | None = None,
        resume: bool = False,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self.hub_org = hub_org
        self.hub_repo = hub_repo
        self.device = device
        self.dtype = dtype
        self.dataset_key = dataset_key
        self.quantization = quantization
        self.methods = validate_tourney_methods(methods)
        self.requested_output_dir = Path(output_dir).expanduser()
        self.output_dir = resolve_tourney_output_dir(
            self.requested_output_dir,
            resume=resume,
        )
        self.resume = resume
        self.trust_remote_code = trust_remote_code

        # A pre-marker checkpoint can still be resumed in place.  Only adopt it
        # as a non-exclusive run when its configuration matches this runner;
        # otherwise route the fresh work to a safe owned directory.
        owner = _owner_marker(self.output_dir)
        if resume and owner is None and _has_legacy_resume_checkpoint(self.output_dir):
            checkpoint = _load_checkpoint(self.output_dir)
            marker_available = not _path_exists(
                self.output_dir / TOURNEY_OWNER_FILENAME
            )
            if marker_available and checkpoint is not None and _checkpoint_matches(
                checkpoint,
                self.model_name,
                self.dataset_key,
                self.quantization,
            ):
                _initialize_resume_run_dir(self.output_dir)
                owner = _owner_marker(self.output_dir)
            else:
                self.output_dir = resolve_tourney_output_dir(
                    self.requested_output_dir,
                    resume=False,
                )
                owner = _owner_marker(self.output_dir)

        # A fresh run may only reset a directory that this tool created and
        # marked as exclusively owned.  An unrelated non-empty path is handled
        # by ``resolve_tourney_output_dir`` via a dedicated child directory.
        # Resume never cleans its run directory.
        if resume and owner is not None:
            pass
        elif not resume and _exclusive_owner_marker(self.output_dir) is not None:
            _reset_owned_run_dir(self.output_dir)
        elif not _path_exists(self.output_dir) or _directory_is_empty(self.output_dir):
            _initialize_owned_run_dir(self.output_dir)
        else:  # Defend against a path replacement between resolve and init.
            raise TourneyOutputSafetyError(
                "Tournament output became non-empty before it could be claimed; "
                f"no files were removed: {self.output_dir}"
            )
        self._on_log = on_log or _noop_log
        self._on_round = on_round or _noop_round

    def log(self, msg: str):
        self._on_log(msg)

    def _register_candidate_dir(self, candidate: Path) -> Path:
        """Record a direct child as tool-owned before a method may write it."""

        owner = _owner_marker(self.output_dir)
        if owner is None:
            raise TourneyOutputSafetyError(
                f"Tournament run directory lost its ownership marker: {self.output_dir}"
            )
        resolved_root = self.output_dir.resolve()
        resolved_candidate = candidate.resolve()
        if resolved_candidate.parent != resolved_root or candidate.is_symlink():
            raise TourneyOutputSafetyError(
                f"Candidate output must be a direct child of the tournament run: {candidate}"
            )

        name = resolved_candidate.name
        owned_children = list(owner["owned_children"])
        if _path_exists(candidate) and name not in owned_children:
            raise TourneyOutputSafetyError(
                f"Refusing to replace unowned candidate output: {candidate}"
            )
        if name not in owned_children:
            owned_children.append(name)
            owner["owned_children"] = owned_children
            _write_owner_marker(self.output_dir, owner)
        return candidate

    def _candidate_dir(self, round_num: int, method: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", method):
            raise TourneyOutputSafetyError(
                f"Candidate method is not a safe directory component: {method!r}"
            )
        return self._register_candidate_dir(
            self.output_dir / f"r{round_num}_{method}"
        )

    def _remove_owned_candidate_dir(self, candidate: str | os.PathLike[str]) -> bool:
        """Remove a candidate only when the root manifest proves ownership."""

        path = Path(candidate)
        owner = _owner_marker(self.output_dir)
        if owner is None:
            self.log(f"  Skipping cleanup without run ownership marker: {path}")
            return False
        try:
            relative = path.resolve().relative_to(self.output_dir.resolve())
        except (OSError, ValueError):
            self.log(f"  Skipping cleanup outside tournament output: {path}")
            return False
        if len(relative.parts) != 1 or relative.name not in owner["owned_children"]:
            self.log(f"  Skipping cleanup of unowned candidate output: {path}")
            return False
        if not _path_exists(path):
            return False
        if path.is_symlink() or not path.is_dir():
            self.log(f"  Skipping unsafe candidate cleanup target: {path}")
            return False

        shutil.rmtree(path)
        owner["owned_children"] = [
            name for name in owner["owned_children"] if name != relative.name
        ]
        _write_owner_marker(self.output_dir, owner)
        return True

    def _load_prompts(self, volume: int) -> tuple[list[str], list[str]]:
        from obliteratus.prompts import load_dataset_source
        harmful, harmless = load_dataset_source(self.dataset_key)
        n = min(volume, len(harmful), len(harmless))
        return harmful[:n], harmless[:n]

    def _load_prompt_sets(
        self,
        volume: int,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Return disjoint direction-discovery and locked evaluation pairs."""

        from obliteratus.evaluation.prompt_split import split_prompt_pairs
        from obliteratus.prompts import load_dataset_source

        harmful, harmless = load_dataset_source(self.dataset_key)
        split = split_prompt_pairs(
            harmful,
            harmless,
            holdout_fraction=0.15,
            seed=42,
            min_holdout=32,
        )
        n = min(
            volume,
            len(split.discovery_harmful),
            len(split.discovery_harmless),
        )
        return (
            list(split.discovery_harmful[:n]),
            list(split.discovery_harmless[:n]),
            list(split.holdout_harmful),
            list(split.holdout_harmless),
        )

    def _run_method(
        self,
        method: str,
        harmful: list[str],
        harmless: list[str],
        save_dir: str,
        verify_sample_size: int = 30,
        evaluation_harmful: list[str] | None = None,
        evaluation_harmless: list[str] | None = None,
    ) -> Contender:
        """Run a single abliteration method and return its Contender result."""
        import torch

        t0 = time.time()
        contender = Contender(method=method)

        try:
            # Use informed pipeline for 'informed' method
            method_log = _MethodLogger(self._on_log, method)

            if method == "informed":
                from obliteratus.informed_pipeline import InformedAbliterationPipeline
                pipeline = InformedAbliterationPipeline(
                    model_name=self.model_name,
                    output_dir=save_dir,
                    device=self.device,
                    dtype=self.dtype,
                    quantization=self.quantization,
                    trust_remote_code=self.trust_remote_code,
                    harmful_prompts=harmful,
                    harmless_prompts=harmless,
                    evaluation_harmful_prompts=evaluation_harmful,
                    evaluation_harmless_prompts=evaluation_harmless,
                    damage_gate_enabled=True,
                    on_log=method_log,
                )
                pipeline.run_informed()
            else:
                from obliteratus.abliterate import AbliterationPipeline
                pipeline = AbliterationPipeline(
                    model_name=self.model_name,
                    output_dir=save_dir,
                    device=self.device,
                    dtype=self.dtype,
                    method=method,
                    quantization=self.quantization,
                    trust_remote_code=self.trust_remote_code,
                    harmful_prompts=harmful,
                    harmless_prompts=harmless,
                    evaluation_harmful_prompts=evaluation_harmful,
                    evaluation_harmless_prompts=evaluation_harmless,
                    damage_gate_enabled=True,
                    verify_sample_size=verify_sample_size,
                    on_log=method_log,
                )
                pipeline.run()

            contender.metrics = add_acceptance_evidence(
                getattr(pipeline, "_quality_metrics", {}) or {},
                getattr(pipeline, "_damage_assessment", None),
            )
            contender.score = composite_score(contender.metrics)
            if contender.score < 0.0:
                raise CandidateEvidenceError(
                    "candidate did not produce complete accepted damage evidence"
                )
            contender.output_dir = save_dir
            contender.direction_method = getattr(pipeline, "direction_method", "")
            contender.spectral_cert = contender.metrics.get("spectral_certification", "") or ""

            # Free pipeline to reclaim GPU
            del pipeline
        except Exception as e:
            # Re-raise GPU quota / expired-token errors so the
            # tournament aborts immediately rather than letting every
            # remaining method fail for the same reason.
            if self._is_quota_error(e):
                raise
            import traceback
            contender.error = f"{type(e).__name__}: {e}"
            contender.score = -1.0  # errors sort to bottom
            self.log(f"  [{method}] ERROR: {contender.error}")
            self.log(f"  [{method}] TRACEBACK:\n{traceback.format_exc()}")
        finally:
            # Always clean up GPU between methods — including when
            # re-raising quota errors, to avoid leaking the pipeline.
            gc.collect()
            try:
                from obliteratus import device as dev
                dev.empty_cache()
            except Exception:
                pass

        contender.time_s = time.time() - t0
        return contender

    def _run_round(
        self,
        round_num: int,
        name: str,
        methods: list[str],
        prompt_volume: int,
        advance_count: int,
        verify_sample_size: int = 30,
    ) -> TourneyRound:
        """Execute one round of the tournament."""
        self.log("")
        self.log("=" * 60)
        self.log(f"ROUND {round_num}: {name}")
        self.log(f"  {len(methods)} contenders | {prompt_volume} prompt pairs | "
                 f"top {advance_count} advance")
        self.log("=" * 60)

        harmful, harmless, evaluation_harmful, evaluation_harmless = (
            self._load_prompt_sets(prompt_volume)
        )

        rnd = TourneyRound(
            round_num=round_num,
            name=name,
            prompt_volume=prompt_volume,
        )

        for i, method in enumerate(methods, 1):
            self.log(f"\n[{i}/{len(methods)}] Running: {method}")
            save_dir = str(self._candidate_dir(round_num, method))
            contender = self._run_method(
                method,
                harmful,
                harmless,
                save_dir,
                verify_sample_size,
                evaluation_harmful,
                evaluation_harmless,
            )
            rnd.contenders.append(contender)
            self.log(
                f"  {method}: score={contender.score:.4f} "
                f"(refusal={contender.metrics.get('refusal_rate', '?')}, "
                f"coherence={contender.metrics.get('coherence', '?')}) "
                f"[{contender.time_s:.0f}s]"
            )

            # Free checkpoint for non-finalists as we go (save disk)
            # We'll keep them until we know who advances

        # Rank by score
        ranked, advanced, eliminated = _rank_and_select(
            rnd.contenders,
            advance_count,
        )
        rnd.advanced_to = [c.method for c in advanced]
        rnd.eliminated = [c.method for c in eliminated]

        # Mark eliminated
        for c in eliminated:
            c.round_eliminated = round_num

        self.log(f"\n{'─' * 40}")
        self.log(f"Round {round_num} results:")
        for i, c in enumerate(ranked, 1):
            status = "ADVANCE" if c.method in rnd.advanced_to else "OUT"
            self.log(f"  {i}. {c.method}: {c.score:.4f} [{status}]")

        # Clean up eliminated checkpoints to free disk
        for c in eliminated:
            if c.output_dir:
                self._remove_owned_candidate_dir(c.output_dir)

        self._on_round(rnd)
        return rnd

    def run(self) -> TourneyResult:
        """Execute the full tournament. Returns TourneyResult with winner."""
        t_start = time.time()
        result = TourneyResult(
            model=self.model_name,
            timestamp=datetime.now().isoformat(),
        )

        n_methods = len(self.methods)
        self.log(f"OBLITERATUS TOURNEY")
        self.log(f"Model: {self.model_name}")
        self.log(f"Contenders: {n_methods} methods")
        self.log(f"Dataset: {self.dataset_key}")

        # Pre-flight disk space check
        try:
            disk = shutil.disk_usage(self.output_dir)
            free_gb = disk.free / 1e9
            self.log(f"Disk space: {free_gb:.1f} GB free on {self.output_dir}")
            if free_gb < 5.0:
                self.log(
                    f"WARNING: Low disk space ({free_gb:.1f} GB free). "
                    f"Tournament may fail saving checkpoints."
                )
        except Exception:
            pass

        # ── Round 1: Qualifiers — all methods, reduced prompts ────────
        r1_advance = max(2, math.ceil(n_methods / 2))
        r1 = self._run_round(
            round_num=1,
            name="Qualifiers",
            methods=self.methods,
            prompt_volume=64,       # fast qualifier round
            advance_count=r1_advance,
            verify_sample_size=30,
        )
        result.rounds.append(r1)
        alive = list(r1.advanced_to)

        if len(alive) <= 1:
            # Only 1 survivor — they win
            pass
        else:
            # ── Round 2: Semifinals — survivors, full prompts ─────────
            r2_advance = max(2, math.ceil(len(alive) / 2))
            r2 = self._run_round(
                round_num=2,
                name="Semifinals",
                methods=alive,
                prompt_volume=128,
                advance_count=r2_advance,
                verify_sample_size=30,
            )
            result.rounds.append(r2)
            alive = list(r2.advanced_to)

            if len(alive) > 2:
                # ── Round 3: Finals — top contenders, max prompts ─────
                r3 = self._run_round(
                    round_num=3,
                    name="Finals",
                    methods=alive,
                    prompt_volume=256,
                    advance_count=1,
                    verify_sample_size=50,
                )
                result.rounds.append(r3)
                alive = list(r3.advanced_to)
            elif len(alive) == 2:
                # Head-to-head final
                r3 = self._run_round(
                    round_num=3,
                    name="Championship",
                    methods=alive,
                    prompt_volume=256,
                    advance_count=1,
                    verify_sample_size=50,
                )
                result.rounds.append(r3)
                alive = list(r3.advanced_to)

        # ── Determine winner ──────────────────────────────────────────
        last_round = result.rounds[-1]
        ranked, eligible_finalists, _ = _rank_and_select(
            last_round.contenders,
            len(last_round.contenders),
        )
        winner = eligible_finalists[0] if eligible_finalists else None
        result.winner = winner
        result.total_time_s = time.time() - t_start

        # Clean up non-winner finalist dirs to free disk
        for c in ranked:
            if c is not winner and c.output_dir:
                self._remove_owned_candidate_dir(c.output_dir)

        self.log("")
        self.log("=" * 60)
        if winner:
            self.log(f"CHAMPION: {winner.method} (score: {winner.score:.4f})")
        else:
            n_errors = sum(1 for c in ranked if c.error)
            self.log(f"NO WINNER — {n_errors}/{len(ranked)} methods errored")
        self.log(f"Total tournament time: {result.total_time_s / 60:.1f} minutes")
        self.log("=" * 60)

        # ── Save tournament results ───────────────────────────────────
        results_path = self.output_dir / "tourney_results.json"
        results_path.write_text(json.dumps(result.to_dict(), indent=2))
        self.log(f"Results saved to {results_path}")

        bracket_path = self.output_dir / "tourney_bracket.md"
        bracket_path.write_text(render_bracket(result))
        self.log(f"Bracket saved to {bracket_path}")

        # ── Push winner to HuggingFace Hub ────────────────────────────
        if winner and winner.output_dir and (self.hub_org or self.hub_repo):
            self._push_winner(result)

        return result

    @staticmethod
    def _is_quota_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        if "exceeded" in msg and "gpu quota" in msg:
            return True
        if "expired" in msg and "zerogpu" in msg:
            return True
        return False

    def _run_one_method(
        self,
        method,
        harmful,
        harmless,
        save_dir,
        verify_sz,
        gpu_wrapper,
        evaluation_harmful=None,
        evaluation_harmless=None,
    ):
        """Run a single method, optionally inside a gpu_wrapper."""
        if gpu_wrapper is not None:
            return gpu_wrapper(
                self._run_method, method, harmful, harmless,
                save_dir, verify_sz, evaluation_harmful, evaluation_harmless,
            )
        return self._run_method(
            method,
            harmful,
            harmless,
            save_dir,
            verify_sz,
            evaluation_harmful,
            evaluation_harmless,
        )

    def run_iter(self, gpu_wrapper=None):
        """Generator version of run() — yields (status, result_so_far) after each method.

        Supports automatic resume: if ``self.resume`` is True and a valid
        checkpoint exists from a previous quota-interrupted run with the
        same model/dataset/quantization, completed rounds and methods are
        restored and execution continues from the interruption point.

        When a GPU quota error occurs, a checkpoint is saved to disk and
        the exception is re-raised.  The caller can catch it and inform
        the user that clicking **Run** again will resume automatically.

        Args:
            gpu_wrapper: Optional callable ``gpu_wrapper(fn, *args, **kw)``
                that executes *fn* inside a GPU context.  On ZeroGPU Spaces
                this should be a ``@spaces.GPU``-decorated function so each
                method gets its own GPU allocation (up to 5 min each).

        Yields:
            (status_msg: str, result: TourneyResult | None)
        """

        t_start = time.time()
        resuming = False
        checkpoint = None
        partial_contenders: list[Contender] = []
        resume_remaining: list[str] = []
        resume_round_spec: dict = {}

        # ── Try to resume from checkpoint ────────────────────────────
        if self.resume:
            checkpoint = _load_checkpoint(self.output_dir)
            if checkpoint and _checkpoint_matches(
                checkpoint, self.model_name, self.dataset_key, self.quantization
            ):
                resuming = True
                result, partial_contenders, resume_remaining, resume_round_spec = (
                    _restore_rounds(checkpoint)
                )
                n_completed_rounds = len(result.rounds)
                n_completed_methods = len(partial_contenders)
                self.log("OBLITERATUS TOURNEY — RESUMING")
                self.log(f"Restored {n_completed_rounds} completed round(s), "
                         f"{n_completed_methods} method(s) from interrupted round")
                yield (
                    f"**Resuming tournament** — {n_completed_rounds} round(s) "
                    f"and {n_completed_methods} method(s) restored from checkpoint.",
                    result,
                )

                # Determine alive list from checkpoint
                alive = list(checkpoint.get("alive", self.methods))

                # Remove the checkpoint file now that we've loaded it
                ckpt_path = self.output_dir / CHECKPOINT_FILENAME
                if ckpt_path.exists():
                    ckpt_path.unlink()
            else:
                # Checkpoint doesn't match current config — start fresh
                checkpoint = None

        n_methods = len(self.methods)

        if not resuming:
            result = TourneyResult(
                model=self.model_name,
                timestamp=datetime.now().isoformat(),
            )
            alive = list(self.methods)

            self.log("OBLITERATUS TOURNEY")
            self.log(f"Model: {self.model_name}")
            self.log(f"Contenders: {n_methods} methods")
            self.log(f"Dataset: {self.dataset_key}")

        # Pre-flight disk space check
        try:
            disk = shutil.disk_usage(self.output_dir)
            free_gb = disk.free / 1e9
            self.log(f"Disk space: {free_gb:.1f} GB free on {self.output_dir}")
            if free_gb < 5.0:
                msg = (
                    f"Low disk space: only {free_gb:.1f} GB free. "
                    f"Tournament needs space for multiple model checkpoints. "
                    f"Free up space or use quantization to reduce checkpoint sizes."
                )
                self.log(f"WARNING: {msg}")
                yield (f"**Warning:** {msg}", result)
        except Exception:
            pass

        # Build round schedule
        rounds_schedule: list[tuple] = []

        if resuming and resume_round_spec:
            # We have an interrupted round to finish — schedule it first,
            # then let the dynamic scheduling add subsequent rounds.
            ir = resume_round_spec
            skip_completed_rounds = len(result.rounds)
        else:
            skip_completed_rounds = 0

        # Always build the full schedule starting from round 1.
        # Completed rounds will be skipped below.
        r1_advance = max(2, math.ceil(n_methods / 2))
        rounds_schedule.append((1, "Qualifiers", self.methods, 64, r1_advance, 30))

        for round_spec in rounds_schedule:
            round_num, name, methods, volume, advance_count, verify_sz = round_spec

            # Skip rounds that were already completed in the checkpoint
            if resuming and round_num <= skip_completed_rounds:
                # Re-derive alive and schedule next rounds from completed data
                completed_rnd = result.rounds[round_num - 1]
                alive = list(completed_rnd.advanced_to)
                if round_num == 1 and len(alive) > 1:
                    r2_advance = max(2, math.ceil(len(alive) / 2))
                    rounds_schedule.append((2, "Semifinals", alive, 128, r2_advance, 30))
                elif round_num == 2 and len(alive) > 1:
                    r3_name = "Championship" if len(alive) == 2 else "Finals"
                    rounds_schedule.append((3, r3_name, alive, 256, 1, 50))
                self.log(f"\nSkipping completed Round {round_num}: {name}")
                yield (
                    f"**Round {round_num} ({name}):** already completed (restored from checkpoint)",
                    result,
                )
                continue

            # For the interrupted round, merge checkpoint data
            is_interrupted_round = (
                resuming
                and resume_round_spec
                and round_num == resume_round_spec.get("round_num")
            )

            if is_interrupted_round:
                # Use the interrupted round's parameters
                volume = resume_round_spec.get("prompt_volume", volume)
                advance_count = resume_round_spec.get("advance_count", advance_count)
                verify_sz = resume_round_spec.get("verify_sample_size", verify_sz)
                methods = list(
                    [c.method for c in partial_contenders] + resume_remaining
                )

            self.log("")
            self.log("=" * 60)
            self.log(f"ROUND {round_num}: {name}")
            self.log(f"  {len(methods)} contenders | {volume} prompt pairs | "
                     f"top {advance_count} advance")
            self.log("=" * 60)

            harmful, harmless, evaluation_harmful, evaluation_harmless = (
                self._load_prompt_sets(volume)
            )

            rnd = TourneyRound(
                round_num=round_num,
                name=name,
                prompt_volume=volume,
            )

            # If resuming an interrupted round, restore already-completed
            # contenders and only run the remaining methods.
            methods_to_run = list(methods)
            if is_interrupted_round and partial_contenders:
                for c in partial_contenders:
                    rnd.contenders.append(c)
                    self.log(f"  [restored] {c.method}: score={c.score:.4f}")
                methods_to_run = list(resume_remaining)
                self.log(f"  {len(partial_contenders)} method(s) restored, "
                         f"{len(methods_to_run)} remaining")

            total_in_round = len(rnd.contenders) + len(methods_to_run)

            for i, method in enumerate(methods_to_run, len(rnd.contenders) + 1):
                self.log(f"\n[{i}/{total_in_round}] Running: {method}")
                yield (
                    f"**Round {round_num} ({name}):** running `{method}` [{i}/{total_in_round}]",
                    result,
                )

                save_dir = str(self._candidate_dir(round_num, method))

                try:
                    contender = self._run_one_method(
                        method, harmful, harmless, save_dir, verify_sz,
                        gpu_wrapper, evaluation_harmful, evaluation_harmless,
                    )
                except Exception as exc:
                    if self._is_quota_error(exc):
                        # Save checkpoint so the tournament can resume later.
                        # Include the failed method in remaining so it retries.
                        still_remaining = methods_to_run[methods_to_run.index(method):]
                        _save_checkpoint(
                            output_dir=self.output_dir,
                            result=result,
                            current_round_num=round_num,
                            current_round_name=name,
                            current_round_volume=volume,
                            current_round_advance=advance_count,
                            current_round_verify=verify_sz,
                            completed_methods=list(rnd.contenders),
                            remaining_methods=still_remaining,
                            alive=alive,
                            model_name=self.model_name,
                            dataset_key=self.dataset_key,
                            quantization=self.quantization,
                            methods=self.methods,
                        )
                        self.log(f"\nGPU SESSION INTERRUPTED — checkpoint saved")
                        self.log(f"  Reason: {exc}")
                        self.log(f"  Completed: {len(rnd.contenders)} methods in round {round_num}")
                        self.log(f"  Remaining: {len(still_remaining)} methods")
                        self.log(f"  Click Run again to resume automatically.")
                    raise

                rnd.contenders.append(contender)
                self.log(
                    f"  {method}: score={contender.score:.4f} "
                    f"(refusal={contender.metrics.get('refusal_rate', '?')}, "
                    f"coherence={contender.metrics.get('coherence', '?')}) "
                    f"[{contender.time_s:.0f}s]"
                )

            # Rank, advance, eliminate
            ranked, advanced, eliminated = _rank_and_select(
                rnd.contenders,
                advance_count,
            )
            rnd.advanced_to = [c.method for c in advanced]
            rnd.eliminated = [c.method for c in eliminated]
            for c in eliminated:
                c.round_eliminated = round_num

            self.log(f"\n{'─' * 40}")
            self.log(f"Round {round_num} results:")
            for idx, c in enumerate(ranked, 1):
                status = "ADVANCE" if c.method in rnd.advanced_to else "OUT"
                self.log(f"  {idx}. {c.method}: {c.score:.4f} [{status}]")

            # Clean up eliminated checkpoints
            for c in eliminated:
                if c.output_dir:
                    self._remove_owned_candidate_dir(c.output_dir)

            self._on_round(rnd)
            result.rounds.append(rnd)
            alive = list(rnd.advanced_to)

            # Schedule next round dynamically
            if round_num == 1 and len(alive) > 1:
                r2_advance = max(2, math.ceil(len(alive) / 2))
                rounds_schedule.append((2, "Semifinals", alive, 128, r2_advance, 30))
            elif round_num == 2 and len(alive) > 1:
                r3_name = "Championship" if len(alive) == 2 else "Finals"
                rounds_schedule.append((3, r3_name, alive, 256, 1, 50))

        # ── Determine winner ──────────────────────────────────────────
        last_round = result.rounds[-1]
        ranked, eligible_finalists, _ = _rank_and_select(
            last_round.contenders,
            len(last_round.contenders),
        )
        winner = eligible_finalists[0] if eligible_finalists else None
        result.winner = winner
        result.total_time_s = time.time() - t_start

        # Clean up non-winner finalist dirs to free disk
        for c in ranked:
            if c is not winner and c.output_dir:
                self._remove_owned_candidate_dir(c.output_dir)

        self.log("")
        self.log("=" * 60)
        if winner:
            self.log(f"CHAMPION: {winner.method} (score: {winner.score:.4f})")
        else:
            n_errors = sum(1 for c in ranked if c.error)
            self.log(f"NO WINNER — {n_errors}/{len(ranked)} methods errored")
        self.log(f"Total tournament time: {result.total_time_s / 60:.1f} minutes")
        self.log("=" * 60)

        # Save results
        results_path = self.output_dir / "tourney_results.json"
        results_path.write_text(json.dumps(result.to_dict(), indent=2))
        self.log(f"Results saved to {results_path}")

        bracket_path = self.output_dir / "tourney_bracket.md"
        bracket_path.write_text(render_bracket(result))
        self.log(f"Bracket saved to {bracket_path}")

        # Clean up checkpoint file on successful completion
        ckpt_path = self.output_dir / CHECKPOINT_FILENAME
        if ckpt_path.exists():
            ckpt_path.unlink()

        # Push winner
        if winner and winner.output_dir and (self.hub_org or self.hub_repo):
            self._push_winner(result)

        # Final yield with completed result
        yield ("Tournament complete", result)

    def _push_winner(self, result: TourneyResult):
        """Push the winning model to HuggingFace Hub."""
        winner = result.winner
        if not winner or not winner.output_dir:
            return

        try:
            from huggingface_hub import HfApi

            short_model = self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
            if self.hub_repo:
                repo_id = self.hub_repo
            else:
                repo_id = f"{self.hub_org}/{short_model}-OBLITERATED"
            result.hub_repo = repo_id

            self.log(f"\nPushing winner to Hub: {repo_id}")

            _token = os.environ.get("HF_PUSH_TOKEN") or os.environ.get("HF_TOKEN") or None
            api = HfApi(token=_token) if _token else HfApi()
            api.create_repo(repo_id, exist_ok=True)

            # Write model card
            model_card = generate_model_card(result)
            card_path = Path(winner.output_dir) / "README.md"
            card_path.write_text(model_card)

            # Write tourney results alongside model
            results_dest = Path(winner.output_dir) / "tourney_results.json"
            results_dest.write_text(json.dumps(result.to_dict(), indent=2))

            api.upload_folder(
                folder_path=winner.output_dir,
                repo_id=repo_id,
                commit_message=(
                    f"OBLITERATUS tourney: {winner.method} wins "
                    f"(score {winner.score:.4f}) on {self.model_name}"
                ),
            )
            self.log(f"Pushed to https://huggingface.co/{repo_id}")

        except Exception as e:
            self.log(f"Hub push failed: {e}")
