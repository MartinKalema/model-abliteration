import json
from pathlib import Path

import pytest

from obliteratus.abliterate import UNAVAILABLE_METHODS
from obliteratus.tourney import (
    TOURNEY_OWNER_FILENAME,
    TOURNEY_RUN_SUBDIR,
    TourneyOutputSafetyError,
    TourneyRunner,
)


def _runner(output_dir: Path, *, resume: bool = False) -> TourneyRunner:
    return TourneyRunner(
        "base/model",
        methods=["advanced"],
        output_dir=str(output_dir),
        resume=resume,
    )


def test_unrelated_nonempty_output_is_preserved_and_gets_owned_child(tmp_path):
    requested = tmp_path / "user-files"
    requested.mkdir()
    keep = requested / "keep.txt"
    keep.write_text("do not delete", encoding="utf-8")

    runner = _runner(requested)

    assert keep.read_text(encoding="utf-8") == "do not delete"
    assert runner.requested_output_dir == requested
    assert runner.output_dir == requested / TOURNEY_RUN_SUBDIR
    marker = json.loads(
        (runner.output_dir / TOURNEY_OWNER_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["tool"] == "obliteratus.tourney"
    assert marker["resolved_path"] == str(runner.output_dir.resolve())


def test_fresh_run_only_resets_a_valid_owned_directory(tmp_path):
    requested = tmp_path / "tourney"
    first = _runner(requested)
    stale = first.output_dir / "stale.txt"
    stale.write_text("old run", encoding="utf-8")

    second = _runner(requested)

    assert second.output_dir == requested
    assert not stale.exists()
    assert (requested / TOURNEY_OWNER_FILENAME).is_file()


def test_copied_marker_does_not_authorize_cleanup(tmp_path):
    legitimate = _runner(tmp_path / "legitimate")
    copied_marker = (
        legitimate.output_dir / TOURNEY_OWNER_FILENAME
    ).read_text(encoding="utf-8")

    requested = tmp_path / "user-files"
    occupied_child = requested / TOURNEY_RUN_SUBDIR
    occupied_child.mkdir(parents=True)
    keep = occupied_child / "keep.txt"
    keep.write_text("mine", encoding="utf-8")
    (occupied_child / TOURNEY_OWNER_FILENAME).write_text(
        copied_marker,
        encoding="utf-8",
    )

    runner = _runner(requested)

    assert keep.read_text(encoding="utf-8") == "mine"
    assert runner.output_dir == requested / f"{TOURNEY_RUN_SUBDIR}-2"


def test_resume_preserves_owned_checkpoint_and_outputs(tmp_path):
    requested = tmp_path / "tourney"
    first = _runner(requested)
    checkpoint = requested / "tourney_checkpoint.json"
    checkpoint.write_text('{"version": 1}', encoding="utf-8")
    candidate = first._candidate_dir(1, "advanced")
    candidate.mkdir()
    (candidate / "weights.bin").write_bytes(b"weights")
    original_owner = json.loads(
        (requested / TOURNEY_OWNER_FILENAME).read_text(encoding="utf-8")
    )["owner_id"]

    resumed = _runner(requested, resume=True)

    assert resumed.output_dir == requested
    assert checkpoint.is_file()
    assert (candidate / "weights.bin").read_bytes() == b"weights"
    resumed_owner = json.loads(
        (requested / TOURNEY_OWNER_FILENAME).read_text(encoding="utf-8")
    )["owner_id"]
    assert resumed_owner == original_owner


def test_legacy_checkpoint_resumes_non_exclusively_and_is_never_reset(tmp_path):
    requested = tmp_path / "legacy-tourney"
    requested.mkdir()
    keep = requested / "user-note.txt"
    keep.write_text("keep", encoding="utf-8")
    (requested / "tourney_checkpoint.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model": "base/model",
                "dataset_key": "builtin",
                "quantization": None,
            }
        ),
        encoding="utf-8",
    )

    resumed = _runner(requested, resume=True)

    assert resumed.output_dir == requested
    marker = json.loads(
        (requested / TOURNEY_OWNER_FILENAME).read_text(encoding="utf-8")
    )
    assert marker["kind"] == "resume-run-directory"
    assert keep.read_text(encoding="utf-8") == "keep"

    fresh = _runner(requested)
    assert fresh.output_dir == requested / TOURNEY_RUN_SUBDIR
    assert keep.read_text(encoding="utf-8") == "keep"


def test_candidate_cleanup_requires_manifest_ownership(tmp_path):
    runner = _runner(tmp_path / "tourney")
    owned = runner._candidate_dir(1, "advanced")
    owned.mkdir()
    (owned / "weights.bin").write_bytes(b"weights")

    unowned = runner.output_dir / "notes"
    unowned.mkdir()
    (unowned / "keep.txt").write_text("keep", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()

    assert runner._remove_owned_candidate_dir(unowned) is False
    assert runner._remove_owned_candidate_dir(outside) is False
    assert unowned.is_dir()
    assert outside.is_dir()
    assert runner._remove_owned_candidate_dir(owned) is True
    assert not owned.exists()


def test_candidate_method_cannot_escape_run_directory(tmp_path):
    runner = _runner(tmp_path / "tourney")

    with pytest.raises(TourneyOutputSafetyError, match="safe directory component"):
        runner._candidate_dir(1, "../../outside")


@pytest.mark.parametrize("method", sorted(UNAVAILABLE_METHODS))
def test_unavailable_method_is_rejected_before_output_creation(tmp_path, method):
    requested = tmp_path / "tourney"

    with pytest.raises(ValueError, match="unavailable"):
        TourneyRunner(
            "base/model",
            methods=[method],
            output_dir=str(requested),
        )

    assert not requested.exists()


def test_unknown_method_is_rejected_before_output_creation(tmp_path):
    requested = tmp_path / "tourney"

    with pytest.raises(ValueError, match="not a runnable preset"):
        TourneyRunner(
            "base/model",
            methods=["not-a-method"],
            output_dir=str(requested),
        )

    assert not requested.exists()


def test_string_method_input_is_rejected_before_output_creation(tmp_path):
    requested = tmp_path / "tourney"

    with pytest.raises(TypeError, match="provided as a list"):
        TourneyRunner(
            "base/model",
            methods="advanced",
            output_dir=str(requested),
        )

    assert not requested.exists()
