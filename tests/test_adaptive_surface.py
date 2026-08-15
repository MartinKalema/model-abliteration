from __future__ import annotations

from obliteratus.adaptive_surface import validate_adaptive_overrides


def test_adaptive_overrides_accept_only_valid_edit_parameters():
    accepted, rejected = validate_adaptive_overrides(
        {
            "n_directions": 12,
            "regularization": 0.05,
            "per_expert_directions": True,
            "projection_target": "all",
            "output_dir": "/tmp/replaced",
            "damage_gate_enabled": False,
            "refinement_passes": 0,
            "transplant_blend": float("nan"),
        }
    )

    assert accepted == {
        "n_directions": 12,
        "regularization": 0.05,
        "per_expert_directions": True,
        "projection_target": "all",
    }
    assert set(rejected) == {
        "output_dir",
        "damage_gate_enabled",
        "refinement_passes",
        "transplant_blend",
    }


def test_adaptive_override_types_are_not_coerced():
    accepted, rejected = validate_adaptive_overrides(
        {
            "project_embeddings": "false",
            "n_directions": True,
            "projection_row_fraction": 0.0,
        }
    )

    assert accepted == {}
    assert set(rejected) == {
        "project_embeddings",
        "n_directions",
        "projection_row_fraction",
    }
