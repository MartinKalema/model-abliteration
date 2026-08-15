"""GGUF coverage for YAML study configuration and runner forwarding."""

from __future__ import annotations

import pytest
import yaml

from obliteratus import runner
from obliteratus.config import StudyConfig


def _gguf_study_dict(tmp_path):
    return {
        "model": {
            "name": "quantizer/gpt-oss-20b-GGUF",
            "task": "causal_lm",
            "dtype": "bfloat16",
            "device": "cpu",
            "gguf_file": "Q4/gpt-oss-20b-Q4_K_M.gguf",
            "canonical_model_id": "openai/gpt-oss-20b",
            "tokenizer_source": "/models/gpt-oss-tokenizer",
        },
        "dataset": {"name": "offline-fixture", "max_samples": 1},
        "strategies": [{"name": "layer_removal", "params": {}}],
        "output_dir": str(tmp_path / "results"),
    }


def test_gguf_model_config_roundtrips_through_yaml_and_dict(tmp_path):
    config_path = tmp_path / "gguf-study.yaml"
    config_path.write_text(
        yaml.safe_dump(_gguf_study_dict(tmp_path)),
        encoding="utf-8",
    )

    config = StudyConfig.from_yaml(config_path)
    assert config.model.gguf_file == "Q4/gpt-oss-20b-Q4_K_M.gguf"
    assert config.model.canonical_model_id == "openai/gpt-oss-20b"
    assert config.model.tokenizer_source == "/models/gpt-oss-tokenizer"

    restored = StudyConfig.from_dict(config.to_dict())
    assert restored.model.gguf_file == config.model.gguf_file
    assert restored.model.canonical_model_id == config.model.canonical_model_id
    assert restored.model.tokenizer_source == config.model.tokenizer_source


def test_runner_forwards_gguf_model_fields_to_loader(tmp_path, monkeypatch):
    config = StudyConfig.from_dict(_gguf_study_dict(tmp_path))
    forwarded = {}

    class StopAfterModelLoad(RuntimeError):
        pass

    def fake_load_model(**kwargs):
        forwarded.update(kwargs)
        raise StopAfterModelLoad

    monkeypatch.setattr(runner, "load_model", fake_load_model)

    with pytest.raises(StopAfterModelLoad):
        runner.run_study(config)

    assert forwarded["model_name"] == "quantizer/gpt-oss-20b-GGUF"
    assert forwarded["gguf_file"] == "Q4/gpt-oss-20b-Q4_K_M.gguf"
    assert forwarded["canonical_model_id"] == "openai/gpt-oss-20b"
    assert forwarded["tokenizer_source"] == "/models/gpt-oss-tokenizer"
    assert forwarded["dtype"] == "bfloat16"
