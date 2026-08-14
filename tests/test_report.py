"""Tests for the reporting module."""

from __future__ import annotations

import json

from obliteratus.reporting.report import AblationReport, AblationResult


def _make_report() -> AblationReport:
    report = AblationReport(model_name="test-model")
    report.add_baseline({"perplexity": 25.0, "accuracy": 0.85})
    report.add_result(
        AblationResult(
            strategy="layer_removal",
            component="layer_0",
            description="Remove layer 0",
            metrics={"perplexity": 30.0, "accuracy": 0.80},
        )
    )
    report.add_result(
        AblationResult(
            strategy="layer_removal",
            component="layer_1",
            description="Remove layer 1",
            metrics={"perplexity": 50.0, "accuracy": 0.60},
        )
    )
    return report


class TestAblationReport:
    def test_to_dataframe(self):
        report = _make_report()
        df = report.to_dataframe()
        assert len(df) == 2
        assert "perplexity" in df.columns
        assert "perplexity_delta" in df.columns
        assert "perplexity_pct_change" in df.columns

    def test_save_json(self, tmp_path):
        report = _make_report()
        out = tmp_path / "results.json"
        report.save_json(out)
        data = json.loads(out.read_text())
        assert data["model_name"] == "test-model"
        assert len(data["results"]) == 2
        assert data["baseline_metrics"]["perplexity"] == 25.0

    def test_save_csv(self, tmp_path):
        report = _make_report()
        out = tmp_path / "results.csv"
        report.save_csv(out)
        text = out.read_text()
        assert "layer_0" in text
        assert "perplexity" in text

    def test_delta_calculation(self):
        report = _make_report()
        df = report.to_dataframe()
        row0 = df[df["component"] == "layer_0"].iloc[0]
        assert row0["perplexity_delta"] == 5.0  # 30 - 25
        assert abs(row0["perplexity_pct_change"] - 20.0) < 0.01

    def test_plot_impact(self, tmp_path):
        report = _make_report()
        out = tmp_path / "impact.png"
        report.plot_impact(metric="perplexity", output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0
