"""M21: report sections for non-proportional hazards and for graphical multiple testing."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from trialdesign import multiplicity, report
from trialdesign.cli import app

EX = Path(__file__).parents[1] / "examples"
NPH = str(EX / "phase3_survival_nph.toml")
ACCRUAL = str(EX / "phase3_survival_accrual.toml")
GRAPH = str(EX / "two_doses_graph.toml")


def _report(*args: str) -> dict:
    res = CliRunner().invoke(app, ["report", *args, "--json"])
    assert res.exit_code == 0, res.output
    return json.loads(res.stdout)


def test_nph_section_from_the_example(tmp_path):
    out = _report("--config", NPH, "--nsim", "2000", "--out", str(tmp_path))
    names = {Path(f).name for f in out["files"]}
    assert {"report.md", "survival.png"} <= names
    nph = out["nph"]
    ph, alt = nph["proportional"], nph["alternative"]
    assert nph["event_model"]["model"] == "delayed_effect"
    assert alt["hazard_ratio"] is None and ph["hazard_ratio"] == pytest.approx(0.7)
    # A 4-month delay costs a lot of power with the log-rank test (0.79 to about 0.48).
    assert ph["power"] > 0.75 and alt["power"] < ph["power"] - 0.2
    # The same planned events are reached slightly earlier: control events are unchanged
    # and treatment patients fail at the control rate for the first 4 months.
    assert all(
        a < p for a, p in zip(alt["expected_look_time"], ph["expected_look_time"], strict=True)
    )
    md = (tmp_path / "report.md").read_text()
    assert "## Non-proportional hazards" in md and "survival.png" in md
    assert "delayed effect (HR 1 for 4 months, then 0.7)" in md


def test_delay_flag_shows_the_first_futility_look(tmp_path):
    out = _report("--config", ACCRUAL, "--delay", "4", "--nsim", "2000", "--seed", "2026")
    ph, alt = out["nph"]["proportional"], out["nph"]["alternative"]
    assert len(ph["prob_futility"]) == 3
    # The first futility look stops far more trials under the delay (about 0.09 to 0.40).
    assert alt["prob_futility"][0] > ph["prob_futility"][0] + 0.2


def test_nph_markdown_without_out_is_printed():
    res = CliRunner().invoke(app, ["report", "--config", NPH, "--nsim", "2000"])
    assert res.exit_code == 0, res.output
    assert "## Non-proportional hazards" in res.output
    assert "survival.png" not in res.output


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (
            [
                "--endpoint",
                "survival",
                "--hr",
                "0.7",
                "--median-control",
                "12",
                "--accrual-time",
                "24",
                "--followup",
                "12",
                "--looks",
                "2",
                "--delay",
                "4",
            ],
            "--nsim",
        ),
        (["--config", ACCRUAL, "--test", "fh:0,1", "--nsim", "100"], "event model"),
        (
            [
                "--endpoint",
                "survival",
                "--hr",
                "0.7",
                "--looks",
                "2",
                "--delay",
                "4",
                "--nsim",
                "100",
            ],
            "accrual model",
        ),
    ],
)
def test_nph_report_errors(args, match):
    res = CliRunner().invoke(app, ["report", *args])
    assert res.exit_code == 2
    assert match in res.output


def test_graph_section(tmp_path):
    out = _report("--config", GRAPH, "--nsim", "5000", "--out", str(tmp_path))
    assert {Path(f).name for f in out["files"]} == {"report.md", "graph.png"}
    g = out["graph"]
    assert g["hypotheses"][0]["name"] == "H1 primary high"
    assert g["power"]["nsim"] == 5000 and g["test"]["rejected"] == [True, True, True, False]
    md = (tmp_path / "report.md").read_text()
    assert "## Multiple testing" in md and "graph.png" in md and "### Power" in md
    assert "Bretz" in md


def test_graph_section_printed_without_out():
    res = CliRunner().invoke(app, ["report", "--config", GRAPH, "--nsim", "2000"])
    assert res.exit_code == 0, res.output
    assert "## Multiple testing" in res.output


@pytest.mark.parametrize(
    "graph",
    [
        multiplicity.Graph.holm(2),
        multiplicity.Graph.holm(3),
        multiplicity.Graph.fallback([0.5, 0.3, 0.2]),
        multiplicity.Graph.two_doses(),
        multiplicity.Graph.holm(5),
    ],
)
def test_plot_graph_writes_a_png(graph, tmp_path):
    path = report.plot_graph(graph, tmp_path / "g.png", [0.5] * graph.size)
    assert path.stat().st_size > 10_000


def test_plot_survival_writes_a_png(tmp_path):
    from trialdesign.hazards import EventModel

    model = EventModel.delayed_effect(0.0578, 0.7, 4.0)
    path = report.plot_survival(model, 0.7, 36.0, tmp_path / "s.png")
    assert path.stat().st_size > 10_000
