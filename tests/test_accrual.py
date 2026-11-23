"""M8: survival accrual, follow-up and dropout, validated against rpact.

Reference values in ``tests/accrual_reference_values.json`` come from
``tests/r/accrual_reference_values.R`` (``rpact::getSampleSizeSurvival``). The events are
Schoenfeld's (validated in M1 and M7); these tests check what accrual adds on top: patients,
the end of accrual or the minimum follow-up, the study duration, and the calendar time of each
look of a group-sequential design.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate
from typer.testing import CliRunner

from trialdesign import accrual, fixed, report
from trialdesign.boundaries import group_sequential_design
from trialdesign.cli import app

REF = json.loads((Path(__file__).parent / "accrual_reference_values.json").read_text())["accrual"]
FIXED = [c for c in REF if c["kind"] == "fixed"]
GSD = [c for c in REF if c["kind"] == "gsd"]
EXAMPLES = Path(__file__).parents[1] / "examples"
runner = CliRunner()


def _scalar(v: list[float]) -> float | list[float]:
    return v[0] if len(v) == 1 else v


def _accrual_kwargs(case: dict) -> dict:
    kw: dict = {
        "median_control": case["median_control"],
        "dropout": case["dropout"],
        "allocation": case["allocation"],
    }
    if "accrual_time" in case:
        kw["accrual_time"] = _scalar(case["accrual_time"])
    if "accrual_rate" in case:
        kw["accrual_rate"] = _scalar(case["accrual_rate"])
    if "followup" in case:
        kw["followup"] = case["followup"]
    return kw


def _events(case: dict) -> float:
    return fixed.events_survival(
        case["hazard_ratio"],
        alpha_two_sided=2 * case["alpha_one_sided"],
        power=case["power"],
        allocation=case["allocation"],
        hazard_ratio0=case["hazard_ratio0"],
    ).events


@pytest.mark.parametrize("case", FIXED, ids=lambda c: c["id"])
def test_fixed_design_matches_rpact(case: dict) -> None:
    events = _events(case)
    assert events == pytest.approx(case["events"], abs=1e-4)
    study = accrual.survival_study(events, case["hazard_ratio"], **_accrual_kwargs(case))
    assert study.n_total == pytest.approx(case["n_total"], abs=0.01)
    assert study.accrual_duration == pytest.approx(case["accrual_end"], abs=1e-3)
    assert study.followup == pytest.approx(case["followup_solved"], abs=1e-3)
    assert study.study_duration == pytest.approx(case["study_duration"], abs=1e-3)
    assert study.events_at(study.study_duration) == pytest.approx(events, rel=1e-8)


@pytest.mark.parametrize("case", GSD, ids=lambda c: c["id"])
def test_group_sequential_look_times_match_rpact(case: dict) -> None:
    d = group_sequential_design(
        case["looks"],
        alpha_one_sided=case["alpha_one_sided"],
        power=case["power"],
        efficacy_spending=case["efficacy"],
    )
    max_events = d.max_sample_size(_events(case))
    assert max_events == pytest.approx(case["max_events"], abs=0.01)
    study = accrual.survival_study(max_events, case["hazard_ratio"], **_accrual_kwargs(case))
    assert study.n_total == pytest.approx(case["max_n_total"], abs=0.05)
    times = study.look_times(d.info_rates)
    assert times == pytest.approx(case["look_time"], abs=1e-3)
    patients = [study.patients_at(t) for t in times]
    assert patients == pytest.approx(case["look_patients"], abs=0.05)


# ------------------------------------------------------------------------- properties


def test_expected_events_matches_numerical_integration() -> None:
    times, rates = [6.0, 20.0], [10.0, 25.0]
    lam, hr, eta, r = math.log(2) / 9, 0.75, accrual.dropout_hazard(0.1), 2.0
    t = 27.0

    def density(u: float) -> float:
        a = 10.0 if u < 6 else 25.0
        out = 0.0
        for share, h in ((1 / (1 + r), lam), (r / (1 + r), hr * lam)):
            mu = h + eta
            out += share * a * h / mu * (1 - math.exp(-mu * (t - u)))
        return out

    num = integrate.quad(density, 0, 6)[0] + integrate.quad(density, 6, 20)[0]
    assert accrual.expected_events(t, times, rates, lam, hr, eta, r) == pytest.approx(num)


def test_instant_accrual_no_dropout_gives_exponential_cdf() -> None:
    lam = accrual.hazard_from_median(12)
    e = accrual.expected_events(24.0 + 1e-9, [1e-9], [1e9], lam, 1.0)  # 1 patient at time 0
    assert e == pytest.approx(1 - math.exp(-lam * 24), rel=1e-6)


def test_uniform_prob_event_closed_form() -> None:
    # Lachin and Foulkes (1986): uniform entry over [0, R], follow-up F, hazards lam and eta.
    lam, eta, R, F = math.log(2) / 12, accrual.dropout_hazard(0.05), 24.0, 12.0
    mu = lam + eta
    closed = lam / mu * (1 - (math.exp(-mu * F) - math.exp(-mu * (R + F))) / (mu * R))
    study = accrual.survival_study(100, 1.0, 12, accrual_time=R, followup=F, dropout=0.05)
    assert study.prob_event == pytest.approx(closed)


def test_three_ways_to_specify_agree() -> None:
    d = fixed.events_survival(0.7).events
    a = accrual.survival_study(d, 0.7, 12, accrual_time=24, followup=12, dropout=0.05)
    rate = float(a.accrual_rates[0])
    b = accrual.survival_study(d, 0.7, 12, accrual_rate=rate, followup=12, dropout=0.05)
    c = accrual.survival_study(d, 0.7, 12, accrual_time=24, accrual_rate=rate, dropout=0.05)
    assert b.accrual_duration == pytest.approx(24, abs=1e-6)
    assert c.followup == pytest.approx(12, abs=1e-6)
    assert a.n_total == pytest.approx(b.n_total) == pytest.approx(c.n_total)


def test_dropout_and_longer_followup_change_n_in_the_right_direction() -> None:
    d = 250.0
    base = accrual.survival_study(d, 0.7, 12, accrual_time=24, followup=12).n_total
    assert (
        accrual.survival_study(d, 0.7, 12, accrual_time=24, followup=12, dropout=0.1).n_total > base
    )
    assert accrual.survival_study(d, 0.7, 12, accrual_time=24, followup=24).n_total < base


def test_look_times_increase_and_end_at_study_duration() -> None:
    study = accrual.survival_study(300, 0.7, 12, accrual_time=24, followup=12)
    t = study.look_times([0.25, 0.5, 0.75, 1.0])
    assert np.all(np.diff(t) > 0)
    assert t[-1] == pytest.approx(study.study_duration)
    assert study.patients_at(100) == pytest.approx(study.n_total)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"accrual_time": 24}, "give two of"),
        ({"followup": 12}, "give an accrual time or accrual rates"),
        ({"accrual_time": [6, 24], "followup": 12}, "single accrual time"),
        ({"accrual_time": [6, 24], "accrual_rate": [10, 25], "followup": 12}, "one accrual"),
        ({"accrual_time": 6, "accrual_rate": 1}, "at most"),
        ({"accrual_time": [24, 6], "accrual_rate": [10, 25]}, "increasing"),
        ({"accrual_time": 24, "followup": 12, "dropout": 1.0}, "dropout"),
        ({"accrual_time": [30], "accrual_rate": [100, 10], "followup": 12}, "reached before"),
    ],
)
def test_invalid_input_fails_with_plain_message(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        accrual.survival_study(250, 0.7, 12, **kwargs)


# --------------------------------------------------------------- report, CLI, config


def test_fixed_summary_with_accrual() -> None:
    inputs = {"median_control": 12, "accrual_time": 24, "followup": 12, "dropout": 0.05}
    s = report.fixed_summary("survival", hr=0.7, accrual_inputs=inputs)
    case = next(c for c in FIXED if c["id"] == "acc_A_uniform_dropout")
    assert s["n_total"] == pytest.approx(case["n_total"], abs=0.01)
    assert s["n_total_ceil"] == 386
    assert "month 36.0" in report.accrual_text(s)
    with pytest.raises(ValueError, match="prob_event or an accrual model"):
        report.fixed_summary("survival", hr=0.7, prob_event=0.7, accrual_inputs=inputs)


def test_cli_group_sequential_with_accrual_matches_rpact() -> None:
    case = next(c for c in GSD if c["id"] == "acc_gsd_obf3_uniform")
    args = [
        "design",
        "--endpoint",
        "survival",
        "--hr",
        "0.7",
        "--looks",
        "3",
        "--median-control",
        "12",
        "--accrual-time",
        "24",
        "--followup",
        "12",
        "--dropout",
        "0.05",
    ]
    out = json.loads(runner.invoke(app, [*args, "--json"]).output)
    acc = out["sample_size"]["accrual"]
    assert acc["look_time"] == pytest.approx(case["look_time"], abs=1e-3)
    assert out["sample_size"]["max_n_total"] == pytest.approx(case["max_n_total"], abs=0.05)
    table = runner.invoke(app, args).output
    assert "Month" in table and "final analysis at month 36.0" in table


def test_cli_rejects_accrual_for_other_endpoints() -> None:
    args = ["design", "--endpoint", "binary", "--pc", "0.3", "--pt", "0.45", "--followup", "12"]
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "accrual settings apply only to the survival endpoint" in result.output


def test_cli_requires_median_for_accrual() -> None:
    args = ["design", "--endpoint", "survival", "--hr", "0.7", "--accrual-time", "24"]
    result = runner.invoke(app, [*args, "--followup", "12"])
    assert result.exit_code == 2
    assert "--median-control is required" in result.output


def test_example_config_design_and_report(tmp_path: Path) -> None:
    cfg = str(EXAMPLES / "phase3_survival_accrual.toml")
    out = json.loads(runner.invoke(app, ["design", "--config", cfg, "--json"]).output)
    acc = out["sample_size"]["accrual"]
    assert acc["accrual_rates"] == [10.0, 25.0]
    assert acc["look_time"][-1] == pytest.approx(acc["study_duration"])
    result = runner.invoke(app, ["report", "--config", cfg, "--out", str(tmp_path)])
    assert result.exit_code == 0, result.output
    md = (tmp_path / "report.md").read_text()
    assert "| Month | Patients |" in md and "piecewise accrual" in md
