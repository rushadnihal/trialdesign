"""M9: sample size re-estimation, validated against rpact (tests/ssr_reference_values.json).

Conditional power is checked against ``rpact::getConditionalPower`` to 1e-6. The exact
operating characteristics (one-dimensional integration over Z1) are checked against
``rpact::getSimulationMeans`` within 3 Monte Carlo SEs, using rpact's rule (re-estimate
upwards whenever CP < target, i.e. ``cp_min = 0``), and against this package's own
simulator for the promising-zone rule.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from trialdesign import ssr
from trialdesign.cli import app

REF = json.loads((Path(__file__).parent / "ssr_reference_values.json").read_text())["ssr"]
CP = [c for c in REF if c["kind"] == "cp"]
SIM = [c for c in REF if c["kind"] == "sim"]
EXAMPLES = Path(__file__).parents[1] / "examples"
runner = CliRunner()


@pytest.mark.parametrize("case", CP, ids=lambda c: c["id"])
def test_conditional_power_matches_rpact(case: dict) -> None:
    d = ssr.ssr_design(case["interim"], efficacy_spending=case["efficacy"])
    assert d.c2 == pytest.approx(case["c2"], abs=1e-6)
    # Planned stage 2 = stage2_events (m = 1), so the planned maximum is n2 / (1 - t1) and
    # the drift of the planned design is |log HR| sqrt(pi (1 - pi) D_max).
    d_max = case["stage2_events"] / (1 - case["interim"])
    drift = abs(math.log(case["hazard_ratio"])) * math.sqrt(0.25 * d_max)
    cp = d.conditional_power(case["z1"], drift=drift)
    assert float(cp) == pytest.approx(case["conditional_power"], abs=1e-6)


@pytest.mark.parametrize("case", SIM, ids=lambda c: c["id"])
def test_exact_operating_characteristics_match_rpact_simulation(case: dict) -> None:
    t1, n = case["interim"], case["planned_n_total"]
    d = ssr.ssr_design(
        t1,
        efficacy_spending=case["efficacy"],
        cp_min=0.0,
        cp_target=case["cp_target"],
        max_factor=case["max_n_total"] / n,
    )
    drift = case["delta"] / case["sd"] * math.sqrt(n / 4)
    oc = d.operating_characteristics(drift, use_futility=False)
    n_sim = int(case["n_sim"])
    se = math.sqrt(oc.reject_prob * (1 - oc.reject_prob) / n_sim)
    assert oc.reject_prob == pytest.approx(case["reject_prob"], abs=3 * se)
    p1 = oc.prob_efficacy_interim
    assert p1 == pytest.approx(
        case["prob_efficacy_interim"], abs=3 * math.sqrt(p1 * (1 - p1) / n_sim)
    )
    sim = d.simulate(drift, n_sim=n_sim, seed=1, use_futility=False)
    n_se = sim.expected_n_fraction_se * n
    assert oc.expected_n_fraction * n == pytest.approx(case["expected_n_total"], abs=3 * n_se + 1)


# ------------------------------------------------------------------------- properties


@pytest.mark.parametrize("cp_min", [0.0, 0.2, 0.36, 0.5])
@pytest.mark.parametrize("futility", [None, "obf"])
def test_chw_test_controls_type_i_error_exactly(cp_min: float, futility: str | None) -> None:
    d = ssr.ssr_design(0.5, futility_spending=futility, cp_min=cp_min)
    oc = d.operating_characteristics(0.0, use_futility=False)
    assert oc.reject_prob == pytest.approx(0.025, abs=1e-8)


def test_naive_test_inflates_alpha_when_adapting_in_every_zone() -> None:
    aggressive = ssr.ssr_design(0.5, cp_min=0.0, max_factor=2.0)
    assert (
        aggressive.operating_characteristics(0.0, use_futility=False, test="naive").reject_prob
        > 0.03
    )
    # Increasing n only in the promising zone does not inflate the naive test (Chen, DeMets
    # and Lan 2004; Mehta and Pocock 2011), but the CHW test is used regardless.
    promising = ssr.ssr_design(0.5, cp_min=0.36)
    naive = promising.operating_characteristics(0.0, use_futility=False, test="naive")
    assert naive.reject_prob <= 0.025


def test_never_adapting_reproduces_the_group_sequential_design() -> None:
    d = ssr.ssr_design(0.5, futility_spending="obf", max_factor=1.0)
    for drift in (0.0, d.gsd.drift):
        oc = d.operating_characteristics(drift)
        eff, fut = d.gsd.crossing_probabilities(drift)
        # boundaries.py integrates on a grid (accurate to about 1e-8); ssr.py uses quad
        assert oc.reject_prob == pytest.approx(eff.sum(), abs=1e-7)
        assert oc.prob_futility_interim == pytest.approx(fut[0], abs=1e-12)
        assert oc.expected_n_fraction == pytest.approx(
            d.gsd.expected_info_fraction(drift), abs=1e-9
        )


def test_simulation_agrees_with_exact_values() -> None:
    d = ssr.ssr_design(0.5, futility_spending="pocock")
    for drift in (0.0, d.gsd.drift):
        oc = d.operating_characteristics(drift)
        sim = d.simulate(drift, n_sim=200_000, seed=2026)
        assert abs(sim.reject_prob - oc.reject_prob) < 3 * sim.reject_prob_se
        assert (
            abs(sim.expected_n_fraction - oc.expected_n_fraction) < 3 * sim.expected_n_fraction_se
        )
        for zone, p in oc.prob_zone.items():
            assert abs(sim.prob_zone[zone] - p) < 3 * math.sqrt(p * (1 - p) / 200_000) + 1e-12
        assert sum(oc.prob_zone.values()) == pytest.approx(1.0, abs=1e-9)


def test_promising_zone_hits_cp_target_or_the_cap() -> None:
    d = ssr.ssr_design(0.5, cp_min=0.36, cp_target=0.8, max_factor=2.0)
    z = np.linspace(0.0, d.c1 - 1e-6, 400)
    zones = d.zone(z)
    m = d.stage2_multiplier(z)
    cp = d.conditional_power(z, m=m)
    prom = zones == "promising"
    assert prom.any()
    capped = np.isclose(m, d.m_max)
    assert np.all(np.isclose(cp[prom & ~capped], 0.8, atol=1e-9))
    assert np.all(cp[prom & capped] < 0.8)
    assert np.all(m[~prom] == 1.0)
    assert np.all(np.diff(d.conditional_power(z)) > 0)  # CP at the observed trend increases


def test_adaptation_raises_power_and_expected_n() -> None:
    plain = ssr.ssr_design(0.5, max_factor=1.0)
    adaptive = ssr.ssr_design(0.5, max_factor=2.0)
    drift = 0.8 * plain.gsd.drift  # a smaller effect than planned
    a, b = plain.operating_characteristics(drift), adaptive.operating_characteristics(drift)
    assert b.reject_prob > a.reject_prob
    assert b.expected_n_fraction > a.expected_n_fraction


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"interim": 1.0}, "interim"),
        ({"cp_min": 0.9, "cp_target": 0.8}, "cp_min < cp_target"),
        ({"max_factor": 0.9}, "max_factor"),
    ],
)
def test_invalid_input_fails_with_plain_message(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ssr.ssr_design(**kwargs)


def test_needs_two_looks() -> None:
    from trialdesign.boundaries import group_sequential_design

    with pytest.raises(ValueError, match="exactly two looks"):
        ssr.SSRDesign(group_sequential_design(3))


# --------------------------------------------------------------- CLI, config, report


def test_cli_ssr_json_and_interim_rule() -> None:
    args = ["ssr", "--endpoint", "normal", "--delta", "0.3", "--sd", "1", "--json"]
    out = json.loads(runner.invoke(app, args).output)
    assert out["n_planned"] == 352 and out["n_cap"] == 704 and out["n_interim"] == 176
    assert out["exact"]["type_i_error"] == pytest.approx(0.025, abs=1e-8)
    z_lo, z_hi = out["zone_z1"]["cp_min"], out["zone_z1"]["cp_target"]
    mid = 0.5 * (z_lo + z_hi)
    out = json.loads(runner.invoke(app, [*args, "--z1", str(mid)]).output)
    i = out["interim"]
    assert i["zone"] == "promising"
    assert 352 < i["n_total"] <= 704
    assert i["cp_observed_after"] == pytest.approx(0.8, abs=1e-6) or i["n_total"] == 704


def test_cli_ssr_rejects_survival_and_extra_looks() -> None:
    result = runner.invoke(app, ["ssr", "--endpoint", "survival", "--hr", "0.7"])
    assert result.exit_code == 2
    assert "normal and binary endpoints" in result.output
    cfg = str(EXAMPLES / "phase3_ssr.toml")
    result = runner.invoke(app, ["report", "--config", cfg, "--looks", "3"])
    assert result.exit_code == 2
    assert "two looks" in result.output


def test_example_config_ssr_and_report(tmp_path: Path) -> None:
    cfg = str(EXAMPLES / "phase3_ssr.toml")
    result = runner.invoke(app, ["ssr", "--config", cfg, "--nsim", "20000"])
    assert result.exit_code == 0, result.output
    assert "Promising" in result.output and "Simulated checks" in result.output
    result = runner.invoke(
        app, ["report", "--config", cfg, "--out", str(tmp_path), "--nsim", "20000"]
    )
    assert result.exit_code == 0, result.output
    md = (tmp_path / "report.md").read_text()
    assert "## Sample size re-estimation" in md and "Cui, Hung and Wang" in md
    assert (tmp_path / "ssr.png").stat().st_size > 10_000
    assert "| Look | Info fraction |" in md  # the planned two-look design is reported too
