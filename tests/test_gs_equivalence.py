"""M19b: equivalence by two one-sided group-sequential tests (Jennison & Turnbull, ch. 6).

Reference values in ``tests/gs_equivalence_reference_values.json`` come from
``tests/r/gs_equivalence_reference_values.R``: critical values from
``rpact::getDesignGroupSequential``, stopping probabilities as sums of multivariate normal
rectangle probabilities (``mvtnorm::pmvnorm``) instead of recursive integration, sample size
by ``uniroot`` and, for rates, Farrington-Manning variances from a numerically maximised
restricted likelihood. ``pmvnorm`` is accurate to about 1e-7 here, so probabilities are
compared to 1e-6 and sample sizes to 1e-5 relative.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from trialdesign.cli import app
from trialdesign.equivalence import (
    group_sequential_equivalence_binary,
    group_sequential_equivalence_normal,
    power_equivalence_binary,
    power_equivalence_normal,
    sample_size_equivalence_binary,
    sample_size_equivalence_normal,
)

REF = json.loads((Path(__file__).parent / "gs_equivalence_reference_values.json").read_text())


def _design(c: dict):
    common = {
        "info_rates": c["info_rates"],
        "alpha": c["alpha"],
        "power": c["power"],
        "allocation": c["allocation"],
        "spending": c["spending"],
    }
    if c["endpoint"] == "normal":
        return group_sequential_equivalence_normal(c["margin"], c["sd"], delta=c["delta"], **common)
    return group_sequential_equivalence_binary(
        c["p_control"], c["p_treatment"], c["margin"], **common
    )


@pytest.mark.parametrize("case", REF["designs"], ids=[c["id"] for c in REF["designs"]])
def test_design_matches_r(case):
    g = _design(case)
    np.testing.assert_allclose(g.critical_values, case["critical_values"], atol=1e-6)
    np.testing.assert_allclose([g.sd_lower, g.sd_upper, g.sd_estimate], case["sds"], rtol=1e-9)
    assert g.n_control == pytest.approx(case["n_control"], rel=1e-5)
    # Probabilities at R's sample size, so only the integration is compared.
    n = case["n_control"]
    for theta, key in (
        (None, "stop_probs_design"),
        (case["margin"][1], "stop_probs_upper_margin"),
        (case["margin"][0], "stop_probs_lower_margin"),
    ):
        np.testing.assert_allclose(
            g.stopping_probabilities(theta, n), case[key], rtol=0, atol=1e-6, err_msg=key
        )
    assert g.expected_n_control() == pytest.approx(case["expected_n_control"], rel=1e-5)


def test_one_look_is_the_fixed_tost():
    for margin, sd, delta, r in ((0.5, 1.0, 0.0, 1.0), ((-0.4, 0.6), 2.0, 0.1, 2.0)):
        g = group_sequential_equivalence_normal(margin, sd, 1, delta=delta, allocation=r)
        f = sample_size_equivalence_normal(margin, sd, delta, 0.025, 0.8, r)
        assert g.n_control == pytest.approx(f.n_control, rel=1e-6)
        assert g.power_at() == pytest.approx(
            power_equivalence_normal(g.n_control, margin, sd, delta, 0.025, r), abs=1e-6
        )
    g = group_sequential_equivalence_binary(0.6, 0.65, (-0.1, 0.2), [1.0], alpha=0.05)
    f = sample_size_equivalence_binary(0.6, 0.65, (-0.1, 0.2), 0.05)
    assert g.n_control == pytest.approx(f.n_control, rel=1e-6)
    assert g.power_at(None) == pytest.approx(
        power_equivalence_binary(g.n_control, 0.6, 0.65, (-0.1, 0.2), 0.05), abs=1e-6
    )
    assert g.inflation_factor == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("looks", [2, 3, [0.3, 0.5, 0.8, 1.0]])
@pytest.mark.parametrize("spending", ["obf", "pocock"])
def test_simulated_probability_at_the_margin_is_at_most_alpha(looks, spending):
    g = group_sequential_equivalence_normal(0.5, 1.0, looks, spending=spending)
    for margin in g.margin:
        sim = g.simulate(margin, n_sim=200_000, seed=2026)
        assert sim.prob_equivalence <= g.alpha + 2 * sim.prob_equivalence_se
        assert g.power_at(margin) <= g.alpha


def test_simulation_agrees_with_the_recursion():
    g = group_sequential_equivalence_normal(0.5, 1.0, 3, power=0.9)
    # 105 patients per arm puts every look on whole patients (35, 70, 105).
    sim = g.simulate(n_sim=200_000, seed=11, n_control=105)
    exact = g.stopping_probabilities(None, 105)
    se = np.sqrt(sim.prob_stop * (1 - sim.prob_stop) / sim.n_sim)
    assert np.all(np.abs(sim.prob_stop - exact) <= 3 * np.maximum(se, 1e-4))
    assert abs(sim.prob_equivalence - exact.sum()) <= 2 * sim.prob_equivalence_se
    expected = exact @ (g.info_rates * 105) + (1 - exact.sum()) * 105
    assert sim.expected_n_control == pytest.approx(expected, rel=2e-3)


def test_more_looks_cost_sample_size_and_save_expected_size():
    one = group_sequential_equivalence_normal(0.5, 1.0, 1, power=0.9)
    three = group_sequential_equivalence_normal(0.5, 1.0, 3, power=0.9)
    assert three.n_control > one.n_control
    assert three.expected_n_control() < one.n_control


def test_errors():
    with pytest.raises(ValueError, match="inside"):
        group_sequential_equivalence_normal(0.5, 1.0, 2, delta=0.6)
    g = group_sequential_equivalence_binary(0.7, 0.7, 0.1, 2)
    with pytest.raises(ValueError, match="normal endpoint"):
        g.simulate()


# ------------------------------------------------------------------------------- CLI


def test_cli_design_margin_with_looks():
    out = CliRunner().invoke(
        app,
        [
            "design", "--endpoint", "normal", "--delta", "0", "--sd", "1", "--margin", "0.5",
            "--looks", "3", "--power", "0.9", "--json",
        ],
    )  # fmt: skip
    assert out.exit_code == 0, out.output
    s = json.loads(out.output)
    case = next(c for c in REF["designs"] if c["id"] == "normal_k3_obf")
    assert s["design"] == "group_sequential_equivalence"
    assert s["sample_size"]["max_n_control"] == pytest.approx(case["n_control"], rel=1e-5)
    assert s["critical_values"] == pytest.approx(case["critical_values"], abs=1e-6)
    assert len(s["looks"]) == 3
    assert s["looks"][-1]["prob_equivalence_at_upper_margin"] <= 0.025


def test_cli_design_margin_with_looks_text():
    out = CliRunner().invoke(
        app,
        [
            "design", "--endpoint", "binary", "--pc", "0.7", "--pt", "0.7", "--margin", "0.1",
            "--looks", "3",
        ],
    )  # fmt: skip
    assert out.exit_code == 0, out.output
    assert "group-sequential equivalence" in out.output
    assert "Repeated CI" in out.output or "Look" in out.output


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (["--futility", "obf"], "futility"),
        (["--analyse", "1,2"], "--analyse"),
    ],
)
def test_cli_gs_equivalence_errors(extra, match):
    out = CliRunner().invoke(
        app,
        [
            "design", "--endpoint", "normal", "--delta", "0", "--sd", "1", "--margin", "0.5",
            "--looks", "2", *extra,
        ],
    )  # fmt: skip
    assert out.exit_code == 2
    assert match in out.output


def test_simulate_and_report_refuse_gs_equivalence(tmp_path):
    cfg = tmp_path / "eq.toml"
    cfg.write_text(
        '[design]\nendpoint = "normal"\ndelta = 0\nsd = 1\nmargin = 0.5\n[sequential]\nlooks = 2\n'
    )
    for cmd in (["simulate", "--nsim", "100"], ["report"]):
        out = CliRunner().invoke(app, [*cmd, "--config", str(cfg)])
        assert out.exit_code == 2
        assert "design only" in out.output
