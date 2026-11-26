"""M19a: blinded sample size re-estimation with an internal pilot (Kieser and Friede 2003).

Reference values in ``tests/blinded_reference_values.json`` come from
``tests/r/blinded_reference_values.R``, an independent patient-level simulation in R (200,000
trials per case; the hand-coded t statistic is checked there against ``stats::t.test``). The
Python simulation draws sufficient statistics instead, with its own seed, so the two rates
are independent estimates: they must agree within 3 SEs of their difference. The
re-estimated sizes for given pilot variances are deterministic and must match exactly.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from typer.testing import CliRunner

from trialdesign import fixed
from trialdesign.blinded import BlindedSSR, lumped_variance
from trialdesign.cli import app

REF = json.loads((Path(__file__).parent / "blinded_reference_values.json").read_text())
EXAMPLE = str(Path(__file__).parents[1] / "examples" / "phase3_blinded_ssr.toml")


def _design(case: dict) -> BlindedSSR:
    return BlindedSSR(
        delta=case["delta"],
        sd=case["sd"],
        pilot_n=int(case["pilot_n"]),
        alpha_one_sided=case["alpha_one_sided"],
        power=case["power"],
        allocation=case["allocation"],
        max_factor=case["max_factor"],
        adjusted=case["adjusted"],
        floor=case["floor"],
    )


def _id(c: dict) -> str:
    return f"{c['design']}-d{c['true_delta']:g}-sd{c['true_sd']:g}"


@pytest.mark.parametrize("case", REF["sizes"])
def test_reestimated_sizes_match_r(case):
    d = _design(case)
    assert d.planned_n_control == case["planned_n_control"]
    res = d.reestimate(case["variance_lumped"])
    assert (res.n_control, res.n_treatment) == (case["n_control"], case["n_treatment"])


@pytest.mark.parametrize("case", REF["simulations"], ids=[_id(c) for c in REF["simulations"]])
def test_simulation_matches_patient_level_r(case):
    sim = _design(case).simulate(case["true_delta"], case["true_sd"], 200_000, seed=19)
    r = case["rejection_rate"]
    se_diff = math.sqrt(sim.rejection_se**2 + r * (1 - r) / case["n_sim"])
    assert abs(sim.rejection_rate - r) <= 3 * se_diff, (sim.rejection_rate, r)
    se_n = case["sd_n_total"] * math.sqrt(2 / case["n_sim"])
    assert abs(sim.mean_n_total - case["mean_n_total"]) <= 3 * max(se_n, 1e-3)


@pytest.mark.parametrize("case", [c for c in REF["simulations"] if c["true_delta"] == 0])
def test_type_i_error_stays_near_alpha(case):
    # Kieser and Friede (2003): the blinded rule leaves the type I error essentially at alpha.
    assert case["rejection_rate"] < case["alpha_one_sided"] + 0.003


def test_no_room_to_adapt_reproduces_the_fixed_t_test():
    # Floor and cap both at the planned size: the fixed design, whose power is exact.
    d = BlindedSSR(0.5, 1.0, 60, max_factor=1.0)
    assert d.n_control_floor == d.n_control_cap == d.planned_n_control == 63
    exact = fixed.power_normal(63, 0.5, 1.0, 0.05, method="t")
    sim = d.simulate(n_sim=400_000, seed=3)
    assert abs(sim.rejection_rate - exact) <= 2 * sim.rejection_se
    assert sim.mean_n_total == 126
    h0 = d.simulate(0.0, n_sim=400_000, seed=4)
    assert abs(h0.rejection_rate - 0.025) <= 2 * h0.rejection_se


def test_adjusted_variance_subtracts_the_bias():
    d = BlindedSSR(0.5, 1.0, 60, adjusted=True)
    assert d.bias_correction == pytest.approx(0.25 * 60 / (4 * 59))
    res = d.reestimate(1.5)
    assert res.variance_used == pytest.approx(1.5 - d.bias_correction)
    k = 2 * (1.959963984540054 + 0.8416212335729143) ** 2 / 0.25
    assert res.n_control_required == math.ceil(k * res.variance_used)


def test_lumped_variance_overestimates_by_the_between_arm_term():
    rng = np.random.default_rng(1)
    c, t = rng.normal(0, 1, 30), rng.normal(0.5, 1, 30)
    s2 = lumped_variance(np.concatenate([c, t]))
    within = ((c - c.mean()) ** 2).sum() + ((t - t.mean()) ** 2).sum()
    between = 30 * 30 / 60 * (t.mean() - c.mean()) ** 2
    assert s2 == pytest.approx((within + between) / 59)


def test_bounds_floor_and_cap():
    d = BlindedSSR(0.5, 1.0, 60)
    assert d.reestimate(0.1).n_control == 63 and d.reestimate(0.1).at_floor
    assert d.reestimate(100).n_control == 126 and d.reestimate(100).at_cap
    k = BlindedSSR(1.0, 1.0, 10, floor="pilot", max_factor=3)
    assert k.reestimate(0.01).n_control == 5


def test_negative_delta_tests_the_other_direction():
    up = BlindedSSR(0.5, 1.0, 60).simulate(n_sim=50_000, seed=5)
    down = BlindedSSR(-0.5, 1.0, 60).simulate(n_sim=50_000, seed=5)
    assert down.rejection_rate == pytest.approx(up.rejection_rate, abs=0.02)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"delta": 0.0}, "delta"),
        ({"sd": -1}, "sd"),
        ({"pilot_n": 3}, "two patients"),
        ({"pilot_n": 30.5}, "whole"),
        ({"max_factor": 0.5}, "max_factor"),
        ({"pilot_n": 400}, "cap"),
        ({"floor": "zero"}, "floor"),
    ],
)
def test_invalid_designs(kwargs, match):
    args = {"delta": 0.5, "sd": 1.0, "pilot_n": 60, **kwargs}
    with pytest.raises(ValueError, match=match):
        BlindedSSR(**args)


# ------------------------------------------------------------------------------- CLI


def test_cli_blinded_ssr_json():
    out = CliRunner().invoke(
        app,
        [
            "ssr", "--blinded", "--endpoint", "normal", "--delta", "0.5", "--sd", "1",
            "--pilot-n", "60", "--pilot-sd", "1.2", "--nsim", "20000", "--seed", "7", "--json",
        ],
    )  # fmt: skip
    assert out.exit_code == 0, out.output
    s = json.loads(out.output)
    assert s["design"] == "blinded_ssr"
    assert s["planned_n_control"] == 63
    k = 2 * (stats.norm.isf(0.025) + stats.norm.isf(0.2)) ** 2 / 0.25
    assert s["interim"]["n_control"] == math.ceil(k * 1.2**2)
    rows = s["operating_characteristics"]
    assert [r["true_sd"] for r in rows] == [0.75, 1.0, 1.25, 1.5]
    assert all(abs(r["type_i_error"] - 0.025) < 0.01 for r in rows)


def test_cli_blinded_text_and_example():
    out = CliRunner().invoke(app, ["ssr", "--config", EXAMPLE, "--nsim", "5000"])
    assert out.exit_code == 0, out.output
    assert "blinded sample size re-estimation" in out.output
    assert "Type I error" in out.output


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        (
            ["--endpoint", "binary", "--pc", "0.3", "--pt", "0.4", "--pilot-n", "60"],
            "normal endpoint",
        ),
        (["--endpoint", "normal", "--delta", "0.5", "--sd", "1"], "--pilot-n"),
        (
            ["--endpoint", "normal", "--delta", "0.5", "--sd", "1", "--pilot-n", "60", "--z1", "1"],
            "--z1",
        ),
    ],
)
def test_cli_blinded_errors(extra, match):
    out = CliRunner().invoke(app, ["ssr", "--blinded", *extra])
    assert out.exit_code == 2
    assert match in out.output


def test_unblinded_keys_are_refused_with_blinded_and_vice_versa():
    base = ["ssr", "--endpoint", "normal", "--delta", "0.5", "--sd", "1"]
    out = CliRunner().invoke(app, [*base, "--blinded", "--pilot-n", "60", "--cp-min", "0.3"])
    assert out.exit_code == 2 and "--cp-min" in out.output
    out = CliRunner().invoke(app, [*base, "--pilot-n", "60"])
    assert out.exit_code == 2 and "needs --blinded" in out.output
    out = CliRunner().invoke(app, ["report", "--config", EXAMPLE])
    assert out.exit_code == 2 and "blinded" in out.output
