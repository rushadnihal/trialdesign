"""M15b: equivalence by two one-sided tests (TOST) for fixed normal and binary designs.

No R package for TOST sample sizes is installed in the reference environment, so the checks
are the closed form at a zero true difference (to 1e-4) and simulation of the actual tests.
The normal power formula is exact for a known SD, so the simulation must agree within the
Monte Carlo error (3 SE, as there are several comparisons). The binary formula evaluates the
Farrington-Manning variances at the design rates, while the test uses the observed rates; with
10^6 trials that makes it conservative by 0.1 to 0.5 points at about 250 per arm, so the
simulated power must be no lower than the formula's (within 2 MC SE) and at most one point
higher.
"""

import math

import numpy as np
import pytest
from scipy import stats

from trialdesign.equivalence import (
    power_equivalence_binary,
    power_equivalence_normal,
    sample_size_equivalence_binary,
    sample_size_equivalence_normal,
    tost,
)
from trialdesign.patients import _fm_rates_vec
from trialdesign.simulate import mc_se

N_SIM = 40_000


@pytest.mark.parametrize(
    ("margin", "sd", "alpha", "power", "allocation"),
    [
        (0.5, 1.0, 0.05, 0.80, 1.0),
        (0.3, 1.2, 0.05, 0.90, 1.0),
        (1.0, 2.0, 0.025, 0.80, 2.0),
        (0.2, 0.5, 0.05, 0.85, 0.5),
    ],
)
def test_normal_matches_closed_form_at_zero(margin, sd, alpha, power, allocation):
    n = sample_size_equivalence_normal(margin, sd, 0.0, alpha, power, allocation)
    z = stats.norm.isf(alpha) + stats.norm.isf((1 - power) / 2)
    closed = (1 + 1 / allocation) * sd**2 * z**2 / margin**2
    assert n.n_control == pytest.approx(closed, abs=1e-4)
    assert n.n_treatment == pytest.approx(allocation * closed, abs=1e-4)
    assert power_equivalence_normal(n.n_control, margin, sd, 0.0, alpha, allocation) == (
        pytest.approx(power, abs=1e-8)
    )


def _simulate_normal(n_c, n_t, delta, sd, lo, hi, alpha, rng):
    diff = rng.normal(delta, sd * math.sqrt(1 / n_c + 1 / n_t), N_SIM)
    se = sd * math.sqrt(1 / n_c + 1 / n_t)
    z = stats.norm.isf(alpha)
    return float(np.mean(((diff - lo) / se >= z) & ((diff - hi) / se <= -z)))


@pytest.mark.parametrize(
    ("margin", "delta", "allocation"),
    [(0.5, 0.1, 1.0), ((-0.4, 0.6), 0.0, 2.0)],
)
def test_normal_power_by_simulation(margin, delta, allocation):
    n = sample_size_equivalence_normal(margin, 1.0, delta, allocation=allocation)
    n_c = math.ceil(n.n_control)
    n_t = math.ceil(allocation * n_c)
    lo, hi = (-margin, margin) if np.ndim(margin) == 0 else margin
    exact = power_equivalence_normal(n_c, margin, 1.0, delta, allocation=n_t / n_c)
    sim = _simulate_normal(n_c, n_t, delta, 1.0, lo, hi, 0.05, np.random.default_rng(1))
    assert abs(sim - exact) <= 3 * float(mc_se(exact, N_SIM))


def _simulate_binary(n_c, n_t, pc, pt, lo, hi, alpha, rng):
    """The actual TOST: Farrington-Manning SE at each margin, from the observed rates."""
    xc = rng.binomial(n_c, pc, N_SIM) / n_c
    xt = rng.binomial(n_t, pt, N_SIM) / n_t
    r = np.full(N_SIM, n_t / n_c)
    z = stats.norm.isf(alpha)
    ok = np.ones(N_SIM, dtype=bool)
    for margin, sign in ((lo, 1.0), (hi, -1.0)):
        c0, t0 = _fm_rates_vec(xc, xt, r, margin)
        se = np.sqrt(c0 * (1 - c0) / n_c + t0 * (1 - t0) / n_t)
        stat = np.divide(xt - xc - margin, se, out=np.zeros(N_SIM), where=se > 0)
        ok &= sign * stat >= z
    return float(ok.mean())


@pytest.mark.parametrize(
    ("pc", "pt", "margin", "allocation"),
    [(0.8, 0.8, 0.1, 1.0), (0.75, 0.8, 0.15, 1.0), (0.6, 0.6, (-0.1, 0.15), 2.0)],
)
def test_binary_power_by_simulation(pc, pt, margin, allocation):
    n = sample_size_equivalence_binary(pc, pt, margin, allocation=allocation)
    n_c = math.ceil(n.n_control)
    n_t = math.ceil(allocation * n_c)
    lo, hi = (-margin, margin) if np.ndim(margin) == 0 else margin
    exact = power_equivalence_binary(n_c, pc, pt, margin, allocation=n_t / n_c)
    assert exact >= 0.80
    sim = _simulate_binary(n_c, n_t, pc, pt, lo, hi, 0.05, np.random.default_rng(2))
    se = float(mc_se(exact, N_SIM))
    assert exact - 2 * se <= sim <= exact + 0.01 + 2 * se


def test_type_i_error_at_a_margin():
    """At the margin the rejection rate is at most alpha (about alpha for large n)."""
    n_c = 400
    sim = _simulate_normal(n_c, n_c, 0.5, 1.0, -0.5, 0.5, 0.05, np.random.default_rng(3))
    assert sim <= 0.05 + 2 * float(mc_se(0.05, N_SIM))
    assert power_equivalence_normal(n_c, 0.5, 1.0, 0.5) == pytest.approx(0.05, abs=1e-3)


def test_wide_margins_give_power_near_one():
    assert power_equivalence_normal(50, 1e3, 1.0) == pytest.approx(1.0)
    assert power_equivalence_binary(50, 0.5, 0.5, 0.99) > 0.999


def test_tost_matches_the_confidence_interval_rule():
    for est in (-0.3, 0.0, 0.25, 0.45):
        res = tost(est, 0.15, 0.5)
        inside = -0.5 < res.ci_lower and res.ci_upper < 0.5
        assert res.equivalent == inside
        assert res.level == pytest.approx(0.90)
    res = tost(0.1, 0.1, (-0.3, 0.2))
    assert res.p_value == pytest.approx(stats.norm.cdf((0.1 - 0.2) / 0.1))


def test_input_checks():
    with pytest.raises(ValueError, match="inside"):
        sample_size_equivalence_normal(0.5, 1.0, delta=0.6)
    with pytest.raises(ValueError, match="lower < 0 < upper"):
        sample_size_equivalence_normal((0.1, 0.5), 1.0)
    with pytest.raises(ValueError, match="pair"):
        sample_size_equivalence_normal((-0.1, 0.1, 0.2), 1.0)
    with pytest.raises(ValueError, match="alpha"):
        sample_size_equivalence_binary(0.5, 0.5, 0.1, alpha=0.6)


# --------------------------------------------------------------------------- CLI


def test_cli_design_margin():
    import json

    from typer.testing import CliRunner

    from trialdesign.cli import app

    runner = CliRunner()
    base = ["design", "--endpoint", "normal", "--delta", "0", "--sd", "1", "--margin", "0.5"]
    result = runner.invoke(app, [*base, "--alpha-two-sided", "0.1", "--json"])
    assert result.exit_code == 0, result.output
    out = json.loads(result.output)
    assert out["margin"] == [-0.5, 0.5] and out["alpha_each_test"] == pytest.approx(0.05)
    assert out["n_control"] == pytest.approx(sample_size_equivalence_normal(0.5, 1.0).n_control)
    result = runner.invoke(app, [*base, "--looks", "2"])
    assert result.exit_code == 2 and "fixed designs" in result.output
    result = runner.invoke(app, [*base, "--delta0", "-0.1"])
    assert result.exit_code == 2 and "not both" in result.output
    result = runner.invoke(
        app, ["design", "--endpoint", "binary", "--pc", "0.8", "--pt", "0.8", "--margin=-0.1,0.12"]
    )
    assert result.exit_code == 0 and "equivalence" in result.output
    result = runner.invoke(
        app, ["design", "--endpoint", "survival", "--hr", "1", "--margin", "0.2"]
    )
    assert result.exit_code == 2
