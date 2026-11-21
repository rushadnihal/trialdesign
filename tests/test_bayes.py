"""M4: Bayesian beta-binomial interim analysis.

Acceptance criteria (project outline, M4): posterior P(p_T > p_C), PPoS, go/no-go and
assurance are tested, and the frequentist operating characteristics are simulated with the M3
engine. Reference values in ``tests/bayes_reference_values.json`` come from an independent
base-R implementation (``tests/r/bayes_reference_values.R``). The posterior probability is
also checked against the exact closed form for integer parameters, computed here in rational
arithmetic, and against Monte Carlo. Simulated rates are compared with the exact values at
4 Monte Carlo SE.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from trialdesign import bayes, simulate

REF = json.loads((Path(__file__).parent / "bayes_reference_values.json").read_text())["bayes"]
POST_REF = [c for c in REF if "prob" in c]
PPOS_REF = [c for c in REF if c.get("kind") == "ppos"]
OC_REF = [c for c in REF if c.get("kind") in ("oc", "assurance")]


def _design(case: dict) -> bayes.BayesianInterimDesign:
    return bayes.bayesian_interim_design(
        case["n_interim_control"],
        case["n_final_control"],
        allocation=case["n_final_treatment"] / case["n_final_control"],
        prior_control=tuple(case["prior_c"]),
        prior_treatment=tuple(case["prior_t"]),
        success_threshold=case["success_threshold"],
        go_threshold=case["go_threshold"],
        nogo_threshold=case["nogo_threshold"],
    )


def _law(value: list[float] | float) -> bayes.DesignPrior:
    return tuple(value) if isinstance(value, list) and len(value) == 2 else float(value[0])


@pytest.fixture(scope="module")
def design() -> bayes.BayesianInterimDesign:
    return bayes.bayesian_interim_design(30, 60)


# --------------------------------------------------------------- posterior probability


@pytest.mark.parametrize("case", POST_REF, ids=lambda c: c["id"])
def test_posterior_probability_matches_r(case: dict) -> None:
    p = bayes.prob_superiority(
        case["x_c"],
        case["n_c"],
        case["x_t"],
        case["n_t"],
        tuple(case["prior_c"]),
        tuple(case["prior_t"]),
    )
    assert p == pytest.approx(case["prob"], abs=1e-9)


def test_outline_reference_value() -> None:
    # Validation plan: 12/30 vs 6/30 with Beta(1, 1) priors gives about 0.951 (tol 0.002).
    assert bayes.prob_superiority(6, 30, 12, 30) == pytest.approx(0.951, abs=0.002)


def _exact_prob_superiority(a_c: int, b_c: int, a_t: int, b_t: int) -> Fraction:
    """Closed form for integer parameters, in exact rational arithmetic.

    P(p_T > p_C) = sum_{i=0}^{a_t-1} B(a_c+i, b_c+b_t) / ((b_t+i) B(1+i, b_t) B(a_c, b_c))
    (Cook 2003, "Exact calculation of beta inequalities"; Miller 2015).
    """

    def beta(a: int, b: int) -> Fraction:
        return Fraction(math.factorial(a - 1) * math.factorial(b - 1), math.factorial(a + b - 1))

    return sum(
        (beta(a_c + i, b_c + b_t) / ((b_t + i) * beta(1 + i, b_t) * beta(a_c, b_c)))
        for i in range(a_t)
    )


@pytest.mark.parametrize(
    "x_c, n_c, x_t, n_t",
    [(6, 30, 12, 30), (0, 5, 5, 5), (40, 200, 3, 10), (9, 10, 1, 10), (50, 100, 50, 100)],
)
def test_posterior_probability_matches_closed_form(x_c, n_c, x_t, n_t) -> None:
    exact = _exact_prob_superiority(1 + x_c, 1 + n_c - x_c, 1 + x_t, 1 + n_t - x_t)
    assert bayes.prob_superiority(x_c, n_c, x_t, n_t) == pytest.approx(float(exact), abs=1e-9)


def test_posterior_probability_matches_monte_carlo() -> None:
    p, se = bayes.prob_superiority_mc(6, 30, 12, 30, n_draws=1_000_000, seed=2026)
    assert abs(p - bayes.prob_superiority(6, 30, 12, 30)) <= 4 * se


def test_posterior_probability_properties() -> None:
    assert bayes.prob_superiority(0, 0, 0, 0) == pytest.approx(0.5, abs=1e-12)
    assert bayes.prob_superiority(7, 20, 7, 20) == pytest.approx(0.5, abs=1e-10)
    # Swapping the arms gives the complement.
    p = bayes.prob_superiority(4, 20, 9, 25, (2, 3), (1, 1))
    q = bayes.prob_superiority(9, 25, 4, 20, (1, 1), (2, 3))
    assert p + q == pytest.approx(1.0, abs=1e-10)
    # Vectorised over treatment responders, and increasing in them.
    probs = bayes.prob_superiority(10, 30, np.arange(31), 30)
    assert probs.shape == (31,)
    assert np.all(np.diff(probs) > 0)


def test_posterior_parameters() -> None:
    assert bayes.posterior(12, 30) == (13.0, 19.0)
    assert bayes.posterior(2, 10, (0.5, 0.5)) == (2.5, 8.5)


# ------------------------------------------------------- predictive probability (PPoS)


@pytest.mark.parametrize("case", PPOS_REF, ids=lambda c: c["id"])
def test_ppos_matches_r(case: dict) -> None:
    d = _design(case)
    assert d.ppos(case["x_c"], case["x_t"]) == pytest.approx(case["ppos"], abs=1e-9)
    standalone = bayes.predictive_probability(
        case["x_c"],
        d.n_interim_control,
        case["x_t"],
        d.n_interim_treatment,
        d.n_final_control,
        d.n_final_treatment,
        d.prior_control,
        d.prior_treatment,
        d.success_threshold,
    )
    assert standalone == pytest.approx(case["ppos"], abs=1e-9)


def test_ppos_matches_monte_carlo(design) -> None:
    # Draw the true rates from the posterior, then the remaining patients, as in the outline.
    # The final rule is read from the success table, which test_success_table_matches_brute_force
    # checks against prob_superiority cell by cell.
    rng = np.random.default_rng(11)
    n = 400_000
    p_c, p_t = rng.beta(1 + 6, 1 + 24, n), rng.beta(1 + 12, 1 + 18, n)
    x_c, x_t = 6 + rng.binomial(30, p_c), 12 + rng.binomial(30, p_t)
    win = design.success_table[x_c, x_t]
    ppos = bayes.predictive_probability(6, 30, 12, 30, 60, 60)
    assert abs(win.mean() - ppos) <= 4 * simulate.mc_se(ppos, n)


def test_ppos_with_no_patients_left_is_zero_or_one() -> None:
    for x_c, x_t in [(6, 12), (10, 10), (3, 14), (8, 13)]:
        ppos = bayes.predictive_probability(x_c, 30, x_t, 30, 30, 30)
        win = bayes.prob_superiority(x_c, 30, x_t, 30) >= 0.95
        assert ppos == float(win)


def test_ppos_increases_with_treatment_successes(design) -> None:
    table = design.ppos_table
    assert np.all(np.diff(table, axis=1) >= -1e-12)  # up in treatment responders
    assert np.all(np.diff(table, axis=0) <= 1e-12)  # down in control responders
    assert table[0, -1] == pytest.approx(1.0) and table[-1, 0] == pytest.approx(0.0)


def test_success_table_matches_brute_force() -> None:
    d = bayes.bayesian_interim_design(8, 20, 1.5, (0.5, 0.5), (2, 1), success_threshold=0.9)
    x_c = np.arange(21)[:, None]
    x_t = np.arange(31)[None, :]
    brute = bayes.prob_superiority(x_c, 20, x_t, 30, (0.5, 0.5), (2, 1)) >= 0.9
    np.testing.assert_array_equal(d.success_table, brute)


# ------------------------------------------------------------------------ go / no-go


def test_decision_rule() -> None:
    assert bayes.decide(0.95) == "go"
    assert bayes.decide(0.9) == "go"
    assert bayes.decide(0.5) == "continue"
    assert bayes.decide(0.1) == "no-go"
    assert bayes.decide(0.02, go_threshold=0.8, nogo_threshold=0.05) == "no-go"


def test_design_decisions_follow_ppos(design) -> None:
    for x_c, x_t in [(6, 12), (9, 9), (3, 15), (12, 6)]:
        assert design.decide(x_c, x_t) == bayes.decide(design.ppos(x_c, x_t))
    table = design.decision_table
    assert set(np.unique(table)) <= {-1, 0, 1}
    assert table[0, -1] == 1 and table[-1, 0] == -1


# ------------------------------------------------- operating characteristics, assurance


@pytest.mark.parametrize("case", OC_REF, ids=lambda c: c["id"])
def test_exact_operating_characteristics_match_r(case: dict) -> None:
    d = _design(case)
    if case["kind"] == "oc":
        oc = d.operating_characteristics(case["p_control"], case["p_treatment"])
    else:
        oc = d.assurance(_law(case["design_prior_control"]), _law(case["design_prior_treatment"]))
    for key in ("prob_go", "prob_nogo", "prob_continue", "prob_final_success"):
        assert getattr(oc, key) == pytest.approx(case[key], abs=1e-9), key


def test_exact_oc_bookkeeping(design) -> None:
    oc = design.operating_characteristics(0.3, 0.45)
    assert oc.prob_go + oc.prob_nogo + oc.prob_continue == pytest.approx(1.0)
    assert oc.prob_success == pytest.approx(oc.prob_go + oc.prob_final_success)
    assert oc.expected_n_control == pytest.approx(30 + oc.prob_continue * 30)
    # A point-mass design prior reduces assurance to power.
    assert design.assurance(0.3, 0.45).prob_success == pytest.approx(oc.prob_success)


@pytest.mark.parametrize("p_t", [0.3, 0.45, 0.55])
def test_simulated_ocs_match_exact(design, p_t) -> None:
    exact = design.operating_characteristics(0.3, p_t)
    sim = bayes.simulate_bayesian(design, 0.3, p_t, n_sim=200_000, seed=2026)
    assert isinstance(sim, simulate.OperatingCharacteristics)
    expected = {
        "efficacy": [exact.prob_go, exact.prob_final_success],
        "futility": [exact.prob_nogo, exact.prob_continue - exact.prob_final_success],
    }
    for kind, sim_p in (("efficacy", sim.prob_efficacy), ("futility", sim.prob_futility)):
        p = np.asarray(expected[kind])
        assert np.all(np.abs(sim_p - p) <= 4 * np.maximum(simulate.mc_se(p, sim.n_sim), 1e-5))
    assert abs(sim.reject_prob - exact.prob_success) <= 4 * sim.reject_prob_se
    n_total = design.n_final_control + design.n_final_treatment
    e_n = exact.expected_n_control + exact.expected_n_treatment
    assert abs(sim.expected_info_fraction * n_total - e_n) <= 4 * (
        sim.expected_info_fraction_se * n_total
    )


@pytest.mark.parametrize("laws", [((30, 70), (45, 55)), (0.3, (4.5, 5.5))])
def test_simulated_assurance_matches_exact(design, laws) -> None:
    exact = design.assurance(*laws).prob_success
    sim = bayes.simulate_assurance(design, *laws, n_sim=200_000, seed=7)
    assert abs(sim.reject_prob - exact) <= 4 * sim.reject_prob_se


def test_assurance_is_power_averaged_over_the_prior(design) -> None:
    # Average exact power over a discretised Beta(9, 11) design prior on p_T (control fixed).
    grid = np.linspace(0.0005, 0.9995, 1000)
    w = np.exp((9 - 1) * np.log(grid) + (11 - 1) * np.log1p(-grid))
    w /= w.sum()
    power = np.array([design.operating_characteristics(0.3, p).prob_success for p in grid])
    assert design.assurance(0.3, (9, 11)).prob_success == pytest.approx(w @ power, abs=1e-4)


def test_simulation_is_seeded(design) -> None:
    a = bayes.simulate_bayesian(design, 0.3, 0.45, n_sim=10_000, seed=3)
    b = bayes.simulate_bayesian(design, 0.3, 0.45, n_sim=10_000, seed=3)
    np.testing.assert_array_equal(a.prob_efficacy, b.prob_efficacy)


# ---------------------------------------------------------------------- validation


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="between 0 and the number"):
        bayes.prob_superiority(31, 30, 1, 30)
    with pytest.raises(ValueError, match="whole numbers"):
        bayes.prob_superiority(1.5, 30, 1, 30)
    with pytest.raises(ValueError, match="positive Beta"):
        bayes.prob_superiority(1, 30, 1, 30, prior_c=(0, 1))
    with pytest.raises(ValueError, match="at least the interim"):
        bayes.predictive_probability(1, 30, 1, 30, 20, 40)
    with pytest.raises(ValueError, match="below go_threshold"):
        bayes.bayesian_interim_design(10, 20, go_threshold=0.5, nogo_threshold=0.6)
    with pytest.raises(ValueError, match="strictly between"):
        bayes.bayesian_interim_design(10, 20, success_threshold=1.0)
    with pytest.raises(ValueError, match="n_interim < n_final"):
        bayes.bayesian_interim_design(20, 20)
    with pytest.raises(ValueError, match="response rate"):
        bayes.bayesian_interim_design(10, 20).assurance(1.5, (1, 1))
