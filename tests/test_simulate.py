"""M3: vectorised simulation of operating characteristics.

Acceptance criteria (project outline, M3): 100,000 trials in under 10 s, and simulated type I
error within 2 Monte Carlo SE of the nominal one-sided 0.025. Simulated rates are also checked
against the numerical-integration values of ``boundaries.crossing_probabilities``, which M2
validated against rpact and gsDesign. Every simulation is seeded, so each check is
deterministic; comparisons against exact values use 3 or 4 MC SE so that a change of seed
would rarely fail them by chance. Expected sample sizes are compared with gsDesign once
``tests/reference_values.json`` has been regenerated with ``tests/r/reference_values.R``.
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from trialdesign import boundaries, fixed, simulate

REF = json.loads((Path(__file__).parent / "reference_values.json").read_text()).get(
    "boundaries", []
)
EXPECTED_N_REF = [c for c in REF if "expected_n_h0" in c]
NEEDS_R = "regenerate reference_values.json with gsDesign (expected sample sizes)"

ALPHA = 0.025
N_SIM = 200_000

DESIGNS = {
    "obf_k3": dict(info_rates=3, efficacy_spending="obf"),
    "pocock_k3": dict(info_rates=3, efficacy_spending="pocock"),
    "obf_k5": dict(info_rates=5, efficacy_spending="obf"),
    "pocock_k4_uneven": dict(info_rates=[0.2, 0.45, 0.8, 1.0], efficacy_spending="pocock"),
    "obf_obf_k3": dict(info_rates=3, efficacy_spending="obf", futility_spending="obf"),
    "obf_pocock_k3": dict(info_rates=3, efficacy_spending="obf", futility_spending="pocock"),
    "pocock_obf_t30_70": dict(
        info_rates=[0.3, 0.7, 1.0], efficacy_spending="pocock", futility_spending="obf"
    ),
}


@pytest.fixture(scope="module", params=list(DESIGNS), ids=list(DESIGNS))
def sim(request) -> simulate.DesignSimulation:
    d = boundaries.group_sequential_design(
        alpha_one_sided=ALPHA, power=0.8, **DESIGNS[request.param]
    )
    return simulate.simulate_design(d, n_sim=N_SIM, seed=2026)


# ------------------------------------------------------------------ acceptance criteria


def test_type_i_error_within_two_mc_se(sim: simulate.DesignSimulation) -> None:
    oc = sim.h0_ignore_futility
    assert abs(oc.reject_prob - ALPHA) <= 2 * simulate.mc_se(ALPHA, oc.n_sim)


def test_power_within_three_mc_se(sim: simulate.DesignSimulation) -> None:
    oc = sim.h1
    assert abs(oc.reject_prob - 0.8) <= 3 * simulate.mc_se(0.8, oc.n_sim)


def test_100k_trials_under_10_seconds() -> None:
    d = boundaries.group_sequential_design(5, futility_spending="obf")
    start = time.perf_counter()
    simulate.simulate_trials(d.info_rates, d.efficacy, d.futility, d.drift, 100_000, seed=1)
    assert time.perf_counter() - start < 10.0


# ------------------------------------------------- agreement with numerical integration


def _check_against_numerics(oc: simulate.OperatingCharacteristics, d, futility) -> None:
    up, low = boundaries.crossing_probabilities(d.info_rates, d.efficacy, futility, oc.drift)
    se_up = np.maximum(simulate.mc_se(up, oc.n_sim), 1e-5)
    se_low = np.maximum(simulate.mc_se(low, oc.n_sim), 1e-5)
    assert np.all(np.abs(oc.prob_efficacy - up) <= 4 * se_up)
    assert np.all(np.abs(oc.prob_futility - low) <= 4 * se_low)
    e_t = boundaries.expected_info_fraction(d.info_rates, d.efficacy, futility, oc.drift)
    assert abs(oc.expected_info_fraction - e_t) <= 4 * oc.expected_info_fraction_se


def test_per_look_probabilities_match_numerics(sim: simulate.DesignSimulation) -> None:
    d = sim.design
    _check_against_numerics(sim.h0_ignore_futility, d, None)
    _check_against_numerics(sim.h0, d, d.futility)
    _check_against_numerics(sim.h1, d, d.futility)


@pytest.mark.skipif(not EXPECTED_N_REF, reason=NEEDS_R)
@pytest.mark.parametrize("case", EXPECTED_N_REF, ids=lambda c: c["id"])
def test_expected_sample_size_matches_gsdesign(case: dict) -> None:
    d = boundaries.group_sequential_design(
        case["info_rates"],
        case["alpha_one_sided"],
        case["power"],
        efficacy_spending=case.get("efficacy_spending", case.get("spending")),
        futility_spending=case.get("futility_spending"),
    )
    en_h0 = d.inflation_factor * d.expected_info_fraction(drift=0.0)
    en_h1 = d.inflation_factor * d.expected_info_fraction()
    assert en_h0 == pytest.approx(case["expected_n_h0"], abs=1e-3)
    assert en_h1 == pytest.approx(case["expected_n_h1"], abs=1e-3)


# ----------------------------------------------------------------------- properties


def test_canonical_joint_distribution() -> None:
    t = np.array([0.25, 0.6, 1.0])
    z = simulate.simulate_z(t, drift=2.5, n_sim=400_000, seed=7)
    np.testing.assert_allclose(z.mean(axis=0), 2.5 * np.sqrt(t), atol=0.01)
    cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    np.testing.assert_allclose(np.cov(z, rowvar=False), cov, atol=0.01)


def test_single_look_is_fixed_design() -> None:
    c = stats.norm.isf(ALPHA)
    oc = simulate.simulate_trials(1, [c], n_sim=N_SIM, seed=3)
    assert abs(oc.reject_prob - ALPHA) <= 3 * oc.reject_prob_se
    assert oc.expected_info_fraction == 1.0
    assert oc.expected_info_fraction_se == 0.0


def test_following_futility_never_raises_rejections(sim: simulate.DesignSimulation) -> None:
    # Common random numbers: every path that rejects with futility followed also rejects
    # when futility is ignored, so the inequality is exact, not just in expectation.
    assert sim.h0.reject_prob <= sim.h0_ignore_futility.reject_prob
    if sim.design.futility is None:
        assert sim.h0 is sim.h0_ignore_futility


def test_stop_probabilities_sum_to_one(sim: simulate.DesignSimulation) -> None:
    for oc in (sim.h0_ignore_futility, sim.h0, sim.h1):
        assert oc.prob_stop.sum() == pytest.approx(1.0)
        assert np.all(oc.prob_stop >= 0)
        assert 0 < oc.expected_info_fraction <= 1


def test_expected_sample_size_scales() -> None:
    d = boundaries.group_sequential_design(3, futility_spending="obf")
    oc = simulate.simulate_trials(d.info_rates, d.efficacy, d.futility, d.drift, 10_000, seed=5)
    n_max = d.max_sample_size(fixed.events_survival(0.70).events)
    en, se = oc.expected_sample_size(n_max)
    assert en == pytest.approx(n_max * oc.expected_info_fraction)
    assert se == pytest.approx(n_max * oc.expected_info_fraction_se)
    with pytest.raises(ValueError, match="positive"):
        oc.expected_sample_size(0)


def test_seed_reproducible_and_chunk_invariant(monkeypatch) -> None:
    d = boundaries.group_sequential_design(3, futility_spending="pocock")
    args = (d.info_rates, d.efficacy, d.futility, d.drift, 30_001)
    a = simulate.simulate_trials(*args, seed=11)
    b = simulate.simulate_trials(*args, seed=11)
    monkeypatch.setattr(simulate, "_CHUNK", 7_000)
    c = simulate.simulate_trials(*args, seed=11)
    for other in (b, c):
        np.testing.assert_array_equal(a.prob_efficacy, other.prob_efficacy)
        np.testing.assert_array_equal(a.prob_futility, other.prob_futility)
        assert a.expected_info_fraction == pytest.approx(other.expected_info_fraction, abs=1e-12)


def test_unseeded_design_simulation_records_its_seed() -> None:
    d = boundaries.group_sequential_design(2)
    first = simulate.simulate_design(d, n_sim=5_000)
    again = simulate.simulate_design(d, n_sim=5_000, seed=first.seed)
    assert first.type_i_error == again.type_i_error
    assert first.power == again.power


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="one boundary per look"):
        simulate.simulate_trials(3, [2.0, 2.0])
    with pytest.raises(ValueError, match="must not exceed"):
        simulate.simulate_trials(2, [2.0, 2.0], [2.5, 2.0])
    with pytest.raises(ValueError, match="positive integer"):
        simulate.simulate_trials(1, [1.96], n_sim=0)
    with pytest.raises(ValueError, match="new patient"):
        simulate.simulate_binary(0.3, 0.4, 1, 3, [3.0, 2.5, 2.0])
    with pytest.raises(ValueError, match="p_control"):
        simulate.simulate_binary(1.2, 0.4, 100, 1, [1.96])


# -------------------------------------------------------- patient-level binary check


def test_binary_patient_level_matches_canonical_at_large_n() -> None:
    d = boundaries.group_sequential_design(3, futility_spending="obf")
    n_fix = fixed.sample_size_binary(0.30, 0.45, alpha_two_sided=2 * ALPHA).n_control
    n_max = math.ceil(d.max_sample_size(n_fix))
    h0 = simulate.simulate_binary(0.30, 0.30, n_max, d.info_rates, d.efficacy, None, seed=1)
    h1 = simulate.simulate_binary(0.30, 0.45, n_max, d.info_rates, d.efficacy, d.futility, seed=2)
    assert h0.drift is None
    # Normal approximation at about 180 patients per arm: within 0.004 of the design values.
    assert abs(h0.reject_prob - ALPHA) < 0.004
    assert abs(h1.reject_prob - 0.80) < 0.015


def test_binary_degenerate_responses_give_zero_z() -> None:
    oc = simulate.simulate_binary(0.0, 0.0, 50, 2, [2.5, 2.0], n_sim=1_000, seed=0)
    assert oc.reject_prob == 0.0
