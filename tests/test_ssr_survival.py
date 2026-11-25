"""M14: promising-zone re-estimation of the number of events in survival trials.

Reference values in ``tests/ssr_survival_reference_values.json`` come from
``tests/r/ssr_survival_reference_values.R``: ``rpact::getConditionalPower`` for a survival
dataset, and ``rpact::getSimulationSurvival`` with event re-estimation (100,000 trials per case),
which is the promising-zone rule with ``cp_min = 0``. The simulated comparisons allow 3 MC SE of
the difference, as there are about 25 of them.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from trialdesign import accrual
from trialdesign.patients import simulate_survival, simulate_survival_ssr
from trialdesign.simulate import mc_se
from trialdesign.ssr import SurvivalSSR, survival_ssr

REF = json.loads((Path(__file__).parent / "ssr_survival_reference_values.json").read_text())
N_SIM = 20_000


def _close(ours: float, ours_se: float, ref: float, ref_se: float, k: float = 3.0) -> None:
    band = k * math.hypot(ours_se, ref_se)
    assert abs(ours - ref) <= band, f"{ours:.5g} vs {ref:.5g} (band {band:.3g})"


def _study(hr=0.7, median=12.0, accrual_time=18.0, allocation=1.0, dropout=0.0, events=248):
    return accrual.survival_study(
        events,
        hr,
        median,
        accrual_time=accrual_time,
        followup=12,
        allocation=allocation,
        dropout=dropout,
    )


def _adaptive(case: dict) -> SurvivalSSR:
    study = _study(
        case["design_hazard_ratio"],
        case["median_control"],
        case["accrual_time"],
        case["allocation"],
        case["dropout"],
        case["events"],
    )
    return survival_ssr(
        study,
        events=case["events"],
        n_total=case["n_total"],
        futility_spending=None if case["futility"] is None else "obf",
        cp_min=0.0,
        max_factor=case["max_factor"],
    )


# --------------------------------------------------------------------------- vs rpact


@pytest.mark.parametrize("case", REF["conditional_power"])
def test_conditional_power_matches_rpact(case):
    d1 = case["interim_events"]
    study = _study(case["hazard_ratio"], allocation=case["allocation"])
    adaptive = survival_ssr(study, events=2 * d1, n_total=20 * d1)
    # nPlanned stage-2 events as a multiple of the planned (1 - t1) D
    m = case["stage2_events"] / d1
    cp = adaptive.design.conditional_power(case["z1"], adaptive.drift(case["hazard_ratio"]), m)
    assert float(cp) == pytest.approx(case["conditional_power"], abs=1e-6)


@pytest.fixture(scope="module", params=REF["simulation"], ids=[c["id"] for c in REF["simulation"]])
def pair(request):
    case = request.param
    adaptive = _adaptive(case)
    return (
        case,
        adaptive,
        simulate_survival_ssr(adaptive, case["hazard_ratio"], n_sim=N_SIM, seed=5),
    )


def test_boundaries_match_rpact(pair):
    case, adaptive, _ = pair
    gsd = adaptive.design.gsd
    np.testing.assert_allclose(gsd.efficacy, case["efficacy"], atol=1e-6)
    if case["futility"] is not None:
        np.testing.assert_allclose(gsd.futility[0], case["futility"][0], atol=1e-6)


def test_patient_simulation_matches_rpact(pair):
    case, _, sim = pair
    ref_n = case["n_sim"]
    for ours, ref in (
        (sim.reject_prob, case["reject_prob"]),
        (sim.prob_zone["efficacy"], case["reject_interim"]),
        (sim.prob_zone["futility"], case["futility_interim"]),
    ):
        _close(ours, float(mc_se(ours, N_SIM)), ref, float(mc_se(ref, ref_n)))
    scale = math.sqrt(N_SIM / ref_n)
    for ours, se, ref in (
        (sim.expected_events, sim.expected_events_se, case["expected_events"]),
        (sim.expected_duration, sim.expected_duration_se, case["study_duration"]),
        (sim.expected_patients, sim.expected_patients_se, case["expected_patients"]),
    ):
        _close(ours, se, ref, se * scale)
    assert sim.events_not_reached == 0.0


def test_exact_characteristics_match_patient_simulation_at_1to1():
    """Equal allocation: the Schoenfeld-scale numbers agree with patient-level simulation.

    (With unequal allocation the log-rank information departs from D r / (1 + r)^2 and the
    exact power is off by about a point; the patient engine is the reference there.)
    """
    case = next(c for c in REF["simulation"] if c["id"] == "obf_1to1_hr0.7")
    adaptive = _adaptive(case)
    oc = adaptive.operating_characteristics(case["hazard_ratio"])
    sim = simulate_survival_ssr(adaptive, case["hazard_ratio"], n_sim=N_SIM, seed=8)
    _close(oc.reject_prob, 0.0, sim.reject_prob, sim.reject_prob_se)
    _close(oc.expected_events, 0.0, sim.expected_events, sim.expected_events_se, k=4.0)
    _close(oc.expected_duration, 0.0, sim.expected_duration, sim.expected_duration_se, k=4.0)
    _close(oc.expected_patients, 0.0, sim.expected_patients, sim.expected_patients_se, k=4.0)
    for z, p in oc.prob_zone.items():
        _close(p, 0.0, sim.prob_zone[z], float(mc_se(p, N_SIM)))


# --------------------------------------------------------------------------- type I error


def test_type_i_error_of_default_design():
    adaptive = survival_ssr(_study(), events=248, n_total=600)
    sim = simulate_survival_ssr(adaptive, 1.0, use_futility=False, n_sim=N_SIM, seed=21)
    assert abs(sim.reject_prob - 0.025) <= 2 * float(mc_se(0.025, N_SIM)) + 1e-3
    assert adaptive.operating_characteristics(1.0, use_futility=False).reject_prob == (
        pytest.approx(0.025, abs=1e-6)
    )


def test_naive_log_rank_inflates_and_chw_does_not():
    """cp_min = 0 and a cap of 2: the pooled log-rank test exceeds alpha (exact 0.0303)."""
    adaptive = survival_ssr(_study(), events=248, n_total=800, cp_min=0.0, max_factor=2.0)
    n = 40_000
    sim = simulate_survival_ssr(adaptive, 1.0, use_futility=False, n_sim=n, seed=22)
    se = float(mc_se(0.025, n))
    assert sim.reject_prob_naive > 0.025 + 3 * se
    assert abs(sim.reject_prob - 0.025) <= 2 * se + 1e-3


def test_no_increase_allowed_is_the_plain_design():
    study = _study()
    adaptive = survival_ssr(study, events=248, n_total=600, max_factor=1.0)
    assert np.all(adaptive.total_events(np.linspace(-3, 3, 61)) == 248)
    oc = adaptive.operating_characteristics(0.7)
    gsd = adaptive.design.gsd
    plain = simulate_survival(
        study,
        gsd.info_rates,
        gsd.efficacy,
        hazard_ratio=0.7,
        n_total=600,
        events=248,
        n_sim=N_SIM,
        seed=4,
    )
    sim = simulate_survival_ssr(adaptive, 0.7, n_sim=N_SIM, seed=4)
    _close(sim.reject_prob, sim.reject_prob_se, plain.reject_prob, plain.reject_prob_se)
    _close(
        sim.expected_duration,
        sim.expected_duration_se,
        plain.expected_duration,
        plain.expected_duration_se,
    )
    assert oc.expected_events == pytest.approx(plain.expected_events, rel=0.01)


# --------------------------------------------------------------------------- building blocks


def test_total_events_rule():
    adaptive = survival_ssr(_study(), events=248, n_total=600)
    d = adaptive.design
    assert adaptive.interim_events == 124
    assert adaptive.max_events == 372
    z = np.linspace(-1, 2.9, 200)
    total = adaptive.total_events(z)
    assert total.min() >= 248 and total.max() == 372
    promising = d.zone(z) == "promising"
    assert np.all(total[~promising] == 248)
    # where the cap does not bind, the new total reaches the target conditional power
    m = (total - 124) / 124
    cp = d.conditional_power(z, None, m)
    free = promising & (total < 372)
    assert np.all(cp[free] >= d.cp_target - 1e-12)


def test_calendar_grows_with_events():
    adaptive = survival_ssr(_study(), events=248, n_total=600)
    t = [adaptive.time_of_events(e, 0.7) for e in (124, 248, 372)]
    assert t[0] < t[1] < t[2]
    time, patients = adaptive.final_analysis(372, 0.7)
    assert time == pytest.approx(t[2]) and patients == pytest.approx(600)


def test_input_checks():
    study = _study()
    with pytest.raises(ValueError, match="cannot give"):
        survival_ssr(study, events=248, n_total=370)  # cap 372 events from 370 patients
    with pytest.raises(ValueError, match="extend"):
        survival_ssr(study, events=248, n_total=600, extend="calendar")
    adaptive = survival_ssr(study, events=248, n_total=370, extend="accrual")
    with pytest.raises(ValueError, match="followup"):
        simulate_survival_ssr(adaptive, n_sim=10)
    with pytest.raises(ValueError, match="hazard_ratio"):
        simulate_survival_ssr(survival_ssr(study, events=248, n_total=600), 0.0, n_sim=10)
