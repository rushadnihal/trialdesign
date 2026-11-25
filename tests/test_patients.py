"""M12: patient-level simulation of event-driven survival trials.

Reference values in ``tests/patients_reference_values.json`` come from
``tests/r/patients_reference_values.R`` (``rpact::getSimulationSurvival``, 100,000 trials per
case). Both sides carry Monte Carlo error, so each comparison allows a multiple of the MC SE of
the *difference*: 2 for the rejection probability, and 3 for the per-look and calendar
quantities, which add up to about 60 comparisons (at 2 SE a few would fail by chance alone).
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from trialdesign import accrual
from trialdesign.boundaries import group_sequential_design
from trialdesign.patients import entry_times, logrank_z, simulate_survival
from trialdesign.simulate import mc_se, simulate_trials

REF = json.loads((Path(__file__).parent / "patients_reference_values.json").read_text())
CASES = REF["survival"]
N_SIM = 20_000


def _study(case: dict) -> accrual.SurvivalStudy:
    kw = {
        "median_control": case["median_control"],
        "dropout": case["dropout"],
        "allocation": case["allocation"],
    }
    if case["accrual_rate"] is None:
        return accrual.survival_study(
            case["design_events"],
            case["design_hazard_ratio"],
            accrual_time=case["accrual_time"][0],
            followup=case["followup"],
            **kw,
        )
    # piecewise: rpact's segment ends (the last stretched to enrol n_total exactly)
    return accrual.survival_study(
        case["design_events"],
        case["design_hazard_ratio"],
        accrual_time=case["accrual_end"],
        accrual_rate=case["accrual_rate"],
        **kw,
    )


def _simulate(case: dict, n_sim: int = N_SIM, seed: int = 2026, **kw):
    events = case["events"]
    return simulate_survival(
        _study(case),
        [e / events[-1] for e in events],
        case["efficacy"],
        case["futility"],
        hazard_ratio=case["hazard_ratio"],
        n_total=case["n_total"],
        events=events[-1],
        n_sim=n_sim,
        seed=seed,
        **kw,
    )


@pytest.fixture(scope="module", params=CASES, ids=[c["id"] for c in CASES])
def pair(request):
    case = request.param
    return case, _simulate(case)


def _close(ours: float, ours_se: float, ref: float, ref_se: float, k: float = 2.0) -> None:
    band = k * math.hypot(ours_se, ref_se)
    assert abs(ours - ref) <= band, f"{ours:.5g} vs rpact {ref:.5g} (band {band:.3g})"


# --------------------------------------------------------------------------- vs rpact


def test_reject_prob_matches_rpact(pair):
    case, sim = pair
    ref = case["reject_prob"]
    _close(sim.reject_prob, sim.reject_prob_se, ref, float(mc_se(ref, case["n_sim"])))


def test_per_look_probabilities_match_rpact(pair):
    case, sim = pair
    # rpact counts no futility stop at the last look; trialdesign counts z < b_K there (with
    # b_K = c_K that is every trial that fails), so the last look's futility is not compared.
    for ours, ref in (
        (sim.oc.prob_efficacy, case["reject_per_look"]),
        (sim.oc.prob_futility[:-1], case["futility_per_look"][:-1]),
    ):
        for o, r in zip(ours, ref, strict=True):
            _close(o, float(mc_se(o, sim.oc.n_sim)), r, float(mc_se(r, case["n_sim"])), k=3.0)


def test_calendar_quantities_match_rpact(pair):
    """Mean look times, study duration, patients and events at the stop."""
    case, sim = pair
    scale = math.sqrt(N_SIM / case["n_sim"])  # rpact's SE is ours scaled by its run length
    looks = zip(sim.analysis_time, sim.analysis_time_se, case["analysis_time"], strict=True)
    for o, se, r in looks:
        _close(o, se, r, se * scale, k=3.0)
    for o, se, r in (
        (sim.expected_duration, sim.expected_duration_se, case["study_duration"]),
        (sim.expected_patients, sim.expected_patients_se, case["expected_patients"]),
        (sim.expected_events, sim.expected_events_se, case["expected_events"]),
    ):
        _close(o, se, r, se * scale, k=3.0)
    assert sim.events_not_reached == 0.0


def test_mean_look_times_match_expected_event_times():
    """With no early stop, the mean look time is close to the M8 expected-events time."""
    d = group_sequential_design(3)
    study = accrual.survival_study(250, 0.7, 12, accrual_time=24, followup=12)
    sim = simulate_survival(study, d.info_rates, [np.inf] * 3, hazard_ratio=0.7, n_sim=4000, seed=1)
    np.testing.assert_allclose(sim.analysis_time, study.look_times(sim.oc.info_rates), atol=0.1)


# --------------------------------------------------------------------------- type I error


H0_CASES = [c for c in CASES if c["hazard_ratio"] == 1]


@pytest.mark.parametrize("case", H0_CASES, ids=[c["id"] for c in H0_CASES])
def test_type_i_error_within_two_se_of_alpha(case):
    sim = _simulate(case, seed=7)
    assert abs(sim.reject_prob - 0.025) <= 2 * float(mc_se(0.025, sim.oc.n_sim)) + 1e-3, (
        "a 1e-3 allowance covers the normal approximation of the boundaries"
    )


@pytest.mark.slow
def test_type_i_error_100k():
    """On demand: ``pytest -m slow``. 100,000 trials of the 3-look OBF design under H0."""
    case = next(c for c in CASES if c["id"] == "obf3_uniform_hr1")
    sim = _simulate(case, n_sim=100_000, seed=11)
    assert abs(sim.reject_prob - 0.025) <= 2 * float(mc_se(0.025, 100_000))


def test_patient_engine_agrees_with_score_engine():
    """At 600 patients the log-rank z behaves like the canonical process."""
    d = group_sequential_design(3, futility_spending="obf")
    study = accrual.survival_study(400, 0.75, 12, accrual_time=18, followup=12)
    sim = simulate_survival(
        study, d.info_rates, d.efficacy, d.futility, n_total=600, events=400, n_sim=10_000, seed=3
    )
    drift = -math.log(0.75) * math.sqrt(400 / 4)
    score = simulate_trials(d.info_rates, d.efficacy, d.futility, drift, 200_000, seed=3)
    _close(sim.reject_prob, sim.reject_prob_se, score.reject_prob, score.reject_prob_se)
    oc = sim.oc
    _close(oc.futility_prob, oc.futility_prob_se, score.futility_prob, score.futility_prob_se)


def test_speed_10k_trials_of_600_patients():
    d = group_sequential_design(3)
    study = accrual.survival_study(400, 0.75, 12, accrual_time=18, followup=12)
    start = time.perf_counter()
    simulate_survival(
        study, d.info_rates, d.efficacy, n_total=600, events=400, n_sim=10_000, seed=1
    )
    assert time.perf_counter() - start < 30


# --------------------------------------------------------------------------- building blocks


def test_logrank_matches_scipy():
    rng = np.random.default_rng(5)
    for _ in range(5):
        n = 80
        treat = rng.integers(0, 2, n)
        t_event = rng.exponential(1, n) / np.where(treat == 1, 0.6, 1.0)
        cens = rng.exponential(2, n)
        time_ = np.minimum(t_event, cens)
        event = t_event <= cens
        z = logrank_z(time_[None, :], event[None, :], treat[None, :])[0]
        data_t = stats.CensoredData.right_censored(time_[treat == 1], ~event[treat == 1])
        data_c = stats.CensoredData.right_censored(time_[treat == 0], ~event[treat == 0])
        ref = stats.logrank(data_c, data_t, alternative="two-sided").statistic
        assert z == pytest.approx(ref, abs=1e-10)


def test_logrank_ignores_patients_not_yet_enrolled():
    time_ = np.array([[1.0, 2.0, 3.0, 4.0]])
    event = np.array([[True, True, False, True]])
    treat = np.array([[1, 0, 1, 0]])
    z = logrank_z(time_, event, treat)
    padded = logrank_z(
        np.hstack([time_, [[-1.0, -1.0]]]),
        np.hstack([event, [[False, False]]]),
        np.hstack([treat, [[1, 0]]]),
    )
    assert padded == pytest.approx(z)


def test_fixed_entry_is_evenly_spaced():
    u = entry_times(4, [24.0], [10.0])
    np.testing.assert_allclose(u, [6, 12, 18, 24])
    pw = entry_times(85, [6.0, 9.0], [10.0, 5.0])  # 60 then 15 patients
    assert (pw <= 6).sum() == 68  # rescaled to 85 over the same period: 85 * 60 / 75
    assert pw[-1] == pytest.approx(9.0)


def test_random_entry_follows_the_accrual_density():
    u = entry_times(200, [6.0, 9.0], [10.0, 5.0], "random", 500, np.random.default_rng(1))
    assert u.shape == (500, 200)
    assert np.all(np.diff(u, axis=1) >= 0)
    assert (u <= 6).mean() == pytest.approx(60 / 75, abs=0.01)


def test_mean_look_time_with_instant_accrual_matches_order_statistics():
    """No dropout, everyone enrolled at time ~0, HR 1: E(T_(k:n)) = sum 1/((n - i) lambda)."""
    n, lam = 200, math.log(2) / 12
    study = accrual.survival_study(150, 1.0, 12, accrual_time=1e-6, followup=60)
    sim = simulate_survival(
        study, [0.5, 1.0], [np.inf, np.inf], n_total=n, events=150, n_sim=4000, seed=2
    )
    for k, d in enumerate((75, 150)):
        expected = sum(1 / ((n - i) * lam) for i in range(d))
        assert sim.analysis_time[k] == pytest.approx(expected, abs=3 * sim.analysis_time_se[k])


def test_same_seed_same_result_and_random_entry_runs():
    case = CASES[0]
    a = _simulate(case, n_sim=500, seed=9)
    b = _simulate(case, n_sim=500, seed=9)
    assert a.reject_prob == b.reject_prob
    np.testing.assert_array_equal(a.analysis_time, b.analysis_time)
    r = _simulate(case, n_sim=500, seed=9, entry="random")
    assert 0 < r.reject_prob < 1


def test_input_checks():
    study = accrual.survival_study(100, 0.7, 12, accrual_time=12, followup=12)
    with pytest.raises(ValueError, match="hazard_ratio"):
        simulate_survival(study, 2, [2.8, 1.97], hazard_ratio=0)
    with pytest.raises(ValueError, match="cannot come from"):
        simulate_survival(study, 2, [2.8, 1.97], n_total=50, events=100)
    with pytest.raises(ValueError, match="entry"):
        simulate_survival(study, 2, [2.8, 1.97], entry="poisson")
    with pytest.raises(ValueError, match="allocation"):
        simulate_survival(
            accrual.survival_study(100, 0.7, 12, accrual_time=12, followup=12, allocation=0.7071),
            2,
            [2.8, 1.97],
        )
