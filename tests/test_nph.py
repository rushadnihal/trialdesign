"""M17: non-proportional hazards, Weibull event times and the Fleming-Harrington test.

Reference values in ``tests/nph_reference_values.json`` come from
``tests/r/nph_reference_values.R``: ``rpact::getEventProbabilities`` (exact),
``rpact::getSimulationSurvival`` with piecewise hazards or Weibull (100,000 trials per case),
and the FH weighted log-rank computed in R and checked there against ``survival::survdiff``.
Simulation comparisons allow a multiple of the MC SE of the difference, as in
``test_patients.py``: 2 for the rejection probability, 3 for the per-look and calendar values.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from trialdesign import accrual
from trialdesign.boundaries import group_sequential_design
from trialdesign.hazards import ArmHazard, EventModel, expected_events
from trialdesign.patients import (
    logrank_z,
    simulate_design_patients,
    simulate_survival,
    weighted_logrank_z,
)
from trialdesign.simulate import mc_se

REF = json.loads((Path(__file__).parent / "nph_reference_values.json").read_text())
N_SIM = 20_000


def _model(case: dict) -> EventModel:
    if case["kappa"] == 1:
        return EventModel.piecewise(case["times"], case["control"], case["treatment"])
    return EventModel(
        ArmHazard.weibull(case["control"][0], case["kappa"]),
        ArmHazard.weibull(case["treatment"][0], case["kappa"]),
    )


def _close(ours: float, ours_se: float, ref: float, ref_se: float, k: float = 2.0) -> None:
    band = k * math.hypot(ours_se, ref_se)
    assert abs(ours - ref) <= band, f"{ours:.5g} vs rpact {ref:.5g} (band {band:.3g})"


# ------------------------------------------------------------------- event distributions


@pytest.mark.parametrize(
    "case", REF["event_probabilities"], ids=[c["id"] for c in REF["event_probabilities"]]
)
def test_expected_events_match_rpact(case):
    model = _model(case)
    eta = accrual.dropout_hazard(case["dropout"])
    for t, ref in zip(case["time"], case["events"], strict=True):
        ours = expected_events(t, case["accrual_end"], case["accrual_rate"], model, eta)
        assert ours / case["n_total"] == pytest.approx(ref / case["n_total"], abs=1e-6)


def test_exponential_model_matches_accrual_closed_form():
    lam = accrual.hazard_from_median(9)
    model = EventModel.exponential(lam, 0.75)
    for t in (3.0, 10.0, 26.0, 40.0):
        ours = expected_events(t, [6, 24], [10, 25], model, 0.01, 2.0)
        ref = accrual.expected_events(t, [6, 24], [10, 25], lam, 0.75, 0.01, 2.0)
        assert ours == pytest.approx(ref, rel=1e-9)


def test_observed_event_prob_without_dropout_is_the_cdf():
    for arm in (
        ArmHazard.piecewise([0, 2, 5], [0.1, 0.0, 0.3]),
        ArmHazard.weibull(0.08, 1.7),
        ArmHazard.weibull(0.08, 0.6),
    ):
        s = np.array([0.5, 2.0, 3.0, 7.5, 20.0])
        np.testing.assert_allclose(arm.observed_event_prob(s), 1 - arm.survival(s), atol=1e-12)


def test_observed_event_prob_weibull_with_dropout_by_simulation():
    arm = ArmHazard.weibull(0.07, 1.5)
    rng = np.random.default_rng(3)
    e = rng.exponential(1.0, 400_000)
    t_event = arm.inverse_cumulative_hazard(e)
    t_drop = rng.exponential(1 / 0.02, e.size)
    p = np.mean((t_event <= t_drop) & (t_event <= 12.0))
    assert float(arm.observed_event_prob(12.0, 0.02)) == pytest.approx(p, abs=4 * mc_se(p, e.size))


def test_inverse_cumulative_hazard_round_trips():
    arm = ArmHazard.piecewise([0, 4, 10], [0.05, 0.0, 0.2])
    t = np.array([0.0, 1.0, 3.99, 4.5, 9.0, 10.0, 30.0])
    back = arm.inverse_cumulative_hazard(arm.cumulative_hazard(t))
    # inside the zero-hazard interval [4, 10) every time maps to the same cumulative hazard,
    # and the inverse returns the end of that interval
    expected = np.where((t > 4) & (t < 10), 10.0, t)
    np.testing.assert_allclose(back, expected, atol=1e-12)
    w = ArmHazard.weibull(0.1, 1.3)
    np.testing.assert_allclose(w.inverse_cumulative_hazard(w.cumulative_hazard(t)), t)
    assert EventModel.weibull(12, 0.7, 1.5).control.median() == pytest.approx(12)


def test_model_constructors_and_hazard_ratio():
    lam = accrual.hazard_from_median(12)
    assert EventModel.exponential(lam, 0.7).hazard_ratio == pytest.approx(0.7)
    assert EventModel.weibull(12, 0.7, 1.5).hazard_ratio == pytest.approx(0.7)
    assert EventModel.delayed_effect(lam, 0.7, 4).hazard_ratio is None
    assert EventModel.waning_effect(lam, 0.7, 12).hazard_ratio is None
    assert EventModel.delayed_effect(lam, 0.7, 0).hazard_ratio == pytest.approx(0.7)
    null = EventModel.delayed_effect(lam, 0.7, 4).null()
    assert null.treatment == null.control
    with pytest.raises(ValueError, match="start at 0"):
        ArmHazard.piecewise([1, 4], [0.1, 0.1])
    with pytest.raises(ValueError, match="last one positive"):
        ArmHazard.piecewise([0, 4], [0.1, 0.0])
    with pytest.raises(ValueError, match="single interval"):
        ArmHazard((0.0, 2.0), (0.1, 0.1), kappa=2.0)


def test_look_times_under_a_model():
    study = accrual.survival_study(260, 0.7, 12, accrual_time=24, followup=12)
    t = [1 / 3, 2 / 3, 1]
    np.testing.assert_allclose(study.look_times(t, study.event_model()), study.look_times(t))
    # a delayed effect gives more events early, so every look comes sooner
    delayed = study.look_times(t, EventModel.delayed_effect(study.hazard_control, 0.7, 4))
    assert np.all(delayed < study.look_times(t))
    # a model with fewer events pushes the final look past the planned end
    slow = EventModel.exponential(study.hazard_control / 2, 0.7)
    assert study.time_of_events(260, slow) > study.study_duration


# ----------------------------------------------------------------- simulation vs rpact


@pytest.fixture(scope="module", params=REF["simulation"], ids=[c["id"] for c in REF["simulation"]])
def pair(request):
    case = request.param
    study = accrual.survival_study(
        case["design_events"],
        case["design_hazard_ratio"],
        case["median_control"],
        accrual_time=case["accrual_time"],
        followup=case["followup"],
        dropout=case["dropout"],
    )
    events = case["events"]
    sim = simulate_survival(
        study,
        [e / events[-1] for e in events],
        case["efficacy"],
        case["futility"],
        model=_model(case),
        n_total=case["n_total"],
        events=events[-1],
        n_sim=N_SIM,
        seed=2026,
    )
    return case, study, sim


def test_reject_prob_matches_rpact(pair):
    case, _, sim = pair
    ref = case["reject_prob"]
    _close(sim.reject_prob, sim.reject_prob_se, ref, float(mc_se(ref, case["n_sim"])))


def test_per_look_probabilities_match_rpact(pair):
    case, _, sim = pair
    for ours, ref in (
        (sim.oc.prob_efficacy, case["reject_per_look"]),
        (sim.oc.prob_futility[:-1], case["futility_per_look"][:-1]),
    ):
        for o, r in zip(ours, ref, strict=True):
            _close(o, float(mc_se(o, sim.oc.n_sim)), r, float(mc_se(r, case["n_sim"])), k=3.0)


def test_calendar_quantities_match_rpact(pair):
    case, _, sim = pair
    scale = math.sqrt(N_SIM / case["n_sim"])
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


def test_mean_look_times_near_expected_event_times(pair):
    """The model's expected-events times sit within 1% of the simulated mean look times."""
    case, study, sim = pair
    events = case["events"]
    model = _model(case)
    expected = [study.time_of_events(e * study.events / events[-1], model) for e in events]
    # the study's events are unrounded; scale to the planned counts the simulator uses
    np.testing.assert_allclose(sim.analysis_time, expected, rtol=0.01)


def test_null_piecewise_case_is_a_type_one_error(pair):
    case, _, sim = pair
    if case["model"] != "null_piecewise":
        pytest.skip("only the shared-hazard case")
    assert sim.hazard_ratio == pytest.approx(1.0)
    assert abs(sim.reject_prob - 0.025) <= 2 * sim.reject_prob_se


# ------------------------------------------------------------ properties of the engine


@pytest.fixture(scope="module")
def study():
    return accrual.survival_study(260, 0.7, 12, accrual_time=24, followup=12)


def test_single_interval_reproduces_the_exponential_engine(study):
    """Same seed, same trials: the v0.3 numbers are unchanged."""
    d = group_sequential_design(2)
    a = simulate_survival(study, d.info_rates, d.efficacy, n_sim=2_000, seed=11)
    model = EventModel.piecewise([0], [study.hazard_control], [0.7 * study.hazard_control])
    b = simulate_survival(study, d.info_rates, d.efficacy, n_sim=2_000, seed=11, model=model)
    assert a.reject_prob == b.reject_prob
    np.testing.assert_array_equal(a.analysis_time, b.analysis_time)
    assert a.expected_events == b.expected_events


def test_zero_delay_is_proportional_hazards(study):
    d = group_sequential_design(2)
    kw = {"n_sim": 2_000, "seed": 12}
    ph = simulate_survival(study, d.info_rates, d.efficacy, **kw)
    zero = EventModel.delayed_effect(study.hazard_control, 0.7, 0.0)
    assert simulate_survival(study, d.info_rates, d.efficacy, model=zero, **kw).reject_prob == (
        ph.reject_prob
    )


def test_delay_beyond_the_study_gives_the_type_one_error(study):
    d = group_sequential_design(2)
    never = EventModel.delayed_effect(study.hazard_control, 0.7, 1_000.0)
    sim = simulate_survival(study, d.info_rates, d.efficacy, model=never, n_sim=N_SIM, seed=13)
    assert abs(sim.reject_prob - 0.025) <= 2 * sim.reject_prob_se
    assert math.isnan(sim.hazard_ratio)


def test_delayed_effect_costs_power(study):
    d = group_sequential_design(2)
    kw = {"n_sim": 4_000, "seed": 14}
    ph = simulate_survival(study, d.info_rates, d.efficacy, **kw)
    delayed = EventModel.delayed_effect(study.hazard_control, 0.7, 4)
    nph = simulate_survival(study, d.info_rates, d.efficacy, model=delayed, **kw)
    assert nph.reject_prob < ph.reject_prob - 0.1


def test_design_simulation_uses_the_null_model_under_h0(study):
    d = group_sequential_design(2)
    delayed = EventModel.delayed_effect(study.hazard_control, 0.7, 4)
    sim = simulate_design_patients(d, "survival", study=study, model=delayed, n_sim=N_SIM, seed=15)
    assert abs(sim.h0.reject_prob - 0.025) <= 2 * sim.h0.reject_prob_se
    assert 0.4 < sim.h1.reject_prob < 0.6


def test_model_and_hazard_ratio_are_exclusive(study):
    d = group_sequential_design(2)
    with pytest.raises(ValueError, match="not both"):
        simulate_survival(
            study, d.info_rates, d.efficacy, hazard_ratio=0.8, model=study.event_model()
        )


# ------------------------------------------------------------------------ the FH test


@pytest.mark.parametrize("case", REF["fh"], ids=[c["id"] for c in REF["fh"]])
def test_fh_statistic_matches_r(case):
    time = np.array([case["time"]], dtype=float)
    event = np.array([case["event"]], dtype=bool)
    treat = np.array([case["treat"]])
    for key, ref in case["z"].items():
        rho, gamma = (float(x) for x in key.split(","))
        assert weighted_logrank_z(time, event, treat, rho, gamma)[0] == pytest.approx(
            ref, abs=1e-10
        )


def test_fh_00_is_the_logrank_test():
    rng = np.random.default_rng(4)
    time = rng.exponential(10, (50, 80))
    event = rng.random((50, 80)) < 0.7
    treat = np.tile([0, 1], (50, 40))
    time[:, :5] = -1.0  # not enrolled: no event, never at risk
    event[:, :5] = False
    np.testing.assert_allclose(
        weighted_logrank_z(time, event, treat, 0, 0), logrank_z(time, event, treat), atol=1e-12
    )


def test_fh_holds_alpha_and_helps_with_a_delayed_effect(study):
    d = group_sequential_design(1)
    delayed = EventModel.delayed_effect(study.hazard_control, 0.7, 4)
    kw = {"n_sim": N_SIM, "seed": 16}
    h0 = simulate_survival(study, d.info_rates, d.efficacy, model=delayed.null(), fh=(0, 1), **kw)
    assert abs(h0.reject_prob - 0.025) <= 2 * h0.reject_prob_se
    assert h0.test == "fh(0,1)"
    fh = simulate_survival(study, d.info_rates, d.efficacy, model=delayed, fh=(0, 1), **kw)
    lr = simulate_survival(study, d.info_rates, d.efficacy, model=delayed, **kw)
    assert fh.reject_prob > lr.reject_prob + 0.05


def test_fh_needs_a_single_look(study):
    d = group_sequential_design(2)
    with pytest.raises(ValueError, match="single final analysis"):
        simulate_survival(study, d.info_rates, d.efficacy, fh=(0, 1))


# ------------------------------------------------------------------------ command line

SURVIVAL = [
    "simulate",
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
    "--engine",
    "patient",
    "--nsim",
    "1000",
    "--seed",
    "1",
]


def _run(*args: str):
    from typer.testing import CliRunner

    from trialdesign.cli import app

    return CliRunner().invoke(app, [*SURVIVAL, *args])


def test_cli_delayed_effect_json():
    result = _run("--looks", "2", "--delay", "4", "--json")
    assert result.exit_code == 0, result.output
    sim = json.loads(result.output)["simulation"]
    assert sim["event_model"]["model"] == "delayed_effect"
    assert sim["event_model"]["hazard_ratio"] is None
    assert sim["test"] == "logrank"
    assert sim["power"] < 0.65


def test_cli_text_names_the_model_and_test():
    result = _run("--looks", "1", "--waning", "12", "--hr-after", "0.6", "--test", "fh:1,0")
    assert result.exit_code == 0, result.output
    assert "a waning effect (HR 0.6 for 12 months, then 1)" in result.output
    assert "FH(1,0) weighted log-rank" in result.output


def test_cli_piecewise_and_weibull_from_config(tmp_path):
    cfg = tmp_path / "nph.toml"
    cfg.write_text(
        "[hazards]\nhazard_times = [0, 4]\nhazards_control = [0.0578, 0.0578]\n"
        "hazards_treatment = [0.0578, 0.0404]\n"
    )
    result = _run("--looks", "2", "--config", str(cfg), "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["simulation"]["event_model"]["model"] == "custom"
    result = _run("--looks", "2", "--kappa", "1.5", "--json")
    assert result.exit_code == 0, result.output
    em = json.loads(result.output)["simulation"]["event_model"]
    assert em["hazard_ratio"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--looks", "2", "--delay", "4", "--kappa", "1.5"), "one event model"),
        (("--looks", "2", "--hr-after", "0.6"), "need --delay"),
        (("--looks", "2", "--hazard-times", "0,4"), "needs --hazards-control"),
        (("--looks", "2", "--test", "fh:0,1"), "single final analysis"),
        (("--looks", "1", "--test", "wilcoxon"), "fh:<rho>,<gamma>"),
        (("--looks", "2", "--delay", "4", "--engine", "score"), "needs --engine patient"),
    ],
)
def test_cli_rejects_bad_event_model_input(args, message):
    result = _run(*args)
    assert result.exit_code != 0
    assert message in result.output


def test_cli_hazards_need_the_survival_endpoint():
    from typer.testing import CliRunner

    from trialdesign.cli import app

    result = CliRunner().invoke(
        app,
        [
            *("simulate", "--endpoint", "normal", "--delta", "0.3", "--sd", "1"),
            *("--looks", "2", "--engine", "patient", "--delay", "4"),
        ],
    )
    assert result.exit_code != 0
    assert "survival endpoint" in result.output


def test_example_config_runs():
    from typer.testing import CliRunner

    from trialdesign.cli import app

    cfg = str(Path(__file__).parents[1] / "examples" / "phase3_survival_nph.toml")
    result = CliRunner().invoke(app, ["simulate", "--config", cfg, "--nsim", "500", "--json"])
    assert result.exit_code == 0, result.output
    sim = json.loads(result.output)["simulation"]
    assert sim["event_model"]["model"] == "delayed_effect"
    assert sim["engine"] == "patient"
