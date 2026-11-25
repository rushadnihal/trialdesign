"""M13: patient-level normal and binary simulation, and the patient engine for whole designs.

Reference values in ``tests/patients_endpoints_reference_values.json`` come from
``tests/r/patients_endpoints_reference_values.R`` (``rpact::getSimulationMeans`` and
``getSimulationRates``, 100,000 trials per case). Every comparison allows 3 MC SE of the
difference: with about 35 comparisons, a 2 SE band would fail one or two by chance alone.

For the normal endpoint, rpact's ``normalApproximation = FALSE`` is compared with
trialdesign's default ``test="t"`` (t turned into the z-value with the same p-value) at 86 or
more patients per arm per stage, where they agree. At 30 per arm per stage they do not: rpact
gives a type I error of about 0.0267 for three O'Brien-Fleming looks, ``test="t"`` keeps 0.025,
and comparing t directly with the z boundaries (``test="z"``) also exceeds it. That case is
checked as a property below.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from trialdesign import accrual, fixed, simulate
from trialdesign.boundaries import group_sequential_design
from trialdesign.cli import app
from trialdesign.patients import (
    _fm_rates_vec,
    simulate_binary,
    simulate_design_patients,
    simulate_normal,
)
from trialdesign.simulate import mc_se, simulate_trials

REF = json.loads((Path(__file__).parent / "patients_endpoints_reference_values.json").read_text())
CASES = REF["cases"]
N_SIM = 100_000
runner = CliRunner()


def _run(case: dict, n_sim: int = N_SIM, seed: int = 2026):
    n_c, n_t = case["n_control"], case["n_treatment"]
    t = [n / n_c[-1] for n in n_c]
    kw = {
        "n_treatment": int(n_t[-1]),
        "delta0": case["delta0"],
        "n_sim": n_sim,
        "seed": seed,
    }
    if case["endpoint"] == "normal":
        args = (case["delta"], case["sd"], int(n_c[-1]), t, case["efficacy"], case["futility"])
        return simulate_normal(*args, **kw)
    args = (case["p_control"], case["p_treatment"], int(n_c[-1]), t, case["efficacy"])
    return simulate_binary(*args, case["futility"], **kw)


@pytest.fixture(scope="module", params=CASES, ids=[c["id"] for c in CASES])
def pair(request):
    case = request.param
    return case, _run(case)


def _close(ours: float, ours_se: float, ref: float, ref_se: float, k: float = 3.0) -> None:
    band = k * math.hypot(ours_se, ref_se)
    assert abs(ours - ref) <= band, f"{ours:.5g} vs rpact {ref:.5g} (band {band:.3g})"


def _se(p: float, n: int) -> float:
    return float(mc_se(p, n))


def test_reject_prob_matches_rpact(pair):
    case, oc = pair
    ref = case["reject_prob"]
    _close(oc.reject_prob, oc.reject_prob_se, ref, _se(ref, case["n_sim"]))


def test_per_look_probabilities_match_rpact(pair):
    case, oc = pair
    # rpact counts no futility stop at the last look (see test_patients.py)
    for ours, ref in (
        (oc.prob_efficacy, case["reject_per_look"]),
        (oc.prob_futility[:-1], case["futility_per_look"][:-1]),
    ):
        for o, r in zip(ours, ref, strict=True):
            _close(o, _se(o, oc.n_sim), r, _se(r, case["n_sim"]))


def test_expected_sample_size_matches_rpact(pair):
    case, oc = pair
    n_max = case["n_control"][-1] + case["n_treatment"][-1]
    ours, se = oc.expected_sample_size(n_max)
    _close(ours, se, case["expected_n_total"], se)


# --------------------------------------------------------------------------- properties


def test_small_sample_t_test_keeps_alpha():
    """30 per arm per stage: the p-value-matched t keeps alpha; t against z boundaries does not.

    Comparing the t statistic directly with the normal boundaries (``test="z"``) rejects too
    often at this size, which is why ``test="t"`` is the default.
    """
    d = group_sequential_design(3)
    args = (0.0, 1.0, 90, d.info_rates, d.efficacy)
    t_test = simulate_normal(*args, n_sim=400_000, seed=1)
    z_test = simulate_normal(*args, test="z", n_sim=400_000, seed=1)
    assert abs(t_test.reject_prob - 0.025) <= 2 * _se(0.025, 400_000)
    assert z_test.reject_prob > 0.025 + 2 * _se(0.025, 400_000)


def test_normal_engine_agrees_with_score_engine_at_large_n():
    """500 per arm: estimating the variance costs about 0.001 of power, within 3 MC SE."""
    d = group_sequential_design(3, futility_spending="obf")
    n = 500
    delta, sd = 0.18, 1.0
    oc = simulate_normal(delta, sd, n, d.info_rates, d.efficacy, d.futility, n_sim=100_000, seed=4)
    drift = delta / (sd * math.sqrt(2 / n))
    score = simulate_trials(d.info_rates, d.efficacy, d.futility, drift, 200_000, seed=4)
    _close(oc.reject_prob, oc.reject_prob_se, score.reject_prob, score.reject_prob_se)
    _close(oc.futility_prob, oc.futility_prob_se, score.futility_prob, score.futility_prob_se)


def test_vectorised_farrington_manning_matches_fixed():
    pc = np.array([0.3, 0.7, 0.05, 0.6])
    pt = np.array([0.45, 0.6, 0.02, 0.6])
    for r in (1.0, 2.0, 0.5):
        for d0 in (0.0, -0.1, 0.05):
            vc, vt = _fm_rates_vec(pc, pt, np.full(pc.size, r), d0)
            for i in range(pc.size):
                ec, et = fixed.farrington_manning_rates(pc[i], pt[i], r, d0)
                assert vc[i] == pytest.approx(ec, abs=1e-12)
                assert vt[i] == pytest.approx(et, abs=1e-12)


def test_binary_superiority_is_unchanged_from_v02():
    """With delta0 = 0 the moved simulator draws the same numbers as the v0.2 function."""
    d = group_sequential_design(3, futility_spending="obf")
    old = simulate.simulate_binary(0.3, 0.45, 180, d.info_rates, d.efficacy, d.futility, seed=8)
    new = simulate_binary(0.3, 0.45, 180, d.info_rates, d.efficacy, d.futility, seed=8)
    np.testing.assert_array_equal(old.prob_efficacy, new.prob_efficacy)
    np.testing.assert_array_equal(old.prob_futility, new.prob_futility)


def test_input_checks():
    with pytest.raises(ValueError, match="sd"):
        simulate_normal(0.2, 0.0, 50, 2, [2.8, 1.97])
    with pytest.raises(ValueError, match="test"):
        simulate_normal(0.2, 1.0, 50, 2, [2.8, 1.97], test="wald")
    with pytest.raises(ValueError, match="at least 3"):
        simulate_normal(0.2, 1.0, 2, [0.5, 1.0], [2.8, 1.97])
    with pytest.raises(ValueError, match="delta0"):
        simulate_binary(0.3, 0.4, 50, 2, [2.8, 1.97], delta0=-1.0)


# --------------------------------------------------------------------------- whole designs


@pytest.mark.parametrize("futility", [None, "obf"])
def test_design_simulation_normal_and_binary(futility):
    d = group_sequential_design(3, futility_spending=futility)
    n = math.ceil(fixed.sample_size_normal(0.25, 1.0, 0.05, 0.80).n_control * d.inflation_factor)
    sim = simulate_design_patients(
        d, "normal", n_control=n, n_treatment=n, delta=0.25, sd=1.0, n_sim=50_000, seed=1
    )
    assert abs(sim.type_i_error - 0.025) <= 2 * _se(0.025, 50_000) + 1e-3
    assert sim.power == pytest.approx(0.80, abs=0.02)
    assert sim.h0.reject_prob <= sim.h0_ignore_futility.reject_prob
    b = simulate_design_patients(
        d, "binary", n_control=180, p_control=0.3, p_treatment=0.45, n_sim=50_000, seed=1
    )
    assert abs(b.type_i_error - 0.025) <= 2 * _se(0.025, 50_000) + 2e-3
    assert b.seed == 1 and sim.seed == 1


def test_design_simulation_non_inferiority_h0_sits_on_the_margin():
    d = group_sequential_design(2)
    sim = simulate_design_patients(
        d,
        "binary",
        n_control=300,
        p_control=0.7,
        p_treatment=0.7,
        delta0=-0.1,
        n_sim=50_000,
        seed=2,
    )
    assert abs(sim.type_i_error - 0.025) <= 2 * _se(0.025, 50_000) + 2e-3
    assert sim.power > 0.5


def test_design_simulation_survival():
    d = group_sequential_design(2)
    study = accrual.survival_study(250, 0.7, 12, accrual_time=24, followup=12)
    sim = simulate_design_patients(d, "survival", study=study, n_sim=5_000, seed=3)
    assert abs(sim.type_i_error - 0.025) <= 3 * _se(0.025, 5_000)
    assert sim.power == pytest.approx(0.80, abs=0.04)


def test_design_simulation_needs_its_inputs():
    d = group_sequential_design(2)
    with pytest.raises(ValueError, match="accrual model"):
        simulate_design_patients(d, "survival", n_sim=10)
    with pytest.raises(ValueError, match="n_control, delta and sd"):
        simulate_design_patients(d, "normal", n_sim=10)
    with pytest.raises(ValueError, match="endpoint"):
        simulate_design_patients(d, "ordinal", n_sim=10)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- command line


def _cli_json(*args: str) -> dict:
    res = runner.invoke(app, ["simulate", *args, "--json"])
    assert res.exit_code == 0, res.output
    return json.loads(res.output)


@pytest.mark.parametrize(
    "args",
    [
        ["--endpoint", "normal", "--delta", "0.25", "--sd", "1"],
        ["--endpoint", "binary", "--pc", "0.3", "--pt", "0.45", "--futility", "obf"],
        ["--endpoint", "binary", "--pc", "0.7", "--pt", "0.7", "--delta0", "-0.1"],
        [
            "--endpoint", "survival", "--hr", "0.7", "--median-control", "12",
            "--accrual-time", "24", "--followup", "12",
        ],
    ],
    ids=["normal", "binary", "binary_ni", "survival"],
)  # fmt: skip
def test_cli_patient_engine(args):
    out = _cli_json(*args, "--looks", "2", "--engine", "patient", "--nsim", "4000", "--seed", "5")
    sim = out["simulation"]
    assert sim["engine"] == "patient"
    assert sim["nsim"] == 4000 and sim["seed"] == 5
    assert abs(sim["type_i_error"] - 0.025) <= 4 * _se(0.025, 4000)
    assert sim["power"] == pytest.approx(0.80, abs=4 * _se(0.8, 4000) + 0.01)


def test_cli_score_engine_is_the_default_and_unchanged():
    args = ["--endpoint", "normal", "--delta", "0.25", "--sd", "1", "--looks", "3"]
    default = _cli_json(*args, "--nsim", "20000", "--seed", "1")["simulation"]
    score = _cli_json(*args, "--nsim", "20000", "--seed", "1", "--engine", "score")["simulation"]
    assert default["engine"] == "score"
    assert default == score


def test_cli_engine_from_config_and_patient_default_nsim(tmp_path):
    cfg = tmp_path / "d.toml"
    cfg.write_text(
        '[design]\nendpoint = "binary"\npc = 0.3\npt = 0.45\n'
        '[sequential]\nlooks = 2\n[simulation]\nengine = "patient"\nseed = 3\n'
    )
    sim = _cli_json("--config", str(cfg))["simulation"]
    assert sim["engine"] == "patient" and sim["nsim"] == 10_000


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--endpoint", "survival", "--hr", "0.7", "--engine", "patient"], "accrual model"),
        (
            [
                "--endpoint", "survival", "--hr", "1.0", "--hr0", "1.3",
                "--median-control", "12", "--accrual-time", "24", "--followup", "12",
                "--engine", "patient",
            ],
            "hr0 = 1",
        ),
        (["--endpoint", "normal", "--delta", "0.3", "--sd", "1", "--engine", "mc"], "engine"),
    ],
)  # fmt: skip
def test_cli_patient_engine_errors(args, message):
    res = runner.invoke(app, ["simulate", *args, "--looks", "2", "--nsim", "100"])
    assert res.exit_code != 0
    assert message in res.output
