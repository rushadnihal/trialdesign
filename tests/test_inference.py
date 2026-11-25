"""M15a: inference at the end of a group-sequential trial.

Reference values in ``tests/inference_reference_values.json`` come from
``tests/r/inference_reference_values.R`` (``rpact`` 3.4.0: getRepeatedConfidenceIntervals,
getRepeatedPValues, getFinalPValue, getFinalConfidenceInterval) on survival datasets, so the
estimates are hazard ratios. The tolerance is 1e-3 as planned; the values agree to about 1e-6.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from trialdesign.boundaries import group_sequential_design
from trialdesign.inference import (
    analyse,
    median_unbiased_estimate,
    repeated_confidence_intervals,
    repeated_p_values,
    stagewise_probability,
)
from trialdesign.simulate import mc_se

REF = json.loads((Path(__file__).parent / "inference_reference_values.json").read_text())
CASES = REF["cases"]
TOL = 1e-3


def _design(case: dict):
    return group_sequential_design(
        case["info_rates"],
        efficacy_spending=case["efficacy_spending"],
        futility_spending=case["futility_spending"],
        binding_futility=case["binding_futility"],
    )


@pytest.fixture(scope="module", params=CASES, ids=[c["id"] for c in CASES])
def pair(request):
    case = request.param
    r = case["allocation"]
    info_max = case["events"] * r / (1 + r) ** 2
    # log HR = -drift / sqrt(I_max): positive z favours treatment
    return case, analyse(_design(case), case["z"], effect_per_drift=-1 / math.sqrt(info_max))


def _hr(result, drift):
    return np.exp(result.to_effect(drift))


def test_boundaries_match_rpact(pair):
    case, _ = pair
    d = _design(case)
    np.testing.assert_allclose(d.efficacy, case["efficacy"], atol=1e-5)
    if case["futility"] is not None:
        np.testing.assert_allclose(d.futility[:-1], case["futility"], atol=1e-5)


def test_repeated_intervals_and_p_values_match_rpact(pair):
    case, res = pair
    # the HR interval's lower end comes from the drift interval's upper end
    np.testing.assert_allclose(_hr(res, res.rci_upper_drift), case["rci_lower_hr"], atol=TOL)
    np.testing.assert_allclose(_hr(res, res.rci_lower_drift), case["rci_upper_hr"], atol=TOL)
    np.testing.assert_allclose(res.repeated_p, case["repeated_p"], atol=TOL)


def test_stagewise_inference_matches_rpact(pair):
    case, res = pair
    if case["final_p"] is None:
        assert res.p_value is None and res.median_unbiased_drift is None
        assert not res.crossed and not res.final
        return
    assert res.p_value == pytest.approx(case["final_p"], abs=TOL)
    assert _hr(res, res.median_unbiased_drift) == pytest.approx(case["median_unbiased_hr"], abs=TOL)
    assert _hr(res, res.ci_upper_drift) == pytest.approx(case["ci_lower_hr"], abs=TOL)
    assert _hr(res, res.ci_lower_drift) == pytest.approx(case["ci_upper_hr"], abs=TOL)


# --------------------------------------------------------------------------- properties


def test_one_look_gives_the_fixed_design_wald_interval():
    d = group_sequential_design([1.0])
    res = analyse(d, [2.5])
    z = stats.norm.isf(0.025)
    assert res.rci_lower_drift[0] == pytest.approx(2.5 - z)
    assert res.rci_upper_drift[0] == pytest.approx(2.5 + z)
    # with one look the stage-wise quantities are the usual ones
    assert res.p_value == pytest.approx(stats.norm.sf(2.5))
    assert res.median_unbiased_drift == pytest.approx(2.5)
    assert res.ci_lower_drift == pytest.approx(2.5 - z)
    assert res.repeated_p[0] == pytest.approx(stats.norm.sf(2.5), rel=1e-6)


def test_stagewise_p_value_at_the_boundary_is_the_alpha_spent():
    d = group_sequential_design(3)
    for k in range(3):
        z = [0.0] * k + [float(d.efficacy[k])]
        spent = stagewise_probability(d, z, 0.0)
        up, _ = d.crossing_probabilities(0.0, use_futility=False)
        assert spent == pytest.approx(up[: k + 1].sum(), abs=1e-9)
    assert stagewise_probability(d, [0.0, 0.0, d.efficacy[2]]) == pytest.approx(0.025, abs=1e-9)


def test_median_unbiased_estimate_is_median_unbiased():
    """Simulate trials at a drift; the estimate falls below the truth half of the time."""
    d = group_sequential_design(3)
    delta, n = 2.2, 1000
    rng = np.random.default_rng(3)
    t = d.info_rates
    incr = rng.normal(delta * np.diff(t, prepend=0.0), np.sqrt(np.diff(t, prepend=0.0)), (n, 3))
    z = np.cumsum(incr, axis=1) / np.sqrt(t)
    below = 0
    for row in z:
        k = next((j for j in range(3) if row[j] >= d.efficacy[j]), 2)
        below += median_unbiased_estimate(d, row[: k + 1]) < delta
    assert abs(below / n - 0.5) <= 3 * float(mc_se(0.5, n))


def test_repeated_p_value_rejects_exactly_when_the_boundary_is_crossed():
    d = group_sequential_design(3, efficacy_spending="pocock")
    lo, _ = repeated_confidence_intervals(d, [1.0, 2.5])
    p = repeated_p_values(d, [1.0, 2.5])
    assert (p[1] <= 0.025) == (2.5 >= d.efficacy[1]) == (lo[1] > 0)
    at = repeated_p_values(d, [d.efficacy[0]])
    assert at[0] == pytest.approx(0.025, rel=1e-6)


def test_input_checks():
    d = group_sequential_design(3)
    with pytest.raises(ValueError, match="crossed before"):
        analyse(d, [5.0, 1.0])
    with pytest.raises(ValueError, match="between 1 and 3"):
        analyse(d, [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="finite"):
        analyse(d, [np.nan])


# --------------------------------------------------------------------------- CLI


def test_cli_design_analyse():
    from typer.testing import CliRunner

    from trialdesign.cli import app

    runner = CliRunner()
    args = ["design", "--endpoint", "survival", "--hr", "0.7", "--looks", "0.25,0.5,0.75,1"]
    result = runner.invoke(app, [*args, "--analyse", "1.5,3.1", "--json"])
    assert result.exit_code == 0, result.output
    a = json.loads(result.output)["analysis"]
    assert a["effect"] == "hazard_ratio" and a["crossed"] and a["stage"] == 2
    assert a["p_value"] == pytest.approx(REF["cases"][0]["final_p"], abs=1e-6)
    assert a["ci_lower"] < a["median_unbiased"] < a["ci_upper"] < 1
    assert a["looks"][1]["rci_lower"] < a["looks"][1]["rci_upper"] < 1
    result = runner.invoke(app, [*args, "--analyse", "0.5"])
    assert result.exit_code == 0 and "trial continues" in result.output
    result = runner.invoke(
        app, ["design", "--endpoint", "normal", "--delta", "0.3", "--sd", "1", "--analyse", "2.0"]
    )
    assert result.exit_code == 2 and "--looks" in result.output
