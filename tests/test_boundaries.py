"""M2: group-sequential boundaries, inflation factor and non-binding futility.

R reference values come from ``tests/r/gsd_reference_values.R`` (rpact for efficacy boundaries
and inflation factors, gsDesign ``test.type = 4`` for non-binding futility) and are skipped
until ``tests/reference_values.json`` is regenerated. The scipy multivariate normal
cross-check and the property tests always run.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from trialdesign import boundaries, fixed, spending
from trialdesign.boundaries import _grid

REF = json.loads((Path(__file__).parent / "reference_values.json").read_text()).get(
    "boundaries", []
)
EFFICACY_REF = [c for c in REF if "efficacy" in c and "futility" not in c]
FUTILITY_REF = [c for c in REF if c["id"].startswith("futility_")]
MAX_EVENTS_REF = [c for c in REF if c["id"].startswith("max_events_")]
NEEDS_R = "regenerate reference_values.json with rpact and gsDesign"


# ---------------------------------------------------------------- R reference values


@pytest.mark.skipif(not EFFICACY_REF, reason=NEEDS_R)
@pytest.mark.parametrize("case", EFFICACY_REF, ids=lambda c: c["id"])
def test_efficacy_bounds_match_rpact(case: dict) -> None:
    c = boundaries.efficacy_bounds(case["info_rates"], case["alpha_one_sided"], case["spending"])
    np.testing.assert_allclose(c, case["efficacy"], atol=1e-3, rtol=0)


@pytest.mark.skipif(not EFFICACY_REF, reason=NEEDS_R)
@pytest.mark.parametrize("case", EFFICACY_REF, ids=lambda c: c["id"])
def test_inflation_factor_matches_rpact(case: dict) -> None:
    d = boundaries.group_sequential_design(
        case["info_rates"], case["alpha_one_sided"], case["power"], case["spending"]
    )
    assert d.inflation_factor == pytest.approx(case["inflation_factor"], abs=1e-3)


@pytest.mark.skipif(not FUTILITY_REF, reason=NEEDS_R)
@pytest.mark.parametrize("case", FUTILITY_REF, ids=lambda c: c["id"])
def test_nonbinding_futility_matches_gsdesign(case: dict) -> None:
    d = boundaries.group_sequential_design(
        case["info_rates"],
        case["alpha_one_sided"],
        case["power"],
        efficacy_spending=case["efficacy_spending"],
        futility_spending=case["futility_spending"],
    )
    np.testing.assert_allclose(d.efficacy, case["efficacy"], atol=1e-3, rtol=0)
    np.testing.assert_allclose(d.futility, case["futility"], atol=1e-3, rtol=0)
    assert d.inflation_factor == pytest.approx(case["inflation_factor"], abs=1e-3)


@pytest.mark.skipif(not MAX_EVENTS_REF, reason=NEEDS_R)
@pytest.mark.parametrize("case", MAX_EVENTS_REF, ids=lambda c: c["id"])
def test_max_events_is_fixed_events_times_inflation(case: dict) -> None:
    ev = fixed.events_survival(
        case["hazard_ratio"], alpha_two_sided=2 * case["alpha_one_sided"], power=case["power"]
    )
    assert ev.events == pytest.approx(case["events_fixed"], abs=1e-6)
    d = boundaries.group_sequential_design(
        case["info_rates"], case["alpha_one_sided"], case["power"], case["spending"]
    )
    assert d.max_sample_size(ev.events) == pytest.approx(case["events_max"], abs=0.05)


# -------------------------------------------------------------- outline's target values


def test_outline_validation_table() -> None:
    """The M2 rows of the outline's validation plan (tolerance 1e-3)."""
    np.testing.assert_allclose(
        boundaries.efficacy_bounds(3, 0.025, "obf"), [3.7103, 2.5114, 1.9931], atol=1e-3
    )
    np.testing.assert_allclose(
        boundaries.efficacy_bounds(3, 0.025, "pocock"), [2.2794, 2.2949, 2.2959], atol=1e-3
    )
    np.testing.assert_allclose(
        boundaries.efficacy_bounds([0.5, 1.0], 0.025, "obf"), [2.9626, 1.9686], atol=1e-3
    )
    obf = boundaries.group_sequential_design(3, 0.025, 0.8, "obf")
    pocock = boundaries.group_sequential_design(3, 0.025, 0.8, "pocock")
    assert obf.inflation_factor == pytest.approx(1.0128, abs=1e-3)
    assert pocock.inflation_factor == pytest.approx(1.1704, abs=1e-3)


# --------------------------------------------- cross-check: scipy multivariate normal


@pytest.mark.parametrize("name", ["obf", "pocock"])
@pytest.mark.parametrize(
    "info_rates", [2, 3, 4, [0.5, 1.0], [0.3, 0.7, 1.0], [0.2, 0.45, 0.8, 1.0]], ids=str
)
def test_recursive_integration_agrees_with_mvn(name: str, info_rates) -> None:
    rec = boundaries.efficacy_bounds(info_rates, 0.025, name)
    mvn = boundaries.efficacy_bounds_mvn(info_rates, 0.025, name)
    np.testing.assert_allclose(rec, mvn, atol=1e-4, rtol=0)


# ------------------------------------------------------------------- property checks


@pytest.mark.parametrize("name", ["obf", "pocock"])
def test_single_look_is_fixed_critical_value(name: str) -> None:
    assert boundaries.efficacy_bounds(1, 0.025, name)[0] == pytest.approx(
        stats.norm.isf(0.025), abs=1e-9
    )
    d = boundaries.group_sequential_design(1, 0.025, 0.8, name)
    assert d.inflation_factor == pytest.approx(1.0, abs=1e-9)


def test_simpson_grid_integrates_normal_density() -> None:
    z, w = _grid(0.7, -math.inf, math.inf, 18)
    # Simpson on the J&T grid with r = 18 is accurate to about 1e-7.
    assert w @ stats.norm.pdf(z - 0.7) == pytest.approx(1.0, abs=1e-6)
    z, w = _grid(0.0, -1.0, 2.0, 18)
    assert w @ stats.norm.pdf(z) == pytest.approx(stats.norm.cdf(2) - stats.norm.cdf(-1), abs=1e-6)


@pytest.mark.parametrize("name", ["obf", "pocock"])
@pytest.mark.parametrize("info_rates", [3, 5, [0.25, 0.6, 1.0]], ids=str)
def test_alpha_spent_per_look_equals_spending_function(name: str, info_rates) -> None:
    c = boundaries.efficacy_bounds(info_rates, 0.025, name)
    up, low = boundaries.crossing_probabilities(info_rates, c)
    np.testing.assert_allclose(
        np.cumsum(up), spending.cumulative_spending(name, info_rates, 0.025), atol=1e-9
    )
    assert np.all(low == 0)


def test_obf_boundaries_decrease_and_pocock_are_nearly_flat() -> None:
    obf = boundaries.efficacy_bounds(5, 0.025, "obf")
    pocock = boundaries.efficacy_bounds(5, 0.025, "pocock")
    assert np.all(np.diff(obf) < 0)
    assert np.ptp(pocock) < 0.06
    assert obf[-1] < pocock[-1]


@pytest.mark.parametrize("name", ["obf", "pocock"])
def test_design_has_target_power_at_its_drift(name: str) -> None:
    d = boundaries.group_sequential_design([0.3, 0.7, 1.0], 0.025, 0.9, name)
    up, _ = d.crossing_probabilities()
    assert up.sum() == pytest.approx(0.9, abs=1e-6)
    fixed_drift = stats.norm.isf(0.025) + stats.norm.isf(0.1)
    assert d.drift == pytest.approx(fixed_drift * math.sqrt(d.inflation_factor))
    assert d.inflation_factor > 1


def test_pocock_inflates_more_than_obf_and_futility_inflates_more() -> None:
    obf = boundaries.group_sequential_design(3).inflation_factor
    pocock = boundaries.group_sequential_design(3, efficacy_spending="pocock").inflation_factor
    obf_fut = boundaries.group_sequential_design(3, futility_spending="obf").inflation_factor
    assert 1 < obf < pocock
    assert obf_fut > obf


@pytest.mark.parametrize(("eff", "fut"), [("obf", "obf"), ("obf", "pocock"), ("pocock", "obf")])
def test_nonbinding_futility_properties(eff: str, fut: str) -> None:
    t = [0.3, 0.7, 1.0]
    d = boundaries.group_sequential_design(t, 0.025, 0.8, eff, fut)
    # Efficacy boundaries do not depend on futility (non-binding).
    np.testing.assert_array_equal(d.efficacy, boundaries.efficacy_bounds(t, 0.025, eff))
    assert d.futility is not None
    assert np.all(d.futility <= d.efficacy)
    assert d.futility[-1] == d.efficacy[-1]
    # Under H1, beta is spent as the beta-spending function says and power is on target.
    up, low = d.crossing_probabilities()
    np.testing.assert_allclose(np.cumsum(low), spending.cumulative_spending(fut, t, 0.2), atol=1e-6)
    assert up.sum() == pytest.approx(0.8, abs=1e-6)
    # Under H0, alpha is exactly 0.025 if futility is ignored and smaller if it is followed.
    up0, _ = d.crossing_probabilities(drift=0.0, use_futility=False)
    assert up0.sum() == pytest.approx(0.025, abs=1e-9)
    up0_fut, _ = d.crossing_probabilities(drift=0.0)
    assert up0_fut.sum() < 0.025


def test_first_futility_bound_closed_form() -> None:
    """At look 1, P_delta(Z_1 < b_1) = beta(t_1), so b_1 = delta sqrt(t_1) + z_{beta(t_1)}."""
    d = boundaries.group_sequential_design([0.4, 1.0], futility_spending="pocock")
    beta_1 = spending.spending_pocock(0.4, 0.2)
    assert d.futility is not None
    assert d.futility[0] == pytest.approx(d.drift * math.sqrt(0.4) + stats.norm.ppf(beta_1))


def test_max_sample_size_reproduces_survival_drift() -> None:
    """Max events D_max = D_fix * IF gives drift sqrt(D_max / 4) |log HR| (Schoenfeld, 1:1)."""
    d = boundaries.group_sequential_design(3, 0.025, 0.8, "obf")
    ev = fixed.events_survival(0.7, alpha_two_sided=0.05, power=0.8)
    d_max = d.max_sample_size(ev.events)
    assert d_max == pytest.approx(ev.events * d.inflation_factor)
    assert math.sqrt(d_max / 4) * abs(math.log(0.7)) == pytest.approx(d.drift)
    np.testing.assert_allclose(d.look_sample_sizes(ev.events), d_max * d.info_rates)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: boundaries.efficacy_bounds(3, 0.6), "alpha_one_sided"),
        (lambda: boundaries.efficacy_bounds([0.6, 0.4, 1.0]), "increase"),
        (lambda: boundaries.efficacy_bounds(3, spending="linear"), "spending must be one of"),
        (lambda: boundaries.group_sequential_design(3, power=0.01), "power"),
        (lambda: boundaries.crossing_probabilities(2, [2.0]), "one boundary per look"),
        (lambda: boundaries.crossing_probabilities(2, [2, 2], [3, 2]), "must not exceed"),
        (lambda: boundaries.group_sequential_design(3).max_sample_size(0), "n_fixed"),
    ],
)
def test_invalid_input_fails_with_plain_message(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
