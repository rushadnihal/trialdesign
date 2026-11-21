"""M1: fixed-design sample size and power, validated against R (tests/reference_values.json).

Reference values come from ``tests/r/fixed_reference_values.R`` (base R ``power.t.test``,
``power.prop.test`` and the Schoenfeld formula evaluated in R).
"""

import json
import math
from pathlib import Path

import pytest
from scipy import stats

from trialdesign import fixed

REF = json.loads((Path(__file__).parent / "reference_values.json").read_text())["fixed"]


def _cases(source: str, key: str) -> list[dict]:
    return [c for c in REF if c["source"] == source and key in c and "power" in c]


def _ref(case_id: str) -> dict:
    return next(c for c in REF if c["id"] == case_id)


# ---------------------------------------------------------------- R reference values


@pytest.mark.parametrize("case", ["normal_n_d0.5", "normal_n_d3_sd10_a01_p90"])
def test_normal_t_matches_power_t_test(case: str) -> None:
    c = _ref(case)
    res = fixed.sample_size_normal(
        c["delta"], c["sd"], alpha_two_sided=c["alpha_two_sided"], power=c["power"], method="t"
    )
    assert res.n_control == pytest.approx(c["n_per_arm"], abs=1e-4)
    assert res.n_control_ceil == math.ceil(c["n_per_arm"])


def test_normal_t_power_matches_power_t_test() -> None:
    c = _ref("normal_power_n40")
    p = fixed.power_normal(c["n_per_arm"], c["delta"], c["sd"], method="t")
    assert p == pytest.approx(c["power"], abs=1e-8)


@pytest.mark.parametrize("case", ["binary_n_30_45", "binary_n_10_20_p90"])
def test_binary_matches_power_prop_test(case: str) -> None:
    c = _ref(case)
    res = fixed.sample_size_binary(
        c["p_control"], c["p_treatment"], alpha_two_sided=c["alpha_two_sided"], power=c["power"]
    )
    assert res.n_control == pytest.approx(c["n_per_arm"], abs=1e-4)
    assert res.n_control_ceil == math.ceil(c["n_per_arm"])


def test_binary_power_matches_power_prop_test() -> None:
    c = _ref("binary_power_n100")
    p = fixed.power_binary(c["n_per_arm"], c["p_control"], c["p_treatment"])
    assert p == pytest.approx(c["power"], abs=1e-8)


@pytest.mark.parametrize("case", ["survival_events_hr0.7", "survival_events_hr0.75_2to1"])
def test_survival_matches_schoenfeld_in_r(case: str) -> None:
    c = _ref(case)
    res = fixed.events_survival(
        c["hazard_ratio"],
        alpha_two_sided=c["alpha_two_sided"],
        power=c["power"],
        allocation=c["allocation"],
    )
    assert res.events == pytest.approx(c["events"], abs=1e-4)
    assert res.events_ceil == math.ceil(c["events"])


# -------------------------------------------------------------- outline's target values


def test_outline_validation_table() -> None:
    """The three M1 rows of the outline's validation plan (tolerance 0.01)."""
    assert fixed.sample_size_normal(0.5, 1.0).n_control == pytest.approx(62.79, abs=0.01)
    assert fixed.sample_size_normal(0.5, 1.0, method="t").n_control == pytest.approx(
        63.77, abs=0.01
    )
    assert fixed.sample_size_binary(0.30, 0.45).n_control == pytest.approx(162.33, abs=0.01)
    assert fixed.events_survival(0.70).events == pytest.approx(246.79, abs=0.01)


# ------------------------------------------------------------------- property checks


@pytest.mark.parametrize("allocation", [0.5, 1.0, 2.0, 3.0])
@pytest.mark.parametrize("method", ["z", "t"])
def test_normal_power_inverts_sample_size(allocation: float, method: str) -> None:
    res = fixed.sample_size_normal(2.0, 5.0, power=0.85, allocation=allocation, method=method)
    assert res.n_treatment == pytest.approx(allocation * res.n_control)
    p = fixed.power_normal(res.n_control, 2.0, 5.0, allocation=allocation, method=method)
    assert p == pytest.approx(0.85, abs=1e-6)


@pytest.mark.parametrize("allocation", [0.5, 1.0, 2.0])
def test_binary_power_inverts_sample_size(allocation: float) -> None:
    res = fixed.sample_size_binary(0.2, 0.35, power=0.9, allocation=allocation)
    p = fixed.power_binary(res.n_control, 0.2, 0.35, allocation=allocation)
    assert p == pytest.approx(0.9, abs=1e-9)


@pytest.mark.parametrize("allocation", [0.5, 1.0, 2.0])
def test_survival_power_inverts_events(allocation: float) -> None:
    res = fixed.events_survival(0.8, power=0.9, allocation=allocation)
    assert fixed.power_survival(res.events, 0.8, allocation=allocation) == pytest.approx(0.9)


def test_normal_z_closed_form_unequal_allocation() -> None:
    """n_C = (1 + 1/r) sigma^2 (z_a + z_b)^2 / delta^2 with r = 2."""
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    expected = 1.5 * 4.0 * z**2 / 1.0
    assert fixed.sample_size_normal(1.0, 2.0, allocation=2.0).n_control == pytest.approx(expected)


def test_unequal_allocation_needs_more_patients_in_total() -> None:
    eq = fixed.sample_size_normal(0.5, 1.0).n_total
    for r in (0.5, 2.0, 3.0):
        assert fixed.sample_size_normal(0.5, 1.0, allocation=r).n_total > eq
    # 1:1 minimises events for the log-rank test
    assert fixed.events_survival(0.7, allocation=2.0).events > fixed.events_survival(0.7).events


def test_symmetry_in_direction_of_effect() -> None:
    assert fixed.sample_size_normal(-0.5, 1.0).n_control == pytest.approx(
        fixed.sample_size_normal(0.5, 1.0).n_control
    )
    assert fixed.sample_size_binary(0.45, 0.30).n_control == pytest.approx(
        fixed.sample_size_binary(0.30, 0.45).n_control
    )
    assert fixed.events_survival(1 / 0.7).events == pytest.approx(fixed.events_survival(0.7).events)


def test_survival_total_patients_from_event_probability() -> None:
    res = fixed.events_survival(0.7, prob_event=0.6)
    assert res.n_total == pytest.approx(res.events / 0.6)
    assert fixed.events_survival(0.7).n_total is None


def test_power_at_null_is_one_tail_alpha() -> None:
    assert fixed.power_survival(300, 1.0) == pytest.approx(0.025)


# ---------------------------------------------------------------------- input errors


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: fixed.sample_size_normal(0.0, 1.0), "delta must be non-zero"),
        (lambda: fixed.sample_size_normal(0.5, -1.0), "sd must be positive"),
        (lambda: fixed.sample_size_normal(0.5, 1.0, alpha_two_sided=1.5), "alpha_two_sided"),
        (lambda: fixed.sample_size_normal(0.5, 1.0, power=0.01), "power"),
        (lambda: fixed.sample_size_normal(0.5, 1.0, allocation=0), "allocation"),
        (lambda: fixed.sample_size_normal(0.5, 1.0, method="x"), "method"),
        (lambda: fixed.sample_size_binary(0.3, 0.3), "must differ"),
        (lambda: fixed.sample_size_binary(0.3, 1.0), "p_treatment"),
        (lambda: fixed.events_survival(1.0), "hazard_ratio"),
        (lambda: fixed.events_survival(0.7, prob_event=0.0), "prob_event"),
    ],
)
def test_invalid_input_fails_with_plain_message(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
