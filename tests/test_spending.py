"""M2: Lan-DeMets spending functions.

R reference values come from ``tests/r/gsd_reference_values.R`` (``gsDesign::sfLDOF`` and
``gsDesign::sfLDPocock``) and are skipped until ``tests/reference_values.json`` is regenerated.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from trialdesign import spending

REF = json.loads((Path(__file__).parent / "reference_values.json").read_text()).get(
    "boundaries", []
)
SPENDING_REF = [c for c in REF if c["id"].startswith("spending_")]


@pytest.mark.skipif(not SPENDING_REF, reason="regenerate reference_values.json with gsDesign")
@pytest.mark.parametrize("case", SPENDING_REF, ids=lambda c: c["id"])
def test_spending_matches_gsdesign(case: dict) -> None:
    f = spending.get_spending_function(case["spending"])
    got = f(np.array(case["info_rates"]), case["total"])
    np.testing.assert_allclose(got, case["cumulative"], rtol=1e-9, atol=1e-12)


def test_obf_closed_form() -> None:
    """2 - 2 Phi(z_{1 - a/2} / sqrt(t)) at t = 0.5, a = 0.025."""
    expected = 2 * stats.norm.sf(stats.norm.isf(0.0125) / np.sqrt(0.5))
    assert spending.spending_obf(0.5, 0.025) == pytest.approx(expected, rel=1e-12)


def test_pocock_closed_form() -> None:
    assert spending.spending_pocock(0.5, 0.025) == pytest.approx(
        0.025 * np.log(1 + (np.e - 1) * 0.5), rel=1e-12
    )


@pytest.mark.parametrize("name", ["obf", "pocock"])
def test_spending_is_zero_at_start_total_at_end_and_increasing(name: str) -> None:
    f = spending.get_spending_function(name)
    t = np.linspace(0, 1, 101)
    s = f(t, 0.025)
    assert s[0] == 0.0
    assert s[-1] == pytest.approx(0.025, rel=1e-12)
    assert np.all(np.diff(s) > 0)


def test_obf_spends_less_early_than_pocock() -> None:
    t = np.array([0.25, 0.5, 0.75])
    assert np.all(spending.spending_obf(t, 0.025) < spending.spending_pocock(t, 0.025))


def test_cumulative_spending_ends_exactly_at_total() -> None:
    s = spending.cumulative_spending("obf", 3, 0.2)
    assert s[-1] == 0.2
    assert s.shape == (3,)


def test_integer_looks_are_equally_spaced() -> None:
    np.testing.assert_allclose(spending.check_info_rates(4), [0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: spending.get_spending_function("hsd"), "needs a parameter"),
        (lambda: spending.spending_obf(0.5, 0.0), "between 0 and 1"),
        (lambda: spending.spending_pocock(1.5, 0.025), r"lie in \[0, 1\]"),
        (lambda: spending.check_info_rates([0.5, 0.9]), "end at 1.0"),
        (lambda: spending.check_info_rates([0.5, 0.5, 1.0]), "increase"),
        (lambda: spending.check_info_rates([0.0, 1.0]), "positive"),
        (lambda: spending.check_info_rates(0), "at least 1"),
        (lambda: spending.check_info_rates([]), "non-empty"),
    ],
)
def test_invalid_input_fails_with_plain_message(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
