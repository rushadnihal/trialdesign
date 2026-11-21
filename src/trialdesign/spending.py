"""Lan-DeMets spending functions: O'Brien-Fleming-type and Pocock-type.

A spending function :math:`f(t; \\gamma)` is non-decreasing in the information fraction
:math:`t \\in [0, 1]` with :math:`f(0) = 0` and :math:`f(1) = \\gamma`, where :math:`\\gamma`
is the total error to spend: the one-sided alpha for efficacy boundaries, or beta for
futility boundaries. The same two families are used for both.

References
----------
Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
*Biometrika* 70, 659-663.
Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
Clinical Trials*. Chapman & Hall/CRC. Section 7.2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "SPENDING_FUNCTIONS",
    "SpendingName",
    "check_info_rates",
    "cumulative_spending",
    "get_spending_function",
    "spending_obf",
    "spending_pocock",
]

SpendingName = Literal["obf", "pocock"]
SpendingFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


def _check_total(total: float) -> None:
    if not 0.0 < total < 1.0:
        raise ValueError("the total error to spend must be between 0 and 1")


def _as_fraction(t: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(t, dtype=float)
    if np.any((arr < 0.0) | (arr > 1.0)) or np.any(np.isnan(arr)):
        raise ValueError("information fractions must lie in [0, 1]")
    return arr


def spending_obf(t: ArrayLike, total: float) -> NDArray[np.float64]:
    r"""Lan-DeMets O'Brien-Fleming-type spending function.

    .. math:: f(t; \gamma) = 2 - 2\,\Phi\!\left(\frac{z_{1-\gamma/2}}{\sqrt t}\right),
              \qquad f(0) = 0.

    Spends almost nothing early, so the final critical value stays close to the fixed-design
    one. Equivalent to ``gsDesign::sfLDOF`` and ``rpact`` ``typeOfDesign = "asOF"``.

    Parameters
    ----------
    t : array_like
        Information fractions in [0, 1].
    total : float
        Total error to spend at :math:`t = 1` (one-sided alpha, or beta).

    References
    ----------
    Lan & DeMets (1983); Jennison & Turnbull (2000), section 7.2.
    """
    _check_total(total)
    arr = _as_fraction(t)
    z = stats.norm.isf(total / 2)
    with np.errstate(divide="ignore"):
        out = 2.0 * stats.norm.sf(z / np.sqrt(arr))
    return np.where(arr > 0, out, 0.0)


def spending_pocock(t: ArrayLike, total: float) -> NDArray[np.float64]:
    r"""Lan-DeMets Pocock-type spending function.

    .. math:: f(t; \gamma) = \gamma \log\{1 + (e - 1)\,t\}.

    Spends error roughly evenly, giving nearly constant critical values. Equivalent to
    ``gsDesign::sfLDPocock`` and ``rpact`` ``typeOfDesign = "asP"``.

    Parameters
    ----------
    t : array_like
        Information fractions in [0, 1].
    total : float
        Total error to spend at :math:`t = 1` (one-sided alpha, or beta).

    References
    ----------
    Lan & DeMets (1983); Jennison & Turnbull (2000), section 7.2.
    """
    _check_total(total)
    arr = _as_fraction(t)
    return total * np.log1p((np.e - 1.0) * arr)


SPENDING_FUNCTIONS: dict[str, SpendingFunction] = {
    "obf": spending_obf,
    "pocock": spending_pocock,
}


def get_spending_function(name: str) -> SpendingFunction:
    """Look up a spending function by name (``"obf"`` or ``"pocock"``)."""
    try:
        return SPENDING_FUNCTIONS[name]
    except KeyError:
        options = ", ".join(f'"{k}"' for k in SPENDING_FUNCTIONS)
        raise ValueError(f"spending must be one of {options}, got {name!r}") from None


def check_info_rates(info_rates: int | Sequence[float]) -> NDArray[np.float64]:
    """Validate information fractions, or build ``k`` equally spaced ones from an integer.

    Returns a float array that is strictly increasing, positive and ends at 1.0.
    """
    if isinstance(info_rates, int | np.integer):
        k = int(info_rates)
        if k < 1:
            raise ValueError("the number of looks must be at least 1")
        return np.arange(1, k + 1, dtype=float) / k
    t = np.asarray(info_rates, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("looks must be a non-empty list of information fractions")
    if np.any(np.isnan(t)) or t[0] <= 0.0 or np.any(np.diff(t) <= 0.0) or t[-1] != 1.0:
        raise ValueError("looks must be positive, increase and end at 1.0")
    return t


def cumulative_spending(
    name: str, info_rates: int | Sequence[float], total: float
) -> NDArray[np.float64]:
    """Cumulative error spent at each look, :math:`f(t_k; \\gamma)`, ending exactly at ``total``.

    References
    ----------
    Lan & DeMets (1983).
    """
    t = check_info_rates(info_rates)
    out = get_spending_function(name)(t, total)
    out[-1] = total  # remove rounding error at t = 1
    return out
