"""Error-spending functions for group-sequential boundaries.

A spending function :math:`f(t; \\gamma)` is non-decreasing in the information fraction
:math:`t \\in [0, 1]` with :math:`f(0) = 0` and :math:`f(1) = \\gamma`, where :math:`\\gamma`
is the total error to spend: the one-sided alpha for efficacy boundaries, or beta for
futility boundaries. Every family is available for both.

A spending choice is written as a short string, used the same way in Python, on the command
line and in config files:

``"obf"``
    Lan-DeMets O'Brien-Fleming-type.
``"pocock"``
    Lan-DeMets Pocock-type.
``"hsd:<gamma>"``
    Hwang-Shih-DeCani with parameter gamma, e.g. ``"hsd:-4"`` (close to O'Brien-Fleming) or
    ``"hsd:1"`` (close to Pocock).
``"kd:<rho>"``
    Kim-DeMets power family :math:`\\gamma t^\\rho`, e.g. ``"kd:3"``.
``"user:<f1>,<f2>,...,1"``
    User-specified cumulative fractions of the total error at each look, e.g.
    ``"user:0.1,0.4,1"`` spends 10%, 40% and 100% of alpha by looks 1, 2 and 3.

References
----------
Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
*Biometrika* 70, 659-663.
Hwang, I. K., Shih, W. J. and DeCani, J. S. (1990). Group sequential designs using a family
of type I error probability spending functions. *Statistics in Medicine* 9, 1439-1445.
Kim, K. and DeMets, D. L. (1987). Design and analysis of group sequential tests based on the
type I error spending rate function. *Biometrika* 74, 149-154.
Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
Clinical Trials*. Chapman & Hall/CRC. Section 7.2.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "SPENDING_FUNCTIONS",
    "Spending",
    "SpendingName",
    "check_info_rates",
    "cumulative_spending",
    "get_spending_function",
    "parse_spending",
    "spending_hsd",
    "spending_kd",
    "spending_name",
    "spending_obf",
    "spending_pocock",
]

SpendingName = Literal["obf", "pocock", "hsd", "kd", "user"]
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


def spending_hsd(t: ArrayLike, total: float, gamma: float) -> NDArray[np.float64]:
    r"""Hwang-Shih-DeCani spending function.

    .. math:: f(t; \gamma_{\mathrm{tot}}) = \gamma_{\mathrm{tot}}
              \frac{1 - e^{-\gamma t}}{1 - e^{-\gamma}}, \qquad \gamma \ne 0,

    and :math:`\gamma_{\mathrm{tot}}\, t` for :math:`\gamma = 0`. Negative gamma spends late
    (gamma = -4 is close to O'Brien-Fleming), positive gamma spends early (gamma = 1 is close
    to Pocock). Equivalent to ``gsDesign::sfHSD`` and ``rpact`` ``typeOfDesign = "asHSD"``.

    Parameters
    ----------
    t : array_like
        Information fractions in [0, 1].
    total : float
        Total error to spend at :math:`t = 1`.
    gamma : float
        Shape parameter.

    References
    ----------
    Hwang, Shih & DeCani (1990).
    """
    _check_total(total)
    arr = _as_fraction(t)
    if not math.isfinite(gamma):
        raise ValueError("the Hwang-Shih-DeCani gamma must be finite")
    if gamma == 0.0:
        return total * arr
    return total * np.expm1(-gamma * arr) / math.expm1(-gamma)


def spending_kd(t: ArrayLike, total: float, rho: float) -> NDArray[np.float64]:
    r"""Kim-DeMets power spending function.

    .. math:: f(t; \gamma) = \gamma\, t^{\rho}, \qquad \rho > 0.

    Larger rho spends later: rho = 3 is close to O'Brien-Fleming, rho = 1 is close to Pocock.
    Equivalent to ``gsDesign::sfPower`` and ``rpact`` ``typeOfDesign = "asKD"``.

    Parameters
    ----------
    t : array_like
        Information fractions in [0, 1].
    total : float
        Total error to spend at :math:`t = 1`.
    rho : float
        Power, positive.

    References
    ----------
    Kim & DeMets (1987).
    """
    _check_total(total)
    arr = _as_fraction(t)
    if not (math.isfinite(rho) and rho > 0.0):
        raise ValueError("the Kim-DeMets rho must be positive")
    return total * arr**rho


SPENDING_FUNCTIONS: dict[str, SpendingFunction] = {
    "obf": spending_obf,
    "pocock": spending_pocock,
}
_PARAM_FUNCTIONS = {"hsd": spending_hsd, "kd": spending_kd}
_PARAM = {"hsd": "gamma", "kd": "rho"}
_FAMILIES = ("obf", "pocock", "hsd", "kd", "user")


def _g(x: float) -> str:
    return f"{x:g}"


@dataclass(frozen=True)
class Spending:
    """A parsed spending choice.

    Attributes
    ----------
    family : {"obf", "pocock", "hsd", "kd", "user"}
        Spending family.
    param : float or None
        Gamma for ``"hsd"``, rho for ``"kd"``, None otherwise.
    fractions : tuple of float or None
        Cumulative fractions of the total error at each look for ``"user"``.
    """

    family: SpendingName
    param: float | None = None
    fractions: tuple[float, ...] | None = None

    @property
    def label(self) -> str:
        """Canonical string form, e.g. ``"hsd:-4"``; :func:`parse_spending` reads it back."""
        if self.family in _PARAM_FUNCTIONS:
            return f"{self.family}:{_g(cast(float, self.param))}"
        if self.family == "user":
            return "user:" + ",".join(_g(f) for f in cast(tuple, self.fractions))
        return self.family

    @property
    def name(self) -> str:
        """Readable name, e.g. ``"Hwang-Shih-DeCani (gamma = -4)"``."""
        if self.family == "obf":
            return "Lan-DeMets O'Brien-Fleming-type"
        if self.family == "pocock":
            return "Lan-DeMets Pocock-type"
        if self.family == "hsd":
            return f"Hwang-Shih-DeCani (gamma = {_g(cast(float, self.param))})"
        if self.family == "kd":
            return f"Kim-DeMets (rho = {_g(cast(float, self.param))})"
        shown = ", ".join(_g(f) for f in cast(tuple, self.fractions))
        return f"user-specified (cumulative fractions {shown})"

    def function(self) -> SpendingFunction:
        """The spending function :math:`f(t; \\gamma)`; not defined for ``"user"``."""
        if self.family == "user":
            raise ValueError("user-specified spending is defined only at the looks")
        if self.family in _PARAM_FUNCTIONS:
            f = _PARAM_FUNCTIONS[self.family]
            return functools.partial(f, **{_PARAM[self.family]: self.param})
        return SPENDING_FUNCTIONS[self.family]


def _spec_error(spec: object) -> ValueError:
    return ValueError(
        'spending must be "obf", "pocock", "hsd:<gamma>", "kd:<rho>" or '
        f'"user:<f1>,...,1", got {spec!r}'
    )


def parse_spending(spec: str | Spending) -> Spending:
    """Parse a spending string such as ``"obf"``, ``"hsd:-4"``, ``"kd:3"`` or ``"user:0.2,1"``.

    User-specified fractions must lie in [0, 1], not decrease and end at 1; that they match
    the number of looks is checked by :func:`cumulative_spending`.
    """
    if isinstance(spec, Spending):
        return spec
    text = str(spec).strip().lower().replace(" ", "")
    family, sep, arg = text.partition(":")
    if family not in _FAMILIES:
        raise _spec_error(spec)
    if family in ("obf", "pocock"):
        if sep:
            raise ValueError(f'"{family}" spending takes no parameter, got {spec!r}')
        return Spending(cast(SpendingName, family))
    if not arg:
        example = {"hsd": "hsd:-4", "kd": "kd:3", "user": "user:0.1,0.4,1"}[family]
        raise ValueError(f'"{family}" spending needs a parameter, e.g. "{example}"')
    try:
        values = tuple(float(v) for v in arg.split(","))
    except ValueError:
        raise _spec_error(spec) from None
    if family in _PARAM_FUNCTIONS:
        if len(values) != 1:
            raise _spec_error(spec)
        out = Spending(cast(SpendingName, family), param=values[0])
        out.function()(np.array([0.5]), 0.5)  # validates the parameter
        return out
    f = np.asarray(values)
    if np.any(np.isnan(f)) or np.any((f < 0) | (f > 1)) or np.any(np.diff(f) < 0) or f[-1] != 1:
        raise ValueError(
            "user-specified spending fractions must lie in [0, 1], not decrease and end at 1"
        )
    return Spending("user", fractions=values)


def spending_name(spec: str | Spending) -> str:
    """Readable name of a spending string, e.g. ``"Kim-DeMets (rho = 3)"``."""
    return parse_spending(spec).name


def get_spending_function(name: str | Spending) -> SpendingFunction:
    """Spending function for a spending string (not available for ``"user"``)."""
    return parse_spending(name).function()


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
    name: str | Spending, info_rates: int | Sequence[float], total: float
) -> NDArray[np.float64]:
    """Cumulative error spent at each look, :math:`f(t_k; \\gamma)`, ending exactly at ``total``.

    For user-specified spending this is ``total`` times the given fractions, which must have
    one entry per look.

    References
    ----------
    Lan & DeMets (1983).
    """
    t = check_info_rates(info_rates)
    spec = parse_spending(name)
    if spec.family == "user":
        _check_total(total)
        fr = cast(tuple, spec.fractions)
        if len(fr) != t.size:
            raise ValueError(
                f"user-specified spending has {len(fr)} fractions but the design has {t.size} looks"
            )
        out = total * np.asarray(fr, dtype=float)
    else:
        out = spec.function()(t, total)
    out[-1] = total  # remove rounding error at t = 1
    return out
