r"""Equivalence by two one-sided tests (TOST) for fixed two-arm designs.

With equivalence margins :math:`\Delta_L < 0 < \Delta_U` on the difference
:math:`\theta = \mu_T - \mu_C` (or :math:`p_T - p_C`), the hypotheses are

.. math:: H_0: \theta \le \Delta_L \text{ or } \theta \ge \Delta_U \quad\text{against}\quad
          H_1: \Delta_L < \theta < \Delta_U,

and equivalence is concluded when both one-sided tests reject at level :math:`\alpha`:
:math:`Z_L = (\hat\theta - \Delta_L)/\mathrm{SE}_L \ge z_{1-\alpha}` and
:math:`Z_U = (\hat\theta - \Delta_U)/\mathrm{SE}_U \le -z_{1-\alpha}` (Schuirmann 1987). This
is the same as the :math:`1 - 2\alpha` confidence interval lying inside the margins, so the
usual :math:`\alpha = 0.05` gives a 90% interval.

Power counts the joint event on one estimate :math:`\hat\theta \sim N(\theta, \sigma_1^2/n_C)`:

.. math:: 1 - \beta = \Phi\!\left(\frac{\Delta_U - z_{1-\alpha}\sigma_U/\sqrt{n_C} -
          \theta}{\sigma_1/\sqrt{n_C}}\right) - \Phi\!\left(\frac{\Delta_L +
          z_{1-\alpha}\sigma_L/\sqrt{n_C} - \theta}{\sigma_1/\sqrt{n_C}}\right),

zero when the bracket is empty. For a normal endpoint with known SD,
:math:`\sigma_L = \sigma_U = \sigma_1 = \sigma\sqrt{1 + 1/r}` and with symmetric margins and
:math:`\theta = 0` the sample size has the closed form
:math:`n_C = (1 + 1/r)\,\sigma^2 (z_{1-\alpha} + z_{1-\beta/2})^2 / \Delta^2`. For a binary
endpoint :math:`\sigma_L` and :math:`\sigma_U` use the Farrington-Manning restricted rates at
each margin and :math:`\sigma_1` the design rates, as in the non-inferiority designs of
:mod:`trialdesign.fixed`. In both cases ``n_C`` is solved with ``brentq``.

References
----------
Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power
approach for assessing the equivalence of average bioavailability. *Journal of
Pharmacokinetics and Biopharmaceutics* 15, 657-680. Chow, Shao & Wang (2008), sections 3.2.4
and 4.2.4. Farrington & Manning (1990).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from trialdesign.fixed import FixedSampleSize, farrington_manning_rates

__all__ = [
    "EquivalenceTest",
    "power_equivalence_binary",
    "power_equivalence_normal",
    "sample_size_equivalence_binary",
    "sample_size_equivalence_normal",
    "tost",
]


def _margins(margin: float | Sequence[float]) -> tuple[float, float]:
    if np.ndim(margin) == 0:
        m = float(margin)  # type: ignore[arg-type]
        lo, hi = -abs(m), abs(m)
    else:
        if len(margin) != 2:  # type: ignore[arg-type]
            raise ValueError("margin must be one number or a (lower, upper) pair")
        lo, hi = (float(x) for x in margin)  # type: ignore[union-attr]
    if not lo < 0.0 < hi:
        raise ValueError("equivalence margins must satisfy lower < 0 < upper")
    return lo, hi


def _check(alpha: float, allocation: float, power: float | None = None) -> None:
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha (the level of each one-sided test) must be in (0, 0.5)")
    if not allocation > 0:
        raise ValueError("allocation (n_T / n_C) must be positive")
    if power is not None and not 0.0 < power < 1.0:
        raise ValueError("power must be between 0 and 1")


def _power(n_c: float, theta: float, lo: float, hi: float, s_lo, s_hi, s1, alpha) -> float:
    z = stats.norm.isf(alpha)
    root = math.sqrt(n_c)
    upper = (hi - z * s_hi / root - theta) / (s1 / root)
    lower = (lo + z * s_lo / root - theta) / (s1 / root)
    return float(max(stats.norm.cdf(upper) - stats.norm.cdf(lower), 0.0))


def _solve_n(power_at, power: float, start: float) -> float:
    lo, hi = 1e-6, max(start, 2.0)
    while power_at(hi) < power:
        hi *= 2.0
        if hi > 1e12:
            raise ValueError(
                "power cannot be reached: the true difference is too close to a margin"
            )
    return float(optimize.brentq(lambda n: power_at(n) - power, lo, hi, xtol=1e-10))


def _inside(theta: float, lo: float, hi: float) -> None:
    if not lo < theta < hi:
        raise ValueError("the true difference must lie strictly inside the equivalence margins")


# ------------------------------------------------------------------------------- normal


def power_equivalence_normal(
    n_control: float,
    margin: float | Sequence[float],
    sd: float,
    delta: float = 0.0,
    alpha: float = 0.05,
    allocation: float = 1.0,
) -> float:
    """Power of TOST for a difference in means with known SD (normal approximation).

    Parameters
    ----------
    n_control : float
        Control-arm size; the treatment arm has ``allocation * n_control``.
    margin : float or (float, float)
        Symmetric margin :math:`\\Delta` (margins :math:`\\pm\\Delta`) or (lower, upper).
    sd : float
        Common standard deviation.
    delta : float
        True difference in means, treatment minus control.
    alpha : float
        Level of each one-sided test (0.05 gives a 90% confidence interval).
    allocation : float
        Allocation ratio :math:`r = n_T / n_C`.

    References
    ----------
    Schuirmann (1987); Chow, Shao & Wang (2008), section 3.2.4.
    """
    _check(alpha, allocation)
    lo, hi = _margins(margin)
    if not sd > 0:
        raise ValueError("sd must be positive")
    if not n_control > 0:
        raise ValueError("n_control must be positive")
    s = sd * math.sqrt(1 + 1 / allocation)
    return _power(n_control, delta, lo, hi, s, s, s, alpha)


def sample_size_equivalence_normal(
    margin: float | Sequence[float],
    sd: float,
    delta: float = 0.0,
    alpha: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
) -> FixedSampleSize:
    """Per-arm sample size for TOST on a difference in means (known SD).

    See :func:`power_equivalence_normal` for the parameters. With symmetric margins and
    ``delta = 0`` this is the closed form in the module docstring.

    References
    ----------
    Schuirmann (1987); Chow, Shao & Wang (2008), section 3.2.4.
    """
    _check(alpha, allocation, power)
    lo, hi = _margins(margin)
    _inside(delta, lo, hi)
    if not sd > 0:
        raise ValueError("sd must be positive")
    s = sd * math.sqrt(1 + 1 / allocation)
    gap = min(hi - delta, delta - lo)
    start = (s * (stats.norm.isf(alpha) + stats.norm.isf((1 - power) / 2)) / gap) ** 2
    n_c = _solve_n(lambda n: _power(n, delta, lo, hi, s, s, s, alpha), power, start)
    return FixedSampleSize(n_control=n_c, n_treatment=allocation * n_c)


# ------------------------------------------------------------------------------- binary


def _binary_sds(pc: float, pt: float, lo: float, hi: float, r: float):
    def sd_at(p_c: float, p_t: float) -> float:
        return math.sqrt(p_c * (1 - p_c) + p_t * (1 - p_t) / r)

    s_lo = sd_at(*farrington_manning_rates(pc, pt, r, lo))
    s_hi = sd_at(*farrington_manning_rates(pc, pt, r, hi))
    return s_lo, s_hi, sd_at(pc, pt)


def _check_rates(pc: float, pt: float) -> None:
    for p, name in ((pc, "p_control"), (pt, "p_treatment")):
        if not 0.0 < p < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1")


def power_equivalence_binary(
    n_control: float,
    p_control: float,
    p_treatment: float,
    margin: float | Sequence[float],
    alpha: float = 0.05,
    allocation: float = 1.0,
) -> float:
    """Power of TOST for a difference in rates :math:`p_T - p_C`.

    Each one-sided test uses the Farrington-Manning variance at its margin (evaluated at the
    design rates); the estimate's spread uses the design rates.

    References
    ----------
    Schuirmann (1987); Farrington & Manning (1990); Chow, Shao & Wang (2008), section 4.2.4.
    """
    _check(alpha, allocation)
    _check_rates(p_control, p_treatment)
    lo, hi = _margins(margin)
    if not n_control > 0:
        raise ValueError("n_control must be positive")
    sds = _binary_sds(p_control, p_treatment, lo, hi, allocation)
    return _power(n_control, p_treatment - p_control, lo, hi, *sds, alpha)


def sample_size_equivalence_binary(
    p_control: float,
    p_treatment: float,
    margin: float | Sequence[float],
    alpha: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
) -> FixedSampleSize:
    """Per-arm sample size for TOST on a difference in rates.

    See :func:`power_equivalence_binary`.

    References
    ----------
    Schuirmann (1987); Farrington & Manning (1990); Chow, Shao & Wang (2008), section 4.2.4.
    """
    _check(alpha, allocation, power)
    _check_rates(p_control, p_treatment)
    lo, hi = _margins(margin)
    theta = p_treatment - p_control
    _inside(theta, lo, hi)
    sds = _binary_sds(p_control, p_treatment, lo, hi, allocation)
    gap = min(hi - theta, theta - lo)
    start = (max(sds) * (stats.norm.isf(alpha) + stats.norm.isf((1 - power) / 2)) / gap) ** 2
    n_c = _solve_n(lambda n: _power(n, theta, lo, hi, *sds, alpha), power, start)
    return FixedSampleSize(n_control=n_c, n_treatment=allocation * n_c)


# ------------------------------------------------------------------------------ analysis


@dataclass(frozen=True)
class EquivalenceTest:
    """Result of TOST on an observed difference."""

    estimate: float
    ci_lower: float
    ci_upper: float
    level: float
    z_lower: float
    z_upper: float
    p_value: float
    equivalent: bool


def tost(
    estimate: float,
    se: float,
    margin: float | Sequence[float],
    alpha: float = 0.05,
    se_lower: float | None = None,
    se_upper: float | None = None,
) -> EquivalenceTest:
    """Two one-sided tests on an estimate with standard error ``se``.

    ``se_lower`` and ``se_upper`` override ``se`` in the test at each margin (e.g. the
    Farrington-Manning restricted SEs for rates). The p-value is the larger of the two
    one-sided p-values; the confidence interval has level ``1 - 2 alpha`` from ``se``.

    References
    ----------
    Schuirmann (1987).
    """
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    if not se > 0:
        raise ValueError("se must be positive")
    lo, hi = _margins(margin)
    s_lo = se if se_lower is None else se_lower
    s_hi = se if se_upper is None else se_upper
    z_lo = (estimate - lo) / s_lo
    z_hi = (estimate - hi) / s_hi
    p = float(max(stats.norm.sf(z_lo), stats.norm.cdf(z_hi)))
    half = stats.norm.isf(alpha) * se
    return EquivalenceTest(
        estimate=float(estimate),
        ci_lower=float(estimate - half),
        ci_upper=float(estimate + half),
        level=1.0 - 2.0 * alpha,
        z_lower=float(z_lo),
        z_upper=float(z_hi),
        p_value=p,
        equivalent=bool(p <= alpha),
    )
