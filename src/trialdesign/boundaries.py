r"""Group-sequential boundaries by recursive numerical integration.

Canonical joint distribution (Jennison & Turnbull 2000, section 3.1): at looks
:math:`k = 1, \dots, K` with information fractions :math:`0 < t_1 < \dots < t_K = 1`, the
z-statistics are jointly normal with

.. math:: E(Z_k) = \delta \sqrt{t_k}, \qquad \operatorname{Cov}(Z_j, Z_k) = \sqrt{t_j / t_k},
          \quad j \le k,

where the *drift* :math:`\delta` is the mean of the final z-statistic. Under H0,
:math:`\delta = 0`. A fixed design with one-sided level :math:`\alpha` and power
:math:`1 - \beta` has :math:`\delta_{\mathrm{fix}} = z_{1-\alpha} + z_{1-\beta}`; a
group-sequential design needs :math:`\delta^2 = \mathrm{IF}\cdot\delta_{\mathrm{fix}}^2`, so its
maximum sample size (or number of events) is the fixed one times the inflation factor IF.

Efficacy boundaries are upper (one-sided) boundaries for :math:`Z_k`. Futility boundaries are
**non-binding**: efficacy boundaries are computed under H0 as if futility never stops the
trial, so type I error is at most :math:`\alpha` whether or not futility is followed.

References
----------
Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
Clinical Trials*. Chapman & Hall/CRC. Chapter 19 (numerical integration), chapter 7
(error spending).
Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
*Biometrika* 70, 659-663.
Anderson, K. M. et al. ``gsDesign`` R package, ``test.type = 4`` (non-binding beta spending).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats

from trialdesign.spending import check_info_rates, cumulative_spending

__all__ = [
    "GroupSequentialDesign",
    "crossing_probabilities",
    "efficacy_bounds",
    "efficacy_bounds_mvn",
    "expected_info_fraction",
    "group_sequential_design",
]

_GRID_R = 18  # grid density of Jennison & Turnbull (2000), section 19.2.1; gsDesign uses 18
_Z_MAX = 40.0  # boundaries are searched in [-_Z_MAX, _Z_MAX]
_XTOL = 1e-12


# ------------------------------------------------------------------ numerical integration


def _grid(mean: float, lower: float, upper: float, r: int) -> tuple[NDArray, NDArray]:
    """Grid points and Simpson weights on ``(lower, upper)`` for a N(``mean``, 1) density.

    Jennison & Turnbull (2000), section 19.2.1: ``6r - 1`` base points, dense within 3 SD of
    the mean and spaced logarithmically in the tails, trimmed to the continuation region and
    with the region's finite end points added. Midpoints are inserted for Simpson's rule.
    """
    i = np.arange(1, 6 * r, dtype=float)
    x = np.where(
        i < r,
        mean - 3.0 - 4.0 * np.log(r / i),
        np.where(
            i <= 5 * r,
            mean - 3.0 + 3.0 * (i - r) / (2.0 * r),
            mean + 3.0 + 4.0 * np.log(r / (6.0 * r - i)),
        ),
    )
    x = x[(x > lower) & (x < upper)]
    pieces = []
    if math.isfinite(lower):
        pieces.append([lower])
    pieces.append(x)
    if math.isfinite(upper):
        pieces.append([upper])
    x = np.concatenate(pieces)
    if x.size < 2:  # region narrower than the grid spacing
        x = np.array([lower, upper]) if x.size == 0 else np.array([x[0], x[0]])
    m = x.size
    z = np.empty(2 * m - 1)
    z[0::2] = x
    z[1::2] = 0.5 * (x[:-1] + x[1:])
    d = np.diff(x)
    w = np.zeros(2 * m - 1)
    w[1::2] = 4.0 * d / 6.0
    w[0:-1:2] += d / 6.0
    w[2::2] += d / 6.0
    return z, w


class _Recursion:
    """Sub-density of :math:`Z_k` on the continuation region, one look at a time."""

    def __init__(self, t: NDArray, drift: float, r: int = _GRID_R) -> None:
        self.t = t
        self.drift = drift
        self.r = r
        self.k = 0  # number of looks already integrated
        self.z: NDArray = np.empty(0)
        self.wh: NDArray = np.empty(0)  # Simpson weight times sub-density

    def _transition_args(self, c: float) -> NDArray | float:
        """Standardised distance from each grid point at look k to ``c`` at look k + 1."""
        k = self.k
        if k == 0:
            return c - self.drift * math.sqrt(self.t[0])
        t0, t1 = self.t[k - 1], self.t[k]
        dt = t1 - t0
        return (c * math.sqrt(t1) - self.z * math.sqrt(t0) - self.drift * dt) / math.sqrt(dt)

    def prob_above(self, c: float) -> float:
        """P(continue to look k + 1 and Z_{k+1} >= c)."""
        if c == math.inf:
            return 0.0
        arg = self._transition_args(c)
        if self.k == 0:
            return float(stats.norm.sf(arg))
        return float(self.wh @ stats.norm.sf(arg))

    def prob_below(self, b: float) -> float:
        """P(continue to look k + 1 and Z_{k+1} < b)."""
        if b == -math.inf:
            return 0.0
        arg = self._transition_args(b)
        if self.k == 0:
            return float(stats.norm.cdf(arg))
        return float(self.wh @ stats.norm.cdf(arg))

    def advance(self, lower: float, upper: float) -> None:
        """Move to the next look, keeping only paths that continue in ``(lower, upper)``."""
        k = self.k
        mean = self.drift * math.sqrt(self.t[k])
        z_new, w_new = _grid(mean, lower, upper, self.r)
        if k == 0:
            h = stats.norm.pdf(z_new - mean)
        else:
            t0, t1 = self.t[k - 1], self.t[k]
            dt = t1 - t0
            sd = math.sqrt(dt)
            arg = (
                z_new[:, None] * math.sqrt(t1) - self.z[None, :] * math.sqrt(t0) - self.drift * dt
            ) / sd
            h = (stats.norm.pdf(arg) * math.sqrt(t1) / sd) @ self.wh
        self.z = z_new
        self.wh = w_new * h
        self.k += 1


def _solve(f, lo: float, hi: float) -> float:
    """Root of a monotone ``f`` on ``[lo, hi]``, clamped to the end points."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        return lo if abs(f_lo) < abs(f_hi) else hi
    return float(optimize.brentq(f, lo, hi, xtol=_XTOL))


# --------------------------------------------------------------------- efficacy (H0)


def _check_alpha(alpha_one_sided: float) -> None:
    if not 0.0 < alpha_one_sided < 0.5:
        raise ValueError("alpha_one_sided must be between 0 and 0.5")


def efficacy_bounds(
    info_rates: int | Sequence[float],
    alpha_one_sided: float = 0.025,
    spending: str = "obf",
) -> NDArray[np.float64]:
    r"""One-sided efficacy z-boundaries from an alpha-spending function.

    Solves, for :math:`k = 1, \dots, K` in turn,

    .. math:: P_0(Z_1 < c_1, \dots, Z_{k-1} < c_{k-1}, Z_k \ge c_k)
              = \alpha(t_k) - \alpha(t_{k-1})

    by recursive numerical integration. Futility is ignored (non-binding).

    Parameters
    ----------
    info_rates : int or sequence of float
        Information fractions at each look, increasing and ending at 1.0, or the number of
        equally spaced looks.
    alpha_one_sided : float
        One-sided significance level.
    spending : {"obf", "pocock"}
        Lan-DeMets alpha-spending family.

    Returns
    -------
    numpy.ndarray
        Efficacy critical values :math:`c_1, \dots, c_K` on the z-scale.

    References
    ----------
    Lan & DeMets (1983); Jennison & Turnbull (2000), sections 7.2 and 19.2.
    """
    _check_alpha(alpha_one_sided)
    t = check_info_rates(info_rates)
    spend = np.diff(cumulative_spending(spending, t, alpha_one_sided), prepend=0.0)
    rec = _Recursion(t, drift=0.0)
    bounds = np.empty(t.size)
    for k in range(t.size):
        target = spend[k]
        if target <= 0.0:
            c = math.inf
        else:
            c = _solve(lambda c, tgt=target: rec.prob_above(c) - tgt, -_Z_MAX, _Z_MAX)
        bounds[k] = c
        if k < t.size - 1:
            rec.advance(-math.inf, c)
    return bounds


def efficacy_bounds_mvn(
    info_rates: int | Sequence[float],
    alpha_one_sided: float = 0.025,
    spending: str = "obf",
    seed: int = 20261121,
) -> NDArray[np.float64]:
    """Cross-check of :func:`efficacy_bounds` using ``scipy.stats.multivariate_normal.cdf``.

    Each boundary solves :math:`P_0(Z_{1:k-1} < c_{1:k-1}, Z_k \\ge c_k)` = alpha spent at
    look k, with the probability from scipy's quasi-Monte Carlo multivariate normal CDF (Genz
    2004). Slow and only accurate to about 1e-6 in probability, so meant for K up to 5 and
    tests only.

    Parameters
    ----------
    info_rates, alpha_one_sided, spending
        As for :func:`efficacy_bounds`.
    seed : int
        Seed of the quasi-Monte Carlo integration, so results are reproducible.

    References
    ----------
    Genz, A. and Bretz, F. (2009). *Computation of Multivariate Normal and t Probabilities*.
    Springer.
    """
    _check_alpha(alpha_one_sided)
    t = check_info_rates(info_rates)
    spend = np.diff(cumulative_spending(spending, t, alpha_one_sided), prepend=0.0)
    cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    bounds = np.empty(t.size)
    bounds[0] = stats.norm.isf(spend[0])
    for k in range(1, t.size):
        mean = np.zeros(k + 1)
        sub_cov = cov[: k + 1, : k + 1]
        upper = np.append(bounds[:k], np.inf)

        def excess(c: float, k: int = k, mean=mean, sub_cov=sub_cov, upper=upper) -> float:
            # A fresh, seeded frozen distribution per call makes the integral deterministic.
            mvn = stats.multivariate_normal(
                mean, sub_cov, seed=seed, maxpts=100_000 * (k + 1), abseps=1e-9, releps=1e-9
            )
            lower = np.append(np.full(k, -np.inf), c)
            return float(mvn.cdf(upper, lower_limit=lower)) - spend[k]

        bounds[k] = optimize.brentq(excess, 0.0, 10.0, xtol=1e-7)
    return bounds


# ----------------------------------------------------------- probabilities under a drift


def crossing_probabilities(
    info_rates: int | Sequence[float],
    efficacy: Sequence[float],
    futility: Sequence[float] | None = None,
    drift: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Probabilities of stopping for efficacy and for futility at each look.

    The trial stops at look k for efficacy if :math:`Z_k \ge c_k` and for futility if
    :math:`Z_k < b_k`, having continued at every earlier look. With ``futility=None`` no
    futility stops are made, which is how the type I error of a non-binding design is
    computed.

    Parameters
    ----------
    info_rates : int or sequence of float
        Information fractions, as for :func:`efficacy_bounds`.
    efficacy : sequence of float
        Efficacy boundaries :math:`c_k`.
    futility : sequence of float, optional
        Futility boundaries :math:`b_k \le c_k`.
    drift : float
        Mean of the final z-statistic, :math:`\delta`; 0 under H0.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Per-look probabilities of stopping for efficacy and for futility. Their sums are the
        overall rejection probability and the probability of stopping for futility.

    References
    ----------
    Jennison & Turnbull (2000), section 19.2.
    """
    t = check_info_rates(info_rates)
    c = np.asarray(efficacy, dtype=float)
    b = np.full(t.size, -np.inf) if futility is None else np.asarray(futility, dtype=float)
    if c.shape != t.shape or b.shape != t.shape:
        raise ValueError("efficacy and futility must have one boundary per look")
    if np.any(b > c):
        raise ValueError("futility boundaries must not exceed efficacy boundaries")
    rec = _Recursion(t, drift=drift)
    up = np.empty(t.size)
    low = np.empty(t.size)
    for k in range(t.size):
        up[k] = rec.prob_above(c[k])
        low[k] = rec.prob_below(b[k])
        if k < t.size - 1:
            rec.advance(b[k], c[k])
    return up, low


def expected_info_fraction(
    info_rates: int | Sequence[float],
    efficacy: Sequence[float],
    futility: Sequence[float] | None = None,
    drift: float = 0.0,
) -> float:
    r"""Expected information fraction at which the trial stops.

    .. math:: E(t_{\text{stop}}) = \sum_{k<K} t_k\, P(\text{stop at } k)
              + \Bigl(1 - \sum_{k<K} P(\text{stop at } k)\Bigr),

    with the stopping probabilities from :func:`crossing_probabilities`. Times the maximum
    sample size (or events) this is the expected sample size; times the inflation factor it
    is the expected sample size relative to the fixed design (``gsDesign``'s ``en`` with
    ``n.fix = 1``).

    Parameters
    ----------
    info_rates, efficacy, futility, drift
        As for :func:`crossing_probabilities`.

    References
    ----------
    Jennison & Turnbull (2000), sections 7.2 and 19.2.
    """
    t = check_info_rates(info_rates)
    up, low = crossing_probabilities(t, efficacy, futility, drift)
    p_stop = (up + low)[:-1]
    return float(t[:-1] @ p_stop + (1.0 - p_stop.sum()))


def _futility_bounds(
    t: NDArray, efficacy: NDArray, beta_spend: NDArray, drift: float
) -> NDArray[np.float64]:
    """Futility boundaries that spend ``beta_spend`` (increments) under ``drift``.

    At the last look the boundary takes whatever value spends the remaining beta, so it
    equals the efficacy boundary only at the drift that gives the target power.
    """
    rec = _Recursion(t, drift=drift)
    b = np.empty(t.size)
    for k in range(t.size):
        target = beta_spend[k]
        bk = _solve(lambda x, tgt=target: rec.prob_below(x) - tgt, -_Z_MAX, _Z_MAX)
        if k < t.size - 1:
            bk = min(bk, efficacy[k])  # never ask to stop for futility above efficacy
            rec.advance(bk, efficacy[k])
        b[k] = bk
    return b


# -------------------------------------------------------------------------- full design


@dataclass(frozen=True)
class GroupSequentialDesign:
    r"""A one-sided group-sequential design with optional non-binding futility.

    Attributes
    ----------
    info_rates : numpy.ndarray
        Information fractions :math:`t_k`.
    efficacy : numpy.ndarray
        Efficacy z-boundaries :math:`c_k`.
    futility : numpy.ndarray or None
        Non-binding futility z-boundaries :math:`b_k`, or None without futility.
    alpha_one_sided, power : float
        Design error rates.
    efficacy_spending, futility_spending : str or None
        Spending families used.
    drift : float
        Mean of the final z-statistic under H1 that gives the target power.
    inflation_factor : float
        :math:`(\delta / (z_{1-\alpha} + z_{1-\beta}))^2`: maximum information relative to
        the fixed design with the same one-sided alpha and power.
    """

    info_rates: NDArray[np.float64]
    efficacy: NDArray[np.float64]
    futility: NDArray[np.float64] | None
    alpha_one_sided: float
    power: float
    efficacy_spending: str
    futility_spending: str | None
    drift: float
    inflation_factor: float

    @property
    def n_looks(self) -> int:
        return int(self.info_rates.size)

    def max_sample_size(self, n_fixed: float) -> float:
        """Maximum sample size (or events): fixed-design ``n_fixed`` times the inflation factor.

        ``n_fixed`` must come from a fixed design with the same one-sided alpha and power,
        e.g. :func:`trialdesign.fixed.events_survival` with
        ``alpha_two_sided = 2 * alpha_one_sided``.
        """
        if not n_fixed > 0:
            raise ValueError("n_fixed must be positive")
        return float(n_fixed * self.inflation_factor)

    def look_sample_sizes(self, n_fixed: float) -> NDArray[np.float64]:
        """Cumulative sample size (or events) at each look."""
        return self.info_rates * self.max_sample_size(n_fixed)

    def crossing_probabilities(
        self, drift: float | None = None, use_futility: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Per-look stopping probabilities at ``drift`` (default: the design alternative)."""
        return crossing_probabilities(
            self.info_rates,
            self.efficacy,
            self.futility if use_futility else None,
            self.drift if drift is None else drift,
        )

    def expected_info_fraction(
        self, drift: float | None = None, use_futility: bool = True
    ) -> float:
        """Expected stopping information fraction at ``drift`` (default: the alternative)."""
        return expected_info_fraction(
            self.info_rates,
            self.efficacy,
            self.futility if use_futility else None,
            self.drift if drift is None else drift,
        )


def group_sequential_design(
    info_rates: int | Sequence[float],
    alpha_one_sided: float = 0.025,
    power: float = 0.80,
    efficacy_spending: str = "obf",
    futility_spending: str | None = None,
) -> GroupSequentialDesign:
    r"""Group-sequential design: boundaries, drift and inflation factor.

    Efficacy boundaries come from :func:`efficacy_bounds`. Without futility the drift
    :math:`\delta` solves :math:`P_\delta(\text{reject}) = 1 - \beta`. With futility, the
    boundaries :math:`b_k` spend beta under :math:`\delta` (Lan-DeMets beta spending), using
    the futility region at earlier looks, and :math:`\delta` is chosen so that the final
    futility boundary meets the final efficacy boundary, :math:`b_K = c_K`. Futility is
    non-binding: it does not change the efficacy boundaries, so it costs power (a larger
    inflation factor) but never inflates alpha. This is ``gsDesign`` ``test.type = 4``.

    Parameters
    ----------
    info_rates : int or sequence of float
        Information fractions at each look, or the number of equally spaced looks.
    alpha_one_sided : float
        One-sided significance level.
    power : float
        Target power at the design alternative.
    efficacy_spending : {"obf", "pocock"}
        Alpha-spending family.
    futility_spending : {"obf", "pocock"}, optional
        Beta-spending family for non-binding futility; None for no futility boundaries.

    Returns
    -------
    GroupSequentialDesign

    References
    ----------
    Jennison & Turnbull (2000), sections 7.2 and 19.2; Lan & DeMets (1983).
    """
    _check_alpha(alpha_one_sided)
    if not alpha_one_sided < power < 1.0:
        raise ValueError("power must be between alpha_one_sided and 1")
    t = check_info_rates(info_rates)
    c = efficacy_bounds(t, alpha_one_sided, efficacy_spending)
    beta = 1.0 - power
    drift_fixed = stats.norm.isf(alpha_one_sided) + stats.norm.isf(beta)

    if futility_spending is None:
        b = None

        def gap(d: float) -> float:
            return float(crossing_probabilities(t, c, None, d)[0].sum()) - power

    else:
        beta_spend = np.diff(cumulative_spending(futility_spending, t, beta), prepend=0.0)

        def gap(d: float) -> float:
            return float(_futility_bounds(t, c, beta_spend, d)[-1] - c[-1])

    # The fixed-design drift is a lower bound; widen upwards until the root is bracketed.
    lo, hi = 0.5 * drift_fixed, 1.5 * drift_fixed
    while gap(hi) < 0:
        lo, hi = hi, 2.0 * hi
    drift = float(optimize.brentq(gap, lo, hi, xtol=_XTOL))
    if futility_spending is not None:
        b = _futility_bounds(t, c, beta_spend, drift)
        b[-1] = c[-1]
    return GroupSequentialDesign(
        info_rates=t,
        efficacy=c,
        futility=b,
        alpha_one_sided=alpha_one_sided,
        power=power,
        efficacy_spending=efficacy_spending,
        futility_spending=futility_spending,
        drift=drift,
        inflation_factor=float((drift / drift_fixed) ** 2),
    )
