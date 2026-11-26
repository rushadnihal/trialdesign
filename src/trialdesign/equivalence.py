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
from numpy.typing import NDArray
from scipy import optimize, stats

from trialdesign.boundaries import _Recursion, efficacy_bounds
from trialdesign.fixed import FixedSampleSize, farrington_manning_rates
from trialdesign.spending import check_info_rates, parse_spending

__all__ = [
    "EquivalenceSimulation",
    "EquivalenceTest",
    "GroupSequentialEquivalence",
    "group_sequential_equivalence_binary",
    "group_sequential_equivalence_normal",
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


# ---------------------------------------------------------------- group-sequential TOST


@dataclass(frozen=True)
class EquivalenceSimulation:
    """Simulated probability of declaring equivalence, as a check of the exact calculation."""

    theta: float
    prob_equivalence: float
    prob_equivalence_se: float
    prob_stop: NDArray[np.float64]
    expected_n_control: float
    n_sim: int
    seed: int | None


@dataclass(frozen=True)
class GroupSequentialEquivalence:
    r"""Equivalence by two one-sided group-sequential tests (Jennison & Turnbull 2000, ch. 6).

    Each one-sided test spends ``alpha`` with the same spending function, so both use the
    critical values :math:`c_1, \dots, c_K` of a one-sided group-sequential test at level
    ``alpha``. Equivalence is declared at the first look :math:`k` at which **both** reject,

    .. math:: \hat\theta_k - c_k\,\mathrm{SE}_{L,k} > \Delta_L \quad\text{and}\quad
              \hat\theta_k + c_k\,\mathrm{SE}_{U,k} < \Delta_U ,

    which is the repeated confidence interval :math:`\hat\theta_k \pm c_k\,\mathrm{SE}_k`
    lying inside the margins. Otherwise the trial continues; at the last look it ends without
    declaring equivalence. There is no futility stop. Each component is a level-``alpha``
    group-sequential test, so by the intersection-union principle the probability of
    declaring equivalence when :math:`\theta \le \Delta_L` or :math:`\theta \ge \Delta_U` is
    at most ``alpha``. With one look the design is the fixed TOST of
    :func:`sample_size_equivalence_normal` and :func:`sample_size_equivalence_binary`.

    Stopping probabilities come from recursive numerical integration over
    :math:`Z_k = \hat\theta_k \sqrt{n_k} / \sigma_1`, whose continuation region at each look
    is the union of two half-lines.

    Attributes
    ----------
    endpoint : str
        ``"normal"`` or ``"binary"``.
    info_rates : numpy.ndarray
        Information fractions :math:`t_k` (fractions of the maximum sample size).
    critical_values : numpy.ndarray
        :math:`c_k`, shared by both one-sided tests.
    alpha : float
        Level of each one-sided test.
    power : float
        Target probability of declaring equivalence at ``theta``.
    spending : str
        Alpha-spending function of each one-sided test.
    margin : (float, float)
        :math:`(\Delta_L, \Delta_U)`.
    theta : float
        True difference at which ``power`` is reached.
    allocation : float
        :math:`r = n_T / n_C`.
    n_control : float
        Maximum control-arm size (unrounded).
    fixed_n_control : float
        Control-arm size of the fixed TOST with the same alpha and power.
    sd_lower, sd_upper, sd_estimate : float
        Per-control-patient SDs of the test at each margin and of the estimate (all
        :math:`\sigma\sqrt{1 + 1/r}` for a normal endpoint; Farrington-Manning at each margin
        for a binary one).
    """

    endpoint: str
    info_rates: NDArray[np.float64]
    critical_values: NDArray[np.float64]
    alpha: float
    power: float
    spending: str
    margin: tuple[float, float]
    theta: float
    allocation: float
    n_control: float
    fixed_n_control: float
    sd_lower: float
    sd_upper: float
    sd_estimate: float

    @property
    def n_treatment(self) -> float:
        return self.allocation * self.n_control

    @property
    def n_looks(self) -> int:
        return int(self.info_rates.size)

    @property
    def inflation_factor(self) -> float:
        """Maximum sample size relative to the fixed TOST."""
        return self.n_control / self.fixed_n_control

    def look_n_control(self) -> NDArray[np.float64]:
        """Cumulative control-arm size at each look."""
        return self.info_rates * self.n_control

    def stopping_probabilities(
        self, theta: float | None = None, n_control: float | None = None
    ) -> NDArray[np.float64]:
        """Probability of declaring equivalence at each look, at true difference ``theta``.

        The estimate's SD is ``sd_estimate`` whatever ``theta`` (exact for a normal endpoint).
        """
        return _gs_stop_probs(
            self.n_control if n_control is None else n_control,
            self.theta if theta is None else theta,
            self.margin,
            self.sd_lower,
            self.sd_upper,
            self.sd_estimate,
            self.info_rates,
            self.critical_values,
        )

    def power_at(self, theta: float | None = None) -> float:
        """Probability of declaring equivalence at true difference ``theta``."""
        return float(self.stopping_probabilities(theta).sum())

    def expected_n_control(self, theta: float | None = None) -> float:
        """Expected control-arm size at stopping (the last look if equivalence is not shown)."""
        p = self.stopping_probabilities(theta)
        n = self.look_n_control()
        return float(p @ n + (1.0 - p.sum()) * n[-1])

    def simulate(
        self,
        theta: float | None = None,
        n_sim: int = 100_000,
        seed: int | None = None,
        sd: float | None = None,
        n_control: float | None = None,
    ) -> EquivalenceSimulation:
        """Simulate the trial with patient-level arm means (normal endpoint, known SD).

        Each look uses whole patients, ``ceil(t_k n_C)`` and ``ceil(r t_k n_C)``, and the
        z-statistics with the known common SD ``sd`` (default: the design SD). ``n_control``
        overrides the maximum control-arm size.
        """
        if self.endpoint != "normal":
            raise ValueError(
                "simulation of group-sequential equivalence is for the normal endpoint"
            )
        theta = self.theta if theta is None else float(theta)
        sigma = self.sd_estimate / math.sqrt(1 + 1 / self.allocation) if sd is None else sd
        if not n_sim >= 1:
            raise ValueError("n_sim must be at least 1")
        rng = np.random.default_rng(seed)
        looks = self.info_rates * (self.n_control if n_control is None else n_control)
        n_c = np.ceil(looks - 1e-9)
        n_t = np.ceil(self.allocation * looks - 1e-9)
        lo, hi = self.margin
        sum_c = np.zeros(n_sim)
        sum_t = np.zeros(n_sim)
        prev_c = prev_t = 0.0
        open_ = np.ones(n_sim, dtype=bool)
        stops = np.zeros(self.n_looks)
        n_stop = np.full(n_sim, n_c[-1])
        for k in range(self.n_looks):
            dc, dt = n_c[k] - prev_c, n_t[k] - prev_t
            sum_c += rng.normal(0.0, sigma * math.sqrt(dc), n_sim) if dc > 0 else 0.0
            sum_t += rng.normal(theta * dt, sigma * math.sqrt(dt), n_sim) if dt > 0 else 0.0
            prev_c, prev_t = n_c[k], n_t[k]
            est = sum_t / n_t[k] - sum_c / n_c[k]
            se = sigma * math.sqrt(1 / n_c[k] + 1 / n_t[k])
            c = self.critical_values[k]
            hit = open_ & (est - c * se > lo) & (est + c * se < hi)
            stops[k] = hit.mean()
            n_stop[hit] = n_c[k]
            open_ &= ~hit
        p = float(stops.sum())
        return EquivalenceSimulation(
            theta=theta,
            prob_equivalence=p,
            prob_equivalence_se=math.sqrt(p * (1 - p) / n_sim),
            prob_stop=stops,
            expected_n_control=float(n_stop.mean()),
            n_sim=int(n_sim),
            seed=seed,
        )


def _gs_stop_probs(
    n_c: float,
    theta: float,
    margin: tuple[float, float],
    s_lo: float,
    s_hi: float,
    s1: float,
    t: NDArray,
    c: NDArray,
) -> NDArray[np.float64]:
    """Per-look probabilities of declaring equivalence (recursive integration)."""
    lo, hi = margin
    rec = _Recursion(t, drift=theta * math.sqrt(n_c) / s1)
    out = np.zeros(t.size)
    for k in range(t.size):
        root = math.sqrt(t[k] * n_c)
        a = (lo * root + c[k] * s_lo) / s1 if math.isfinite(c[k]) else math.inf
        b = (hi * root - c[k] * s_hi) / s1 if math.isfinite(c[k]) else -math.inf
        if a < b:
            out[k] = max(rec.prob_below(b) - rec.prob_below(a), 0.0)
        if k < t.size - 1:
            rec.advance_regions(
                [(-math.inf, a), (b, math.inf)] if a < b else [(-math.inf, math.inf)]
            )
    return out


def _gs_equivalence(
    endpoint: str,
    lo: float,
    hi: float,
    theta: float,
    sds: tuple[float, float, float],
    fixed_n: float,
    info_rates: int | Sequence[float],
    alpha: float,
    power: float,
    allocation: float,
    spending: str,
) -> GroupSequentialEquivalence:
    t = check_info_rates(info_rates)
    label = parse_spending(spending).label
    c = efficacy_bounds(t, alpha, label)
    s_lo, s_hi, s1 = sds

    def power_at(n: float) -> float:
        return float(_gs_stop_probs(n, theta, (lo, hi), s_lo, s_hi, s1, t, c).sum())

    n_c = _solve_n(power_at, power, 1.5 * fixed_n)
    return GroupSequentialEquivalence(
        endpoint=endpoint,
        info_rates=t,
        critical_values=c,
        alpha=alpha,
        power=power,
        spending=label,
        margin=(lo, hi),
        theta=theta,
        allocation=allocation,
        n_control=n_c,
        fixed_n_control=fixed_n,
        sd_lower=s_lo,
        sd_upper=s_hi,
        sd_estimate=s1,
    )


def group_sequential_equivalence_normal(
    margin: float | Sequence[float],
    sd: float,
    info_rates: int | Sequence[float],
    delta: float = 0.0,
    alpha: float = 0.025,
    power: float = 0.80,
    allocation: float = 1.0,
    spending: str = "obf",
) -> GroupSequentialEquivalence:
    """Group-sequential TOST for a difference in means with known SD.

    Parameters
    ----------
    margin, sd, delta, power, allocation
        As for :func:`sample_size_equivalence_normal`.
    info_rates : int or sequence of float
        Number of equally spaced looks, or information fractions ending at 1.
    alpha : float
        Level of each one-sided test (0.025 by default, as for other group-sequential
        designs; 0.05 gives the usual 90% interval at a single look).
    spending : str
        Alpha-spending function of each one-sided test (see :mod:`trialdesign.spending`).

    References
    ----------
    Jennison & Turnbull (2000), chapter 6; Schuirmann (1987).
    """
    _check(alpha, allocation, power)
    lo, hi = _margins(margin)
    _inside(delta, lo, hi)
    fixed_n = sample_size_equivalence_normal((lo, hi), sd, delta, alpha, power, allocation)
    s = sd * math.sqrt(1 + 1 / allocation)
    return _gs_equivalence(
        "normal",
        lo,
        hi,
        delta,
        (s, s, s),
        fixed_n.n_control,
        info_rates,
        alpha,
        power,
        allocation,
        spending,
    )


def group_sequential_equivalence_binary(
    p_control: float,
    p_treatment: float,
    margin: float | Sequence[float],
    info_rates: int | Sequence[float],
    alpha: float = 0.025,
    power: float = 0.80,
    allocation: float = 1.0,
    spending: str = "obf",
) -> GroupSequentialEquivalence:
    """Group-sequential TOST for a difference in rates (Farrington-Manning variances).

    See :func:`group_sequential_equivalence_normal` and :func:`power_equivalence_binary`.

    References
    ----------
    Jennison & Turnbull (2000), chapter 6; Farrington & Manning (1990).
    """
    _check(alpha, allocation, power)
    _check_rates(p_control, p_treatment)
    lo, hi = _margins(margin)
    theta = p_treatment - p_control
    _inside(theta, lo, hi)
    fixed_n = sample_size_equivalence_binary(
        p_control, p_treatment, (lo, hi), alpha, power, allocation
    )
    sds = _binary_sds(p_control, p_treatment, lo, hi, allocation)
    return _gs_equivalence(
        "binary",
        lo,
        hi,
        theta,
        sds,
        fixed_n.n_control,
        info_rates,
        alpha,
        power,
        allocation,
        spending,
    )
