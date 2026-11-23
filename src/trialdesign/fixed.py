"""Fixed-design (single-analysis) sample size and power for two-arm trials.

All functions use a **two-sided** significance level, named ``alpha_two_sided``, and an
allocation ratio ``allocation = n_T / n_C``. Results are returned unrounded so they can be
compared with reference software; round up with :func:`math.ceil` for a real design.

Superiority is the default. For non-inferiority (or super-superiority), give the effect under
H0: ``delta0`` (normal), ``delta0`` (binary, a difference in rates T - C) or
``hazard_ratio0`` (survival). The test is then one-sided in the direction of the alternative,
at level ``alpha_two_sided / 2``: with ``delta0 = -0.1`` and ``delta = 0`` it tests
H0: delta <= -0.1 against delta > -0.1. This is the parameterisation of ``gsDesign``
(``delta0``, ``hr0``) and ``rpact`` (``thetaH0``).

References
----------
Chow, S.-C., Shao, J. and Wang, H. (2008). *Sample Size Calculations in Clinical Research*,
2nd ed. Chapman & Hall/CRC. Sections 3.2 (normal) and 4.2 (binary).
Fleiss, J. L., Levin, B. and Paik, M. C. (2003). *Statistical Methods for Rates and
Proportions*, 3rd ed. Wiley. Section 4.2 (pooled/unpooled variance formula).
Farrington, C. P. and Manning, G. (1990). Test statistics and sample size formulae for
comparative binomial trials with null hypothesis of non-zero risk difference or non-unity
relative risk. *Statistics in Medicine* 9, 1447-1454.
Schoenfeld, D. A. (1983). Sample-size formula for the proportional-hazards regression model.
*Biometrics* 39, 499-503.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy import optimize, stats

__all__ = [
    "FixedSampleSize",
    "SurvivalEvents",
    "events_survival",
    "farrington_manning_rates",
    "power_binary",
    "power_normal",
    "power_survival",
    "sample_size_binary",
    "sample_size_normal",
]

NormalMethod = Literal["z", "t"]


@dataclass(frozen=True)
class FixedSampleSize:
    """Per-arm sample size of a fixed two-arm design (unrounded)."""

    n_control: float
    n_treatment: float

    @property
    def n_total(self) -> float:
        return self.n_control + self.n_treatment

    @property
    def n_control_ceil(self) -> int:
        return math.ceil(self.n_control - 1e-9)

    @property
    def n_treatment_ceil(self) -> int:
        return math.ceil(self.n_treatment - 1e-9)

    @property
    def n_total_ceil(self) -> int:
        return self.n_control_ceil + self.n_treatment_ceil


@dataclass(frozen=True)
class SurvivalEvents:
    """Required number of events (and optionally patients) for a log-rank design."""

    events: float
    n_total: float | None = None

    @property
    def events_ceil(self) -> int:
        return math.ceil(self.events - 1e-9)

    @property
    def n_total_ceil(self) -> int | None:
        return None if self.n_total is None else math.ceil(self.n_total - 1e-9)


# --------------------------------------------------------------------------- validation


def _check_alpha_power(alpha_two_sided: float, power: float | None = None) -> None:
    if not 0.0 < alpha_two_sided < 1.0:
        raise ValueError("alpha_two_sided must be between 0 and 1")
    if power is not None and not alpha_two_sided / 2 < power < 1.0:
        raise ValueError("power must be between alpha_two_sided / 2 and 1")


def _check_allocation(allocation: float) -> None:
    if not allocation > 0.0:
        raise ValueError("allocation (n_T / n_C) must be positive")


def _check_n(n_control: float, minimum: float = 0.0) -> None:
    if not n_control > minimum:
        raise ValueError(f"n_control must be greater than {minimum:g}")


def _check_prob(p: float, name: str) -> None:
    if not 0.0 < p < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")


# ------------------------------------------------------------------------------- normal


def _t_power(n_control: float, std_effect: float, alpha_two_sided: float, r: float) -> float:
    """Power of the two-sample t-test (one rejection tail, as R's ``power.t.test``)."""
    n_t = r * n_control
    df = n_control + n_t - 2
    ncp = std_effect / math.sqrt(1 / n_control + 1 / n_t)
    crit = stats.t.ppf(1 - alpha_two_sided / 2, df)
    return float(stats.nct.sf(crit, df, ncp))


def _effect_gap(effect: float, effect0: float, message: str) -> float:
    """``|effect - effect0|``, or a plain error when the alternative equals H0."""
    if math.isclose(effect, effect0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(message)
    return abs(effect - effect0)


def sample_size_normal(
    delta: float,
    sd: float,
    alpha_two_sided: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
    method: NormalMethod = "z",
    delta0: float = 0.0,
) -> FixedSampleSize:
    r"""Per-arm sample size for a difference in means with common SD.

    With ``method="z"`` (normal approximation)

    .. math:: n_C = (1 + 1/r)\,\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2
              / (\Delta - \Delta_0)^2, \quad n_T = r\,n_C.

    With ``method="t"`` the noncentral-t power of the two-sample t-test is solved for
    :math:`n_C`, matching R's ``power.t.test`` (``strict = FALSE``) when ``allocation = 1``.

    Parameters
    ----------
    delta : float
        Difference in means under the alternative, treatment minus control.
    sd : float
        Common standard deviation.
    alpha_two_sided : float
        Two-sided significance level.
    power : float
        Target power.
    allocation : float
        Allocation ratio :math:`r = n_T / n_C`.
    method : {"z", "t"}
        Normal approximation or exact noncentral-t calculation.
    delta0 : float
        Difference in means under H0: 0 for superiority, ``-margin`` for non-inferiority
        when larger is better. The sign of ``delta`` is ignored only when ``delta0 = 0``.

    References
    ----------
    Chow, Shao & Wang (2008), sections 3.2 (superiority) and 3.2.3 (non-inferiority).
    """
    _check_alpha_power(alpha_two_sided, power)
    _check_allocation(allocation)
    if not sd > 0:
        raise ValueError("sd must be positive")
    r = allocation
    std_effect = (
        _effect_gap(
            delta,
            delta0,
            "delta must be non-zero" if delta0 == 0 else "delta must differ from delta0",
        )
        / sd
    )
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    z_b = stats.norm.ppf(power)
    n_c = (1 + 1 / r) * (z_a + z_b) ** 2 / std_effect**2
    if method == "t":
        # The z answer is a lower bound; bracket upwards and solve exactly.
        lower = max(n_c, 2.0 / (1 + r) + 1e-6)
        upper = 2 * lower + 10
        while _t_power(upper, std_effect, alpha_two_sided, r) < power:
            upper *= 2
        n_c = optimize.brentq(
            lambda n: _t_power(n, std_effect, alpha_two_sided, r) - power,
            lower,
            upper,
            xtol=1e-10,
        )
    elif method != "z":
        raise ValueError('method must be "z" or "t"')
    return FixedSampleSize(n_control=float(n_c), n_treatment=float(r * n_c))


def power_normal(
    n_control: float,
    delta: float,
    sd: float,
    alpha_two_sided: float = 0.05,
    allocation: float = 1.0,
    method: NormalMethod = "z",
    delta0: float = 0.0,
) -> float:
    """Power for a difference in means given the control-arm size ``n_control``.

    The treatment arm has ``allocation * n_control`` patients. Only the rejection tail in
    the direction of ``delta - delta0`` is counted, as in R's ``power.t.test`` with
    ``strict = FALSE``.

    References
    ----------
    Chow, Shao & Wang (2008), section 3.2.
    """
    _check_alpha_power(alpha_two_sided)
    _check_allocation(allocation)
    if not sd > 0:
        raise ValueError("sd must be positive")
    r = allocation
    std_effect = abs(delta - delta0) / sd
    if method == "t":
        _check_n(n_control, 2.0 / (1 + r))
        return _t_power(n_control, std_effect, alpha_two_sided, r)
    if method != "z":
        raise ValueError('method must be "z" or "t"')
    _check_n(n_control)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    se = math.sqrt(1 / n_control + 1 / (r * n_control))
    return float(stats.norm.cdf(std_effect / se - z_a))


# ------------------------------------------------------------------------------- binary


def farrington_manning_rates(
    p_control: float, p_treatment: float, allocation: float = 1.0, delta0: float = 0.0
) -> tuple[float, float]:
    r"""Rates that maximise the likelihood subject to :math:`p_T - p_C = \delta_0`.

    Closed-form root of the score equation's cubic (Farrington and Manning 1990, eq. 2.4;
    Miettinen and Nurminen 1985). Evaluated at the design rates, these give the variance under
    H0 that ``gsDesign::nBinomial`` and ``rpact::getSampleSizeRates`` use. With
    ``delta0 = 0`` they reduce to the pooled rate :math:`(p_C + r p_T) / (1 + r)`.

    Parameters
    ----------
    p_control, p_treatment : float
        Response rates (or observed proportions) in each arm.
    allocation : float
        Allocation ratio :math:`r = n_T / n_C`.
    delta0 : float
        Difference :math:`p_T - p_C` under H0.

    Returns
    -------
    (p_control_tilde, p_treatment_tilde)

    References
    ----------
    Farrington & Manning (1990), section 2.
    """
    _check_allocation(allocation)
    if not -1.0 < delta0 < 1.0:
        raise ValueError("delta0 must be between -1 and 1")
    if delta0 == 0.0:
        p_bar = (p_control + allocation * p_treatment) / (1 + allocation)
        return p_bar, p_bar
    # Group 1 = treatment, group 2 = control; theta = n2 / n1; H0: p1 - p2 = delta0.
    p1, p2, d, theta = p_treatment, p_control, delta0, 1.0 / allocation
    a = 1.0 + theta
    b = -(1.0 + theta + p1 + theta * p2 + d * (theta + 2.0))
    c = d * d + d * (2.0 * p1 + theta + 1.0) + p1 + theta * p2
    e = -p1 * d * (1.0 + d)
    v = b**3 / (27 * a**3) - b * c / (6 * a**2) + e / (2 * a)
    u = math.copysign(math.sqrt(max(b * b / (9 * a * a) - c / (3 * a), 0.0)), v)
    ratio = 0.0 if u == 0.0 else max(-1.0, min(1.0, v / u**3))
    w = (math.pi + math.acos(ratio)) / 3.0
    p1_tilde = 2.0 * u * math.cos(w) - b / (3.0 * a)
    p2_tilde = p1_tilde - d
    return float(p2_tilde), float(p1_tilde)


def _binary_terms(
    p_control: float, p_treatment: float, r: float, delta0: float = 0.0
) -> tuple[float, float]:
    """Per-``n_C`` standard deviations of the difference under H0 (restricted MLE) and H1."""
    pc0, pt0 = farrington_manning_rates(p_control, p_treatment, r, delta0)
    sd0 = math.sqrt(pc0 * (1 - pc0) + pt0 * (1 - pt0) / r)
    sd1 = math.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment) / r)
    return sd0, sd1


def sample_size_binary(
    p_control: float,
    p_treatment: float,
    alpha_two_sided: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
    delta0: float = 0.0,
) -> FixedSampleSize:
    r"""Per-arm sample size for comparing two proportions (no continuity correction).

    Uses the variance at the restricted MLE under H0 (:func:`farrington_manning_rates`) and
    the unrestricted variance under H1:

    .. math:: n_C = \frac{\left(z_{1-\alpha/2}\sqrt{\tilde p_C(1-\tilde p_C)
              + \tilde p_T(1-\tilde p_T)/r}
              + z_{1-\beta}\sqrt{p_C(1-p_C) + p_T(1-p_T)/r}\right)^2}{(p_T - p_C - \delta_0)^2}.

    For superiority (``delta0 = 0``) the restricted rates are the pooled rate
    :math:`\bar p = (p_C + r p_T)/(1 + r)`, and for ``allocation = 1`` this is the formula
    solved by R's ``power.prop.test`` (``strict = FALSE``). For non-inferiority it matches
    ``gsDesign::nBinomial(delta0 = )``.

    Parameters
    ----------
    delta0 : float
        Difference :math:`p_T - p_C` under H0: 0 for superiority, ``-margin`` for
        non-inferiority when a higher rate is better, ``+margin`` when lower is better.

    References
    ----------
    Fleiss, Levin & Paik (2003), section 4.2; Farrington & Manning (1990).
    """
    _check_alpha_power(alpha_two_sided, power)
    _check_allocation(allocation)
    _check_prob(p_control, "p_control")
    _check_prob(p_treatment, "p_treatment")
    r = allocation
    gap = _effect_gap(
        p_treatment - p_control,
        delta0,
        "p_control and p_treatment must differ"
        if delta0 == 0
        else "p_treatment - p_control must differ from delta0",
    )
    sd0, sd1 = _binary_terms(p_control, p_treatment, r, delta0)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    z_b = stats.norm.ppf(power)
    n_c = (z_a * sd0 + z_b * sd1) ** 2 / gap**2
    return FixedSampleSize(n_control=float(n_c), n_treatment=float(r * n_c))


def power_binary(
    n_control: float,
    p_control: float,
    p_treatment: float,
    alpha_two_sided: float = 0.05,
    allocation: float = 1.0,
    delta0: float = 0.0,
) -> float:
    """Power for comparing two proportions given the control-arm size ``n_control``.

    Inverse of :func:`sample_size_binary`.

    References
    ----------
    Fleiss, Levin & Paik (2003), section 4.2.
    """
    _check_alpha_power(alpha_two_sided)
    _check_allocation(allocation)
    _check_prob(p_control, "p_control")
    _check_prob(p_treatment, "p_treatment")
    _check_n(n_control)
    sd0, sd1 = _binary_terms(p_control, p_treatment, allocation, delta0)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    diff = abs(p_treatment - p_control - delta0)
    return float(stats.norm.cdf((math.sqrt(n_control) * diff - z_a * sd0) / sd1))


# ----------------------------------------------------------------------------- survival


def events_survival(
    hazard_ratio: float,
    alpha_two_sided: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
    prob_event: float | None = None,
    hazard_ratio0: float = 1.0,
) -> SurvivalEvents:
    r"""Required number of events for the log-rank test (Schoenfeld 1983).

    .. math:: D = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}
              {\pi(1-\pi)\,(\log \mathrm{HR} - \log \mathrm{HR}_0)^2},
              \quad \pi = r / (1 + r).

    Parameters
    ----------
    hazard_ratio : float
        Hazard ratio, treatment vs control (either side of 1).
    alpha_two_sided, power, allocation
        As for :func:`sample_size_normal`.
    prob_event : float, optional
        Probability that a randomised patient has an event by the final analysis. If given,
        the total number of patients is ``events / prob_event``.
    hazard_ratio0 : float
        Hazard ratio under H0: 1 for superiority, the margin (above 1 when a lower hazard is
        better) for non-inferiority. Matches ``gsDesign::nEvents(hr0 = )``.

    References
    ----------
    Schoenfeld, D. A. (1983). *Biometrics* 39, 499-503.
    """
    _check_alpha_power(alpha_two_sided, power)
    _check_allocation(allocation)
    if not hazard_ratio > 0 or not hazard_ratio0 > 0:
        raise ValueError("hazard_ratio and hazard_ratio0 must be positive")
    gap = _effect_gap(
        math.log(hazard_ratio),
        math.log(hazard_ratio0),
        "hazard_ratio must be positive and not equal to 1"
        if hazard_ratio0 == 1
        else "hazard_ratio must differ from hazard_ratio0",
    )
    pi = allocation / (1 + allocation)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    z_b = stats.norm.ppf(power)
    d = (z_a + z_b) ** 2 / (pi * (1 - pi) * gap**2)
    n_total = None
    if prob_event is not None:
        if not 0.0 < prob_event <= 1.0:
            raise ValueError("prob_event must be in (0, 1]")
        n_total = float(d / prob_event)
    return SurvivalEvents(events=float(d), n_total=n_total)


def power_survival(
    events: float,
    hazard_ratio: float,
    alpha_two_sided: float = 0.05,
    allocation: float = 1.0,
    hazard_ratio0: float = 1.0,
) -> float:
    """Power of the log-rank test given the number of events (Schoenfeld 1983).

    References
    ----------
    Schoenfeld, D. A. (1983). *Biometrics* 39, 499-503.
    """
    _check_alpha_power(alpha_two_sided)
    _check_allocation(allocation)
    if not events > 0:
        raise ValueError("events must be positive")
    if not hazard_ratio > 0 or not hazard_ratio0 > 0:
        raise ValueError("hazard_ratio and hazard_ratio0 must be positive")
    pi = allocation / (1 + allocation)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    gap = abs(math.log(hazard_ratio) - math.log(hazard_ratio0))
    drift = math.sqrt(events * pi * (1 - pi)) * gap
    return float(stats.norm.cdf(drift - z_a))
