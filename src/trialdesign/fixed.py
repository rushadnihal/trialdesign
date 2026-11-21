"""Fixed-design (single-analysis) sample size and power for two-arm superiority trials.

All functions use a **two-sided** significance level, named ``alpha_two_sided``, and an
allocation ratio ``allocation = n_T / n_C``. Results are returned unrounded so they can be
compared with reference software; round up with :func:`math.ceil` for a real design.

References
----------
Chow, S.-C., Shao, J. and Wang, H. (2008). *Sample Size Calculations in Clinical Research*,
2nd ed. Chapman & Hall/CRC. Sections 3.2 (normal) and 4.2 (binary).
Fleiss, J. L., Levin, B. and Paik, M. C. (2003). *Statistical Methods for Rates and
Proportions*, 3rd ed. Wiley. Section 4.2 (pooled/unpooled variance formula).
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


def sample_size_normal(
    delta: float,
    sd: float,
    alpha_two_sided: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
    method: NormalMethod = "z",
) -> FixedSampleSize:
    r"""Per-arm sample size for a difference in means with common SD.

    With ``method="z"`` (normal approximation)

    .. math:: n_C = (1 + 1/r)\,\sigma^2 (z_{1-\alpha/2} + z_{1-\beta})^2 / \Delta^2,
              \quad n_T = r\,n_C.

    With ``method="t"`` the noncentral-t power of the two-sample t-test is solved for
    :math:`n_C`, matching R's ``power.t.test`` (``strict = FALSE``) when ``allocation = 1``.

    Parameters
    ----------
    delta : float
        Difference in means, treatment minus control (sign is ignored).
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

    References
    ----------
    Chow, Shao & Wang (2008), section 3.2.
    """
    _check_alpha_power(alpha_two_sided, power)
    _check_allocation(allocation)
    if delta == 0:
        raise ValueError("delta must be non-zero")
    if not sd > 0:
        raise ValueError("sd must be positive")
    r = allocation
    std_effect = abs(delta) / sd
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
) -> float:
    """Power for a difference in means given the control-arm size ``n_control``.

    The treatment arm has ``allocation * n_control`` patients. Only the rejection tail in
    the direction of ``delta`` is counted, as in R's ``power.t.test`` with ``strict = FALSE``.

    References
    ----------
    Chow, Shao & Wang (2008), section 3.2.
    """
    _check_alpha_power(alpha_two_sided)
    _check_allocation(allocation)
    if not sd > 0:
        raise ValueError("sd must be positive")
    r = allocation
    std_effect = abs(delta) / sd
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


def _binary_terms(p_control: float, p_treatment: float, r: float) -> tuple[float, float]:
    """Per-``n_C`` standard deviations of the difference under H0 (pooled) and H1."""
    p_bar = (p_control + r * p_treatment) / (1 + r)
    sd0 = math.sqrt(p_bar * (1 - p_bar) * (1 + 1 / r))
    sd1 = math.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment) / r)
    return sd0, sd1


def sample_size_binary(
    p_control: float,
    p_treatment: float,
    alpha_two_sided: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
) -> FixedSampleSize:
    r"""Per-arm sample size for comparing two proportions (no continuity correction).

    Uses the pooled variance under H0 and the unpooled variance under H1:

    .. math:: n_C = \frac{\left(z_{1-\alpha/2}\sqrt{\bar p(1-\bar p)(1+1/r)}
              + z_{1-\beta}\sqrt{p_C(1-p_C) + p_T(1-p_T)/r}\right)^2}{(p_T - p_C)^2},

    with :math:`\bar p = (p_C + r p_T)/(1 + r)`. For ``allocation = 1`` this is the
    formula solved by R's ``power.prop.test`` (``strict = FALSE``).

    References
    ----------
    Fleiss, Levin & Paik (2003), section 4.2; Chow, Shao & Wang (2008), section 4.2.
    """
    _check_alpha_power(alpha_two_sided, power)
    _check_allocation(allocation)
    _check_prob(p_control, "p_control")
    _check_prob(p_treatment, "p_treatment")
    if p_control == p_treatment:
        raise ValueError("p_control and p_treatment must differ")
    r = allocation
    sd0, sd1 = _binary_terms(p_control, p_treatment, r)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    z_b = stats.norm.ppf(power)
    n_c = (z_a * sd0 + z_b * sd1) ** 2 / (p_treatment - p_control) ** 2
    return FixedSampleSize(n_control=float(n_c), n_treatment=float(r * n_c))


def power_binary(
    n_control: float,
    p_control: float,
    p_treatment: float,
    alpha_two_sided: float = 0.05,
    allocation: float = 1.0,
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
    sd0, sd1 = _binary_terms(p_control, p_treatment, allocation)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    diff = abs(p_treatment - p_control)
    return float(stats.norm.cdf((math.sqrt(n_control) * diff - z_a * sd0) / sd1))


# ----------------------------------------------------------------------------- survival


def events_survival(
    hazard_ratio: float,
    alpha_two_sided: float = 0.05,
    power: float = 0.80,
    allocation: float = 1.0,
    prob_event: float | None = None,
) -> SurvivalEvents:
    r"""Required number of events for the log-rank test (Schoenfeld 1983).

    .. math:: D = \frac{(z_{1-\alpha/2} + z_{1-\beta})^2}{\pi(1-\pi)\,(\log \mathrm{HR})^2},
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

    References
    ----------
    Schoenfeld, D. A. (1983). *Biometrics* 39, 499-503.
    """
    _check_alpha_power(alpha_two_sided, power)
    _check_allocation(allocation)
    if not hazard_ratio > 0 or hazard_ratio == 1:
        raise ValueError("hazard_ratio must be positive and not equal to 1")
    pi = allocation / (1 + allocation)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    z_b = stats.norm.ppf(power)
    d = (z_a + z_b) ** 2 / (pi * (1 - pi) * math.log(hazard_ratio) ** 2)
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
    if not hazard_ratio > 0:
        raise ValueError("hazard_ratio must be positive")
    pi = allocation / (1 + allocation)
    z_a = stats.norm.ppf(1 - alpha_two_sided / 2)
    drift = math.sqrt(events * pi * (1 - pi)) * abs(math.log(hazard_ratio))
    return float(stats.norm.cdf(drift - z_a))
