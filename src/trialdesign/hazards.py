r"""Event-time models beyond the exponential: piecewise-constant hazards and Weibull.

The design formulas in :mod:`trialdesign.fixed` and :mod:`trialdesign.accrual` assume
exponential event times with proportional hazards. Immunotherapy trials often break that:
the treatment effect starts only after a few months (a **delayed effect**), fades (a
**waning effect**), or the hazards cross. This module describes such event-time
distributions so that the patient-level simulator (:func:`trialdesign.patients.simulate_survival`)
can draw from them and :func:`expected_events` can say when each look is expected.

An arm's event time has either

- a **piecewise-constant hazard** :math:`\lambda_j` on :math:`[s_{j-1}, s_j)`, with
  :math:`0 = s_0 < s_1 < \dots` (the last interval is open-ended), so the cumulative hazard
  :math:`\Lambda(t)` is piecewise linear; or
- a **Weibull** distribution :math:`S(t) = \exp\{-(\lambda t)^\kappa\}`, the parameterisation
  of ``rpact`` (``lambda2``, ``kappa``). With the same shape :math:`\kappa` in both arms the
  hazard ratio is :math:`(\lambda_T/\lambda_C)^\kappa`, constant over time, so
  :meth:`EventModel.weibull` keeps proportional hazards with a non-exponential baseline.

Event times are drawn by inversion: :math:`T = \Lambda^{-1}(E)` with :math:`E \sim
\text{Exp}(1)`. For one interval this is :math:`E/\lambda`, exactly the exponential draw of
v0.3, so the same seed gives the same trials.

With dropout at hazard :math:`\eta` (both arms), a patient followed for :math:`s` has an
observed event with probability

.. math:: G(s) = \int_0^s h(x)\, S(x)\, e^{-\eta x}\, dx,

closed form for piecewise hazards and a numerical integral for Weibull, and the expected number
of events by calendar time :math:`t` under accrual intensity :math:`a(u)` is
:math:`\sum_{\text{arm}} \pi_{\text{arm}} \int_0^t a(u)\, G_{\text{arm}}(t - u)\, du`, the
Lachin and Foulkes (1986) integral with a general event distribution.

References
----------
Lachin, J. M. and Foulkes, M. A. (1986). Evaluation of sample size and power for analyses of
survival with allowance for nonuniform patient entry, losses to follow-up, noncompliance, and
stratification. *Biometrics* 42, 507-519.
Lakatos, E. (1988). Sample sizes based on the log-rank statistic in complex clinical trials.
*Biometrics* 44, 229-241.
Wassmer, G. and Brannath, W. (2016). *Group Sequential and Confirmatory Adaptive Designs in
Clinical Trials*. Springer. Section 4.4 (piecewise exponential and Weibull in ``rpact``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

__all__ = [
    "ArmHazard",
    "EventModel",
    "expected_events",
]

_QUAD = {"epsabs": 1e-12, "epsrel": 1e-10, "limit": 200}


@dataclass(frozen=True)
class ArmHazard:
    r"""Event-time distribution of one arm: piecewise-constant hazard, or Weibull.

    Attributes
    ----------
    times : tuple of float
        Start of each hazard interval, beginning at 0 and increasing. One entry for a
        single hazard (exponential or Weibull).
    hazards : tuple of float
        Hazard on each interval (piecewise), or the Weibull scale :math:`\lambda` (one entry).
        Hazards are zero or positive, and the last one is positive.
    kappa : float
        Weibull shape; 1 is exponential. Only with a single interval.
    """

    times: tuple[float, ...]
    hazards: tuple[float, ...]
    kappa: float = 1.0

    def __post_init__(self) -> None:
        t = np.asarray(self.times, dtype=float)
        h = np.asarray(self.hazards, dtype=float)
        if t.ndim != 1 or t.size == 0 or t.shape != h.shape:
            raise ValueError("give one hazard per interval start")
        if t[0] != 0.0 or np.any(np.diff(t) <= 0):
            raise ValueError("hazard intervals must start at 0 and increase")
        if np.any(~np.isfinite(h)) or np.any(h < 0) or not h[-1] > 0:
            raise ValueError("hazards must be zero or positive, and the last one positive")
        if not (math.isfinite(self.kappa) and self.kappa > 0):
            raise ValueError("kappa (Weibull shape) must be positive")
        if self.kappa != 1.0 and t.size > 1:
            raise ValueError("a Weibull shape (kappa != 1) needs a single interval")

    # ------------------------------------------------------------------ construction

    @classmethod
    def exponential(cls, hazard: float) -> ArmHazard:
        return cls((0.0,), (float(hazard),))

    @classmethod
    def piecewise(cls, times: Sequence[float], hazards: Sequence[float]) -> ArmHazard:
        return cls(tuple(float(x) for x in times), tuple(float(x) for x in hazards))

    @classmethod
    def weibull(cls, scale: float, kappa: float) -> ArmHazard:
        return cls((0.0,), (float(scale),), float(kappa))

    # ------------------------------------------------------------------- properties

    @property
    def is_exponential(self) -> bool:
        return len(self.hazards) == 1 and self.kappa == 1.0

    def _cum_at_starts(self) -> NDArray[np.float64]:
        t = np.asarray(self.times)
        h = np.asarray(self.hazards)
        return np.concatenate([[0.0], np.cumsum(h[:-1] * np.diff(t))])

    def cumulative_hazard(self, t: NDArray | float) -> NDArray[np.float64]:
        r""":math:`\Lambda(t)`, for :math:`t \ge 0`."""
        x = np.maximum(np.asarray(t, dtype=float), 0.0)
        if self.kappa != 1.0:
            return (self.hazards[0] * x) ** self.kappa
        starts = np.asarray(self.times)
        h = np.asarray(self.hazards)
        j = np.searchsorted(starts, x, side="right") - 1
        return self._cum_at_starts()[j] + h[j] * (x - starts[j])

    def survival(self, t: NDArray | float) -> NDArray[np.float64]:
        return np.exp(-self.cumulative_hazard(t))

    def median(self) -> float:
        return float(self.inverse_cumulative_hazard(math.log(2)))

    def inverse_cumulative_hazard(self, e: NDArray | float) -> NDArray[np.float64]:
        r""":math:`\Lambda^{-1}(e)`: the event time for a unit exponential draw ``e``."""
        x = np.asarray(e, dtype=float)
        if self.kappa != 1.0:
            return x ** (1.0 / self.kappa) / self.hazards[0]
        if self.is_exponential:
            return x / self.hazards[0]
        starts = np.asarray(self.times)
        h = np.asarray(self.hazards)
        cum = self._cum_at_starts()
        j = np.searchsorted(cum, x, side="right") - 1
        # within interval j the hazard is positive (zero-hazard intervals have the same cum
        # as the next start, so 'right' skips past them)
        return starts[j] + (x - cum[j]) / h[j]

    def observed_event_prob(self, s: NDArray | float, dropout: float = 0.0) -> NDArray:
        r""":math:`G(s)`: probability of an observed event within ``s`` of entry.

        The event must come before an exponential dropout with hazard ``dropout``.
        """
        x = np.maximum(np.asarray(s, dtype=float), 0.0)
        if self.kappa != 1.0:
            if dropout == 0.0:
                return 1.0 - self.survival(x)
            lam, k = self.hazards[0], self.kappa

            def dens(y: float) -> float:
                if y <= 0.0:
                    return 0.0 if k > 1 else (math.inf if k < 1 else lam)
                return k * lam * (lam * y) ** (k - 1) * math.exp(-((lam * y) ** k) - dropout * y)

            def one(v: float) -> float:
                return integrate.quad(dens, 0.0, v, **_QUAD)[0] if v > 0 else 0.0

            return np.vectorize(one, otypes=[float])(x)
        starts = np.asarray(self.times)
        ends = np.append(starts[1:], np.inf)
        h = np.asarray(self.hazards)
        mu = h + dropout
        s_start = np.exp(-self._cum_at_starts() - dropout * starts)  # P(no event, no drop)
        length = np.clip(x[..., None] - starts, 0.0, ends - starts)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(mu > 0, -np.expm1(-mu * length) / np.where(mu > 0, mu, 1.0), length)
        return np.sum(h * s_start * frac, axis=-1)

    def describe(self) -> dict[str, Any]:
        if self.kappa != 1.0:
            return {"weibull_scale": self.hazards[0], "kappa": self.kappa}
        return {"times": list(self.times), "hazards": list(self.hazards)}


@dataclass(frozen=True)
class EventModel:
    """Event-time distributions of the control and treatment arms.

    Build one with :meth:`exponential`, :meth:`delayed_effect`, :meth:`waning_effect`,
    :meth:`piecewise` or :meth:`weibull`. :meth:`null` gives the H0 model, with the
    treatment arm given the control distribution.
    """

    control: ArmHazard
    treatment: ArmHazard
    name: str = "custom"

    @classmethod
    def exponential(cls, hazard_control: float, hazard_ratio: float) -> EventModel:
        """Proportional hazards with exponential event times (the v0.3 model)."""
        _check_hr(hazard_ratio)
        return cls(
            ArmHazard.exponential(hazard_control),
            ArmHazard.exponential(hazard_ratio * hazard_control),
            "exponential",
        )

    @classmethod
    def delayed_effect(cls, hazard_control: float, hazard_ratio: float, delay: float) -> EventModel:
        """Hazard ratio 1 for ``delay`` time units after entry, then ``hazard_ratio``."""
        _check_hr(hazard_ratio)
        if not delay >= 0:
            raise ValueError("delay must be zero or positive")
        if delay == 0:
            m = cls.exponential(hazard_control, hazard_ratio)
            return cls(m.control, m.treatment, "delayed_effect")
        return cls(
            ArmHazard.piecewise((0.0, delay), (hazard_control, hazard_control)),
            ArmHazard.piecewise((0.0, delay), (hazard_control, hazard_ratio * hazard_control)),
            "delayed_effect",
        )

    @classmethod
    def waning_effect(cls, hazard_control: float, hazard_ratio: float, until: float) -> EventModel:
        """Hazard ratio ``hazard_ratio`` until ``until`` time units after entry, then 1."""
        _check_hr(hazard_ratio)
        if not until > 0:
            raise ValueError("the end of the effect (waning) must be positive")
        return cls(
            ArmHazard.piecewise((0.0, until), (hazard_control, hazard_control)),
            ArmHazard.piecewise((0.0, until), (hazard_ratio * hazard_control, hazard_control)),
            "waning_effect",
        )

    @classmethod
    def piecewise(
        cls, times: Sequence[float], control: Sequence[float], treatment: Sequence[float]
    ) -> EventModel:
        """Any piecewise-constant hazards per arm, on shared interval starts ``times``."""
        return cls(ArmHazard.piecewise(times, control), ArmHazard.piecewise(times, treatment))

    @classmethod
    def weibull(cls, median_control: float, hazard_ratio: float, kappa: float) -> EventModel:
        r"""Weibull event times with shape ``kappa`` and proportional hazards.

        The control scale gives the median: :math:`\lambda_C = (\log 2)^{1/\kappa} / m_C`;
        the treatment scale is :math:`\lambda_C\,\mathrm{HR}^{1/\kappa}`, as in ``rpact``.
        """
        _check_hr(hazard_ratio)
        if not median_control > 0:
            raise ValueError("median_control must be positive")
        if not kappa > 0:
            raise ValueError("kappa (Weibull shape) must be positive")
        lam_c = math.log(2) ** (1.0 / kappa) / median_control
        return cls(
            ArmHazard.weibull(lam_c, kappa),
            ArmHazard.weibull(lam_c * hazard_ratio ** (1.0 / kappa), kappa),
            "weibull",
        )

    def null(self) -> EventModel:
        """The H0 model: both arms follow the control distribution."""
        return EventModel(self.control, self.control, self.name)

    @property
    def hazard_ratio(self) -> float | None:
        """The hazard ratio if it is constant over time (proportional hazards), else None."""
        c, t = self.control, self.treatment
        if c.kappa != t.kappa:
            return None
        if c.kappa != 1.0:
            return (t.hazards[0] / c.hazards[0]) ** c.kappa
        times = sorted(set(c.times) | set(t.times))
        hc = _hazards_at(c, times)
        ht = _hazards_at(t, times)
        if np.any(hc == 0):
            return None
        ratio = ht / hc
        return float(ratio[0]) if np.allclose(ratio, ratio[0], rtol=1e-12, atol=0) else None

    def describe(self) -> dict[str, Any]:
        hr = self.hazard_ratio
        return {
            "model": self.name,
            "hazard_ratio": hr,
            "control": self.control.describe(),
            "treatment": self.treatment.describe(),
        }


def _hazards_at(arm: ArmHazard, times: Sequence[float]) -> NDArray[np.float64]:
    starts = np.asarray(arm.times)
    j = np.searchsorted(starts, np.asarray(times, dtype=float), side="right") - 1
    return np.asarray(arm.hazards)[j]


def _check_hr(hr: float) -> None:
    if not (math.isfinite(hr) and hr > 0):
        raise ValueError("hazard_ratio must be positive")


def expected_events(
    t: float,
    accrual_times: Sequence[float],
    accrual_rates: Sequence[float],
    model: EventModel,
    dropout: float = 0.0,
    allocation: float = 1.0,
) -> float:
    """Expected number of events (both arms) by calendar time ``t`` under ``model``.

    The same integral as :func:`trialdesign.accrual.expected_events`, which it equals for an
    exponential model, with the event distribution of each arm taken from ``model``.

    Parameters
    ----------
    t : float
        Calendar time since the first patient entered.
    accrual_times, accrual_rates : sequence of float
        End and rate of each accrual segment, as in :mod:`trialdesign.accrual`.
    model : EventModel
        Event-time distributions of the two arms.
    dropout : float
        Exponential dropout hazard, both arms.
    allocation : float
        Allocation ratio :math:`r = n_T / n_C`.

    References
    ----------
    Lachin & Foulkes (1986), section 2; Wassmer & Brannath (2016), section 4.4.
    """
    ends = np.asarray(accrual_times, dtype=float)
    rates = np.asarray(accrual_rates, dtype=float)
    starts = np.concatenate([[0.0], ends[:-1]])
    pi_t = allocation / (1 + allocation)
    if t <= 0:
        return 0.0
    total = 0.0
    for arm, share in ((model.control, 1 - pi_t), (model.treatment, pi_t)):
        kinks = [t - s for s in arm.times[1:] if 0 < t - s < t]
        for lo, hi, rate in zip(starts, ends, rates, strict=True):
            a, b = min(lo, t), min(hi, t)
            if b <= a:
                continue
            pts = [p for p in kinks if a < p < b] or None
            val = integrate.quad(
                lambda u, arm=arm: float(arm.observed_event_prob(t - u, dropout)),
                a,
                b,
                points=pts,
                **_QUAD,
            )[0]
            total += share * rate * val
    return float(total)
