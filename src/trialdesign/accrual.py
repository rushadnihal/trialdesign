r"""Survival accrual, follow-up and dropout: from required events to patients and calendar time.

A log-rank design fixes the number of **events** (:func:`trialdesign.fixed.events_survival`,
times the inflation factor for a group-sequential design). How many **patients** that needs,
and **when** each analysis happens, depends on how fast patients enter and how long they are
followed. This module answers that under the standard assumptions of Lachin and Foulkes
(1986), as implemented by ``rpact::getSampleSizeSurvival``:

- patients enter at a piecewise-constant rate :math:`a(u)` (patients per unit time);
- event times are exponential with hazard :math:`\lambda_C = \log 2 / m_C` in the control arm
  (median :math:`m_C`) and :math:`\lambda_T = \mathrm{HR}\,\lambda_C` in the treatment arm;
- dropout times are exponential with hazard :math:`\eta`, the same in both arms; an annual
  dropout probability :math:`p` gives :math:`\eta = -\log(1 - p) / 12` with time in months.

A patient entering at time :math:`u` has an observed event by calendar time :math:`t` with
probability :math:`\frac{\lambda}{\lambda + \eta}\bigl(1 - e^{-(\lambda+\eta)(t-u)}\bigr)`,
so the expected number of events by time :math:`t` is

.. math:: E(t) = \sum_{\text{arm}} \pi_{\text{arm}} \int_0^{t} a(u)\,
          \frac{\lambda}{\lambda+\eta}\bigl(1 - e^{-(\lambda+\eta)(t-u)}\bigr)\,du,

which has a closed form on each accrual segment. Given any two of the accrual duration
:math:`R`, the minimum follow-up :math:`F` and the accrual rate, :func:`survival_study` solves
the third so that :math:`E(R + F)` equals the required events. :meth:`SurvivalStudy.look_times`
then gives the calendar time at which each interim's share of the events is expected.

Time can be in any unit as long as medians, accrual, follow-up and ``dropout_time`` agree;
the defaults assume months.

References
----------
Lachin, J. M. and Foulkes, M. A. (1986). Evaluation of sample size and power for analyses of
survival with allowance for nonuniform patient entry, losses to follow-up, noncompliance, and
stratification. *Biometrics* 42, 507-519.
Wassmer, G. and Brannath, W. (2016). *Group Sequential and Confirmatory Adaptive Designs in
Clinical Trials*. Springer. Section 4.4 (the method behind ``rpact``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from trialdesign import hazards
from trialdesign.hazards import EventModel

__all__ = [
    "SurvivalStudy",
    "dropout_hazard",
    "expected_events",
    "hazard_from_median",
    "survival_study",
]

_XTOL = 1e-10


def hazard_from_median(median: float) -> float:
    """Exponential hazard with the given median survival time: ``log(2) / median``."""
    if not median > 0:
        raise ValueError("median must be positive")
    return math.log(2) / median


def dropout_hazard(prob: float, time: float = 12.0) -> float:
    """Exponential dropout hazard that loses a share ``prob`` of patients by ``time``.

    With time in months, ``time = 12`` turns an annual dropout probability into a hazard.
    """
    if not 0.0 <= prob < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not time > 0:
        raise ValueError("dropout_time must be positive")
    return -math.log1p(-prob) / time


def _arm_events(
    t: float, starts: NDArray, ends: NDArray, rates: NDArray, hazard: float, dropout: float
) -> float:
    """Expected events by calendar time ``t`` per unit allocation share, one arm."""
    mu = hazard + dropout
    lo = np.minimum(starts, t)
    hi = np.minimum(ends, t)
    # integral over u in [lo, hi] of 1 - exp(-mu (t - u))
    inner = (hi - lo) - (np.exp(-mu * (t - hi)) - np.exp(-mu * (t - lo))) / mu
    return float(hazard / mu * np.sum(rates * inner))


def expected_events(
    t: float,
    accrual_times: Sequence[float],
    accrual_rates: Sequence[float],
    hazard_control: float,
    hazard_ratio: float,
    dropout: float = 0.0,
    allocation: float = 1.0,
) -> float:
    """Expected number of events (both arms) by calendar time ``t``.

    Parameters
    ----------
    t : float
        Calendar time since the first patient entered.
    accrual_times : sequence of float
        End of each accrual segment, increasing; segment ``j`` runs from the previous end
        (0 for the first) to ``accrual_times[j]``.
    accrual_rates : sequence of float
        Patients entering per unit time in each segment, both arms together.
    hazard_control : float
        Event hazard in the control arm.
    hazard_ratio : float
        Treatment vs control hazard ratio.
    dropout : float
        Dropout hazard (see :func:`dropout_hazard`), the same in both arms.
    allocation : float
        Allocation ratio :math:`r = n_T / n_C`.

    References
    ----------
    Lachin & Foulkes (1986), section 2.
    """
    ends = np.asarray(accrual_times, dtype=float)
    rates = np.asarray(accrual_rates, dtype=float)
    starts = np.concatenate([[0.0], ends[:-1]])
    pi_t = allocation / (1 + allocation)
    lam_c = hazard_control
    lam_t = hazard_ratio * hazard_control
    return (1 - pi_t) * _arm_events(t, starts, ends, rates, lam_c, dropout) + pi_t * _arm_events(
        t, starts, ends, rates, lam_t, dropout
    )


@dataclass(frozen=True)
class SurvivalStudy:
    """Accrual and follow-up that deliver a required number of events (all unrounded).

    Attributes
    ----------
    events : float
        Required events at the final analysis.
    n_total : float
        Patients randomised (both arms).
    accrual_times, accrual_rates : numpy.ndarray
        End time and rate (patients per unit time) of each accrual segment.
    followup : float
        Minimum follow-up: time from the last patient's entry to the final analysis.
    hazard_control, hazard_ratio, dropout, allocation : float
        Model inputs, as for :func:`expected_events`.
    """

    events: float
    n_total: float
    accrual_times: NDArray[np.float64]
    accrual_rates: NDArray[np.float64]
    followup: float
    hazard_control: float
    hazard_ratio: float
    dropout: float
    allocation: float

    @property
    def accrual_duration(self) -> float:
        return float(self.accrual_times[-1])

    @property
    def study_duration(self) -> float:
        """Calendar time of the final analysis: accrual duration plus minimum follow-up."""
        return self.accrual_duration + self.followup

    @property
    def prob_event(self) -> float:
        """Share of randomised patients with an event by the final analysis."""
        return self.events / self.n_total

    def event_model(self) -> EventModel:
        """The study's exponential, proportional-hazards event model."""
        return EventModel.exponential(self.hazard_control, self.hazard_ratio)

    def events_at(self, t: float, model: EventModel | None = None) -> float:
        """Expected events by calendar time ``t``.

        With ``model`` (see :mod:`trialdesign.hazards`), under that event-time model instead of
        the study's exponential one, with the study's accrual, dropout and allocation.
        """
        if model is not None:
            return hazards.expected_events(
                t, self.accrual_times, self.accrual_rates, model, self.dropout, self.allocation
            )
        return expected_events(
            t,
            self.accrual_times,
            self.accrual_rates,
            self.hazard_control,
            self.hazard_ratio,
            self.dropout,
            self.allocation,
        )

    def patients_at(self, t: float) -> float:
        """Patients randomised by calendar time ``t``."""
        ends = self.accrual_times
        starts = np.concatenate([[0.0], ends[:-1]])
        return float(np.sum(self.accrual_rates * (np.minimum(ends, t) - np.minimum(starts, t))))

    def time_of_events(self, events: float, model: EventModel | None = None) -> float:
        """Calendar time at which ``events`` events are expected.

        Without ``model``, at most the study end (the planned final events). With ``model``,
        the time may fall after the planned study end, when that model gives events more
        slowly; a ``ValueError`` says so if the patients can never produce that many.
        """
        if model is None:
            if not 0 < events <= self.events * (1 + 1e-9):
                raise ValueError("events must be positive and at most the final number of events")
            end = self.study_duration
            if events >= self.events:
                return end
            return float(
                optimize.brentq(lambda t: self.events_at(t) - events, 0.0, end, xtol=_XTOL)
            )
        if not events > 0:
            raise ValueError("events must be positive")
        hi = self.study_duration
        for _ in range(60):
            if self.events_at(hi, model) >= events:
                break
            hi *= 2
        else:
            raise ValueError(f"{events:g} events are not expected under this event model")
        return float(
            optimize.brentq(lambda t: self.events_at(t, model) - events, 0.0, hi, xtol=_XTOL)
        )

    def look_times(
        self, info_rates: Sequence[float], model: EventModel | None = None
    ) -> NDArray[np.float64]:
        """Calendar time of each analysis, at information fractions of the final events.

        With ``model``, the expected times under that event-time model for the same planned
        event counts.
        """
        return np.array([self.time_of_events(f * self.events, model) for f in info_rates])


def _segments(accrual_time: float | Sequence[float]) -> NDArray[np.float64]:
    ends = np.atleast_1d(np.asarray(accrual_time, dtype=float))
    if ends.size and (np.any(ends <= 0) or np.any(np.diff(ends) <= 0)):
        raise ValueError("accrual times must be positive and increasing")
    return ends


def survival_study(
    events: float,
    hazard_ratio: float,
    median_control: float,
    *,
    accrual_time: float | Sequence[float] | None = None,
    accrual_rate: float | Sequence[float] | None = None,
    followup: float | None = None,
    dropout: float = 0.0,
    dropout_time: float = 12.0,
    allocation: float = 1.0,
) -> SurvivalStudy:
    r"""Patients, accrual and follow-up that give ``events`` expected events at the final analysis.

    Give two of accrual time, accrual rate and follow-up; the third is solved:

    ============================  ==============================================================
    Given                         Solved
    ============================  ==============================================================
    ``accrual_time``, ``followup``  the (uniform) accrual rate, so :math:`n = D / P(\text{event})`
    ``accrual_rate``, ``followup``  the end of accrual; ``accrual_time`` may hold the breakpoints
                                  before the last, open-ended segment (one fewer than rates)
    ``accrual_time``, ``accrual_rate``  the minimum follow-up; one end time per rate
    ============================  ==============================================================

    Parameters
    ----------
    events : float
        Required events at the final analysis.
    hazard_ratio : float
        Treatment vs control hazard ratio under the alternative.
    median_control : float
        Median survival time in the control arm.
    accrual_time : float or sequence of float, optional
        Accrual duration, or the end of each accrual segment.
    accrual_rate : float or sequence of float, optional
        Patients per unit time in each accrual segment, both arms together.
    followup : float, optional
        Minimum follow-up after the last patient enters.
    dropout : float
        Probability of dropping out by ``dropout_time`` (exponential), the same in both arms.
    dropout_time : float
        Time at which ``dropout`` applies; 12 for an annual rate in months.
    allocation : float
        Allocation ratio :math:`r = n_T / n_C`.

    Returns
    -------
    SurvivalStudy

    References
    ----------
    Lachin & Foulkes (1986); Wassmer & Brannath (2016), section 4.4.
    """
    if not events > 0:
        raise ValueError("events must be positive")
    if not hazard_ratio > 0:
        raise ValueError("hazard_ratio must be positive")
    if not allocation > 0:
        raise ValueError("allocation (n_T / n_C) must be positive")
    lam_c = hazard_from_median(median_control)
    eta = dropout_hazard(dropout, dropout_time)
    if followup is not None and not followup >= 0:
        raise ValueError("followup must be zero or positive")
    if followup is None and (accrual_time is None or accrual_rate is None):
        raise ValueError("give two of accrual time, accrual rate and follow-up")
    if followup is not None and accrual_time is None and accrual_rate is None:
        raise ValueError("with follow-up, give an accrual time or accrual rates")
    times = None if accrual_time is None else _segments(accrual_time)
    rates = None if accrual_rate is None else np.atleast_1d(np.asarray(accrual_rate, dtype=float))
    if rates is not None and np.any(rates <= 0):
        raise ValueError("accrual rates must be positive")

    def events_by(t: float, ends: NDArray, r: NDArray) -> float:
        return expected_events(t, ends, r, lam_c, hazard_ratio, eta, allocation)

    def study(ends: NDArray, r: NDArray, f: float) -> SurvivalStudy:
        n = float(np.sum(r * np.diff(np.concatenate([[0.0], ends]))))
        return SurvivalStudy(
            events=float(events),
            n_total=n,
            accrual_times=ends,
            accrual_rates=r,
            followup=float(f),
            hazard_control=lam_c,
            hazard_ratio=float(hazard_ratio),
            dropout=eta,
            allocation=float(allocation),
        )

    if rates is None:  # accrual time and follow-up: solve the uniform rate
        assert times is not None and followup is not None
        if times.size != 1:
            raise ValueError("without accrual rates, give a single accrual time (uniform accrual)")
        per_unit_rate = events_by(times[0] + followup, times, np.ones(1))
        return study(times, np.array([events / per_unit_rate]), followup)

    if followup is not None:  # rates and follow-up: solve the end of accrual
        breaks = np.empty(0) if times is None else times
        if breaks.size != rates.size - 1:
            raise ValueError(
                "with accrual rates and follow-up, give one accrual breakpoint fewer than rates "
                "(the last segment's end is solved)"
            )
        start = float(breaks[-1]) if breaks.size else 0.0

        def gap(end: float) -> float:
            return events_by(end + followup, np.append(breaks, end), rates) - events

        if gap(start + 1e-9) >= 0:
            raise ValueError(
                "the required events are reached before the last accrual segment starts; "
                "drop a segment or shorten the follow-up"
            )
        hi = max(2 * start, 1.0)
        while gap(hi) < 0:
            hi *= 2
            if hi > 1e7:
                raise ValueError("accrual cannot reach the required events")
        end = float(optimize.brentq(gap, start + 1e-9, hi, xtol=_XTOL))
        return study(np.append(breaks, end), rates, followup)

    # accrual times and rates: solve the minimum follow-up
    assert times is not None
    if times.size != rates.size:
        raise ValueError("give one accrual end time per accrual rate")
    n = float(np.sum(rates * np.diff(np.concatenate([[0.0], times]))))
    pi_t = allocation / (1 + allocation)
    lam_t = hazard_ratio * lam_c
    ceiling = n * ((1 - pi_t) * lam_c / (lam_c + eta) + pi_t * lam_t / (lam_t + eta))
    if ceiling <= events:
        raise ValueError(
            f"{n:.1f} patients can give at most {ceiling:.1f} events, fewer than the "
            f"{events:.1f} required; accrue more patients"
        )
    end = float(times[-1])

    def gap_f(f: float) -> float:
        return events_by(end + f, times, rates) - events

    if gap_f(0.0) >= 0:
        return study(times, rates, 0.0)
    hi = max(end, 1.0)
    while gap_f(hi) < 0:
        hi *= 2
    f = float(optimize.brentq(gap_f, 0.0, hi, xtol=_XTOL))
    return study(times, rates, f)
