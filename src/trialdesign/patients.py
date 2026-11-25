r"""Patient-level simulation of event-driven survival trials with group-sequential looks.

The score-process engine in :mod:`trialdesign.simulate` draws the z-statistics directly from
their asymptotic joint distribution. This module simulates the patients instead, so it checks
how far that approximation holds with real accrual, dropout and a log-rank test, and it gives
the calendar quantities that only patients have: when each look happens, how many patients
are enrolled by then, and how long the study runs.

Each simulated trial:

1. enrols ``n_total`` patients over the accrual period of a
   :class:`~trialdesign.accrual.SurvivalStudy` (piecewise-constant intensity), either at
   evenly spaced times (``entry="fixed"``, as ``rpact::getSimulationSurvival`` does) or as
   ordered uniform draws from the accrual density (``entry="random"``);
2. assigns arms by permuted blocks (``allocation`` as a small integer ratio, times
   ``block_multiplier``);
3. draws exponential event times with hazard :math:`\lambda_C = \log 2 / m_C` in control and
   :math:`\lambda_T = \mathrm{HR}\,\lambda_C` in treatment, and exponential dropout times with
   hazard :math:`\eta` in both arms;
4. takes look k at calendar time :math:`\tau_k`, the time of the :math:`D_k`-th observed
   event (an event observed before dropout), with :math:`D_k = \operatorname{round}(t_k D)`.
   Patients enrolled by :math:`\tau_k` are censored administratively there;
5. computes the unweighted log-rank statistic (Mantel 1966; Peto and Peto 1972) at each look,
   oriented so that **positive favours treatment** (fewer treatment events than expected):

   .. math:: Z_k = \frac{\sum_j \bigl(d_j\, n_{Tj}/n_j - e_{Tj}\bigr)}
             {\sqrt{\sum_j d_j \frac{n_{Tj} n_{Cj}}{n_j^2} \frac{n_j - d_j}{n_j - 1}}},

   summed over distinct event times. With continuous times there are no ties, so every
   :math:`d_j = 1` and the tie correction is 1.

The trial then stops at the first look with :math:`Z_k \ge c_k` (efficacy) or
:math:`Z_k < b_k` (futility), exactly as in :func:`trialdesign.simulate.simulate_trials`.
The test is of H0: HR = 1 (superiority); non-inferiority margins on the hazard ratio need a
shifted log-rank test and are not simulated here.

Everything is vectorised over trials (one ``numpy`` sort per look for a batch of trials), with
a Python loop over looks and batches only.

References
----------
Mantel, N. (1966). Evaluation of survival data and two new rank order statistics arising in
its consideration. *Cancer Chemotherapy Reports* 50, 163-170.
Peto, R. and Peto, J. (1972). Asymptotically efficient rank invariant test procedures.
*Journal of the Royal Statistical Society A* 135, 185-207.
Tsiatis, A. A. (1981). The asymptotic joint distribution of the efficient scores test for
the proportional hazards model calculated over time. *Biometrika* 68, 311-315.
Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
Clinical Trials*. Chapman & Hall/CRC. Section 13.1 (survival data and the log-rank test).
Wassmer, G. and Brannath, W. (2016). *Group Sequential and Confirmatory Adaptive Designs in
Clinical Trials*. Springer. Section 4.4 (the simulation behind ``rpact``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from trialdesign.accrual import SurvivalStudy
from trialdesign.simulate import (
    OperatingCharacteristics,
    _check_bounds,
    _check_n_sim,
    _rng,
    _Tally,
)
from trialdesign.spending import check_info_rates

__all__ = [
    "SurvivalSimulation",
    "entry_times",
    "logrank_z",
    "simulate_survival",
]

Entry = Literal["fixed", "random"]

_CELLS = 4_000_000  # patients x trials per batch; bounds memory at about 32 MB per array


# ----------------------------------------------------------------------------- building blocks


def entry_times(
    n_total: int,
    accrual_times: Sequence[float],
    accrual_rates: Sequence[float],
    entry: Entry = "fixed",
    size: int = 1,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    r"""Entry times of ``n_total`` patients under a piecewise-constant accrual intensity.

    The accrual period runs to the end of the last segment, where the cumulative intensity
    :math:`A(t)` reaches its total; if ``n_total`` differs from that total (it is usually the
    total rounded up), the intensity is rescaled so that exactly ``n_total`` patients enter
    over the same period.

    ``entry="fixed"`` places patient i at :math:`A^{-1}(i / n)`, :math:`i = 1, \dots, n`: the
    evenly spaced entry of ``rpact::getSimulationSurvival``. ``entry="random"`` draws ``size``
    independent sets of ordered uniform entry times from the accrual density.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_total,)`` for ``"fixed"``; ``(size, n_total)`` for ``"random"``, each row
        sorted.
    """
    if not (isinstance(n_total, int | np.integer) and n_total >= 2):
        raise ValueError("n_total must be an integer of at least 2")
    ends = np.asarray(accrual_times, dtype=float)
    rates = np.asarray(accrual_rates, dtype=float)
    if ends.shape != rates.shape or ends.size == 0:
        raise ValueError("give one accrual end time per accrual rate")
    starts = np.concatenate([[0.0], ends[:-1]])
    cum = np.concatenate([[0.0], np.cumsum(rates * (ends - starts))])
    share = cum / cum[-1]  # A(t) / A(end) at each breakpoint

    def inverse(u: NDArray) -> NDArray:
        j = np.clip(np.searchsorted(share, u, side="left") - 1, 0, ends.size - 1)
        return starts[j] + (u - share[j]) / (share[j + 1] - share[j]) * (ends[j] - starts[j])

    if entry == "fixed":
        return inverse(np.arange(1, n_total + 1) / n_total)
    if entry == "random":
        gen = _rng(rng)
        return inverse(np.sort(gen.random((size, n_total)), axis=1))
    raise ValueError("entry must be 'fixed' or 'random'")


def _block(allocation: float, multiplier: int) -> NDArray[np.int8]:
    frac = Fraction(allocation).limit_denominator(20)
    if abs(float(frac) - allocation) > 1e-9:
        raise ValueError("allocation must be a ratio of small integers, such as 1, 2 or 1.5")
    n_t, n_c = frac.numerator, frac.denominator
    return np.array([1] * (n_t * multiplier) + [0] * (n_c * multiplier), dtype=np.int8)


def _arms(n_total: int, block: NDArray[np.int8], size: int, rng: np.random.Generator) -> NDArray:
    """Permuted-block treatment indicators (1 = treatment), shape ``(size, n_total)``."""
    n_blocks = -(-n_total // block.size)
    keys = rng.random((size, n_blocks, block.size))
    order = np.argsort(keys, axis=2)
    arms = block[order].reshape(size, n_blocks * block.size)
    return arms[:, :n_total]


def logrank_z(
    time: NDArray[np.float64], event: NDArray[np.bool_], treat: NDArray
) -> NDArray[np.float64]:
    r"""Unweighted log-rank z-statistic for each row, positive when treatment does better.

    Parameters
    ----------
    time : numpy.ndarray
        Observed times, shape ``(m, n)``. Patients who should not count (not yet enrolled)
        are given a negative time and no event: they sort before everyone and never join a
        risk set.
    event : numpy.ndarray of bool
        Event indicators, same shape.
    treat : numpy.ndarray
        1 for treatment, 0 for control, same shape.

    Returns
    -------
    numpy.ndarray
        Shape ``(m,)``. Rows with no events, or no variance, get 0.

    Notes
    -----
    Ties between an event and a censoring at the same time are broken by sorting censorings
    after events (the usual convention: a patient censored at t is at risk at t). Ties between
    events have probability 0 with continuous times and are treated as consecutive events,
    which differs from the tie-corrected statistic only in that case.

    References
    ----------
    Mantel (1966); Peto and Peto (1972); Jennison & Turnbull (2000), section 13.1.
    """
    m, n = time.shape
    # sort by time, events before censorings at the same time
    order = _order(time, event)
    rows = np.arange(m)[:, None]
    ev = event[rows, order]
    tr = treat[rows, order].astype(np.float64)
    # at risk at sorted position i: everyone at position >= i
    n_risk = np.arange(n, 0, -1, dtype=np.float64)[None, :]
    n_t = np.cumsum(tr[:, ::-1], axis=1)[:, ::-1]
    p_t = n_t / n_risk
    num = np.where(ev, p_t - tr, 0.0).sum(axis=1)
    var = np.where(ev, p_t * (1.0 - p_t), 0.0).sum(axis=1)
    return np.divide(num, np.sqrt(var), out=np.zeros(m), where=var > 0)


def _order(time: NDArray, event: NDArray) -> NDArray:
    # events sort before censorings at the same time: nudge censored times up by a hair
    key = np.where(event, time, np.nextafter(time, np.inf))
    return np.argsort(key, axis=1, kind="stable")


# ------------------------------------------------------------------------------- results


@dataclass(frozen=True)
class SurvivalSimulation:
    r"""Patient-level operating characteristics of an event-driven survival design.

    Attributes
    ----------
    oc : OperatingCharacteristics
        Stopping probabilities per look and overall, each with a Monte Carlo SE. Its
        ``expected_info_fraction`` is the mean of :math:`D_k / D` at the stop.
    hazard_ratio : float
        True hazard ratio simulated.
    n_total : int
        Patients per trial if it runs to the end of accrual.
    events : numpy.ndarray
        Planned cumulative events :math:`D_k` at each look.
    analysis_time, analysis_time_se : numpy.ndarray
        Mean calendar time of look k among trials that reach look k (``rpact``'s
        ``analysisTime``), and its Monte Carlo SE.
    expected_duration, expected_duration_se : float
        Mean calendar time at which the trial stops (``rpact``'s ``studyDuration``).
    expected_patients, expected_patients_se : float
        Mean number of patients enrolled by the stop (``expectedNumberOfSubjects``).
    expected_events, expected_events_se : float
        Mean number of events at the stop (``expectedNumberOfEvents``).
    events_not_reached : float
        Share of trials in which some look's event count was never reached; such a look is
        taken when the last patient's follow-up ends. Should be 0 for a sensible design.
    seed : int
        Entropy of the seed sequence, so the run can be repeated.
    """

    oc: OperatingCharacteristics
    hazard_ratio: float
    n_total: int
    events: NDArray[np.int64]
    analysis_time: NDArray[np.float64]
    analysis_time_se: NDArray[np.float64]
    expected_duration: float
    expected_duration_se: float
    expected_patients: float
    expected_patients_se: float
    expected_events: float
    expected_events_se: float
    events_not_reached: float
    seed: int

    @property
    def reject_prob(self) -> float:
        """Type I error under HR = 1, power otherwise."""
        return self.oc.reject_prob

    @property
    def reject_prob_se(self) -> float:
        return self.oc.reject_prob_se


class _Moments:
    """Running sum and sum of squares, for a mean and its Monte Carlo SE."""

    def __init__(self, shape: int | tuple[int, ...] = ()) -> None:
        self.n = np.zeros(shape)
        self.s = np.zeros(shape)
        self.s2 = np.zeros(shape)

    def add(self, x: NDArray, mask: NDArray | None = None, axis: int = 0) -> None:
        if mask is None:
            mask = np.ones_like(x, dtype=bool)
        xm = np.where(mask, x, 0.0)
        self.n += mask.sum(axis=axis)
        self.s += xm.sum(axis=axis)
        self.s2 += (xm * xm).sum(axis=axis)

    def mean_se(self) -> tuple[NDArray, NDArray]:
        n = np.maximum(self.n, 1)
        mean = self.s / n
        var = np.maximum(self.s2 / n - mean * mean, 0.0) * n / np.maximum(n - 1, 1)
        return mean, np.sqrt(var / n)


# ---------------------------------------------------------------------------- simulation


def simulate_survival(
    study: SurvivalStudy,
    info_rates: int | Sequence[float],
    efficacy: Sequence[float],
    futility: Sequence[float] | None = None,
    *,
    hazard_ratio: float | None = None,
    n_total: int | None = None,
    events: int | None = None,
    entry: Entry = "fixed",
    block_multiplier: int = 2,
    n_sim: int = 10_000,
    seed: int | None = None,
) -> SurvivalSimulation:
    r"""Simulate patients in an event-driven group-sequential survival trial.

    Parameters
    ----------
    study : SurvivalStudy
        Accrual, control hazard, dropout and allocation, from
        :func:`trialdesign.accrual.survival_study`.
    info_rates : int or sequence of float
        Information (event) fractions of the looks, or the number of equally spaced looks.
    efficacy : sequence of float
        Efficacy z-boundaries :math:`c_k`.
    futility : sequence of float, optional
        Futility z-boundaries :math:`b_k`. None means futility is ignored.
    hazard_ratio : float, optional
        True treatment vs control hazard ratio; defaults to the study's design value. Use 1
        for the type I error.
    n_total : int, optional
        Patients enrolled if the trial runs to the end of accrual; defaults to the study's
        ``n_total`` rounded up.
    events : int, optional
        Events at the final look; defaults to the study's ``events`` rounded up. Look k is at
        :math:`D_k = \operatorname{round}(t_k D)` events.
    entry : {"fixed", "random"}
        Evenly spaced entry (as ``rpact``) or random entry from the accrual density.
    block_multiplier : int
        Permuted blocks hold ``block_multiplier`` times the smallest integer allocation ratio
        (4 patients for 1:1 with the default 2).
    n_sim : int
        Number of simulated trials.
    seed : int, optional
        Seed; if None, fresh entropy is drawn and stored in the result.

    Returns
    -------
    SurvivalSimulation

    References
    ----------
    Jennison & Turnbull (2000), section 13.1; Tsiatis (1981); Wassmer & Brannath (2016),
    section 4.4.
    """
    t = check_info_rates(info_rates)
    c, b = _check_bounds(t, efficacy, futility)
    _check_n_sim(n_sim)
    hr = study.hazard_ratio if hazard_ratio is None else float(hazard_ratio)
    if not hr > 0:
        raise ValueError("hazard_ratio must be positive")
    n = math.ceil(study.n_total - 1e-9) if n_total is None else int(n_total)
    d_max = math.ceil(study.events - 1e-9) if events is None else int(events)
    d_k = np.rint(t * d_max).astype(np.int64)
    if d_k[0] < 1 or np.any(np.diff(d_k) < 1):
        raise ValueError("each look needs at least one new event")
    if d_max > n:
        raise ValueError(f"{d_max} events cannot come from {n} patients")
    if not (isinstance(block_multiplier, int) and block_multiplier >= 1):
        raise ValueError("block_multiplier must be a positive integer")
    if entry not in ("fixed", "random"):
        raise ValueError("entry must be 'fixed' or 'random'")
    block = _block(study.allocation, block_multiplier)

    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)
    lam_c = study.hazard_control
    lam_t = hr * lam_c
    eta = study.dropout
    fixed_entry = (
        entry_times(n, study.accrual_times, study.accrual_rates, "fixed")
        if entry == "fixed"
        else None
    )

    tally = _Tally(t, c, b)
    look_time = _Moments(t.size)
    duration, patients, n_events = _Moments(), _Moments(), _Moments()
    not_reached = 0
    chunk = max(1, _CELLS // n)
    k_idx = d_k - 1
    for start in range(0, n_sim, chunk):
        m = min(chunk, n_sim - start)
        u = (
            np.broadcast_to(fixed_entry, (m, n))
            if fixed_entry is not None
            else entry_times(n, study.accrual_times, study.accrual_rates, "random", m, rng)
        )
        arm = _arms(n, block, m, rng)
        lam = np.where(arm == 1, lam_t, lam_c)
        t_event = rng.exponential(1.0, (m, n)) / lam
        t_drop = rng.exponential(1.0, (m, n)) / eta if eta > 0 else np.full((m, n), np.inf)
        has_event = t_event <= t_drop
        t_obs = np.minimum(t_event, t_drop)  # time from entry to event or dropout
        cal_event = np.where(has_event, u + t_event, np.inf)  # calendar time of each event
        sorted_cal = np.sort(cal_event, axis=1)
        tau = sorted_cal[:, k_idx]  # calendar time of the D_k-th event, (m, K)
        short = ~np.isfinite(tau)
        if short.any():
            not_reached += int(short.any(axis=1).sum())
            last = np.max(u + t_obs, axis=1, keepdims=True)
            tau = np.where(short, last, tau)
        z = np.empty((m, t.size))
        for k in range(t.size):
            cut = tau[:, k : k + 1]
            in_trial = u < cut
            follow = cut - u
            time_k = np.where(in_trial, np.minimum(t_obs, follow), -1.0)
            event_k = in_trial & has_event & (t_event <= follow)
            z[:, k] = logrank_z(time_k, event_k, arm)
        tally.add(z)
        # stop look per trial, as in _Tally
        stop = (z >= c) | (z < b)
        stop[:, -1] = True
        k_stop = stop.argmax(axis=1)
        reached = np.arange(t.size)[None, :] <= k_stop[:, None]
        look_time.add(tau, reached)
        rows = np.arange(m)
        tau_stop = tau[rows, k_stop]
        duration.add(tau_stop)
        patients.add((u < tau_stop[:, None]).sum(axis=1).astype(float))
        n_events.add((cal_event <= tau_stop[:, None]).sum(axis=1).astype(float))

    oc = tally.result(None, futility is not None)
    at_mean, at_se = look_time.mean_se()
    dur_mean, dur_se = duration.mean_se()
    pat_mean, pat_se = patients.mean_se()
    ev_mean, ev_se = n_events.mean_se()
    oc = OperatingCharacteristics(
        info_rates=d_k / d_max,
        drift=None,
        n_sim=oc.n_sim,
        prob_efficacy=oc.prob_efficacy,
        prob_futility=oc.prob_futility,
        futility_followed=oc.futility_followed,
        expected_info_fraction=float(ev_mean) / d_max,
        expected_info_fraction_se=float(ev_se) / d_max,
    )
    return SurvivalSimulation(
        oc=oc,
        hazard_ratio=hr,
        n_total=n,
        events=d_k,
        analysis_time=np.asarray(at_mean, dtype=float),
        analysis_time_se=np.asarray(at_se, dtype=float),
        expected_duration=float(dur_mean),
        expected_duration_se=float(dur_se),
        expected_patients=float(pat_mean),
        expected_patients_se=float(pat_se),
        expected_events=float(ev_mean),
        expected_events_se=float(ev_se),
        events_not_reached=not_reached / n_sim,
        seed=int(ss.entropy),
    )
