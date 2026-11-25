r"""Patient-level simulation of group-sequential trials: survival, normal and binary endpoints.

The score-process engine in :mod:`trialdesign.simulate` draws the z-statistics directly from
their asymptotic joint distribution. This module simulates the patients instead, so it checks
how far that approximation holds with real accrual, dropout and a log-rank test, and it gives
the calendar quantities that only patients have: when each look happens, how many patients
are enrolled by then, and how long the study runs.

Normal and binary endpoints (:func:`simulate_normal`, :func:`simulate_binary`) draw each
arm's sufficient statistics per look increment and test with the estimated variance;
:func:`simulate_design_patients` runs any endpoint under H0 and H1, like
:func:`trialdesign.simulate.simulate_design`. The rest of this docstring describes the
survival simulator.

Each simulated survival trial:

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
from scipy import stats

from trialdesign.accrual import SurvivalStudy
from trialdesign.boundaries import GroupSequentialDesign
from trialdesign.simulate import (
    DesignSimulation,
    OperatingCharacteristics,
    SeedLike,
    _check_bounds,
    _check_n_sim,
    _rng,
    _Tally,
)
from trialdesign.spending import check_info_rates
from trialdesign.ssr import ZONES, SurvivalSSR

__all__ = [
    "SurvivalSSRSimulation",
    "SurvivalSimulation",
    "entry_times",
    "logrank_score",
    "logrank_z",
    "simulate_binary",
    "simulate_design_patients",
    "simulate_normal",
    "simulate_survival",
    "simulate_survival_ssr",
]

NormalTest = Literal["t", "z"]
_CHUNK = 250_000  # trials per batch for the normal and binary engines

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
    num, var = logrank_score(time, event, treat)
    return np.divide(num, np.sqrt(var), out=np.zeros(num.size), where=var > 0)


def logrank_score(
    time: NDArray[np.float64], event: NDArray[np.bool_], treat: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Log-rank score :math:`S = \sum_j (d_j n_{Tj}/n_j - e_{Tj})` and its variance :math:`V`.

    Same inputs and conventions as :func:`logrank_z`, which returns :math:`S / \sqrt{V}`.
    The score has approximately independent increments over calendar time (Tsiatis 1981), so
    the increment between two analyses, standardised by the increment of :math:`V`, is the
    stage-2 statistic of an adaptive design.
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
    return num, var


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


# ------------------------------------------------------------------ normal and binary


def _look_counts(t: NDArray, n_control: int, n_treatment: int) -> tuple[NDArray, NDArray]:
    n_c = np.rint(t * n_control).astype(np.int64)
    n_t = np.rint(t * n_treatment).astype(np.int64)
    if n_c[0] < 1 or n_t[0] < 1 or np.any(np.diff(n_c) < 1) or np.any(np.diff(n_t) < 1):
        raise ValueError("each look needs at least one new patient per arm")
    return n_c, n_t


def _n_treatment(n_control: int, n_treatment: int | None, allocation: float) -> int:
    if not allocation > 0:
        raise ValueError("allocation must be positive")
    return round(allocation * n_control) if n_treatment is None else int(n_treatment)


def _arm_moments(
    rng: np.random.Generator, m: int, inc: NDArray, mean: float, sd: float
) -> tuple[NDArray, NDArray]:
    """Cumulative sums and sums of squares about the mean, per look, for one arm."""
    xbar = rng.normal(mean, sd / np.sqrt(inc), (m, inc.size))  # stage means
    ss = sd * sd * rng.chisquare(np.maximum(inc - 1, 1), (m, inc.size))
    ss = np.where(inc > 1, ss, 0.0)  # within-stage sum of squares
    n = np.cumsum(inc)
    total = np.cumsum(inc * xbar, axis=1)
    ss_cum = np.cumsum(ss + inc * xbar * xbar, axis=1) - total * total / n
    return total / n, np.maximum(ss_cum, 0.0)


def simulate_normal(
    delta: float,
    sd: float,
    n_control: int,
    info_rates: int | Sequence[float],
    efficacy: Sequence[float],
    futility: Sequence[float] | None = None,
    *,
    allocation: float = 1.0,
    n_treatment: int | None = None,
    delta0: float = 0.0,
    test: NormalTest = "t",
    n_sim: int = 100_000,
    seed: SeedLike = None,
) -> OperatingCharacteristics:
    r"""Patient-level simulation of a two-arm normal trial with group-sequential looks.

    Each arm's stage mean and within-stage sum of squares are drawn from their exact
    distributions (:math:`N(\mu, \sigma^2/m)` and :math:`\sigma^2\chi^2_{m-1}`, independent),
    which is the same as drawing every patient. At look k, with the pooled variance
    :math:`s_k^2` on :math:`n_{Ck} + n_{Tk} - 2` degrees of freedom,

    .. math:: T_k = \frac{\bar x_{Tk} - \bar x_{Ck} - \delta_0}
              {s_k \sqrt{1/n_{Ck} + 1/n_{Tk}}}.

    With ``test="t"`` the boundaries are applied to :math:`Z_k = \Phi^{-1}(F_{t,\nu}(T_k))`,
    the z-value with the same one-sided p-value (``rpact``'s ``normalApproximation = FALSE``);
    with ``test="z"``, to :math:`T_k` itself.

    Parameters
    ----------
    delta : float
        True difference :math:`\mu_T - \mu_C`.
    sd : float
        True common standard deviation.
    n_control : int
        Maximum control patients; treatment gets ``n_treatment`` or
        ``round(allocation * n_control)``.
    info_rates, efficacy, futility
        As for :func:`trialdesign.simulate.simulate_trials`. Look k has
        ``round(t_k * n)`` patients per arm.
    allocation : float
        Ratio :math:`n_T / n_C`.
    n_treatment : int, optional
        Maximum treatment patients, overriding ``allocation``.
    delta0 : float
        Difference under H0 (negative for non-inferiority); 0 for superiority.
    test : {"t", "z"}
        Reference distribution of the statistic, see above.
    n_sim : int
        Number of simulated trials.
    seed : int, numpy.random.SeedSequence, numpy.random.Generator or None
        Seed or generator.

    Returns
    -------
    OperatingCharacteristics
        With ``drift=None``.

    References
    ----------
    Jennison & Turnbull (2000), sections 3.1 and 3.4 (unknown variance); Wassmer & Brannath
    (2016), section 3.3.
    """
    if not sd > 0:
        raise ValueError("sd must be positive")
    if test not in ("t", "z"):
        raise ValueError("test must be 't' or 'z'")
    t = check_info_rates(info_rates)
    c, b = _check_bounds(t, efficacy, futility)
    _check_n_sim(n_sim)
    n_t_max = _n_treatment(n_control, n_treatment, allocation)
    n_c, n_t = _look_counts(t, n_control, n_t_max)
    if n_c[0] + n_t[0] < 3:
        raise ValueError("the first look needs at least 3 patients to estimate the variance")
    inc_c, inc_t = np.diff(n_c, prepend=0), np.diff(n_t, prepend=0)
    df = n_c + n_t - 2
    rng = _rng(seed)
    tally = _Tally(t, c, b)
    for start in range(0, n_sim, _CHUNK):
        m = min(_CHUNK, n_sim - start)
        mean_c, ss_c = _arm_moments(rng, m, inc_c, 0.0, sd)
        mean_t, ss_t = _arm_moments(rng, m, inc_t, delta, sd)
        s = np.sqrt((ss_c + ss_t) / df)
        stat = (mean_t - mean_c - delta0) / (s * np.sqrt(1.0 / n_c + 1.0 / n_t))
        z = stats.norm.isf(stats.t.sf(stat, df)) if test == "t" else stat
        tally.add(z)
    return tally.result(None, futility is not None)


def _fm_rates_vec(pc: NDArray, pt: NDArray, r: NDArray, d: float) -> tuple[NDArray, NDArray]:
    """Vectorised :func:`trialdesign.fixed.farrington_manning_rates`."""
    if d == 0.0:
        p_bar = (pc + r * pt) / (1 + r)
        return p_bar, p_bar
    p1, p2, theta = pt, pc, 1.0 / r
    a = 1.0 + theta
    b_ = -(1.0 + theta + p1 + theta * p2 + d * (theta + 2.0))
    c_ = d * d + d * (2.0 * p1 + theta + 1.0) + p1 + theta * p2
    e = -p1 * d * (1.0 + d)
    v = b_**3 / (27 * a**3) - b_ * c_ / (6 * a**2) + e / (2 * a)
    u = np.copysign(np.sqrt(np.maximum(b_ * b_ / (9 * a * a) - c_ / (3 * a), 0.0)), v)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(u == 0.0, 0.0, np.clip(v / u**3, -1.0, 1.0))
    w = (np.pi + np.arccos(ratio)) / 3.0
    p1_tilde = 2.0 * u * np.cos(w) - b_ / (3.0 * a)
    return p1_tilde - d, p1_tilde


def simulate_binary(
    p_control: float,
    p_treatment: float,
    n_control: int,
    info_rates: int | Sequence[float],
    efficacy: Sequence[float],
    futility: Sequence[float] | None = None,
    allocation: float = 1.0,
    n_sim: int = 100_000,
    seed: SeedLike = None,
    *,
    n_treatment: int | None = None,
    delta0: float = 0.0,
) -> OperatingCharacteristics:
    r"""Patient-level simulation of a two-arm binary trial with group-sequential looks.

    At look k each arm has :math:`n_{ik} = \operatorname{round}(t_k n_i)` patients, with
    binomial successes drawn per look increment. The statistic uses the variance at the
    rates that maximise the likelihood under H0 (Farrington and Manning 1990):

    .. math:: Z_k = \frac{\hat p_{T} - \hat p_{C} - \delta_0}
              {\sqrt{\tilde p_C (1 - \tilde p_C)/n_{Ck} + \tilde p_T (1 - \tilde p_T)/n_{Tk}}},

    which for superiority (:math:`\delta_0 = 0`) is the pooled-variance score test behind
    ``prop.test`` without continuity correction. Z is set to 0 when the variance is 0.
    Positive z favours treatment.

    Parameters
    ----------
    p_control, p_treatment : float
        True response probabilities.
    n_control : int
        Maximum number of control patients; treatment gets ``n_treatment`` or
        ``round(allocation * n_control)``.
    info_rates, efficacy, futility
        As for :func:`trialdesign.simulate.simulate_trials`. Looks are placed by sample size,
        so the information fractions hold only approximately.
    allocation : float
        Ratio :math:`n_T / n_C`.
    n_sim : int
        Number of simulated trials.
    seed : int, numpy.random.SeedSequence, numpy.random.Generator or None
        Seed or generator.
    n_treatment : int, optional
        Maximum treatment patients, overriding ``allocation``.
    delta0 : float
        Difference :math:`p_T - p_C` under H0 (negative for non-inferiority).

    Returns
    -------
    OperatingCharacteristics
        With ``drift=None``; ``expected_info_fraction`` is the planned fraction at the stop.

    References
    ----------
    Jennison & Turnbull (2000), section 3.1; Farrington, C. P. and Manning, G. (1990). Test
    statistics and sample size formulae for comparative binomial trials with null hypothesis
    of non-zero risk difference or non-unity relative risk. *Statistics in Medicine* 9,
    1447-1454.
    """
    for p, name in ((p_control, "p_control"), (p_treatment, "p_treatment")):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not -1.0 < delta0 < 1.0:
        raise ValueError("delta0 must be between -1 and 1")
    t = check_info_rates(info_rates)
    c, b = _check_bounds(t, efficacy, futility)
    _check_n_sim(n_sim)
    n_t_max = _n_treatment(n_control, n_treatment, allocation)
    n_c, n_t = _look_counts(t, n_control, n_t_max)
    inc_c, inc_t = np.diff(n_c, prepend=0), np.diff(n_t, prepend=0)
    r = n_t / n_c
    rng = _rng(seed)
    tally = _Tally(t, c, b)
    for start in range(0, n_sim, _CHUNK):
        m = min(_CHUNK, n_sim - start)
        x_c = np.cumsum(rng.binomial(inc_c, p_control, size=(m, t.size)), axis=1)
        x_t = np.cumsum(rng.binomial(inc_t, p_treatment, size=(m, t.size)), axis=1)
        ph_c, ph_t = x_c / n_c, x_t / n_t
        pc0, pt0 = _fm_rates_vec(ph_c, ph_t, r, delta0)
        var = pc0 * (1.0 - pc0) / n_c + pt0 * (1.0 - pt0) / n_t
        diff = ph_t - ph_c - delta0
        z = np.divide(diff, np.sqrt(np.maximum(var, 0.0)), out=np.zeros_like(diff), where=var > 0)
        tally.add(z)
    return tally.result(None, futility is not None)


# ------------------------------------------------------------------------ whole design


def simulate_design_patients(
    design: GroupSequentialDesign,
    endpoint: Literal["normal", "binary", "survival"],
    *,
    n_control: int | None = None,
    n_treatment: int | None = None,
    delta: float | None = None,
    sd: float | None = None,
    p_control: float | None = None,
    p_treatment: float | None = None,
    delta0: float = 0.0,
    test: NormalTest = "t",
    study: SurvivalStudy | None = None,
    n_total: int | None = None,
    events: int | None = None,
    n_sim: int = 10_000,
    seed: int | None = None,
) -> DesignSimulation:
    """Patient-level operating characteristics of a design under H0 and H1.

    The patient-level counterpart of :func:`trialdesign.simulate.simulate_design`, with the
    same three scenarios: H0 with futility ignored (the type I error of a non-binding
    design), H0 with futility followed, and H1 with futility followed. H0 sits on its
    boundary: the treatment effect equals ``delta0`` (normal, binary) or HR = 1 (survival).
    All three scenarios start from the same seed.

    Parameters
    ----------
    design : GroupSequentialDesign
        Boundaries to apply.
    endpoint : {"normal", "binary", "survival"}
        Which simulator to run.
    n_control, n_treatment : int
        Maximum patients per arm (normal, binary).
    delta, sd : float
        Design difference and standard deviation (normal).
    p_control, p_treatment : float
        Design response rates (binary).
    delta0 : float
        Difference under H0 (normal, binary).
    test : {"t", "z"}
        Reference distribution for the normal statistic.
    study : SurvivalStudy
        Accrual and event model (survival); its hazard ratio is the H1 value.
    n_total, events : int, optional
        Patients and final events (survival); default to the study's, rounded up.
    n_sim : int
        Simulated trials per scenario.
    seed : int, optional
        Seed; if None, fresh entropy is drawn and stored in the result.

    Returns
    -------
    DesignSimulation
    """
    ss = np.random.SeedSequence(seed)
    entropy = int(ss.entropy)
    t, c, b = design.info_rates, design.efficacy, design.futility

    def run(futility: NDArray | None, h1: bool) -> OperatingCharacteristics:
        rng_seed = np.random.SeedSequence(entropy)
        if endpoint == "normal":
            if n_control is None or delta is None or sd is None:
                raise ValueError("the normal endpoint needs n_control, delta and sd")
            return simulate_normal(
                delta if h1 else delta0,
                sd,
                n_control,
                t,
                c,
                futility,
                n_treatment=n_treatment,
                delta0=delta0,
                test=test,
                n_sim=n_sim,
                seed=rng_seed,
            )
        if endpoint == "binary":
            if n_control is None or p_control is None or p_treatment is None:
                raise ValueError("the binary endpoint needs n_control, p_control and p_treatment")
            p_t = p_treatment if h1 else p_control + delta0
            if not 0.0 <= p_t <= 1.0:
                raise ValueError("p_control + delta0 must lie in [0, 1]")
            return simulate_binary(
                p_control,
                p_t,
                n_control,
                t,
                c,
                futility,
                n_sim=n_sim,
                seed=rng_seed,
                n_treatment=n_treatment,
                delta0=delta0,
            )
        if endpoint == "survival":
            if study is None:
                raise ValueError(
                    "the patient-level survival simulator needs an accrual model "
                    "(median control, accrual and follow-up)"
                )
            return simulate_survival(
                study,
                t,
                c,
                futility,
                hazard_ratio=None if h1 else 1.0,
                n_total=n_total,
                events=events,
                n_sim=n_sim,
                seed=int(rng_seed.generate_state(1)[0]),
            ).oc
        raise ValueError(f'endpoint must be "normal", "binary" or "survival", got {endpoint!r}')

    h0_ignore = run(None, False)
    h0 = h0_ignore if b is None else run(b, False)
    return DesignSimulation(
        design=design, seed=entropy, h0_ignore_futility=h0_ignore, h0=h0, h1=run(b, True)
    )


# ----------------------------------------------------------- survival: event re-estimation


@dataclass(frozen=True)
class SurvivalSSRSimulation:
    """Patient-level operating characteristics of an event-driven adaptive survival trial.

    Every rate and mean carries its Monte Carlo SE. ``reject_prob_naive`` uses the pooled
    log-rank statistic at the final analysis instead of the CHW combination, to show the
    inflation that re-estimating events causes without the weights.
    """

    hazard_ratio: float
    n_sim: int
    seed: int
    reject_prob: float
    reject_prob_se: float
    reject_prob_naive: float
    reject_prob_naive_se: float
    prob_zone: dict[str, float]
    expected_events: float
    expected_events_se: float
    expected_duration: float
    expected_duration_se: float
    expected_patients: float
    expected_patients_se: float
    events_not_reached: float


def _cut_at(sorted_cal: NDArray, last: NDArray, k: NDArray) -> NDArray:
    """Calendar time of the ``k``-th event per trial, or the last observation if never reached."""
    tau = np.take_along_axis(sorted_cal, (k - 1)[:, None], axis=1)[:, 0]
    return np.where(np.isfinite(tau), tau, last)


def _score_at(
    tau: NDArray,
    u: NDArray,
    t_obs: NDArray,
    has_event: NDArray,
    t_event: NDArray,
    arm: NDArray,
) -> tuple[NDArray, NDArray]:
    """Log-rank score and variance of each trial cut at its calendar time ``tau``."""
    cut = tau[:, None]
    in_trial = u < cut
    follow = cut - u
    time_k = np.where(in_trial, np.minimum(t_obs, follow), -1.0)
    event_k = in_trial & has_event & (t_event <= follow)
    return logrank_score(time_k, event_k, arm)


def simulate_survival_ssr(
    adaptive: SurvivalSSR,
    hazard_ratio: float | None = None,
    *,
    use_futility: bool = True,
    entry: Entry = "fixed",
    block_multiplier: int = 2,
    n_sim: int = 10_000,
    seed: int | None = None,
) -> SurvivalSSRSimulation:
    r"""Simulate patients in a survival trial that re-estimates its events at the interim.

    The interim is taken at the calendar time of the :math:`D_1`-th event. Its log-rank
    statistic :math:`Z_1` decides the zone (:meth:`trialdesign.ssr.SSRDesign.zone`) and, for
    a continuing trial, the new total :math:`D_2`
    (:meth:`trialdesign.ssr.SurvivalSSR.total_events`).
    The final analysis is at the :math:`D_2`-th event, with the stage-2 statistic from the
    log-rank score increment,

    .. math:: Z_2^\ast = \frac{S(\tau_2) - S(\tau_1)}{\sqrt{V(\tau_2) - V(\tau_1)}}, \qquad
              Z_{\mathrm{CHW}} = \sqrt{t_1}\,Z_1 + \sqrt{1 - t_1}\,Z_2^\ast,

    rejecting H0 when :math:`Z_{\mathrm{CHW}} \ge c_2`. The naive test compares
    :math:`S(\tau_2)/\sqrt{V(\tau_2)}` with :math:`c_2`. Only ``extend="followup"`` is
    simulated: the enrolled patients are followed until the new number of events.

    Parameters
    ----------
    adaptive : SurvivalSSR
        The design, from :func:`trialdesign.ssr.survival_ssr`.
    hazard_ratio : float, optional
        True hazard ratio; defaults to the study's design value. Use 1 for the type I error.
    use_futility : bool
        Whether interim futility stops are made (ignore them for the type I error of a
        non-binding design).
    entry, block_multiplier
        As for :func:`simulate_survival`.
    n_sim : int
        Number of simulated trials.
    seed : int, optional
        Seed; if None, fresh entropy is drawn and stored in the result.

    Returns
    -------
    SurvivalSSRSimulation

    References
    ----------
    Tsiatis (1981); Cui, Hung & Wang (1999); Schäfer and Müller (2001); Wassmer & Brannath
    (2016), section 4.4.
    """
    if adaptive.extend != "followup":
        raise ValueError('the patient-level simulation supports extend="followup" only')
    if entry not in ("fixed", "random"):
        raise ValueError("entry must be 'fixed' or 'random'")
    _check_n_sim(n_sim)
    study, d = adaptive.study, adaptive.design
    hr = study.hazard_ratio if hazard_ratio is None else float(hazard_ratio)
    if not hr > 0:
        raise ValueError("hazard_ratio must be positive")
    n = adaptive.n_total
    d1, d_max = adaptive.interim_events, adaptive.max_events
    if d_max > n:
        raise ValueError(f"the capped {d_max} events cannot come from {n} patients")
    block = _block(study.allocation, block_multiplier)
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)
    ends = study.accrual_times
    rates = study.accrual_rates
    fixed_entry = entry_times(n, ends, rates, "fixed") if entry == "fixed" else None
    lam_c, eta = study.hazard_control, study.dropout
    lam_t = hr * lam_c
    sqrt_t1, sqrt_t2 = math.sqrt(d.t1), math.sqrt(1.0 - d.t1)

    acc = {k: _Moments() for k in ("rej", "naive", "events", "duration", "patients")}
    zone_counts = {z: 0 for z in ZONES}
    not_reached = 0
    chunk = max(1, _CELLS // n)
    for start in range(0, n_sim, chunk):
        m = min(chunk, n_sim - start)
        u = (
            np.broadcast_to(fixed_entry, (m, n))
            if fixed_entry is not None
            else entry_times(n, ends, rates, "random", m, rng)
        )
        arm = _arms(n, block, m, rng)
        lam = np.where(arm == 1, lam_t, lam_c)
        t_event = rng.exponential(1.0, (m, n)) / lam
        t_drop = rng.exponential(1.0, (m, n)) / eta if eta > 0 else np.full((m, n), np.inf)
        has_event = t_event <= t_drop
        t_obs = np.minimum(t_event, t_drop)
        cal_event = np.where(has_event, u + t_event, np.inf)
        sorted_cal = np.sort(cal_event, axis=1)
        last = np.max(u + t_obs, axis=1)

        tau1 = _cut_at(sorted_cal, last, np.full(m, d1))
        s1, v1 = _score_at(tau1, u, t_obs, has_event, t_event, arm)
        z1 = np.divide(s1, np.sqrt(v1), out=np.zeros(m), where=v1 > 0)
        labels = d.zone(z1, use_futility)
        for z in ZONES:
            zone_counts[z] += int(np.sum(labels == z))
        stop = (labels == "efficacy") | (labels == "futility")
        d2 = np.where(stop, d1, adaptive.total_events(z1))
        tau2 = _cut_at(sorted_cal, last, d2)
        kth = np.take_along_axis(sorted_cal, (d2 - 1)[:, None], axis=1)[:, 0]
        not_reached += int(np.sum(~np.isfinite(kth)))
        s2, v2 = _score_at(tau2, u, t_obs, has_event, t_event, arm)
        dv = v2 - v1
        z2 = np.divide(s2 - s1, np.sqrt(np.maximum(dv, 0.0)), out=np.zeros(m), where=dv > 0)
        chw = sqrt_t1 * z1 + sqrt_t2 * z2
        z_pool = np.divide(s2, np.sqrt(v2), out=np.zeros(m), where=v2 > 0)
        eff = labels == "efficacy"
        cont = ~stop
        acc["rej"].add((eff | (cont & (chw >= d.c2))).astype(float))
        acc["naive"].add((eff | (cont & (z_pool >= d.c2))).astype(float))
        acc["events"].add(np.minimum(d2, (cal_event <= tau2[:, None]).sum(axis=1)).astype(float))
        acc["duration"].add(tau2)
        acc["patients"].add((u < tau2[:, None]).sum(axis=1).astype(float))

    out = {k: tuple(float(x) for x in v.mean_se()) for k, v in acc.items()}
    return SurvivalSSRSimulation(
        hazard_ratio=hr,
        n_sim=n_sim,
        seed=int(ss.entropy),
        reject_prob=out["rej"][0],
        reject_prob_se=out["rej"][1],
        reject_prob_naive=out["naive"][0],
        reject_prob_naive_se=out["naive"][1],
        prob_zone={z: c / n_sim for z, c in zone_counts.items()},
        expected_events=out["events"][0],
        expected_events_se=out["events"][1],
        expected_duration=out["duration"][0],
        expected_duration_se=out["duration"][1],
        expected_patients=out["patients"][0],
        expected_patients_se=out["patients"][1],
        events_not_reached=not_reached / n_sim,
    )
