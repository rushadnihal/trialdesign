r"""Vectorised Monte Carlo operating characteristics of group-sequential trials.

The canonical simulation (Jennison & Turnbull 2000, section 3.1) draws the increments of the
score process rather than individual patients. With information fractions
:math:`0 < t_1 < \dots < t_K = 1` and drift :math:`\delta` (the mean of the final
z-statistic), the score :math:`S_k` has independent increments

.. math:: S_k - S_{k-1} \sim N\!\left(\delta\,(t_k - t_{k-1}),\; t_k - t_{k-1}\right),
          \qquad Z_k = S_k / \sqrt{t_k},

so :math:`E(Z_k) = \delta\sqrt{t_k}` and :math:`\operatorname{Cov}(Z_j, Z_k) = \sqrt{t_j/t_k}`
for :math:`j \le k`. One ``numpy`` call gives an ``(n_sim, K)`` matrix of z-statistics, with
no Python loop over trials.

A trial stops at the first look k with :math:`Z_k \ge c_k` (efficacy) or :math:`Z_k < b_k`
(futility). Every rate is reported with its Monte Carlo standard error
:math:`\sqrt{p(1 - p) / n_{\mathrm{sim}}}`.

:func:`simulate_binary` is a patient-level check for a binary endpoint: it draws binomial
counts per arm and look and uses the pooled-variance z-test, so it shows how far the normal
approximation behind the boundaries holds at small sample sizes.

References
----------
Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
Clinical Trials*. Chapman & Hall/CRC. Sections 3.1 (canonical joint distribution) and 7.2.
Lan, K. K. G. and DeMets, D. L. (1983). Discrete sequential boundaries for clinical trials.
*Biometrika* 70, 659-663.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from trialdesign.boundaries import GroupSequentialDesign
from trialdesign.spending import check_info_rates

__all__ = [
    "DesignSimulation",
    "OperatingCharacteristics",
    "SeedLike",
    "mc_se",
    "simulate_binary",
    "simulate_design",
    "simulate_trials",
    "simulate_z",
]

SeedLike = int | np.random.SeedSequence | np.random.Generator | None

_CHUNK = 250_000  # trials per numpy batch; bounds memory at about 2 MB per look per batch


def mc_se(p: float | NDArray[np.float64], n_sim: int) -> float | NDArray[np.float64]:
    r"""Monte Carlo standard error :math:`\sqrt{p(1-p)/n_{\mathrm{sim}}}` of a simulated rate."""
    return np.sqrt(np.asarray(p) * (1.0 - np.asarray(p)) / n_sim)


def _rng(seed: SeedLike) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _check_n_sim(n_sim: int) -> None:
    if not (isinstance(n_sim, int | np.integer) and n_sim >= 1):
        raise ValueError("n_sim must be a positive integer")


def _check_bounds(
    t: NDArray, efficacy: Sequence[float], futility: Sequence[float] | None
) -> tuple[NDArray, NDArray]:
    c = np.asarray(efficacy, dtype=float)
    b = np.full(t.size, -np.inf) if futility is None else np.asarray(futility, dtype=float)
    if c.shape != t.shape or b.shape != t.shape:
        raise ValueError("efficacy and futility must have one boundary per look")
    if np.any(b > c):
        raise ValueError("futility boundaries must not exceed efficacy boundaries")
    return c, b


# ------------------------------------------------------------------------ results


@dataclass(frozen=True)
class OperatingCharacteristics:
    r"""Simulated operating characteristics of one design under one true effect.

    Attributes
    ----------
    info_rates : numpy.ndarray
        Information fractions :math:`t_k` of the looks.
    drift : float or None
        Mean of the final z-statistic used to simulate; None for a patient-level simulation.
    n_sim : int
        Number of simulated trials.
    prob_efficacy, prob_futility : numpy.ndarray
        Per-look proportions of trials that stop for efficacy and for futility. At the last
        look, a trial that neither crosses :math:`c_K` nor falls below :math:`b_K` is counted
        in neither (this only happens when :math:`b_K < c_K`, e.g. without futility).
    futility_followed : bool
        Whether futility boundaries stopped trials in this simulation.
    expected_info_fraction : float
        Mean information fraction at which the trial stops, :math:`E(t_{\text{stop}})`.
    expected_info_fraction_se : float
        Its Monte Carlo standard error, :math:`\mathrm{sd}(t_{\text{stop}})/\sqrt{n_{sim}}`.
    """

    info_rates: NDArray[np.float64]
    drift: float | None
    n_sim: int
    prob_efficacy: NDArray[np.float64]
    prob_futility: NDArray[np.float64]
    futility_followed: bool
    expected_info_fraction: float
    expected_info_fraction_se: float

    @property
    def reject_prob(self) -> float:
        """Overall probability of rejecting H0: type I error under H0, power under H1."""
        return float(self.prob_efficacy.sum())

    @property
    def reject_prob_se(self) -> float:
        return float(mc_se(self.reject_prob, self.n_sim))

    @property
    def futility_prob(self) -> float:
        """Overall probability of stopping for futility."""
        return float(self.prob_futility.sum())

    @property
    def futility_prob_se(self) -> float:
        return float(mc_se(self.futility_prob, self.n_sim))

    @property
    def prob_efficacy_se(self) -> NDArray[np.float64]:
        return np.asarray(mc_se(self.prob_efficacy, self.n_sim))

    @property
    def prob_futility_se(self) -> NDArray[np.float64]:
        return np.asarray(mc_se(self.prob_futility, self.n_sim))

    @property
    def prob_stop(self) -> NDArray[np.float64]:
        """Per-look probability that the trial ends there; the last look takes the rest."""
        p = self.prob_efficacy + self.prob_futility
        p[-1] = 1.0 - p[:-1].sum()
        return p

    @property
    def prob_stop_se(self) -> NDArray[np.float64]:
        return np.asarray(mc_se(self.prob_stop, self.n_sim))

    def expected_sample_size(self, n_max: float) -> tuple[float, float]:
        """Expected sample size (or events) and its Monte Carlo SE, for a maximum ``n_max``.

        Assumes the looks fall at ``info_rates * n_max``, as in
        :meth:`trialdesign.boundaries.GroupSequentialDesign.look_sample_sizes`.
        """
        if not n_max > 0:
            raise ValueError("n_max must be positive")
        return n_max * self.expected_info_fraction, n_max * self.expected_info_fraction_se


class _Tally:
    """Accumulates stopping counts over batches of simulated z-statistics."""

    def __init__(self, t: NDArray, c: NDArray, b: NDArray) -> None:
        self.t, self.c, self.b = t, c, b
        self.n = 0
        self.eff = np.zeros(t.size, dtype=np.int64)
        self.fut = np.zeros(t.size, dtype=np.int64)
        self.sum_t = 0.0
        self.sum_t2 = 0.0

    def add(self, z: NDArray) -> None:
        eff = z >= self.c
        fut = z < self.b
        stop = eff | fut
        stop[:, -1] = True  # every trial ends at the last look
        k = stop.argmax(axis=1)  # first look at which the trial stops
        rows = np.arange(z.shape[0])
        size = self.t.size
        self.eff += np.bincount(k[eff[rows, k]], minlength=size)
        self.fut += np.bincount(k[fut[rows, k]], minlength=size)
        t_stop = self.t[k]
        self.sum_t += float(t_stop.sum())
        self.sum_t2 += float((t_stop * t_stop).sum())
        self.n += z.shape[0]

    def result(self, drift: float | None, futility_followed: bool) -> OperatingCharacteristics:
        n = self.n
        mean_t = self.sum_t / n
        var_t = max(self.sum_t2 / n - mean_t * mean_t, 0.0) * n / max(n - 1, 1)
        return OperatingCharacteristics(
            info_rates=self.t,
            drift=drift,
            n_sim=n,
            prob_efficacy=self.eff / n,
            prob_futility=self.fut / n,
            futility_followed=futility_followed,
            expected_info_fraction=mean_t,
            expected_info_fraction_se=math.sqrt(var_t / n),
        )


# ------------------------------------------------------------------ canonical simulation


def simulate_z(
    info_rates: int | Sequence[float],
    drift: float,
    n_sim: int,
    seed: SeedLike = None,
) -> NDArray[np.float64]:
    r"""Simulate z-statistics at every look from the canonical joint distribution.

    Draws independent increments :math:`N(\delta\,\Delta t_k, \Delta t_k)` of the score
    process, cumulates them and divides by :math:`\sqrt{t_k}`.

    Parameters
    ----------
    info_rates : int or sequence of float
        Information fractions, increasing and ending at 1.0, or the number of equally spaced
        looks.
    drift : float
        Mean of the final z-statistic, :math:`\delta`; 0 under H0.
    n_sim : int
        Number of simulated trials (rows).
    seed : int, numpy.random.SeedSequence, numpy.random.Generator or None
        Seed or generator. Chunked calls on one generator give the same stream as one call.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_sim, K)``; column k holds :math:`Z_k`.

    References
    ----------
    Jennison & Turnbull (2000), section 3.1.
    """
    t = check_info_rates(info_rates)
    _check_n_sim(n_sim)
    dt = np.diff(t, prepend=0.0)
    inc = _rng(seed).standard_normal((n_sim, t.size))
    inc *= np.sqrt(dt)
    inc += drift * dt
    return np.cumsum(inc, axis=1) / np.sqrt(t)


def simulate_trials(
    info_rates: int | Sequence[float],
    efficacy: Sequence[float],
    futility: Sequence[float] | None = None,
    drift: float = 0.0,
    n_sim: int = 100_000,
    seed: SeedLike = None,
) -> OperatingCharacteristics:
    r"""Simulate whole group-sequential trials and tally when and why they stop.

    Parameters
    ----------
    info_rates : int or sequence of float
        Information fractions of the looks, or the number of equally spaced looks.
    efficacy : sequence of float
        Efficacy z-boundaries :math:`c_k`.
    futility : sequence of float, optional
        Futility z-boundaries :math:`b_k \le c_k`. None means futility is ignored, which is
        how the type I error of a non-binding design is defined.
    drift : float
        Mean of the final z-statistic, :math:`\delta`; 0 under H0.
    n_sim : int
        Number of simulated trials.
    seed : int, numpy.random.SeedSequence, numpy.random.Generator or None
        Seed or generator, for reproducibility.

    Returns
    -------
    OperatingCharacteristics
        Per-look stopping probabilities, rejection probability and expected information,
        each with a Monte Carlo standard error.

    References
    ----------
    Jennison & Turnbull (2000), sections 3.1 and 7.2.
    """
    t = check_info_rates(info_rates)
    c, b = _check_bounds(t, efficacy, futility)
    _check_n_sim(n_sim)
    rng = _rng(seed)
    tally = _Tally(t, c, b)
    for start in range(0, n_sim, _CHUNK):
        tally.add(simulate_z(t, drift, min(_CHUNK, n_sim - start), rng))
    return tally.result(drift, futility is not None)


# ------------------------------------------------------------------- whole design


@dataclass(frozen=True)
class DesignSimulation:
    """Operating characteristics of a group-sequential design under H0 and H1.

    Attributes
    ----------
    design : GroupSequentialDesign
        The simulated design.
    seed : int or None
        Entropy of the seed sequence used, so the run can be repeated.
    h0_ignore_futility : OperatingCharacteristics
        Under H0 with futility ignored: ``reject_prob`` is the type I error that a regulator
        will ask about for a non-binding design.
    h0 : OperatingCharacteristics
        Under H0 with futility followed: rejection rate, futility stops and expected
        information when the sponsor does stop for futility.
    h1 : OperatingCharacteristics
        Under the design alternative with futility followed: power and expected information.
    """

    design: GroupSequentialDesign
    seed: int | None
    h0_ignore_futility: OperatingCharacteristics
    h0: OperatingCharacteristics
    h1: OperatingCharacteristics

    @property
    def type_i_error(self) -> float:
        return self.h0_ignore_futility.reject_prob

    @property
    def power(self) -> float:
        return self.h1.reject_prob


def simulate_design(
    design: GroupSequentialDesign,
    n_sim: int = 100_000,
    seed: int | None = None,
    drift_h1: float | None = None,
) -> DesignSimulation:
    """Simulate a design's operating characteristics under H0 and H1.

    All three scenarios reuse the same standard-normal draws (common random numbers), so
    differences between them carry less Monte Carlo noise than independent runs.

    Parameters
    ----------
    design : GroupSequentialDesign
        From :func:`trialdesign.boundaries.group_sequential_design`.
    n_sim : int
        Number of simulated trials per scenario.
    seed : int, optional
        Seed. If None, fresh entropy is drawn and stored in the result.
    drift_h1 : float, optional
        Drift under H1; defaults to the design drift that gives the target power.

    Returns
    -------
    DesignSimulation

    References
    ----------
    Jennison & Turnbull (2000), sections 3.1 and 7.2.
    """
    ss = np.random.SeedSequence(seed)
    entropy = int(ss.entropy)
    t, c, b = design.info_rates, design.efficacy, design.futility
    d1 = design.drift if drift_h1 is None else drift_h1

    def run(futility: NDArray | None, drift: float) -> OperatingCharacteristics:
        return simulate_trials(t, c, futility, drift, n_sim, np.random.SeedSequence(entropy))

    h0_ignore = run(None, 0.0)
    h0 = h0_ignore if b is None else run(b, 0.0)
    return DesignSimulation(
        design=design,
        seed=entropy,
        h0_ignore_futility=h0_ignore,
        h0=h0,
        h1=run(b, d1),
    )


# ------------------------------------------------------------- patient-level (binary)


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
) -> OperatingCharacteristics:
    r"""Patient-level simulation of a two-arm binary trial with group-sequential looks.

    At look k each arm has :math:`n_{ik} = \operatorname{round}(t_k n_i)` patients, with
    binomial successes drawn per look increment. The test statistic is the pooled-variance
    z-test (the score test behind ``prop.test`` without continuity correction),

    .. math:: Z_k = \frac{\hat p_{T} - \hat p_{C}}
              {\sqrt{\bar p (1 - \bar p)\,(1/n_{Ck} + 1/n_{Tk})}},

    set to 0 when :math:`\bar p \in \{0, 1\}`. Positive z favours treatment.

    Parameters
    ----------
    p_control, p_treatment : float
        True response probabilities.
    n_control : int
        Maximum number of control patients; treatment gets ``round(allocation * n_control)``.
    info_rates, efficacy, futility
        As for :func:`simulate_trials`. Looks are placed by sample size, so the information
        fractions hold only approximately.
    allocation : float
        Ratio :math:`n_T / n_C`.
    n_sim : int
        Number of simulated trials.
    seed : int, numpy.random.SeedSequence, numpy.random.Generator or None
        Seed or generator.

    Returns
    -------
    OperatingCharacteristics
        With ``drift=None``; ``expected_info_fraction`` is the planned fraction at the stop.

    References
    ----------
    Jennison & Turnbull (2000), section 3.1 (binary responses as an approximately canonical
    process); Fleiss, Levin and Paik (2003), *Statistical Methods for Rates and Proportions*,
    section 4.2.
    """
    for p, name in ((p_control, "p_control"), (p_treatment, "p_treatment")):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1]")
    if not allocation > 0:
        raise ValueError("allocation must be positive")
    t = check_info_rates(info_rates)
    c, b = _check_bounds(t, efficacy, futility)
    _check_n_sim(n_sim)
    n_c = np.rint(t * n_control).astype(np.int64)
    n_t = np.rint(t * round(allocation * n_control)).astype(np.int64)
    if n_c[0] < 1 or n_t[0] < 1 or np.any(np.diff(n_c) < 1) or np.any(np.diff(n_t) < 1):
        raise ValueError("each look needs at least one new patient per arm")
    inc_c, inc_t = np.diff(n_c, prepend=0), np.diff(n_t, prepend=0)
    rng = _rng(seed)
    tally = _Tally(t, c, b)
    for start in range(0, n_sim, _CHUNK):
        m = min(_CHUNK, n_sim - start)
        x_c = np.cumsum(rng.binomial(inc_c, p_control, size=(m, t.size)), axis=1)
        x_t = np.cumsum(rng.binomial(inc_t, p_treatment, size=(m, t.size)), axis=1)
        p_bar = (x_c + x_t) / (n_c + n_t)
        var = p_bar * (1.0 - p_bar) * (1.0 / n_c + 1.0 / n_t)
        diff = x_t / n_t - x_c / n_c
        z = np.divide(diff, np.sqrt(var), out=np.zeros_like(diff), where=var > 0)
        tally.add(z)
    return tally.result(None, futility is not None)
