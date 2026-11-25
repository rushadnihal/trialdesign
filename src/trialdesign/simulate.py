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

Patient-level simulators for all three endpoints are in :mod:`trialdesign.patients`;
:func:`simulate_binary` stays here as a thin wrapper so v0.2 code keeps working.

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
        will ask about for a non-binding design (for a binding design it shows how far alpha
        rises if futility is not followed).
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
    def _h0_alpha(self) -> OperatingCharacteristics:
        return self.h0 if self.design.binding_futility else self.h0_ignore_futility

    @property
    def type_i_error(self) -> float:
        """Futility ignored for a non-binding design, followed for a binding one."""
        return self._h0_alpha.reject_prob

    @property
    def type_i_error_se(self) -> float:
        return self._h0_alpha.reject_prob_se

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
    """Patient-level binary simulation; kept here for v0.2 code.

    Moved to :func:`trialdesign.patients.simulate_binary`, which also takes ``delta0``
    (non-inferiority) and ``n_treatment``.
    """
    from trialdesign.patients import simulate_binary as _simulate_binary

    return _simulate_binary(
        p_control, p_treatment, n_control, info_rates, efficacy, futility, allocation, n_sim, seed
    )
