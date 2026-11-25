r"""Inference at the end of a group-sequential trial.

Works on the canonical scale of :mod:`trialdesign.boundaries`: the z-statistics :math:`Z_k`
at information fractions :math:`t_k` have :math:`E(Z_k) = \delta \sqrt{t_k}`, where the drift
:math:`\delta` is the mean of the final z-statistic. The effect is linear in the drift,
:math:`\theta = \theta_0 + s\,\delta`, with :math:`s = 1/\sqrt{I_{\max}}` when larger
effects favour treatment (a difference in means or rates) and :math:`s = -1/\sqrt{I_{\max}}`
for a log hazard ratio. Every estimate is given on the drift scale and, with ``effect_per_drift``
(:math:`s`), on the effect scale.

Three kinds of inference, as in ``rpact``:

- **Repeated confidence intervals** (Jennison and Turnbull 1989): at look :math:`k`,
  :math:`\hat\delta_k \pm c_k / \sqrt{t_k}` with :math:`\hat\delta_k = Z_k / \sqrt{t_k}`.
  They hold their joint coverage :math:`1 - 2\alpha` whatever the stopping rule, and the
  **repeated p-value** at look :math:`k` is the smallest :math:`\alpha` whose boundary
  :math:`c_k(\alpha)` the observed :math:`Z_k` reaches.
- **Stage-wise ordering** (Armitage 1957; Fairbanks and Madsen 1982; Tsiatis, Rosner and
  Mehta 1984): an outcome is more extreme if it stops for efficacy at an earlier look, or at
  the same look with a larger z. For a trial that stops for efficacy at look :math:`k^\ast`
  with :math:`Z_{k^\ast} = z^\ast`, or reaches the last look, let

  .. math:: f(\delta) = \sum_{j < k^\ast} P_\delta(\text{continue to } j,\ Z_j \ge c_j)
            + P_\delta(\text{continue to } k^\ast,\ Z_{k^\ast} \ge z^\ast).

  The p-value is :math:`f(0)`, the median-unbiased estimate solves :math:`f(\delta) = 0.5`,
  and the :math:`1 - 2\alpha` confidence interval is :math:`f(\delta) = \alpha` and
  :math:`f(\delta) = 1 - \alpha`.

Futility is followed in the continuation region only for a binding design; for a non-binding
design it is ignored, as the type I error of the design itself is.

References
----------
Jennison, C. and Turnbull, B. W. (2000). *Group Sequential Methods with Applications to
Clinical Trials*. Chapman & Hall/CRC. Chapters 8 (stage-wise ordering) and 9 (repeated
confidence intervals). Jennison, C. and Turnbull, B. W. (1989). Interim analyses: the repeated
confidence interval approach. *JRSS B* 51, 305-361. Tsiatis, A. A., Rosner, G. L. and Mehta,
C. R. (1984). Exact confidence intervals following a group sequential test. *Biometrics* 40,
797-803. Wassmer, G. and Brannath, W. (2016). *Group Sequential and Confirmatory Adaptive
Designs in Clinical Trials*. Springer. Section 4.1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize

from trialdesign.boundaries import (
    GroupSequentialDesign,
    crossing_probabilities,
    efficacy_bounds,
    group_sequential_design,
)

__all__ = [
    "GroupSequentialInference",
    "analyse",
    "median_unbiased_estimate",
    "repeated_confidence_intervals",
    "repeated_p_values",
    "stagewise_probability",
]

_P_MIN = 1e-10  # repeated p-values are searched in [_P_MIN, 0.5)
_P_MAX = 0.5 - 1e-10
_DRIFT_SPAN = 60.0


def _check_z(design: GroupSequentialDesign, z: ArrayLike) -> NDArray[np.float64]:
    z = np.atleast_1d(np.asarray(z, dtype=float))
    if z.ndim != 1 or z.size == 0 or z.size > design.n_looks:
        raise ValueError(f"give between 1 and {design.n_looks} observed z-statistics, one per look")
    if not np.all(np.isfinite(z)):
        raise ValueError("observed z-statistics must be finite")
    return z


def repeated_confidence_intervals(
    design: GroupSequentialDesign, z: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Repeated confidence intervals for the drift at each look analysed.

    .. math:: \frac{Z_k - c_k}{\sqrt{t_k}} \le \delta \le \frac{Z_k + c_k}{\sqrt{t_k}}

    Jointly they cover :math:`\delta` with probability at least :math:`1 - 2\alpha`.

    Parameters
    ----------
    design : GroupSequentialDesign
    z : array_like
        Observed z-statistics at looks 1 to :math:`k`, positive favouring treatment.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Lower and upper limits on the drift scale.

    References
    ----------
    Jennison & Turnbull (2000), section 9.1.
    """
    z = _check_z(design, z)
    k = z.size
    c, s = design.efficacy[:k], np.sqrt(design.info_rates[:k])
    return (z - c) / s, (z + c) / s


def repeated_p_values(design: GroupSequentialDesign, z: ArrayLike) -> NDArray[np.float64]:
    r"""Repeated p-values: the smallest one-sided alpha at which look :math:`k` rejects.

    Solves :math:`c_k(\alpha) = Z_k` with the design's alpha-spending function and
    information fractions; for a binding design the whole design (with its beta spending and
    power) is rebuilt at each level, as ``rpact`` does. Values are capped at 0.5 (an observed
    z at or below the boundary of a level-0.5 design) and floored at 1e-10.

    References
    ----------
    Jennison & Turnbull (2000), section 9.2.
    """
    z = _check_z(design, z)
    t, spending = design.info_rates, design.efficacy_spending

    def bounds(a: float) -> NDArray[np.float64]:
        if not design.binding_futility:
            return efficacy_bounds(t, a, spending)
        # binding: the efficacy boundaries depend on the futility ones, rebuilt at level a
        return group_sequential_design(
            t, a, design.power, spending, design.futility_spending, binding_futility=True
        ).efficacy

    out = np.empty(z.size)
    for k, zk in enumerate(z):

        def gap(a: float, k: int = k, zk: float = float(zk)) -> float:
            return float(bounds(a)[k]) - zk

        if gap(_P_MAX) >= 0:
            out[k] = 0.5
        elif gap(_P_MIN) <= 0:
            out[k] = _P_MIN
        else:
            out[k] = optimize.brentq(gap, _P_MIN, _P_MAX, xtol=1e-12, rtol=1e-10)
    return out


def stagewise_probability(design: GroupSequentialDesign, z: ArrayLike, drift: float = 0.0) -> float:
    r"""Probability, under ``drift``, of an outcome at least as extreme in stage-wise order.

    ``z`` holds the z-statistics up to the look :math:`k^\ast` where the trial stopped; the
    last is the observed :math:`z^\ast`. See the module docstring for :math:`f(\delta)`.

    References
    ----------
    Jennison & Turnbull (2000), section 8.4.
    """
    z = _check_z(design, z)
    k = z.size
    t = design.info_rates[:k] / design.info_rates[k - 1]
    eff = np.append(design.efficacy[: k - 1], z[-1])
    fut = None
    if design.binding_futility and design.futility is not None:
        fut = np.append(np.minimum(design.futility[: k - 1], eff[: k - 1]), -np.inf)
    up, _ = crossing_probabilities(t, eff, fut, drift * math.sqrt(design.info_rates[k - 1]))
    return float(up.sum())


def _solve_drift(design: GroupSequentialDesign, z: NDArray, target: float) -> float:
    centre = z[-1] / math.sqrt(design.info_rates[z.size - 1])
    lo, hi = centre - _DRIFT_SPAN, centre + _DRIFT_SPAN
    return float(
        optimize.brentq(lambda d: stagewise_probability(design, z, d) - target, lo, hi, xtol=1e-10)
    )


def median_unbiased_estimate(design: GroupSequentialDesign, z: ArrayLike) -> float:
    r"""Median-unbiased estimate of the drift under stage-wise ordering: :math:`f(\delta) = 0.5`.

    ``z`` ends at the look where the trial stopped (or the last look).

    References
    ----------
    Jennison & Turnbull (2000), section 8.3.
    """
    return _solve_drift(design, _check_z(design, z), 0.5)


@dataclass(frozen=True)
class GroupSequentialInference:
    """Inference after looks 1 to ``stage``.

    Drift-scale quantities end in ``_drift``; :meth:`to_effect` maps them to
    ``theta0 + effect_per_drift * drift`` (None when ``effect_per_drift`` was not given). The
    stage-wise p-value, median-unbiased estimate and final confidence interval are None
    unless the trial stopped for efficacy at ``stage`` or ``stage`` is the last look.
    """

    stage: int
    z: NDArray[np.float64]
    info_rates: NDArray[np.float64]
    efficacy: NDArray[np.float64]
    level: float
    crossed: bool
    final: bool
    estimate_drift: NDArray[np.float64]
    rci_lower_drift: NDArray[np.float64]
    rci_upper_drift: NDArray[np.float64]
    repeated_p: NDArray[np.float64]
    p_value: float | None
    median_unbiased_drift: float | None
    ci_lower_drift: float | None
    ci_upper_drift: float | None
    effect_per_drift: float | None
    theta0: float

    def to_effect(self, drift):
        """Drift-scale value(s) on the effect scale (None if ``effect_per_drift`` is unknown)."""
        if drift is None or self.effect_per_drift is None:
            return None
        return self.theta0 + self.effect_per_drift * np.asarray(drift)


def analyse(
    design: GroupSequentialDesign,
    z: ArrayLike,
    effect_per_drift: float | None = None,
    theta0: float = 0.0,
) -> GroupSequentialInference:
    r"""End-of-trial (or interim) inference for a group-sequential design.

    Parameters
    ----------
    design : GroupSequentialDesign
        The design whose boundaries were used. The looks are taken at its information
        fractions; if the observed information differs, rebuild the design at the observed
        fractions first (the spending function then gives the boundaries actually used).
    z : array_like
        Observed z-statistics at looks 1 to :math:`k`, positive favouring treatment.
    effect_per_drift : float, optional
        Effect per unit of drift, :math:`s` in :math:`\theta = \theta_0 + s\,\delta`:
        :math:`1/\sqrt{I_{\max}}` with the maximum information :math:`I_{\max}`, negated for
        a log hazard ratio. Given the design effect :math:`\theta_1` it is
        :math:`(\theta_1 - \theta_0) / \delta_1` with the design drift :math:`\delta_1`.
    theta0 : float
        Effect under H0 (0 for superiority, or a log hazard ratio of 0; the margin for
        non-inferiority).

    Returns
    -------
    GroupSequentialInference

    References
    ----------
    Jennison & Turnbull (2000), chapters 8 and 9; Wassmer & Brannath (2016), section 4.1.
    """
    z = _check_z(design, z)
    k = z.size
    if np.any(z[:-1] >= design.efficacy[: k - 1]):
        raise ValueError(
            "an efficacy boundary was crossed before the last look given; the trial "
            "would have stopped there"
        )
    crossed = bool(z[-1] >= design.efficacy[k - 1])
    final = k == design.n_looks
    lo, up = repeated_confidence_intervals(design, z)
    p = mu = ci_lo = ci_hi = None
    a = design.alpha_one_sided
    if crossed or final:
        p = stagewise_probability(design, z, 0.0)
        mu = _solve_drift(design, z, 0.5)
        ci_lo = _solve_drift(design, z, a)
        ci_hi = _solve_drift(design, z, 1.0 - a)
    return GroupSequentialInference(
        stage=k,
        z=z,
        info_rates=design.info_rates[:k].copy(),
        efficacy=design.efficacy[:k].copy(),
        level=1.0 - 2.0 * a,
        crossed=crossed,
        final=final,
        estimate_drift=z / np.sqrt(design.info_rates[:k]),
        rci_lower_drift=lo,
        rci_upper_drift=up,
        repeated_p=repeated_p_values(design, z),
        p_value=p,
        median_unbiased_drift=mu,
        ci_lower_drift=ci_lo,
        ci_upper_drift=ci_hi,
        effect_per_drift=None if effect_per_drift is None else float(effect_per_drift),
        theta0=float(theta0),
    )
