r"""Blinded sample size re-estimation from an internal pilot (Kieser and Friede 2003).

A two-arm trial with a normal endpoint is planned with an assumed SD :math:`\sigma_0`. After
an internal pilot of :math:`n_1` patients (control :math:`n_{1C}`, treatment
:math:`n_{1T}`), the SD is re-estimated **without unblinding** from the one-sample
(lumped) variance of all pilot observations,

.. math:: s^2_{OS} = \frac{1}{n_1 - 1} \sum_{i=1}^{n_1} (y_i - \bar y)^2 ,

which overestimates the within-group variance by :math:`n_{1C} n_{1T}\,\theta^2 /
(n_1 (n_1 - 1))` when the true difference is :math:`\theta`. The optional *adjusted*
estimate subtracts this bias at the design difference :math:`\Delta`,
:math:`s^2_{\mathrm{adj}} = s^2_{OS} - n_{1C} n_{1T} \Delta^2 / (n_1 (n_1 - 1))` (Kieser and
Friede 2003, section 2; :math:`\Delta^2 n_1 / (4 (n_1 - 1))` with equal allocation).

The control-arm size is recomputed with the normal-approximation formula of
:func:`trialdesign.fixed.sample_size_normal`,

.. math:: n_C(s^2) = \left\lceil (1 + 1/r)\, s^2 (z_{1-\alpha} + z_{1-\beta})^2 / \Delta^2
          \right\rceil ,

and bounded below and above: :math:`n_C = \max(n_{\min}, \min(n_{\max}, n_C(s^2)))`. The
lower bound is the planned size (``floor="planned"``, the sample size never falls) or the
pilot (``floor="pilot"``, Kieser and Friede's restricted design); the upper bound is
``max_factor`` times the planned size. The treatment arm has :math:`\lceil r\,n_C\rceil`.
At the end the one-sided two-sample t-test at level :math:`\alpha` is applied to all
patients, ignoring the re-estimation.

The blinded rule changes the sample size only through a statistic that is nearly
independent of the treatment difference, so the type I error stays close to
:math:`\alpha`: Kieser and Friede report inflation that is negligible except for very small
pilots. :meth:`BlindedSSR.simulate` estimates the type I error, power and sample size
distribution by Monte Carlo, drawing the sufficient statistics of each arm and stage.

References
----------
Kieser, M. and Friede, T. (2003). Simple procedures for blinded sample size adjustment that
do not affect the type I error rate. *Statistics in Medicine* 22, 3571-3581.
Friede, T. and Kieser, M. (2006). Sample size recalculation in internal pilot study
designs: a review. *Biometrical Journal* 48, 537-555.
Wittes, J. and Brittain, E. (1990). The role of internal pilot studies in increasing the
efficiency of clinical trials. *Statistics in Medicine* 9, 65-72.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

__all__ = [
    "BlindedReestimate",
    "BlindedSSR",
    "BlindedSSRSimulation",
    "lumped_variance",
]

Floor = Literal["planned", "pilot"]


def lumped_variance(values: ArrayLike) -> float:
    """One-sample variance of the pooled, blinded pilot observations (divisor n - 1)."""
    y = np.asarray(values, dtype=float).ravel()
    if y.size < 2:
        raise ValueError("the pilot needs at least two observations")
    return float(np.var(y, ddof=1))


@dataclass(frozen=True)
class BlindedReestimate:
    """The re-estimated sample size for one observed pilot variance."""

    variance_lumped: float
    variance_used: float
    n_control_required: int
    n_control: int
    n_treatment: int
    n_total: int
    at_floor: bool
    at_cap: bool


@dataclass(frozen=True)
class BlindedSSRSimulation:
    """Monte Carlo operating characteristics of the blinded re-estimation rule."""

    delta: float
    sd: float
    n_sim: int
    seed: int | None
    rejection_rate: float
    rejection_se: float
    mean_n_total: float
    n_total_quantiles: tuple[float, float, float]
    prob_at_floor: float
    prob_at_cap: float


@dataclass(frozen=True)
class BlindedSSR:
    r"""Blinded sample size re-estimation with an internal pilot.

    Parameters
    ----------
    delta : float
        Design difference in means, treatment minus control (its sign sets the direction of
        the one-sided test).
    sd : float
        SD assumed at planning.
    pilot_n : int
        Patients in the internal pilot, split by the allocation ratio.
    alpha_one_sided : float
        Level of the final one-sided t-test.
    power : float
        Target power, used in the planned and the re-estimated sample size.
    allocation : float
        :math:`r = n_T / n_C`.
    max_factor : float
        Cap on the control-arm size as a multiple of the planned one.
    adjusted : bool
        Subtract the bias of the lumped variance at ``delta``.
    floor : {"planned", "pilot"}
        Lower bound on the re-estimated size: the planned size or the pilot.
    """

    delta: float
    sd: float
    pilot_n: int
    alpha_one_sided: float = 0.025
    power: float = 0.80
    allocation: float = 1.0
    max_factor: float = 2.0
    adjusted: bool = False
    floor: Floor = "planned"

    def __post_init__(self) -> None:
        if self.delta == 0 or not math.isfinite(self.delta):
            raise ValueError("delta must be non-zero")
        if not self.sd > 0:
            raise ValueError("sd must be positive")
        if not 0.0 < self.alpha_one_sided < 0.5:
            raise ValueError("alpha_one_sided must be between 0 and 0.5")
        if not self.alpha_one_sided < self.power < 1.0:
            raise ValueError("power must be between alpha_one_sided and 1")
        if not self.allocation > 0:
            raise ValueError("allocation (n_T / n_C) must be positive")
        if self.floor not in ("planned", "pilot"):
            raise ValueError('floor must be "planned" or "pilot"')
        if int(self.pilot_n) != self.pilot_n:
            raise ValueError("pilot_n must be a whole number")
        if self.pilot_control < 2 or self.pilot_treatment < 2:
            raise ValueError("the pilot needs at least two patients per arm")
        if not self.max_factor >= 1.0:
            raise ValueError("max_factor must be at least 1")
        if self.n_control_cap < self.n_control_floor:
            raise ValueError(
                "the cap (max_factor times the planned size) is below the pilot; "
                "raise max_factor or use a smaller pilot"
            )

    # ------------------------------------------------------------------ sizes

    @property
    def pilot_control(self) -> int:
        return round(self.pilot_n / (1 + self.allocation))

    @property
    def pilot_treatment(self) -> int:
        return int(self.pilot_n) - self.pilot_control

    @property
    def _k(self) -> float:
        """n_C per unit variance: (1 + 1/r) (z_a + z_b)^2 / delta^2."""
        z = stats.norm.isf(self.alpha_one_sided) + stats.norm.isf(1 - self.power)
        return (1 + 1 / self.allocation) * z**2 / self.delta**2

    def _treatment(self, n_c: NDArray | int) -> NDArray | int:
        return np.ceil(self.allocation * np.asarray(n_c) - 1e-9).astype(int)

    @property
    def planned_n_control(self) -> int:
        """Control-arm size at the planning SD (normal approximation, rounded up)."""
        return math.ceil(self._k * self.sd**2 - 1e-9)

    @property
    def planned_n_total(self) -> int:
        return self.planned_n_control + int(self._treatment(self.planned_n_control))

    @property
    def n_control_floor(self) -> int:
        pilot = max(self.pilot_control, math.ceil(self.pilot_treatment / self.allocation - 1e-9))
        if self.floor == "pilot":
            return pilot
        return max(pilot, self.planned_n_control)

    @property
    def n_control_cap(self) -> int:
        return math.floor(self.max_factor * self.planned_n_control + 1e-9)

    @property
    def bias_correction(self) -> float:
        """Amount subtracted from the lumped variance when ``adjusted``."""
        n1, n1c, n1t = self.pilot_n, self.pilot_control, self.pilot_treatment
        return n1c * n1t * self.delta**2 / (n1 * (n1 - 1))

    def _n_control(self, s2: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        used = s2 - self.bias_correction if self.adjusted else s2
        used = np.maximum(used, 0.0)
        required = np.ceil(self._k * used - 1e-9).astype(int)
        n_c = np.clip(required, self.n_control_floor, self.n_control_cap)
        return used, required, n_c, self._treatment(n_c)

    def reestimate(self, variance_lumped: float) -> BlindedReestimate:
        """Sample size after the pilot, from the observed lumped (one-sample) variance."""
        if not variance_lumped >= 0:
            raise ValueError("the pilot variance must be non-negative")
        used, req, n_c, n_t = self._n_control(np.array([float(variance_lumped)]))
        c, t = int(n_c[0]), int(n_t[0])
        return BlindedReestimate(
            variance_lumped=float(variance_lumped),
            variance_used=float(used[0]),
            n_control_required=int(req[0]),
            n_control=c,
            n_treatment=t,
            n_total=c + t,
            at_floor=bool(req[0] <= self.n_control_floor),
            at_cap=bool(req[0] >= self.n_control_cap),
        )

    # ------------------------------------------------------------- simulation

    def simulate(
        self,
        delta: float | None = None,
        sd: float | None = None,
        n_sim: int = 100_000,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> BlindedSSRSimulation:
        """Simulate the trial at true difference ``delta`` and SD ``sd`` (defaults: design).

        Per arm and stage the mean is normal and the within-arm sum of squares is
        :math:`\\sigma^2\\chi^2_{n-1}`, which is exactly the distribution of the patient data's
        sufficient statistics. The final test is the one-sided pooled-variance t-test.
        """
        theta = self.delta if delta is None else float(delta)
        sigma = self.sd if sd is None else float(sd)
        if not sigma > 0:
            raise ValueError("sd must be positive")
        if not n_sim >= 1:
            raise ValueError("n_sim must be at least 1")
        rng = np.random.default_rng(seed) if rng is None else rng
        m = int(n_sim)
        n1c, n1t = self.pilot_control, self.pilot_treatment
        n1 = n1c + n1t

        def stage(n: NDArray | int, mean: float) -> tuple[NDArray, NDArray]:
            n = np.broadcast_to(np.asarray(n, dtype=float), (m,))
            safe = np.maximum(n, 1.0)
            xbar = mean + sigma * rng.standard_normal(m) / np.sqrt(safe)
            ss = sigma**2 * rng.gamma(np.maximum(n - 1.0, 0.0) / 2.0, 2.0)
            return np.where(n > 0, xbar, 0.0), ss

        mc1, ssc1 = stage(n1c, 0.0)
        mt1, sst1 = stage(n1t, theta)
        grand = (n1c * mc1 + n1t * mt1) / n1
        s2 = (ssc1 + sst1 + n1c * (mc1 - grand) ** 2 + n1t * (mt1 - grand) ** 2) / (n1 - 1)
        _, req, n_c, n_t = self._n_control(s2)
        m_c, m_t = n_c - n1c, n_t - n1t
        mc2, ssc2 = stage(m_c, 0.0)
        mt2, sst2 = stage(m_t, theta)
        mean_c = (n1c * mc1 + m_c * mc2) / n_c
        mean_t = (n1t * mt1 + m_t * mt2) / n_t
        ss = ssc1 + ssc2 + sst1 + sst2
        ss += n1c * m_c / n_c * (mc1 - mc2) ** 2 + n1t * m_t / n_t * (mt1 - mt2) ** 2
        df = n_c + n_t - 2
        se = np.sqrt(ss / df * (1 / n_c + 1 / n_t))
        tstat = np.sign(self.delta) * (mean_t - mean_c) / se
        reject = tstat >= stats.t.isf(self.alpha_one_sided, df)
        rate = float(reject.mean())
        total = n_c + n_t
        q = np.quantile(total, [0.05, 0.5, 0.95])
        return BlindedSSRSimulation(
            delta=theta,
            sd=sigma,
            n_sim=m,
            seed=seed,
            rejection_rate=rate,
            rejection_se=math.sqrt(rate * (1 - rate) / m),
            mean_n_total=float(total.mean()),
            n_total_quantiles=(float(q[0]), float(q[1]), float(q[2])),
            prob_at_floor=float(np.mean(req <= self.n_control_floor)),
            prob_at_cap=float(np.mean(req >= self.n_control_cap)),
        )

    def operating_characteristics(
        self,
        sd_values: Sequence[float] | None = None,
        n_sim: int = 100_000,
        seed: int | None = None,
    ) -> list[tuple[BlindedSSRSimulation, BlindedSSRSimulation]]:
        """Simulated (H0, H1) pairs for each true SD in ``sd_values``.

        Default true SDs are 0.75, 1, 1.25 and 1.5 times the planning SD. One seed drives all
        runs through a shared generator, so the table is reproducible.
        """
        if sd_values is None:
            sd_values = [f * self.sd for f in (0.75, 1.0, 1.25, 1.5)]
        rng = np.random.default_rng(seed)
        out = []
        for s in sd_values:
            h0 = self.simulate(0.0, s, n_sim, seed, rng=rng)
            h1 = self.simulate(self.delta, s, n_sim, seed, rng=rng)
            out.append((h0, h1))
        return out
