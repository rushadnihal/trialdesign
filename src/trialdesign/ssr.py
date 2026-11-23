r"""Unblinded sample size re-estimation at one interim: conditional power and the promising zone.

A two-look group-sequential design (interim at information fraction :math:`t_1`, efficacy
boundaries :math:`c_1, c_2`, optional non-binding futility :math:`b_1`) is extended with a
rule that may **increase** the second-stage sample size after an unblinded look at the
interim z-statistic :math:`Z_1`.

**Final test.** Stage 2 patients alone give :math:`Z_2^\ast`. The final statistic combines
the stages with weights fixed by the *planned* information (Cui, Hung and Wang 1999):

.. math:: Z_{\mathrm{CHW}} = \sqrt{t_1}\,Z_1 + \sqrt{1 - t_1}\,Z_2^\ast .

Under H0, :math:`Z_2^\ast \sim N(0, 1)` independently of :math:`Z_1` whatever the stage-2
sample size, so :math:`(Z_1, Z_{\mathrm{CHW}})` has the same null distribution as the
unadapted design and the group-sequential boundaries keep the type I error at
``alpha_one_sided``. The naive statistic that pools all patients does not; see
:meth:`SSRDesign.operating_characteristics` with ``test="naive"``.

**Conditional power.** With drift :math:`\delta` (the mean of the final z-statistic of the
planned design) and a stage-2 sample size :math:`m` times the planned one,

.. math:: \mathrm{CP}(z_1; \delta, m) = 1 - \Phi\!\left(\frac{c_2 - \sqrt{t_1}\,z_1}{\sqrt{1-t_1}}
          - \delta\sqrt{(1 - t_1)\,m}\right).

Under the observed trend :math:`\hat\delta = z_1 / \sqrt{t_1}`.

**Promising zone** (Mehta and Pocock 2011). Let :math:`\mathrm{CP}_1` be the conditional
power at the observed trend with the planned stage 2 (:math:`m = 1`). If
:math:`\mathrm{CP}_{\min} \le \mathrm{CP}_1 < \mathrm{CP}_{\text{target}}` (the promising zone),
choose the smallest :math:`m \ge 1` with :math:`\mathrm{CP}(z_1; \hat\delta, m) =
\mathrm{CP}_{\text{target}}`, capped so the total sample size is at most ``max_factor`` times
the planned maximum. Otherwise (unfavourable or favourable zone) keep the planned stage 2.
With ``cp_min = 0`` the rule is the plain conditional-power re-estimation of
``rpact::getSimulationMeans(conditionalPower = , minNumberOfSubjectsPerStage = planned)``.

Operating characteristics are computed **exactly** by one-dimensional integration over
:math:`Z_1` (no Monte Carlo error), and :meth:`SSRDesign.simulate` checks them by simulation.

References
----------
Cui, L., Hung, H. M. J. and Wang, S.-J. (1999). Modification of sample size in group
sequential clinical trials. *Biometrics* 55, 853-857.
Mehta, C. R. and Pocock, S. J. (2011). Adaptive increase in sample size when interim results
are promising: a practical guide with examples. *Statistics in Medicine* 30, 3267-3284.
Gao, P., Ware, J. H. and Mehta, C. (2008). Sample size re-estimation for adaptive sequential
design in clinical trials. *Journal of Biopharmaceutical Statistics* 18, 1184-1196.
Jennison, C. and Turnbull, B. W. (2015). Adaptive sample size modification in clinical
trials: start small then ask for more? *Statistics in Medicine* 34, 3793-3810.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, optimize, stats

from trialdesign.boundaries import GroupSequentialDesign, group_sequential_design

__all__ = [
    "SSRDesign",
    "SSROperatingCharacteristics",
    "SSRSimulation",
    "ssr_design",
]

Test = Literal["chw", "naive"]
ZONES = ("efficacy", "futility", "unfavourable", "promising", "favourable")
_NEG_INF_Z = -10.0  # z1 below this contributes < 1e-23 for any sensible drift


@dataclass(frozen=True)
class SSROperatingCharacteristics:
    """Exact operating characteristics of an adaptive design at one drift.

    Sample sizes are fractions of the planned maximum (multiply by the planned maximum
    sample size for patients).
    """

    drift: float
    reject_prob: float
    prob_efficacy_interim: float
    prob_futility_interim: float
    prob_zone: dict[str, float]
    expected_n_fraction: float
    max_n_fraction: float


@dataclass(frozen=True)
class SSRSimulation:
    """Monte Carlo operating characteristics, each with its Monte Carlo SE."""

    drift: float
    n_sim: int
    seed: int | None
    reject_prob: float
    reject_prob_se: float
    reject_prob_naive: float
    reject_prob_naive_se: float
    prob_zone: dict[str, float]
    expected_n_fraction: float
    expected_n_fraction_se: float


@dataclass(frozen=True)
class SSRDesign:
    """Two-look group-sequential design with promising-zone sample size re-estimation.

    Attributes
    ----------
    gsd : GroupSequentialDesign
        The planned two-look design (interim and final).
    cp_min, cp_target : float
        Promising zone :math:`[\\mathrm{CP}_{\\min}, \\mathrm{CP}_{\\text{target}})` of the
        conditional power at the observed trend, and the conditional power aimed for.
    max_factor : float
        Cap on the total sample size as a multiple of the planned maximum (at least 1).
    """

    gsd: GroupSequentialDesign
    cp_min: float = 0.36
    cp_target: float = 0.80
    max_factor: float = 2.0
    _breaks: tuple[float, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.gsd.n_looks != 2:
            raise ValueError("sample size re-estimation needs a design with exactly two looks")
        if self.gsd.binding_futility:
            raise ValueError("sample size re-estimation supports non-binding futility only")
        if not 0.0 <= self.cp_min < self.cp_target < 1.0:
            raise ValueError("need 0 <= cp_min < cp_target < 1")
        if not self.max_factor >= 1.0:
            raise ValueError("max_factor must be at least 1")
        object.__setattr__(self, "_breaks", self._zone_breaks())

    # ----------------------------------------------------------------- basic quantities

    @property
    def t1(self) -> float:
        return float(self.gsd.info_rates[0])

    @property
    def c1(self) -> float:
        return float(self.gsd.efficacy[0])

    @property
    def c2(self) -> float:
        return float(self.gsd.efficacy[1])

    @property
    def b1(self) -> float | None:
        return None if self.gsd.futility is None else float(self.gsd.futility[0])

    @property
    def m_max(self) -> float:
        """Largest stage-2 multiplier: total n = max_factor times the planned maximum."""
        return (self.max_factor - self.t1) / (1.0 - self.t1)

    def conditional_power(
        self, z1: ArrayLike, drift: float | None = None, m: ArrayLike = 1.0
    ) -> NDArray[np.float64]:
        """Conditional power of the CHW test given :math:`Z_1 = z_1`.

        ``drift`` is the mean of the final z-statistic of the planned design under the
        assumed effect; None uses the observed trend :math:`z_1 / \\sqrt{t_1}`. ``m`` is the
        stage-2 sample size as a multiple of the planned one.

        References
        ----------
        Mehta & Pocock (2011), equation 3; Cui, Hung & Wang (1999).
        """
        z1 = np.asarray(z1, dtype=float)
        t1 = self.t1
        d = z1 / math.sqrt(t1) if drift is None else drift
        b = (self.c2 - math.sqrt(t1) * z1) / math.sqrt(1 - t1)
        return stats.norm.sf(b - d * np.sqrt((1 - t1) * np.asarray(m, dtype=float)))

    def _m_star(self, z1: NDArray) -> NDArray:
        """Stage-2 multiplier that gives cp_target at the observed trend (inf if impossible)."""
        t1 = self.t1
        d = z1 / math.sqrt(t1)
        b = (self.c2 - math.sqrt(t1) * z1) / math.sqrt(1 - t1)
        # CP = target  <=>  b - d sqrt((1 - t1) m) = Phi^{-1}(1 - target)
        need = b - stats.norm.ppf(1 - self.cp_target)
        with np.errstate(divide="ignore", invalid="ignore"):
            m = np.where(d > 0, (need / np.where(d > 0, d, 1.0)) ** 2 / (1 - t1), np.inf)
        return np.where(need <= 0, 0.0, m)

    def stage2_multiplier(self, z1: ArrayLike) -> NDArray[np.float64]:
        """Stage-2 sample size as a multiple of the planned stage 2, for continuing trials."""
        z1 = np.asarray(z1, dtype=float)
        cp1 = self.conditional_power(z1)
        promising = (cp1 >= self.cp_min) & (cp1 < self.cp_target)
        m = np.clip(self._m_star(z1), 1.0, self.m_max)
        return np.where(promising, m, 1.0)

    def zone(self, z1: ArrayLike, use_futility: bool = True) -> NDArray[np.str_]:
        """Interim decision for each :math:`z_1`: one of :data:`ZONES`."""
        z1 = np.asarray(z1, dtype=float)
        cp1 = self.conditional_power(z1)
        out = np.where(
            cp1 < self.cp_min,
            "unfavourable",
            np.where(cp1 < self.cp_target, "promising", "favourable"),
        )
        if use_futility and self.b1 is not None:
            out = np.where(z1 < self.b1, "futility", out)
        return np.where(z1 >= self.c1, "efficacy", out)

    def _z_at_cp(self, cp: float) -> float | None:
        """The z1 at which conditional power at the observed trend (m = 1) equals ``cp``."""
        if cp <= 0.0:
            return None
        f = lambda z: float(self.conditional_power(z)) - cp  # noqa: E731
        if f(_NEG_INF_Z) > 0 or f(self.c1) < 0:
            return None
        return float(optimize.brentq(f, _NEG_INF_Z, self.c1, xtol=1e-12))

    def _zone_breaks(self) -> tuple[float, ...]:
        """Points in z1 where the stage-2 rule has a kink or a jump (for the quadrature)."""
        z_lo, z_hi = self._z_at_cp(self.cp_min), self._z_at_cp(self.cp_target)
        pts = [z for z in (z_lo, z_hi) if z is not None]
        lo = _NEG_INF_Z if z_lo is None else z_lo
        hi = self.c1 if z_hi is None else z_hi
        g = lambda z: float(self._m_star(np.asarray(z))) - self.m_max  # noqa: E731
        lo_cap = max(lo, 1e-9)  # m* is finite only for a positive observed trend
        if lo_cap < hi and g(lo_cap) > 0 > g(hi):
            pts.append(float(optimize.brentq(g, lo_cap, hi, xtol=1e-12)))
        return tuple(sorted(pts))

    # ------------------------------------------------------------ exact characteristics

    def _reject_given_z1(self, z1: NDArray, drift: float, test: Test) -> NDArray:
        t1 = self.t1
        m = self.stage2_multiplier(z1)
        mean2 = drift * np.sqrt((1 - t1) * m)
        if test == "chw":
            crit = (self.c2 - math.sqrt(t1) * z1) / math.sqrt(1 - t1)
        elif test == "naive":  # pooled z over all patients, information t1 + (1 - t1) m
            crit = (self.c2 * np.sqrt(t1 + (1 - t1) * m) - math.sqrt(t1) * z1) / np.sqrt(
                (1 - t1) * m
            )
        else:
            raise ValueError('test must be "chw" or "naive"')
        return stats.norm.sf(crit - mean2)

    def _integrate(self, func, lo: float, hi: float) -> float:
        pts = [p for p in self._breaks if lo < p < hi]
        edges = [lo, *pts, hi]
        return float(
            sum(
                integrate.quad(func, a, b, epsabs=1e-13, epsrel=1e-11, limit=200)[0]
                for a, b in itertools.pairwise(edges)
                if b > a
            )
        )

    def operating_characteristics(
        self, drift: float = 0.0, use_futility: bool = True, test: Test = "chw"
    ) -> SSROperatingCharacteristics:
        """Exact rejection probability, zone probabilities and expected sample size.

        ``drift`` is the mean of the final z-statistic of the planned design (0 under H0,
        ``gsd.drift`` under the design alternative). With ``use_futility=False`` the interim
        futility boundary is ignored, which is how the type I error of a non-binding design
        is defined. ``test="naive"`` replaces the CHW statistic by the pooled one, to show
        the inflation it causes.
        """
        t1, mu1 = self.t1, drift * math.sqrt(self.t1)
        dens = lambda z: stats.norm.pdf(z - mu1)  # noqa: E731
        lo = self.b1 if (use_futility and self.b1 is not None) else mu1 - 12.0
        hi = self.c1
        p_eff = float(stats.norm.sf(hi - mu1))
        p_fut = float(stats.norm.cdf(lo - mu1)) if lo == self.b1 else 0.0
        cont = self._integrate(
            lambda z: dens(z) * float(self._reject_given_z1(np.asarray(z), drift, test)), lo, hi
        )
        zones = {"efficacy": p_eff, "futility": p_fut}
        z_min, z_target = self._z_at_cp(self.cp_min), self._z_at_cp(self.cp_target)
        cut_lo = lo if z_min is None else max(lo, z_min)
        cut_hi = hi if z_target is None else min(hi, max(z_target, lo))
        zones["unfavourable"] = float(
            max(0.0, stats.norm.cdf(cut_lo - mu1) - stats.norm.cdf(lo - mu1))
        )
        zones["promising"] = float(
            max(0.0, stats.norm.cdf(cut_hi - mu1) - stats.norm.cdf(cut_lo - mu1))
        )
        zones["favourable"] = float(
            max(0.0, stats.norm.cdf(hi - mu1) - stats.norm.cdf(cut_hi - mu1))
        )
        extra = self._integrate(
            lambda z: dens(z) * (float(self.stage2_multiplier(np.asarray(z))) - 1.0), lo, hi
        )
        p_cont = 1.0 - p_eff - p_fut
        expected = t1 * (p_eff + p_fut) + p_cont * 1.0 + (1 - t1) * extra
        return SSROperatingCharacteristics(
            drift=float(drift),
            reject_prob=p_eff + cont,
            prob_efficacy_interim=p_eff,
            prob_futility_interim=p_fut,
            prob_zone=zones,
            expected_n_fraction=float(expected),
            max_n_fraction=float(self.max_factor if zones["promising"] > 0 else 1.0),
        )

    # ------------------------------------------------------------------- simulation

    def simulate(
        self,
        drift: float = 0.0,
        n_sim: int = 100_000,
        seed: int | np.random.Generator | None = None,
        use_futility: bool = True,
    ) -> SSRSimulation:
        """Simulate the adaptive trial on the z-scale (vectorised; no loop over trials).

        Draws :math:`Z_1 \\sim N(\\delta\\sqrt{t_1}, 1)`, applies the interim rule, then
        :math:`Z_2^\\ast \\sim N(\\delta\\sqrt{(1-t_1)m}, 1)` with the re-estimated
        :math:`m`, and applies both the CHW and the naive final test.
        """
        if n_sim < 1:
            raise ValueError("n_sim must be at least 1")
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        t1 = self.t1
        z1 = rng.standard_normal(n_sim) + drift * math.sqrt(t1)
        eff = z1 >= self.c1
        fut = (
            (z1 < self.b1) & ~eff if (use_futility and self.b1 is not None) else np.zeros_like(eff)
        )
        cont = ~eff & ~fut
        m = np.where(cont, self.stage2_multiplier(z1), 0.0)
        z2 = rng.standard_normal(n_sim) + drift * np.sqrt((1 - t1) * m)
        chw = math.sqrt(t1) * z1 + math.sqrt(1 - t1) * z2
        with np.errstate(invalid="ignore", divide="ignore"):
            naive = (math.sqrt(t1) * z1 + np.sqrt((1 - t1) * m) * z2) / np.sqrt(t1 + (1 - t1) * m)
        rej = eff | (cont & (chw >= self.c2))
        rej_naive = eff | (cont & (naive >= self.c2))
        n_frac = np.where(cont, t1 + (1 - t1) * m, t1)
        labels = self.zone(z1, use_futility)

        def rate(x: NDArray) -> tuple[float, float]:
            p = float(x.mean())
            return p, math.sqrt(p * (1 - p) / n_sim)

        r, r_se = rate(rej)
        rn, rn_se = rate(rej_naive)
        return SSRSimulation(
            drift=float(drift),
            n_sim=n_sim,
            seed=seed if isinstance(seed, int) else None,
            reject_prob=r,
            reject_prob_se=r_se,
            reject_prob_naive=rn,
            reject_prob_naive_se=rn_se,
            prob_zone={z: float(np.mean(labels == z)) for z in ZONES},
            expected_n_fraction=float(n_frac.mean()),
            expected_n_fraction_se=float(n_frac.std(ddof=1) / math.sqrt(n_sim))
            if n_sim > 1
            else 0.0,
        )


def ssr_design(
    interim: float = 0.5,
    alpha_one_sided: float = 0.025,
    power: float = 0.80,
    efficacy_spending: str = "obf",
    futility_spending: str | None = None,
    cp_min: float = 0.36,
    cp_target: float = 0.80,
    max_factor: float = 2.0,
) -> SSRDesign:
    """Build the planned two-look design and attach the re-estimation rule.

    Parameters
    ----------
    interim : float
        Information fraction of the interim look, strictly between 0 and 1.
    alpha_one_sided, power, efficacy_spending, futility_spending
        As for :func:`trialdesign.boundaries.group_sequential_design`, with looks
        ``[interim, 1]``.
    cp_min, cp_target, max_factor
        The promising-zone rule; see :class:`SSRDesign`.

    References
    ----------
    Mehta & Pocock (2011); Cui, Hung & Wang (1999).
    """
    if not 0.0 < interim < 1.0:
        raise ValueError("interim must be strictly between 0 and 1")
    gsd = group_sequential_design(
        [interim, 1.0],
        alpha_one_sided=alpha_one_sided,
        power=power,
        efficacy_spending=efficacy_spending,
        futility_spending=futility_spending,
    )
    return SSRDesign(gsd, cp_min=cp_min, cp_target=cp_target, max_factor=max_factor)
