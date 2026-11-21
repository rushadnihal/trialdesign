r"""Beta-binomial posterior, predictive probability of success and assurance.

Two-arm binary endpoint with independent :math:`\mathrm{Beta}(a, b)` priors per arm. After
:math:`x` responders out of :math:`n` the posterior is :math:`\mathrm{Beta}(a + x, b + n - x)`.

**Final success rule.** The trial succeeds at the final analysis if
:math:`P(p_T > p_C \mid \text{final data}) \ge \eta`. The posterior probability is computed by
one-dimensional quadrature,

.. math:: P(p_T > p_C) = \int_0^1 S_T\!\left(F_C^{-1}(u)\right)\,du,

where :math:`F_C` is the control posterior CDF and :math:`S_T` the treatment posterior survival
function. The substitution :math:`u = (1 - \cos \pi s)/2` clusters the Gauss-Legendre nodes at
the ends, where :math:`F_C^{-1}` is singular; 128 nodes agree with adaptive quadrature to
about 1e-10.

**Predictive probability of success (PPoS).** At an interim with :math:`x_C/n_C` and
:math:`x_T/n_T`, the remaining :math:`m = N - n` patients per arm follow the posterior
predictive (beta-binomial) distribution, independently by arm. PPoS is the exact average of
the final success indicator over all future outcomes,

.. math:: \mathrm{PPoS} = \sum_{y_C=0}^{m_C} \sum_{y_T=0}^{m_T}
          \mathrm{BB}(y_C)\,\mathrm{BB}(y_T)\;
          \mathbf 1\{P(p_T > p_C \mid x_C + y_C, x_T + y_T) \ge \eta\},

so no Monte Carlo error enters the decision. The interim decision is go if
:math:`\mathrm{PPoS} \ge \gamma_{go}`, no-go if :math:`\mathrm{PPoS} \le \gamma_{nogo}`, and
continue otherwise. Go stops the trial and declares success; no-go stops it for futility.

**Operating characteristics.** Every decision is a lookup in two tables: the final success
table over :math:`(X_C, X_T)` and the interim decision table over :math:`(x_C, x_T)`. Exact
frequentist operating characteristics follow by summing binomial probabilities over the
tables (:meth:`BayesianInterimDesign.operating_characteristics`); :func:`simulate_bayesian`
checks them with the vectorised M3 engine of :mod:`trialdesign.simulate`.

**Assurance** is the probability of success averaged over a design prior on the true response
rates (O'Hagan, Stevens and Campbell 2005). With Beta design priors the data are marginally
beta-binomial, so :meth:`BayesianInterimDesign.assurance` is exact; :func:`simulate_assurance`
estimates it the way the outline describes, by drawing the true rates and then the trial. The
design prior describes what we believe about the truth; the analysis priors are those used to
compute the posterior, and the two need not be equal.

References
----------
Berry, S. M., Carlin, B. P., Lee, J. J. and Muller, P. (2010). *Bayesian Adaptive Methods for
Clinical Trials*. Chapman & Hall/CRC. Chapters 2 and 4.
Lee, J. J. and Liu, D. D. (2008). A predictive probability design for phase II cancer clinical
trials. *Clinical Trials* 5, 93-106.
O'Hagan, A., Stevens, J. W. and Campbell, M. J. (2005). Assurance in clinical trial design.
*Pharmaceutical Statistics* 4, 187-201.
Spiegelhalter, D. J., Abrams, K. R. and Myles, J. P. (2004). *Bayesian Approaches to Clinical
Trials and Health-Care Evaluation*. Wiley. Section 6.6.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special, stats

from trialdesign.simulate import OperatingCharacteristics, SeedLike, _check_n_sim, _rng, _Tally

__all__ = [
    "BayesianInterimDesign",
    "BayesianOC",
    "Decision",
    "DesignPrior",
    "Prior",
    "bayesian_interim_design",
    "decide",
    "posterior",
    "predictive_probability",
    "prob_superiority",
    "prob_superiority_mc",
    "simulate_assurance",
    "simulate_bayesian",
]

Prior = tuple[float, float]
DesignPrior = float | Prior  # a fixed response rate, or a Beta(a, b) design prior
Decision = Literal["go", "no-go", "continue"]

_GO, _CONTINUE, _NOGO = 1, 0, -1
_CHUNK = 250_000

# Gauss-Legendre nodes on s in [0, 1], mapped to u = (1 - cos(pi s)) / 2.
_S, _W = np.polynomial.legendre.leggauss(128)
_S = (_S + 1.0) / 2.0
_U = (1.0 - np.cos(np.pi * _S)) / 2.0
_WU = _W / 2.0 * (np.pi / 2.0) * np.sin(np.pi * _S)


# ------------------------------------------------------------------------- checks


def _check_prior(prior: Prior, name: str) -> tuple[float, float]:
    a, b = (float(v) for v in prior)
    if not (a > 0 and b > 0 and np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"{name} must be a pair of positive Beta parameters (a, b)")
    return a, b


def _check_counts(x: ArrayLike, n: ArrayLike, name: str) -> tuple[NDArray, NDArray]:
    x, n = np.asarray(x), np.asarray(n)
    if np.any(n < 0) or np.any(x < 0) or np.any(x > n):
        raise ValueError(f"{name}: responders must lie between 0 and the number of patients")
    if np.any(x != np.round(x)) or np.any(n != np.round(n)):
        raise ValueError(f"{name}: responders and patients must be whole numbers")
    return x.astype(np.int64), n.astype(np.int64)


def _check_threshold(value: float, name: str) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return float(value)


def _check_design_prior(prior: DesignPrior, name: str) -> DesignPrior:
    if isinstance(prior, tuple):
        return _check_prior(prior, name)
    if not 0.0 <= prior <= 1.0:
        raise ValueError(f"{name} must be a response rate in [0, 1] or a Beta (a, b) pair")
    return float(prior)


# ----------------------------------------------------------------- posterior quantities


def posterior(x: int, n: int, prior: Prior = (1.0, 1.0)) -> Prior:
    """Beta posterior parameters :math:`(a + x, b + n - x)` after ``x`` responders of ``n``.

    References
    ----------
    Spiegelhalter, Abrams and Myles (2004), section 3.6.
    """
    a, b = _check_prior(prior, "prior")
    _check_counts(x, n, "posterior")
    return a + x, b + n - x


def _prob_superiority(a_c: ArrayLike, b_c: ArrayLike, a_t: ArrayLike, b_t: ArrayLike) -> NDArray:
    """P(p_T > p_C) for independent Beta(a_c, b_c) and Beta(a_t, b_t); broadcasts."""
    a_c, b_c, a_t, b_t = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (a_c, b_c, a_t, b_t))
    )
    p = special.betaincinv(a_c[..., None], b_c[..., None], _U)  # control quantiles
    surv_t = special.betainc(b_t[..., None], a_t[..., None], 1.0 - p)  # S_T(p) = I_{1-p}(b, a)
    return np.clip(surv_t @ _WU, 0.0, 1.0)


def prob_superiority(
    x_c: ArrayLike,
    n_c: ArrayLike,
    x_t: ArrayLike,
    n_t: ArrayLike,
    prior_c: Prior = (1.0, 1.0),
    prior_t: Prior = (1.0, 1.0),
) -> float | NDArray[np.float64]:
    r"""Posterior probability :math:`P(p_T > p_C \mid \text{data})`.

    Parameters
    ----------
    x_c, n_c, x_t, n_t : int or array_like of int
        Responders and patients in the control and treatment arms; arrays broadcast.
    prior_c, prior_t : (float, float)
        Beta analysis priors per arm. ``(1, 1)`` is uniform.

    Returns
    -------
    float or numpy.ndarray
        The posterior probability, accurate to about 1e-10.

    References
    ----------
    Spiegelhalter, Abrams and Myles (2004), section 3.6; Berry et al. (2010), section 2.5.
    """
    a_c, b_c = _check_prior(prior_c, "prior_c")
    a_t, b_t = _check_prior(prior_t, "prior_t")
    x_c, n_c = _check_counts(x_c, n_c, "control")
    x_t, n_t = _check_counts(x_t, n_t, "treatment")
    out = _prob_superiority(a_c + x_c, b_c + n_c - x_c, a_t + x_t, b_t + n_t - x_t)
    return float(out) if out.ndim == 0 else out


def prob_superiority_mc(
    x_c: int,
    n_c: int,
    x_t: int,
    n_t: int,
    prior_c: Prior = (1.0, 1.0),
    prior_t: Prior = (1.0, 1.0),
    n_draws: int = 1_000_000,
    seed: SeedLike = None,
) -> tuple[float, float]:
    """Monte Carlo estimate of :math:`P(p_T > p_C)` and its standard error, as a cross-check.

    References
    ----------
    Berry et al. (2010), section 2.5.
    """
    a_c, b_c = posterior(x_c, n_c, prior_c)
    a_t, b_t = posterior(x_t, n_t, prior_t)
    _check_n_sim(n_draws)
    rng = _rng(seed)
    p = float(np.mean(rng.beta(a_t, b_t, n_draws) > rng.beta(a_c, b_c, n_draws)))
    return p, float(np.sqrt(p * (1.0 - p) / n_draws))


# ------------------------------------------------------------- predictive probability


def _success_table(
    n_c: int, n_t: int, prior_c: Prior, prior_t: Prior, threshold: float
) -> NDArray[np.bool_]:
    r"""Final success indicator over all final counts, shape (n_c + 1, n_t + 1).

    :math:`P(p_T > p_C)` increases in the treatment responders (the Beta posterior is
    stochastically increasing in x), so each row is a step: success iff
    :math:`x_T \ge x_T^*(x_C)`. A vectorised bisection finds :math:`x_T^*` for every
    :math:`x_C` in about :math:`\log_2 n_T` quadratures instead of :math:`n_T`.
    """
    x_c = np.arange(n_c + 1)
    (a_c, b_c), (a_t, b_t) = prior_c, prior_t

    def wins(x_t: NDArray) -> NDArray:
        prob = _prob_superiority(a_c + x_c, b_c + n_c - x_c, a_t + x_t, b_t + n_t - x_t)
        return prob >= threshold

    lo = np.zeros(n_c + 1, dtype=np.int64)  # invariant: the step lies in [lo, hi]
    hi = np.full(n_c + 1, n_t + 1, dtype=np.int64)  # n_t + 1 means "never succeeds"
    while np.any(lo < hi):
        mid = (lo + hi) // 2
        ok = wins(np.minimum(mid, n_t)) & (mid <= n_t)
        hi = np.where(ok & (lo < hi), mid, hi)
        lo = np.where(~ok & (lo < hi), mid + 1, lo)
    return np.arange(n_t + 1)[None, :] >= lo[:, None]


def _transition(n: int, n_final: int, law: DesignPrior | Prior, predictive: bool) -> NDArray:
    """Matrix M[x, X] = P(X responders at the final | x at the interim), shape (n+1, N+1).

    ``predictive`` uses the beta-binomial posterior predictive under the Beta ``law``;
    otherwise ``law`` is a fixed response rate and the increment is binomial.
    """
    m = n_final - n
    x = np.arange(n + 1)[:, None]
    y = np.arange(n_final + 1)[None, :] - x  # future responders
    ok = (y >= 0) & (y <= m)
    yy = np.where(ok, y, 0)
    if predictive:
        a, b = law
        pmf = stats.betabinom.pmf(yy, m, a + x, b + n - x)
    else:
        pmf = stats.binom.pmf(yy, m, law)
    return np.where(ok, pmf, 0.0)


def _interim_law(n: int, law: DesignPrior) -> NDArray:
    """Distribution of the interim responder count: binomial, or beta-binomial for a prior."""
    k = np.arange(n + 1)
    if isinstance(law, tuple):
        return stats.betabinom.pmf(k, n, *law)
    return stats.binom.pmf(k, n, law)


def predictive_probability(
    x_c: int,
    n_c: int,
    x_t: int,
    n_t: int,
    n_final_c: int,
    n_final_t: int,
    prior_c: Prior = (1.0, 1.0),
    prior_t: Prior = (1.0, 1.0),
    success_threshold: float = 0.95,
) -> float:
    r"""Predictive probability of success at the final analysis, given interim data.

    Enumerates every future outcome of the remaining patients under the beta-binomial
    posterior predictive and averages the final success indicator
    :math:`\mathbf 1\{P(p_T > p_C \mid \text{final}) \ge \eta\}`, so the result is exact.

    Parameters
    ----------
    x_c, n_c, x_t, n_t : int
        Interim responders and patients per arm.
    n_final_c, n_final_t : int
        Planned patients per arm at the final analysis.
    prior_c, prior_t : (float, float)
        Beta analysis priors.
    success_threshold : float
        :math:`\eta`, the posterior probability needed at the final analysis.

    Returns
    -------
    float
        PPoS. With no patients left it is 0 or 1.

    References
    ----------
    Lee and Liu (2008); Berry et al. (2010), section 4.2; Spiegelhalter et al. (2004),
    section 6.6.
    """
    prior_c = _check_prior(prior_c, "prior_c")
    prior_t = _check_prior(prior_t, "prior_t")
    eta = _check_threshold(success_threshold, "success_threshold")
    _check_counts(x_c, n_c, "control")
    _check_counts(x_t, n_t, "treatment")
    if n_final_c < n_c or n_final_t < n_t:
        raise ValueError("final sample sizes must be at least the interim sample sizes")
    success = _success_table(n_final_c, n_final_t, prior_c, prior_t, eta)
    w_c = _transition(n_c, n_final_c, prior_c, predictive=True)[x_c]
    w_t = _transition(n_t, n_final_t, prior_t, predictive=True)[x_t]
    return float(np.clip(w_c @ success @ w_t, 0.0, 1.0))


def decide(ppos: float, go_threshold: float = 0.9, nogo_threshold: float = 0.1) -> Decision:
    """Interim decision: go if PPoS >= go, no-go if PPoS <= no-go, otherwise continue."""
    go = _check_threshold(go_threshold, "go_threshold")
    nogo = _check_threshold(nogo_threshold, "nogo_threshold")
    if not nogo < go:
        raise ValueError("nogo_threshold must be below go_threshold")
    if ppos >= go:
        return "go"
    if ppos <= nogo:
        return "no-go"
    return "continue"


# ---------------------------------------------------------------------- the design


@dataclass(frozen=True)
class BayesianOC:
    """Exact operating characteristics of a Bayesian interim design.

    Attributes
    ----------
    p_control, p_treatment : float or (float, float)
        True response rates, or the Beta design priors they were averaged over (assurance).
    prob_go, prob_nogo, prob_continue : float
        Probabilities of each interim decision.
    prob_final_success : float
        Probability of continuing past the interim and then succeeding at the final.
    expected_n_control, expected_n_treatment : float
        Expected patients per arm.
    """

    p_control: DesignPrior
    p_treatment: DesignPrior
    prob_go: float
    prob_nogo: float
    prob_continue: float
    prob_final_success: float
    expected_n_control: float
    expected_n_treatment: float

    @property
    def prob_success(self) -> float:
        """Probability the trial declares success (go, or continue and succeed at the final).

        The type I error when ``p_control == p_treatment``, power under an alternative, and
        assurance when averaged over a design prior.
        """
        return self.prob_go + self.prob_final_success


@dataclass(frozen=True)
class BayesianInterimDesign:
    """A two-arm binary design with one Bayesian go/no-go interim and a final analysis.

    Build it with :func:`bayesian_interim_design`. The tables are computed once, so every
    later decision, operating characteristic and simulation is a lookup.

    Attributes
    ----------
    n_interim_control, n_interim_treatment, n_final_control, n_final_treatment : int
        Patients per arm at the interim and at the final analysis.
    prior_control, prior_treatment : (float, float)
        Beta analysis priors.
    success_threshold : float
        :math:`\\eta`: final success needs :math:`P(p_T > p_C \\mid \\text{data}) \\ge \\eta`.
    go_threshold, nogo_threshold : float
        :math:`\\gamma_{go}` and :math:`\\gamma_{nogo}` for PPoS at the interim.
    success_table : numpy.ndarray of bool
        Final success indicator, indexed ``[x_control, x_treatment]`` at the final.
    ppos_table : numpy.ndarray
        PPoS, indexed ``[x_control, x_treatment]`` at the interim.
    """

    n_interim_control: int
    n_interim_treatment: int
    n_final_control: int
    n_final_treatment: int
    prior_control: Prior
    prior_treatment: Prior
    success_threshold: float
    go_threshold: float
    nogo_threshold: float
    success_table: NDArray[np.bool_]
    ppos_table: NDArray[np.float64]

    @property
    def decision_table(self) -> NDArray[np.int8]:
        """Interim decision per ``[x_control, x_treatment]``: 1 go, 0 continue, -1 no-go."""
        d = np.zeros(self.ppos_table.shape, dtype=np.int8)
        d[self.ppos_table >= self.go_threshold] = _GO
        d[self.ppos_table <= self.nogo_threshold] = _NOGO
        return d

    @property
    def interim_fraction(self) -> float:
        """Share of the total sample size enrolled at the interim."""
        n1 = self.n_interim_control + self.n_interim_treatment
        return n1 / (self.n_final_control + self.n_final_treatment)

    def ppos(self, x_control: int, x_treatment: int) -> float:
        """PPoS for interim responder counts."""
        _check_counts(x_control, self.n_interim_control, "control")
        _check_counts(x_treatment, self.n_interim_treatment, "treatment")
        return float(self.ppos_table[x_control, x_treatment])

    def decide(self, x_control: int, x_treatment: int) -> Decision:
        """Interim decision for interim responder counts."""
        return decide(self.ppos(x_control, x_treatment), self.go_threshold, self.nogo_threshold)

    def _oc(self, law_c: DesignPrior, law_t: DesignPrior) -> BayesianOC:
        n1c, n1t = self.n_interim_control, self.n_interim_treatment
        nc, nt = self.n_final_control, self.n_final_treatment
        pi_c, pi_t = _interim_law(n1c, law_c), _interim_law(n1t, law_t)
        # Given the interim counts, the final counts follow the binomial increment for a fixed
        # rate, or the beta-binomial posterior predictive under the design prior.
        m_c = _transition(n1c, nc, law_c, predictive=isinstance(law_c, tuple))
        m_t = _transition(n1t, nt, law_t, predictive=isinstance(law_t, tuple))
        p_win = m_c @ self.success_table @ m_t.T  # P(final success | interim counts)
        d = self.decision_table
        joint = np.outer(pi_c, pi_t)
        p_go = float(joint[d == _GO].sum())
        p_nogo = float(joint[d == _NOGO].sum())
        p_cont = float(joint[d == _CONTINUE].sum())
        p_final = float((joint * p_win)[d == _CONTINUE].sum())
        return BayesianOC(
            p_control=law_c,
            p_treatment=law_t,
            prob_go=p_go,
            prob_nogo=p_nogo,
            prob_continue=p_cont,
            prob_final_success=p_final,
            expected_n_control=n1c + p_cont * (nc - n1c),
            expected_n_treatment=n1t + p_cont * (nt - n1t),
        )

    def operating_characteristics(self, p_control: float, p_treatment: float) -> BayesianOC:
        """Exact frequentist operating characteristics at true response rates.

        Sums binomial probabilities over the interim decision table and the final success
        table. With ``p_control == p_treatment``, ``prob_success`` is the type I error.
        """
        law_c = _check_design_prior(float(p_control), "p_control")
        law_t = _check_design_prior(float(p_treatment), "p_treatment")
        return self._oc(law_c, law_t)

    def assurance(
        self, design_prior_control: DesignPrior, design_prior_treatment: DesignPrior
    ) -> BayesianOC:
        """Exact assurance: operating characteristics averaged over a design prior.

        Each arm takes a fixed response rate or a Beta ``(a, b)`` design prior; the data are
        then marginally beta-binomial. ``prob_success`` is the assurance.

        References
        ----------
        O'Hagan, Stevens and Campbell (2005).
        """
        law_c = _check_design_prior(design_prior_control, "design_prior_control")
        law_t = _check_design_prior(design_prior_treatment, "design_prior_treatment")
        return self._oc(law_c, law_t)


def bayesian_interim_design(
    n_interim: int,
    n_final: int,
    allocation: float = 1.0,
    prior_control: Prior = (1.0, 1.0),
    prior_treatment: Prior = (1.0, 1.0),
    success_threshold: float = 0.95,
    go_threshold: float = 0.9,
    nogo_threshold: float = 0.1,
) -> BayesianInterimDesign:
    r"""Build a two-arm Bayesian go/no-go interim design for a binary endpoint.

    Parameters
    ----------
    n_interim, n_final : int
        Control patients at the interim and at the final analysis. The treatment arm gets
        ``round(allocation * n)`` at each.
    allocation : float
        Ratio :math:`n_T / n_C`.
    prior_control, prior_treatment : (float, float)
        Beta analysis priors.
    success_threshold : float
        :math:`\eta` for final success, :math:`P(p_T > p_C \mid \text{data}) \ge \eta`.
    go_threshold, nogo_threshold : float
        PPoS thresholds :math:`\gamma_{go} > \gamma_{nogo}`.

    Returns
    -------
    BayesianInterimDesign

    References
    ----------
    Lee and Liu (2008); Berry et al. (2010), chapter 4.
    """
    prior_c = _check_prior(prior_control, "prior_control")
    prior_t = _check_prior(prior_treatment, "prior_treatment")
    eta = _check_threshold(success_threshold, "success_threshold")
    go = _check_threshold(go_threshold, "go_threshold")
    nogo = _check_threshold(nogo_threshold, "nogo_threshold")
    if not nogo < go:
        raise ValueError("nogo_threshold must be below go_threshold")
    if not allocation > 0:
        raise ValueError("allocation must be positive")
    if not (isinstance(n_interim, int | np.integer) and isinstance(n_final, int | np.integer)):
        raise ValueError("n_interim and n_final must be whole numbers")
    n1c, nc = int(n_interim), int(n_final)
    n1t, nt = round(allocation * n1c), round(allocation * nc)
    if not (1 <= n1c < nc and 1 <= n1t < nt):
        raise ValueError("need 1 <= n_interim < n_final patients in each arm")
    success = _success_table(nc, nt, prior_c, prior_t, eta)
    w_c = _transition(n1c, nc, prior_c, predictive=True)
    w_t = _transition(n1t, nt, prior_t, predictive=True)
    ppos = np.clip(w_c @ success @ w_t.T, 0.0, 1.0)
    return BayesianInterimDesign(
        n_interim_control=n1c,
        n_interim_treatment=n1t,
        n_final_control=nc,
        n_final_treatment=nt,
        prior_control=prior_c,
        prior_treatment=prior_t,
        success_threshold=eta,
        go_threshold=go,
        nogo_threshold=nogo,
        success_table=success,
        ppos_table=ppos,
    )


# ------------------------------------------------------------------------ simulation


def _simulate(
    design: BayesianInterimDesign,
    draw_rates,
    n_sim: int,
    seed: SeedLike,
) -> OperatingCharacteristics:
    """Patient-count simulation through the M3 tally.

    The look statistics are PPoS at the interim and the final success indicator, with
    efficacy boundaries (go, 0.5) and futility boundaries (just above no-go, 0.5), so
    ``prob_efficacy`` holds P(go) and P(final success) and ``prob_futility`` holds P(no-go)
    and P(final failure).
    """
    _check_n_sim(n_sim)
    rng = _rng(seed)
    n1c, n1t = design.n_interim_control, design.n_interim_treatment
    m_c, m_t = design.n_final_control - n1c, design.n_final_treatment - n1t
    t = np.array([design.interim_fraction, 1.0])
    c = np.array([design.go_threshold, 0.5])
    b = np.array([np.nextafter(design.nogo_threshold, np.inf), 0.5])
    tally = _Tally(t, c, b)
    for start in range(0, n_sim, _CHUNK):
        k = min(_CHUNK, n_sim - start)
        p_c, p_t = (np.broadcast_to(p, (k,)) for p in draw_rates(rng, k))
        x1c, x1t = rng.binomial(n1c, p_c), rng.binomial(n1t, p_t)
        xc, xt = x1c + rng.binomial(m_c, p_c), x1t + rng.binomial(m_t, p_t)
        stat = np.column_stack(
            [design.ppos_table[x1c, x1t], design.success_table[xc, xt].astype(float)]
        )
        tally.add(stat)
    return tally.result(None, futility_followed=True)


def simulate_bayesian(
    design: BayesianInterimDesign,
    p_control: float,
    p_treatment: float,
    n_sim: int = 100_000,
    seed: SeedLike = None,
) -> OperatingCharacteristics:
    """Simulate the design's frequentist operating characteristics with the M3 engine.

    Returns
    -------
    trialdesign.simulate.OperatingCharacteristics
        Two looks (interim, final). ``prob_efficacy`` is (P(go), P(continue and succeed)),
        ``prob_futility`` is (P(no-go), P(continue and fail)), ``reject_prob`` is the
        probability of success with its Monte Carlo SE, and ``expected_info_fraction`` times
        the total sample size is the expected sample size.

    References
    ----------
    Berry et al. (2010), section 2.5.4.
    """
    p_c = _check_design_prior(float(p_control), "p_control")
    p_t = _check_design_prior(float(p_treatment), "p_treatment")
    return _simulate(design, lambda rng, k: (p_c, p_t), n_sim, seed)


def simulate_assurance(
    design: BayesianInterimDesign,
    design_prior_control: DesignPrior,
    design_prior_treatment: DesignPrior,
    n_sim: int = 100_000,
    seed: SeedLike = None,
) -> OperatingCharacteristics:
    """Assurance by simulation: draw the true rates from the design prior, then the trial.

    ``reject_prob`` is the assurance and ``reject_prob_se`` its Monte Carlo SE.

    References
    ----------
    O'Hagan, Stevens and Campbell (2005).
    """
    law_c = _check_design_prior(design_prior_control, "design_prior_control")
    law_t = _check_design_prior(design_prior_treatment, "design_prior_treatment")

    def draw(law: DesignPrior, rng: np.random.Generator, k: int) -> NDArray | float:
        return rng.beta(*law, size=k) if isinstance(law, tuple) else law

    return _simulate(design, lambda rng, k: (draw(law_c, rng, k), draw(law_t, rng, k)), n_sim, seed)
