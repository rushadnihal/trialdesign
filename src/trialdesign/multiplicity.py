r"""Graphical multiple testing: several hypotheses sharing one alpha (Bretz et al. 2009).

A confirmatory trial rarely tests one hypothesis. Two doses against control, a primary and a
key secondary endpoint, or a population and a subgroup each need the family-wise error rate
(FWER) held at :math:`\alpha`. The graphical approach writes the multiple testing strategy as
a directed graph:

- each hypothesis :math:`H_i` has an initial weight :math:`w_i \ge 0`, with
  :math:`\sum_i w_i \le 1`; it is tested at level :math:`w_i \alpha`;
- a transition matrix :math:`G = (g_{ij})` with :math:`g_{ii} = 0` and rows summing to at
  most 1 says where the level of a rejected hypothesis goes.

The sequentially rejective algorithm rejects any :math:`H_j` with :math:`p_j \le w_j\alpha`,
then removes it and updates, for the remaining :math:`l \ne k`,

.. math:: w_l \leftarrow w_l + w_j g_{jl}, \qquad
          g_{lk} \leftarrow \frac{g_{lk} + g_{lj} g_{jk}}{1 - g_{lj} g_{jl}},

and repeats until nothing more can be rejected. The final set of rejections does not depend on
the order in which rejectable hypotheses are taken, and the procedure is a shortcut for a
closed test of weighted Bonferroni tests, so it controls the FWER strongly at :math:`\alpha`
for any dependence between the p-values. Holm, fixed-sequence and fallback procedures are
special graphs.

:func:`graph_test` applies a graph to observed one-sided p-values and returns the rejections,
the adjusted p-values (Bretz et al. 2009, Algorithm 2) and the path through the graph.
:func:`graph_power` simulates one-sided z-statistics from a multivariate normal with given
means and correlation and reports, with Monte Carlo SEs, the probability of rejecting each
hypothesis, at least one, and all of them.

References
----------
Bretz, F., Maurer, W., Brannath, W. and Posch, M. (2009). A graphical approach to sequentially
rejective multiple test procedures. *Statistics in Medicine* 28, 586-604.
Bretz, F., Posch, M., Glimm, E., Klinglmueller, F., Maurer, W. and Rohmeyer, K. (2011).
Graphical approaches for multiple comparison procedures using weighted Bonferroni, Simes, or
parametric tests. *Biometrical Journal* 53, 894-913.
Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian
Journal of Statistics* 6, 65-70.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from trialdesign.simulate import SeedLike, _check_n_sim

__all__ = [
    "Graph",
    "GraphPower",
    "GraphTest",
    "graph_power",
    "graph_test",
]

_TOL = 1e-12
_CHUNK = 200_000


@dataclass(frozen=True)
class Graph:
    """Initial weights and transition matrix of a graphical multiple test procedure.

    Attributes
    ----------
    weights : tuple of float
        Initial weight of each hypothesis: zero or positive, summing to at most 1.
    transitions : tuple of tuple of float
        Square matrix :math:`G`: zero diagonal, entries in [0, 1], rows summing to at most 1.
    names : tuple of str
        Hypothesis labels; default ``H1``, ``H2``, ...
    """

    weights: tuple[float, ...]
    transitions: tuple[tuple[float, ...], ...]
    names: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        g = np.asarray(self.transitions, dtype=float)
        m = w.size
        if w.ndim != 1 or m == 0:
            raise ValueError("give at least one hypothesis weight")
        if g.shape != (m, m):
            raise ValueError(f"the transition matrix must be {m} x {m}, one row per hypothesis")
        if np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() > 1 + 1e-9:
            raise ValueError("weights must be zero or positive and sum to at most 1")
        if np.any(~np.isfinite(g)) or np.any(g < 0) or np.any(g > 1):
            raise ValueError("transition weights must lie in [0, 1]")
        if np.any(np.diag(g) != 0):
            raise ValueError("the transition matrix needs a zero diagonal (no self-loops)")
        if np.any(g.sum(axis=1) > 1 + 1e-9):
            raise ValueError("each row of the transition matrix must sum to at most 1")
        if not self.names:
            object.__setattr__(self, "names", tuple(f"H{i + 1}" for i in range(m)))
        if len(self.names) != m or len(set(self.names)) != m:
            raise ValueError("give one distinct name per hypothesis")
        object.__setattr__(self, "weights", tuple(float(x) for x in w))
        object.__setattr__(self, "transitions", tuple(tuple(float(x) for x in r) for r in g))

    # ------------------------------------------------------------------- presets

    @classmethod
    def bonferroni(cls, m: int) -> Graph:
        """Equal weights and no edges: each hypothesis at :math:`\\alpha/m`, nothing passed on."""
        _check_m(m)
        return cls(tuple([1.0 / m] * m), tuple(tuple([0.0] * m) for _ in range(m)))

    @classmethod
    def holm(cls, m: int) -> Graph:
        """Holm (1979): equal weights, each rejected level split equally among the rest."""
        _check_m(m)
        g = (
            [[0.0 if i == j else 1.0 / (m - 1) for j in range(m)] for i in range(m)]
            if m > 1
            else [[0.0]]
        )
        return cls(tuple([1.0 / m] * m), tuple(tuple(r) for r in g))

    @classmethod
    def fixed_sequence(cls, m: int) -> Graph:
        """Fixed sequence (hierarchical) testing: all alpha on H1, passed on in order."""
        _check_m(m)
        w = [1.0] + [0.0] * (m - 1)
        g = [[1.0 if j == i + 1 else 0.0 for j in range(m)] for i in range(m)]
        return cls(tuple(w), tuple(tuple(r) for r in g))

    @classmethod
    def fallback(cls, weights: Sequence[float]) -> Graph:
        """Fallback procedure (Wiens 2003): level passed from each hypothesis to the next."""
        m = len(weights)
        _check_m(m)
        g = [[1.0 if j == i + 1 else 0.0 for j in range(m)] for i in range(m)]
        return cls(tuple(float(x) for x in weights), tuple(tuple(r) for r in g))

    @classmethod
    def two_doses(cls, split: float = 0.5) -> Graph:
        """Two doses, each with a primary and a key secondary hypothesis (Bretz et al. 2009).

        H1, H2 are the primary hypotheses of the high and low dose, H3, H4 their secondary
        ones. Each primary starts with half of alpha; a rejected primary passes ``split`` to
        its own secondary and the rest to the other primary; a rejected secondary passes all
        of its level to the other dose's primary.
        """
        if not 0 <= split <= 1:
            raise ValueError("split must lie in [0, 1]")
        g = (
            (0.0, 1 - split, split, 0.0),
            (1 - split, 0.0, 0.0, split),
            (0.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        )
        return cls(
            (0.5, 0.5, 0.0, 0.0),
            g,
            ("H1 primary high", "H2 primary low", "H3 secondary high", "H4 secondary low"),
        )

    # ----------------------------------------------------------------- algorithm

    @property
    def size(self) -> int:
        return len(self.weights)

    def weights_for(self, subset: Sequence[int]) -> NDArray[np.float64]:
        """Weights of the intersection hypothesis :math:`H_I` (Bretz et al. 2009, Algorithm 1).

        Removes the hypotheses outside ``subset`` one at a time with the update rule; the
        result has zeros outside ``subset``.
        """
        keep = set(int(i) for i in subset)
        w = np.asarray(self.weights, dtype=float)[None, :].copy()
        g = np.asarray(self.transitions, dtype=float)[None, :, :].copy()
        for j in range(self.size):
            if j not in keep:
                w, g = _remove(w, g, np.array([j]))
        return w[0]

    def describe(self) -> dict[str, object]:
        return {
            "names": list(self.names),
            "weights": list(self.weights),
            "transitions": [list(r) for r in self.transitions],
        }


def _check_m(m: int) -> None:
    if not (isinstance(m, int | np.integer) and m >= 1):
        raise ValueError("the number of hypotheses must be a positive integer")


def _remove(
    w: NDArray[np.float64], g: NDArray[np.float64], j: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remove hypothesis ``j[r]`` from graph ``r`` of a batch (Bretz et al. 2009, Algorithm 1).

    ``w`` has shape ``(n, m)`` and ``g`` ``(n, m, m)``; returns new arrays.
    """
    n, m = w.shape
    rows = np.arange(n)
    wj = w[rows, j]  # (n,)
    g_out = g[rows, j, :]  # g_{jk}, (n, m)
    g_in = g[rows, :, j]  # g_{lj}, (n, m)
    w_new = w + wj[:, None] * g_out
    denom = 1.0 - g_in * g_out  # 1 - g_{lj} g_{jl}, indexed by l, (n, m)
    num = g + g_in[:, :, None] * g_out[:, None, :]  # g_{lk} + g_{lj} g_{jk}
    safe = denom > _TOL
    g_new = np.where(safe[:, :, None], num / np.where(safe, denom, 1.0)[:, :, None], 0.0)
    eye = np.eye(m, dtype=bool)[None, :, :]
    g_new = np.where(eye, 0.0, g_new)
    w_new[rows, j] = 0.0
    g_new[rows, j, :] = 0.0
    g_new[rows, :, j] = 0.0
    return w_new, g_new


@dataclass(frozen=True)
class GraphTest:
    """Result of testing observed p-values through a graph.

    Attributes
    ----------
    graph : Graph
        The graph applied.
    alpha_one_sided : float
        Family-wise level.
    p_values : numpy.ndarray
        Observed one-sided p-values.
    rejected : numpy.ndarray of bool
        Hypotheses rejected.
    adjusted_p : numpy.ndarray
        Adjusted p-values: :math:`H_i` is rejected exactly when its adjusted p-value is at
        most :math:`\\alpha`.
    steps : list of dict
        Each rejection in order: the hypothesis, its p-value, the level it was tested at, and
        the weights after the graph was updated.
    """

    graph: Graph
    alpha_one_sided: float
    p_values: NDArray[np.float64]
    rejected: NDArray[np.bool_]
    adjusted_p: NDArray[np.float64]
    steps: list[dict[str, object]]


def _check_alpha(alpha: float) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha_one_sided must lie in (0, 1)")
    return float(alpha)


def graph_test(
    p_values: Sequence[float], graph: Graph, alpha_one_sided: float = 0.025
) -> GraphTest:
    """Apply a graphical procedure to observed one-sided p-values.

    Parameters
    ----------
    p_values : sequence of float
        One p-value per hypothesis, in the graph's order.
    graph : Graph
        Weights and transitions.
    alpha_one_sided : float
        Family-wise error rate to control.

    Returns
    -------
    GraphTest

    References
    ----------
    Bretz et al. (2009), Algorithms 1 and 2.
    """
    alpha = _check_alpha(alpha_one_sided)
    p = np.asarray(p_values, dtype=float)
    m = graph.size
    if p.shape != (m,):
        raise ValueError(f"give {m} p-values, one per hypothesis")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("p-values must lie in [0, 1]")

    # adjusted p-values (Algorithm 2)
    w = np.asarray(graph.weights, dtype=float)[None, :]
    g = np.asarray(graph.transitions, dtype=float)[None, :, :]
    left = np.ones(m, dtype=bool)
    adj = np.ones(m)
    running = 0.0
    for _ in range(m):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(left & (w[0] > _TOL), p / w[0], np.inf)
        j = int(np.argmin(ratio))
        if not np.isfinite(ratio[j]):
            break  # the rest have zero weight in every remaining graph: adjusted p = 1
        running = max(running, ratio[j])
        adj[j] = min(1.0, running)
        left[j] = False
        w, g = _remove(w, g, np.array([j]))

    # rejections in order, with the path (Algorithm 1)
    w = np.asarray(graph.weights, dtype=float)[None, :]
    g = np.asarray(graph.transitions, dtype=float)[None, :, :]
    rejected = np.zeros(m, dtype=bool)
    steps: list[dict[str, object]] = []
    while True:
        level = w[0] * alpha
        can = (~rejected) & (p <= level * (1 + 1e-12)) & (w[0] > _TOL)
        if not can.any():
            break
        # take the smallest p / w first, which gives the same order as the adjusted p-values
        with np.errstate(divide="ignore"):
            ratio = np.where(can, p / np.where(w[0] > 0, w[0], 1.0), np.inf)
        j = int(np.argmin(ratio))
        tested_at = float(level[j])
        rejected[j] = True
        w, g = _remove(w, g, np.array([j]))
        steps.append(
            {
                "rejected": graph.names[j],
                "p_value": float(p[j]),
                "level": tested_at,
                "weights_after": [float(x) for x in w[0]],
            }
        )
    return GraphTest(graph, alpha, p, rejected, adj, steps)


@dataclass(frozen=True)
class GraphPower:
    """Simulated power of a graphical procedure, each rate with its Monte Carlo SE.

    Attributes
    ----------
    local_power, local_power_se : numpy.ndarray
        Probability of rejecting each hypothesis.
    any_power, any_power_se : float
        Probability of rejecting at least one hypothesis (the FWER when every mean is 0).
    all_power, all_power_se : float
        Probability of rejecting all of them.
    expected_rejections : float
        Mean number of hypotheses rejected.
    n_sim, seed : int
        Trials simulated and the seed entropy.
    """

    graph: Graph
    alpha_one_sided: float
    means: NDArray[np.float64]
    correlation: NDArray[np.float64]
    local_power: NDArray[np.float64]
    local_power_se: NDArray[np.float64]
    any_power: float
    any_power_se: float
    all_power: float
    all_power_se: float
    expected_rejections: float
    n_sim: int
    seed: int


def _reject_batch(p: NDArray[np.float64], graph: Graph, alpha: float) -> NDArray[np.bool_]:
    """Sequentially rejective algorithm for a batch of p-value vectors, shape ``(n, m)``."""
    n, m = p.shape
    w = np.broadcast_to(np.asarray(graph.weights, dtype=float), (n, m)).copy()
    g = np.broadcast_to(np.asarray(graph.transitions, dtype=float), (n, m, m)).copy()
    rejected = np.zeros((n, m), dtype=bool)
    for _ in range(m):
        can = (~rejected) & (w > _TOL) & (p <= w * alpha * (1 + 1e-12))
        active = can.any(axis=1)
        if not active.any():
            break
        idx = np.flatnonzero(active)
        j = can[idx].argmax(axis=1)  # any rejectable one: the result is order-free
        rejected[idx, j] = True
        w[idx], g[idx] = _remove(w[idx], g[idx], j)
    return rejected


def graph_power(
    graph: Graph,
    means: Sequence[float],
    correlation: float | Sequence[Sequence[float]] = 0.0,
    alpha_one_sided: float = 0.025,
    n_sim: int = 100_000,
    seed: SeedLike = None,
) -> GraphPower:
    r"""Power of a graphical procedure by simulating correlated one-sided z-statistics.

    The test statistics are :math:`Z \sim N(\mu, R)` with p-values
    :math:`p_i = 1 - \Phi(Z_i)`. A mean of :math:`z_{1-\alpha} + z_{1-\beta}` gives power
    :math:`1 - \beta` to a single test at level :math:`\alpha`: 2.80 for 80% at one-sided
    0.025. Zero means give the family-wise error rate under the global null.

    Parameters
    ----------
    graph : Graph
        The procedure.
    means : sequence of float
        Mean of each z-statistic (its non-centrality).
    correlation : float or matrix
        One correlation for every pair, or the full correlation matrix (positive
        semi-definite, unit diagonal). Two doses against a shared control with equal group
        sizes have correlation 0.5 between their comparisons.
    alpha_one_sided : float
        Family-wise level.
    n_sim : int
        Simulated trials.
    seed : int or numpy.random.Generator, optional
        Random seed.

    Returns
    -------
    GraphPower

    References
    ----------
    Bretz et al. (2011), section 5 (power and the choice of a graph).
    """
    alpha = _check_alpha(alpha_one_sided)
    _check_n_sim(n_sim)
    m = graph.size
    mu = np.asarray(means, dtype=float)
    if mu.shape != (m,) or np.any(~np.isfinite(mu)):
        raise ValueError(f"give {m} finite means, one per hypothesis")
    if np.ndim(correlation) == 0:
        r = float(correlation)  # type: ignore[arg-type]
        corr = np.full((m, m), r)
        np.fill_diagonal(corr, 1.0)
    else:
        corr = np.asarray(correlation, dtype=float)
    if corr.shape != (m, m) or not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1):
        raise ValueError(f"the correlation must be a symmetric {m} x {m} matrix with unit diagonal")
    eig = np.linalg.eigvalsh(corr)
    if eig.min() < -1e-10:
        raise ValueError("the correlation matrix must be positive semi-definite")
    # a factor that works for singular matrices too
    vals, vecs = np.linalg.eigh(corr)
    factor = vecs * np.sqrt(np.clip(vals, 0.0, None))

    if isinstance(seed, np.random.Generator):
        rng, entropy = seed, 0
    else:
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        rng, entropy = np.random.default_rng(ss), int(ss.entropy)  # type: ignore[arg-type]
    hits = np.zeros(m)
    any_hits = all_hits = 0
    total = 0.0
    for start in range(0, n_sim, _CHUNK):
        k = min(_CHUNK, n_sim - start)
        z = mu + rng.standard_normal((k, m)) @ factor.T
        rej = _reject_batch(stats.norm.sf(z), graph, alpha)
        hits += rej.sum(axis=0)
        any_hits += int(rej.any(axis=1).sum())
        all_hits += int(rej.all(axis=1).sum())
        total += float(rej.sum())

    def se(x: NDArray | float) -> NDArray | float:
        return np.sqrt(np.asarray(x) * (1 - np.asarray(x)) / n_sim)

    local = hits / n_sim
    p_any, p_all = any_hits / n_sim, all_hits / n_sim
    return GraphPower(
        graph=graph,
        alpha_one_sided=alpha,
        means=mu,
        correlation=corr,
        local_power=local,
        local_power_se=np.asarray(se(local)),
        any_power=p_any,
        any_power_se=float(se(p_any)),
        all_power=p_all,
        all_power_se=float(se(p_all)),
        expected_rejections=total / n_sim,
        n_sim=n_sim,
        seed=entropy,
    )
